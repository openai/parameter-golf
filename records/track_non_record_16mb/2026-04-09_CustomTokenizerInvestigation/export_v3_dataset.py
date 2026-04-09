from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable

import numpy as np

DATAFILE_MAGIC = 20240520
DATAFILE_VERSION = 1
DEFAULT_NUM_VAL_DOCS = 50_000
DEFAULT_SHARD_SIZE_TOKENS = 100_000_000
DEFAULT_BATCH_DOCS = 512


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def iter_docs(path: Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            yield obj["text"]


def write_datafile(path: Path, toks: np.ndarray) -> None:
    if toks.dtype != np.uint16:
        raise ValueError(f"expected uint16 tokens, got {toks.dtype}")
    if len(toks) >= 2**31:
        raise ValueError("token count too large for shard header")

    header = np.zeros(256, dtype="<i4")
    header[0] = DATAFILE_MAGIC
    header[1] = DATAFILE_VERSION
    header[2] = len(toks)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(header.tobytes())
        f.write(toks.astype("<u2", copy=False).tobytes())


def is_byte_piece(piece: str) -> bool:
    return piece.startswith("<0x") and piece.endswith(">")


def is_boundary_char(ch: str) -> bool:
    return ch.isspace() or ch in {".", ",", "!", "?", ";", ":", "(", ")", "[", "]", "{", "}", '"', "'"}


def boundary_penalty_for_fallback(ch: str) -> int:
    if ch == "\n":
        return 4
    if ch == "\t":
        return 3
    if ch == " ":
        return 3
    if is_boundary_char(ch):
        return 2
    return 0


class FastCustomDPTokenizer:
    """
    Boundary-aware DP tokenizer for a frozen custom vocab.

    Objective:
    1. minimize token count
    2. minimize fallback token count
    3. minimize fallback runs
    4. minimize boundary fallback penalty
    5. prefer longer first piece
    """

    def __init__(self, vocab_rows: list[dict]):
        self.rows = vocab_rows
        self.id_to_piece_map = {int(row["id"]): row["piece"] for row in vocab_rows}
        self.piece_to_id_map = {row["piece"]: int(row["id"]) for row in vocab_rows}

        if not self.piece_to_id_map:
            raise ValueError("empty vocab")

        self.vocab_size = max(self.id_to_piece_map) + 1
        if self.vocab_size > 65535:
            raise ValueError(f"vocab too large for uint16 shards: {self.vocab_size}")

        self.byte_piece_to_id: dict[int, int] = {}
        for b in range(256):
            piece = f"<0x{b:02X}>"
            pid = self.piece_to_id_map.get(piece)
            if pid is not None:
                self.byte_piece_to_id[b] = pid

        missing = [b for b in range(256) if b not in self.byte_piece_to_id]
        if missing:
            raise ValueError(f"missing byte fallback pieces for bytes: {missing[:10]}")

        candidates_by_first_char: dict[str, list[tuple[str, int, int]]] = defaultdict(list)

        for piece, pid in self.piece_to_id_map.items():
            if not piece or is_byte_piece(piece):
                continue
            L = len(piece)
            candidates_by_first_char[piece[0]].append((piece, pid, L))

        for ch in candidates_by_first_char:
            candidates_by_first_char[ch].sort(key=lambda x: x[2], reverse=True)

        self.candidates_by_first_char = dict(candidates_by_first_char)
        self._char_fallback_cache: dict[str, list[int]] = {}

    def _byte_fallback_ids_for_char(self, ch: str) -> list[int]:
        cached = self._char_fallback_cache.get(ch)
        if cached is not None:
            return cached

        out: list[int] = []
        for b in ch.encode("utf-8"):
            pid = self.byte_piece_to_id.get(b)
            if pid is None:
                raise ValueError(
                    f"missing byte fallback piece for char={repr(ch)} byte=<0x{b:02X}>"
                )
            out.append(pid)

        self._char_fallback_cache[ch] = out
        return out

    def encode_dp(self, text: str) -> list[int]:
        n = len(text)
        if n == 0:
            return []

        inf = 10**9
        sentinel = (inf, inf, inf, inf, 0)

        dp_score = [sentinel for _ in range(n + 1)]
        dp_next: list[tuple[str, int | list[int], int] | None] = [None] * (n + 1)
        dp_kind = [""] * (n + 1)
        dp_score[n] = (0, 0, 0, 0, 0)
        dp_kind[n] = "end"

        for i in range(n - 1, -1, -1):
            first_char = text[i]
            candidates = self.candidates_by_first_char.get(first_char, ())

            for piece, pid, L in candidates:
                if text.startswith(piece, i):
                    tail = dp_score[i + L]
                    score = (
                        1 + tail[0],
                        tail[1],
                        tail[2],
                        tail[3],
                        -L,
                    )
                    if score < dp_score[i]:
                        dp_score[i] = score
                        dp_next[i] = ("piece", pid, L)
                        dp_kind[i] = "piece"

            fallback_ids = self._byte_fallback_ids_for_char(first_char)
            tail = dp_score[i + 1]
            next_is_fallback = dp_kind[i + 1] == "bytes"
            new_run = 0 if next_is_fallback else 1
            fallback_len = len(fallback_ids)
            score = (
                fallback_len + tail[0],
                fallback_len + tail[1],
                new_run + tail[2],
                boundary_penalty_for_fallback(first_char) + tail[3],
                -1,
            )
            if score < dp_score[i]:
                dp_score[i] = score
                dp_next[i] = ("bytes", fallback_ids, 1)
                dp_kind[i] = "bytes"

        ids: list[int] = []
        i = 0
        while i < n:
            choice = dp_next[i]
            if choice is None:
                raise RuntimeError(f"DP reconstruction failed at position {i}")
            kind, payload, advance = choice
            if kind == "piece":
                ids.append(int(payload))
            else:
                ids.extend(payload)  # type: ignore[arg-type]
            i += advance

        return ids


_TOKENIZER: FastCustomDPTokenizer | None = None
_BOS_ID: int | None = None
_EOS_ID: int | None = None
_APPEND_EOS: bool = False


def _worker_init(vocab_rows: list[dict], bos_id: int | None, eos_id: int | None, append_eos: bool) -> None:
    global _TOKENIZER, _BOS_ID, _EOS_ID, _APPEND_EOS
    _TOKENIZER = FastCustomDPTokenizer(vocab_rows)
    _BOS_ID = bos_id
    _EOS_ID = eos_id
    _APPEND_EOS = append_eos


def _tokenize_batch(batch: list[tuple[int, str]]) -> list[tuple[int, np.ndarray]]:
    global _TOKENIZER, _BOS_ID, _EOS_ID, _APPEND_EOS
    if _TOKENIZER is None:
        raise RuntimeError("worker tokenizer not initialized")

    out: list[tuple[int, np.ndarray]] = []

    for doc_idx, text in batch:
        ids: list[int] = []
        if _BOS_ID is not None:
            ids.append(_BOS_ID)
        ids.extend(_TOKENIZER.encode_dp(text))
        if _APPEND_EOS:
            if _EOS_ID is None:
                raise ValueError("--append-eos was set but no --eos-id provided")
            ids.append(_EOS_ID)

        out.append((doc_idx, np.asarray(ids, dtype=np.uint16)))

    return out


def batched_docs(docs_jsonl: Path, batch_docs: int) -> Iterable[list[tuple[int, str]]]:
    batch: list[tuple[int, str]] = []
    for doc_idx, text in enumerate(iter_docs(docs_jsonl)):
        batch.append((doc_idx, text))
        if len(batch) >= batch_docs:
            yield batch
            batch = []
    if batch:
        yield batch


def export_shards_parallel(
    *,
    docs_jsonl: Path,
    vocab_rows: list[dict],
    output_dir: Path,
    num_val_docs: int,
    shard_size_tokens: int,
    bos_id: int | None,
    eos_id: int | None,
    append_eos: bool,
    workers: int,
    batch_docs: int,
    max_shards: int | None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    for pattern in ("fineweb_train_*.bin", "fineweb_val_*.bin"):
        for stale in output_dir.glob(pattern):
            stale.unlink()

    stats = {
        "docs_total": 0,
        "docs_val": 0,
        "docs_train": 0,
        "files_total": 0,
        "files_val": 0,
        "files_train": 0,
        "tokens_total": 0,
        "tokens_val": 0,
        "tokens_train": 0,
        "stopped_early": False,
        "max_shards": max_shards,
    }

    split = "val"
    shard_idx = {"val": 0, "train": 0}
    buf = np.empty((shard_size_tokens,), dtype=np.uint16)
    fill = 0
    stop_requested = False

    def flush(current_split: str) -> None:
        nonlocal fill, stop_requested

        if fill == 0:
            return

        if max_shards is not None and stats["files_total"] >= max_shards:
            stop_requested = True
            return

        out_path = output_dir / f"fineweb_{current_split}_{shard_idx[current_split]:06d}.bin"
        write_datafile(out_path, buf[:fill])

        print(
            f"[flush] wrote {out_path.name} tokens={fill:,} size_bytes={out_path.stat().st_size:,}",
            flush=True,
        )

        stats["files_total"] += 1
        stats[f"files_{current_split}"] += 1
        shard_idx[current_split] += 1
        fill = 0

        if max_shards is not None and stats["files_total"] >= max_shards:
            stop_requested = True

    start_t = time.time()
    next_log_docs = 1_000

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_worker_init,
        initargs=(vocab_rows, bos_id, eos_id, append_eos),
    ) as ex:
        for tokenized_batch in ex.map(_tokenize_batch, batched_docs(docs_jsonl, batch_docs), chunksize=1):
            if stop_requested:
                break

            for doc_idx, ids_np in tokenized_batch:
                if stop_requested:
                    break

                split_for_doc = "val" if doc_idx < num_val_docs else "train"
                if split_for_doc != split:
                    flush(split)
                    if stop_requested:
                        break
                    split = split_for_doc

                stats["docs_total"] += 1
                stats[f"docs_{split}"] += 1
                stats["tokens_total"] += len(ids_np)
                stats[f"tokens_{split}"] += len(ids_np)

                pos = 0
                while pos < len(ids_np):
                    remaining_capacity = shard_size_tokens - fill
                    take = min(remaining_capacity, len(ids_np) - pos)
                    buf[fill:fill + take] = ids_np[pos:pos + take]
                    fill += take
                    pos += take

                    if fill == shard_size_tokens:
                        flush(split)
                        if stop_requested:
                            break

                if stop_requested:
                    break

                if stats["docs_total"] >= next_log_docs:
                    elapsed = time.time() - start_t
                    docs_per_sec = stats["docs_total"] / max(elapsed, 1e-9)
                    toks_per_sec = stats["tokens_total"] / max(elapsed, 1e-9)
                    print(
                        f"docs={stats['docs_total']:,} "
                        f"train_docs={stats['docs_train']:,} "
                        f"val_docs={stats['docs_val']:,} "
                        f"tokens={stats['tokens_total']:,} "
                        f"files={stats['files_total']:,} "
                        f"docs/sec={docs_per_sec:,.1f} "
                        f"tok/sec={toks_per_sec:,.1f}",
                        flush=True,
                    )
                    if next_log_docs < 10_000:
                        next_log_docs += 1_000
                    else:
                        next_log_docs += 10_000

    if stop_requested:
        stats["stopped_early"] = True
        print(f"Reached max_shards={max_shards}, stopping early.", flush=True)
    else:
        flush(split)

    return stats


def build_manifest(
    *,
    output_dir: Path,
    vocab_path: Path,
    stats: dict,
    num_val_docs: int,
    shard_size_tokens: int,
    bos_id: int | None,
    eos_id: int | None,
    append_eos: bool,
    vocab_size: int,
    workers: int,
    batch_docs: int,
    max_shards: int | None,
) -> None:
    manifest = {
        "name": output_dir.name,
        "tokenizer_kind": "custom_dp_jsonl",
        "tokenizer_path": str(vocab_path),
        "vocab_size": vocab_size,
        "bos_id": bos_id,
        "eos_id": eos_id,
        "append_eos": append_eos,
        "num_val_docs": num_val_docs,
        "shard_size_tokens": shard_size_tokens,
        "workers": workers,
        "batch_docs": batch_docs,
        "max_shards": max_shards,
        "scoring_mode": "boundary_aware_v1",
        "stats": stats,
        "train_glob": str(output_dir / "fineweb_train_*.bin"),
        "val_glob": str(output_dir / "fineweb_val_*.bin"),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args():
    p = argparse.ArgumentParser(description="Export FineWeb docs into custom DP-tokenized shards (multiprocess)")
    p.add_argument("--docs-jsonl", default="data/docs_selected.jsonl", help="Path to docs_selected.jsonl")
    p.add_argument("--vocab-jsonl", default="vocab/vocab_best.jsonl", help="Path to frozen custom vocab JSONL")
    p.add_argument("--output-dir", default="data/datasets/fineweb10B_customdp1024_best", help="Output dataset directory")
    p.add_argument("--num-val-docs", type=int, default=DEFAULT_NUM_VAL_DOCS, help="Number of validation docs from the front of docs_selected.jsonl")
    p.add_argument("--shard-size-tokens", type=int, default=DEFAULT_SHARD_SIZE_TOKENS, help="Max tokens per shard")
    p.add_argument("--bos-id", type=int, default=None, help="Optional BOS token id to prepend to every doc")
    p.add_argument("--eos-id", type=int, default=None, help="Optional EOS token id")
    p.add_argument("--append-eos", action="store_true", help="Append EOS to every doc")
    p.add_argument("--workers", type=int, default=max(1, min(16, (os.cpu_count() or 8) - 1)), help="Number of worker processes")
    p.add_argument("--batch-docs", type=int, default=DEFAULT_BATCH_DOCS, help="Documents per worker batch")
    p.add_argument("--max-shards", type=int, default=None, help="Optional cap on total number of shards to write")
    return p.parse_args()


def main():
    args = parse_args()

    docs_jsonl = Path(args.docs_jsonl).resolve()
    vocab_jsonl = Path(args.vocab_jsonl).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not docs_jsonl.is_file():
        raise FileNotFoundError(f"docs file not found: {docs_jsonl}")
    if not vocab_jsonl.is_file():
        raise FileNotFoundError(f"vocab file not found: {vocab_jsonl}")

    vocab_rows = load_jsonl(vocab_jsonl)
    tokenizer_probe = FastCustomDPTokenizer(vocab_rows)

    print(f"docs_jsonl:   {docs_jsonl}")
    print(f"vocab_jsonl:  {vocab_jsonl}")
    print(f"output_dir:   {output_dir}")
    print(f"vocab_size:   {tokenizer_probe.vocab_size}")
    print(f"num_val_docs: {args.num_val_docs}")
    print(f"shard_size:   {args.shard_size_tokens}")
    print(f"bos_id:       {args.bos_id}")
    print(f"eos_id:       {args.eos_id}")
    print(f"append_eos:   {args.append_eos}")
    print(f"workers:      {args.workers}")
    print(f"batch_docs:   {args.batch_docs}")
    print(f"max_shards:   {args.max_shards}")
    print("scoring_mode: boundary_aware_v1")
    print()

    t0 = time.time()

    stats = export_shards_parallel(
        docs_jsonl=docs_jsonl,
        vocab_rows=vocab_rows,
        output_dir=output_dir,
        num_val_docs=args.num_val_docs,
        shard_size_tokens=args.shard_size_tokens,
        bos_id=args.bos_id,
        eos_id=args.eos_id,
        append_eos=args.append_eos,
        workers=args.workers,
        batch_docs=args.batch_docs,
        max_shards=args.max_shards,
    )

    build_manifest(
        output_dir=output_dir,
        vocab_path=vocab_jsonl,
        stats=stats,
        num_val_docs=args.num_val_docs,
        shard_size_tokens=args.shard_size_tokens,
        bos_id=args.bos_id,
        eos_id=args.eos_id,
        append_eos=args.append_eos,
        vocab_size=tokenizer_probe.vocab_size,
        workers=args.workers,
        batch_docs=args.batch_docs,
        max_shards=args.max_shards,
    )

    elapsed = time.time() - t0

    print("\nDONE")
    print(f"elapsed_sec: {elapsed:,.2f}")
    for k, v in stats.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
