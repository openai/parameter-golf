from __future__ import annotations

import argparse
import io
import json
import os
import urllib.request
from pathlib import Path

import numpy as np


DOCS_FILENAME = "docs_selected.jsonl"
SIDECAR_FILENAME = "docs_selected.source_manifest.json"
DATAFILE_MAGIC = 20240520
DATAFILE_VERSION = 1
SHARD_SIZE = 10**8
DEFAULT_REPO_ID = os.environ.get("MATCHED_FINEWEB_REPO_ID", "willdepueoai/parameter-golf")
DEFAULT_REMOTE_ROOT = os.environ.get("MATCHED_FINEWEB_REMOTE_ROOT_PREFIX", "datasets")
DEFAULT_NUM_VAL_DOCS = 50_000
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3
BYTE_OFFSET = 4


def hf_resolve_url(repo_id: str, remote_root: str, filename: str) -> str:
    remote_path = Path(remote_root) / filename if remote_root else Path(filename)
    return f"https://huggingface.co/datasets/{repo_id}/resolve/main/{remote_path.as_posix()}"


def open_remote_text(repo_id: str, remote_root: str, filename: str):
    response = urllib.request.urlopen(hf_resolve_url(repo_id, remote_root, filename), timeout=300)
    return io.TextIOWrapper(response, encoding="utf-8")


def maybe_download_json(repo_id: str, remote_root: str, filename: str) -> dict | None:
    try:
        with urllib.request.urlopen(hf_resolve_url(repo_id, remote_root, filename), timeout=60) as src:
            return json.load(src)
    except Exception:
        return None


def write_datafile(path: Path, toks: np.ndarray) -> None:
    header = np.zeros(256, dtype="<i4")
    header[0] = DATAFILE_MAGIC
    header[1] = DATAFILE_VERSION
    header[2] = int(len(toks))
    with path.open("wb") as f:
        f.write(header.tobytes())
        f.write(toks.astype("<u2", copy=False).tobytes())


def byte_encode(text: str) -> np.ndarray:
    data = text.encode("utf-8", errors="replace")
    return np.frombuffer(data, dtype=np.uint8).astype(np.uint16, copy=False) + BYTE_OFFSET


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the minimum byte260 subset needed for a Runpod experiment")
    parser.add_argument("--output-dir", required=True, help="Directory for fineweb_train_*.bin and fineweb_val_*.bin")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--remote-root", default=DEFAULT_REMOTE_ROOT)
    parser.add_argument("--train-shards", type=int, required=True)
    parser.add_argument("--num-val-docs", type=int, default=None)
    parser.add_argument("--chunk-tokens", type=int, default=SHARD_SIZE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.train_shards < 0:
        raise ValueError("--train-shards must be non-negative")
    if args.chunk_tokens <= 0:
        raise ValueError("--chunk-tokens must be positive")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in output_dir.glob("fineweb_*_*.bin"):
        stale.unlink()

    if args.num_val_docs is not None:
        num_val_docs = int(args.num_val_docs)
    else:
        payload = maybe_download_json(args.repo_id, args.remote_root, SIDECAR_FILENAME)
        num_val_docs = int(payload.get("docs_val", DEFAULT_NUM_VAL_DOCS)) if payload else DEFAULT_NUM_VAL_DOCS

    target_train_tokens = args.train_shards * int(args.chunk_tokens)
    chunk_tokens = int(args.chunk_tokens)
    buf = np.empty((chunk_tokens,), dtype=np.uint16)
    fill = 0
    split = "val"
    shard_idx = {"val": 0, "train": 0}
    stats = {
        "docs_total_seen": 0,
        "docs_val": 0,
        "docs_train": 0,
        "tokens_val": 0,
        "tokens_train": 0,
        "files_val": 0,
        "files_train": 0,
    }

    def flush() -> None:
        nonlocal fill
        if fill == 0:
            return
        path = output_dir / f"fineweb_{split}_{shard_idx[split]:06d}.bin"
        write_datafile(path, buf[:fill])
        shard_idx[split] += 1
        stats[f"files_{split}"] += 1
        fill = 0

    with open_remote_text(args.repo_id, args.remote_root, DOCS_FILENAME) as f:
        for doc_idx, line in enumerate(f):
            doc = json.loads(line)
            text = doc["text"]
            raw_bytes = text.encode("utf-8", errors="replace")
            toks = np.empty((len(raw_bytes) + 1,), dtype=np.uint16)
            toks[0] = BOS_ID
            toks[1:] = np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.uint16, copy=False) + BYTE_OFFSET
            split_for_doc = "val" if doc_idx < num_val_docs else "train"
            if split_for_doc != split:
                flush()
                split = split_for_doc
            stats["docs_total_seen"] += 1
            stats[f"docs_{split}"] += 1

            pos = 0
            while pos < len(toks):
                if split == "train" and stats["tokens_train"] >= target_train_tokens:
                    break
                remaining = len(toks) - pos
                if split == "train":
                    remaining_train_budget = target_train_tokens - stats["tokens_train"]
                    if remaining_train_budget <= 0:
                        break
                    remaining = min(remaining, remaining_train_budget)
                take = min(chunk_tokens - fill, remaining)
                buf[fill : fill + take] = toks[pos : pos + take]
                fill += take
                pos += take
                stats[f"tokens_{split}"] += take
                if fill == chunk_tokens:
                    flush()

            if split == "train" and stats["tokens_train"] >= target_train_tokens:
                break

            if stats["docs_total_seen"] and stats["docs_total_seen"] % 10_000 == 0:
                print(
                    f"docs_seen={stats['docs_total_seen']} val_tokens={stats['tokens_val']} train_tokens={stats['tokens_train']}",
                    flush=True,
                )

    flush()
    summary = {
        "repo_id": args.repo_id,
        "remote_root": args.remote_root,
        "output_dir": str(output_dir),
        "train_shards_requested": args.train_shards,
        "chunk_tokens": chunk_tokens,
        "num_val_docs": num_val_docs,
        "pad_id": PAD_ID,
        "bos_id": BOS_ID,
        "eos_id": EOS_ID,
        "unk_id": UNK_ID,
        "byte_offset": BYTE_OFFSET,
        **stats,
    }
    (output_dir / "bootstrap_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
