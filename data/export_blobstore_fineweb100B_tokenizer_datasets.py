"""Export matched tokenizer datasets from raw FineWeb blobstore shards."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tiktoken

from export_matched_fineweb_tokenizer_datasets import DEMO_CONFIG, build_parser as build_matched_parser
from export_matched_fineweb_tokenizer_datasets import run_export as run_matched_export


RAW_EOT_TOKEN = 50256
RAW_DATAFILE_MAGIC = 20240520
RAW_DATAFILE_VERSION = 1
RAW_HEADER_BYTES = 256 * 4
DEFAULT_SOURCE_ROOT = "az://oaidatasets2/speedrunkits/fineweb100B"
DEFAULT_TARGET_TRAIN_TOKENS = 30_000_000_000
DEFAULT_SP_VOCAB_SIZES = "512,1024,2048,4096"


@dataclass(frozen=True)
class BlobShard:
    path: str
    split: str
    index: int
    size_bytes: int
    tokens: int


def run_command(args: list[str]) -> str:
    return subprocess.run(args, check=True, text=True, capture_output=True).stdout


def list_blobstore_shards(source_root: str) -> list[BlobShard]:
    output = run_command(["bbb", "ls", "-l", "--machine", source_root])
    shards: list[BlobShard] = []
    for line in output.splitlines():
        match = re.match(r"^\s*(\d+)\s+\S+\s+(.+)$", line)
        if match is None:
            continue
        size_bytes = int(match.group(1))
        path = match.group(2).strip()
        shard_match = re.search(r"fineweb_(train|val)_(\d+)\.bin$", path)
        if shard_match is None:
            continue
        if size_bytes < RAW_HEADER_BYTES or (size_bytes - RAW_HEADER_BYTES) % 2 != 0:
            raise ValueError(f"unexpected shard size for {path}: {size_bytes}")
        shards.append(
            BlobShard(
                path=path,
                split=shard_match.group(1),
                index=int(shard_match.group(2)),
                size_bytes=size_bytes,
                tokens=(size_bytes - RAW_HEADER_BYTES) // 2,
            )
        )
    if not shards:
        raise ValueError(f"no raw shards found under {source_root}")
    return sorted(shards, key=lambda shard: (shard.split, shard.index))


def select_even(shards: list[BlobShard], count: int) -> list[BlobShard]:
    if count <= 0:
        return []
    if count >= len(shards):
        return list(shards)
    if count == 1:
        return [shards[0]]
    return [shards[round(i * (len(shards) - 1) / (count - 1))] for i in range(count)]


def select_shards(
    shards: list[BlobShard],
    *,
    count: int,
    selection_mode: str,
    selection_seed: int,
) -> list[BlobShard]:
    if count < 0:
        raise ValueError(f"count must be non-negative, got {count}")
    if count > len(shards):
        raise ValueError(f"requested {count} shards but only {len(shards)} available")
    if selection_mode == "first":
        return list(shards[:count])
    if selection_mode == "random":
        rng = random.Random(selection_seed)
        return sorted(rng.sample(shards, count), key=lambda shard: shard.index)
    if selection_mode == "even":
        return select_even(shards, count)
    raise ValueError(f"unsupported selection_mode={selection_mode}")


def selected_train_shards(
    train_shards: list[BlobShard],
    *,
    requested_train_shards: int | None,
    target_train_tokens: int,
    selection_mode: str,
    selection_seed: int,
) -> list[BlobShard]:
    if requested_train_shards is not None:
        return select_shards(
            train_shards,
            count=requested_train_shards,
            selection_mode=selection_mode,
            selection_seed=selection_seed,
        )
    avg_tokens = sum(shard.tokens for shard in train_shards) / len(train_shards)
    count = max(1, math.ceil(target_train_tokens / avg_tokens))
    while True:
        selected = select_shards(
            train_shards,
            count=min(count, len(train_shards)),
            selection_mode=selection_mode,
            selection_seed=selection_seed,
        )
        if sum(shard.tokens for shard in selected) >= target_train_tokens or len(selected) == len(train_shards):
            return selected
        count += 1


def contiguous_runs(shards: list[BlobShard]) -> list[list[BlobShard]]:
    if not shards:
        return []
    runs = [[shards[0]]]
    for shard in shards[1:]:
        if shard.index == runs[-1][-1].index + 1:
            runs[-1].append(shard)
        else:
            runs.append([shard])
    return runs


def read_blob_tokens(path: str) -> np.ndarray:
    raw = subprocess.run(["bbb", "cat", path], check=True, stdout=subprocess.PIPE).stdout
    if len(raw) < RAW_HEADER_BYTES:
        raise ValueError(f"raw shard is too short: {path}")
    header = np.frombuffer(raw[:RAW_HEADER_BYTES], dtype="<i4", count=256)
    if int(header[0]) != RAW_DATAFILE_MAGIC or int(header[1]) != RAW_DATAFILE_VERSION:
        raise ValueError(f"unexpected raw shard header for {path}: {header[:3].tolist()}")
    token_count = int(header[2])
    tokens = np.frombuffer(raw, dtype="<u2", offset=RAW_HEADER_BYTES, count=token_count)
    if tokens.size != token_count:
        raise ValueError(f"raw shard token count mismatch for {path}: expected {token_count}, got {tokens.size}")
    return tokens


def run_summary(run: list[BlobShard], *, order: int, starts_at_boundary: bool) -> dict[str, Any]:
    return {
        "order": order,
        "start_index": run[0].index,
        "end_index": run[-1].index,
        "starts_at_boundary": starts_at_boundary,
        "tokens": sum(shard.tokens for shard in run),
        "shards": [shard.path for shard in run],
    }


def write_docs_from_runs(
    *,
    split: str,
    runs: list[list[BlobShard]],
    first_available_index: int,
    writer,
    encoder,
    max_docs: int | None,
    target_raw_tokens: int | None = None,
    drop_leading_empty_doc: bool = False,
) -> tuple[int, int, list[dict[str, Any]]]:
    docs_written = 0
    raw_tokens_written = 0
    summaries: list[dict[str, Any]] = []
    for order, run in enumerate(runs):
        starts_at_boundary = run[0].index == first_available_index
        seeking_boundary = not starts_at_boundary
        carry = np.empty((0,), dtype=np.uint16)
        docs_before = docs_written
        raw_tokens_before = raw_tokens_written
        skipped_leading_empty_docs = 0
        for shard in run:
            tokens = read_blob_tokens(shard.path)
            if seeking_boundary:
                eos_positions = np.flatnonzero(tokens == RAW_EOT_TOKEN)
                if eos_positions.size == 0:
                    continue
                tokens = tokens[eos_positions[0] + 1 :]
                seeking_boundary = False
            start = 0
            eos_positions = np.flatnonzero(tokens == RAW_EOT_TOKEN)
            for eos in eos_positions:
                piece = tokens[start:eos]
                doc_tokens = piece
                if carry.size:
                    doc_tokens = np.concatenate((carry, piece)) if piece.size else carry
                    carry = np.empty((0,), dtype=np.uint16)
                raw_tokens = int(doc_tokens.size)
                if (
                    drop_leading_empty_doc
                    and docs_written == 0
                    and raw_tokens_written == 0
                    and raw_tokens == 0
                    and starts_at_boundary
                ):
                    skipped_leading_empty_docs += 1
                    start = eos + 1
                    continue
                text = encoder.decode(doc_tokens.tolist())
                writer.write(json.dumps({"text": text}, ensure_ascii=False))
                writer.write("\n")
                docs_written += 1
                raw_tokens_written += raw_tokens
                if max_docs is not None and docs_written >= max_docs:
                    start = eos + 1
                    break
                if target_raw_tokens is not None and raw_tokens_written >= target_raw_tokens:
                    start = eos + 1
                    break
                start = eos + 1
            if max_docs is not None and docs_written >= max_docs:
                break
            if target_raw_tokens is not None and raw_tokens_written >= target_raw_tokens:
                break
            remainder = tokens[start:]
            if carry.size:
                carry = np.concatenate((carry, remainder)) if remainder.size else carry
            else:
                carry = remainder.copy()
        summaries.append(
            {
                "split": split,
                **run_summary(run, order=order, starts_at_boundary=starts_at_boundary),
                "docs_emitted": docs_written - docs_before,
                "raw_tokens_emitted": raw_tokens_written - raw_tokens_before,
                "skipped_leading_empty_docs": skipped_leading_empty_docs,
                "trailing_tokens_dropped": int(carry.size),
            }
        )
        if max_docs is not None and docs_written >= max_docs:
            break
        if target_raw_tokens is not None and raw_tokens_written >= target_raw_tokens:
            break
    return docs_written, raw_tokens_written, summaries


def build_docs_cache_from_blobstore(
    *,
    source_root: str,
    docs_jsonl: Path,
    selected_val: list[BlobShard],
    selected_train: list[BlobShard],
    selection_mode: str,
    selection_seed: int,
    target_train_tokens: int,
    requested_train_shards: int | None,
    requested_num_val_docs: int | None,
    max_docs_per_split: int | None,
) -> dict[str, Any]:
    encoder = tiktoken.get_encoding("gpt2")
    val_runs = contiguous_runs(selected_val)
    train_runs = contiguous_runs(selected_train)
    shuffled_train_runs = list(train_runs)
    random.Random(selection_seed).shuffle(shuffled_train_runs)

    docs_jsonl.parent.mkdir(parents=True, exist_ok=True)
    val_doc_limit = requested_num_val_docs
    if max_docs_per_split is not None:
        val_doc_limit = max_docs_per_split if val_doc_limit is None else min(val_doc_limit, max_docs_per_split)
    with docs_jsonl.open("w", encoding="utf-8") as writer:
        docs_val, raw_tokens_val, val_summaries = write_docs_from_runs(
            split="val",
            runs=val_runs,
            first_available_index=min(shard.index for shard in selected_val),
            writer=writer,
            encoder=encoder,
            max_docs=val_doc_limit,
            drop_leading_empty_doc=requested_num_val_docs is not None,
        )
        docs_train, raw_tokens_train, train_summaries = write_docs_from_runs(
            split="train",
            runs=shuffled_train_runs,
            first_available_index=min(shard.index for shard in selected_train),
            writer=writer,
            encoder=encoder,
            max_docs=max_docs_per_split,
            target_raw_tokens=target_train_tokens if requested_num_val_docs is not None else None,
        )
    if requested_num_val_docs is not None and max_docs_per_split is None and docs_val != requested_num_val_docs:
        raise ValueError(f"expected {requested_num_val_docs} val docs, wrote {docs_val}")
    if requested_num_val_docs is not None and max_docs_per_split is None and raw_tokens_train < target_train_tokens:
        raise ValueError(
            f"expected at least {target_train_tokens} raw train tokens, only collected {raw_tokens_train}"
        )

    metadata = {
        "source_root": source_root,
        "selection_mode": selection_mode,
        "selection_seed": selection_seed,
        "shuffle_seed": selection_seed,
        "target_train_tokens": target_train_tokens,
        "requested_train_shards": requested_train_shards,
        "requested_num_val_docs": requested_num_val_docs,
        "selected_train_shards": len(selected_train),
        "selected_val_shards": len(selected_val),
        "selected_train_tokens": sum(shard.tokens for shard in selected_train),
        "selected_val_tokens": sum(shard.tokens for shard in selected_val),
        "selected_train_indices": [shard.index for shard in selected_train],
        "selected_val_indices": [shard.index for shard in selected_val],
        "max_docs_per_split": max_docs_per_split,
        "docs_train": docs_train,
        "docs_val": docs_val,
        "raw_tokens_train": raw_tokens_train,
        "raw_tokens_val": raw_tokens_val,
        "num_docs": docs_train + docs_val,
        "val_policy": (
            f"first_{requested_num_val_docs}_docs_from_blobstore_val_stream"
            if requested_num_val_docs is not None
            else "all_docs_from_selected_val_runs"
        ),
        "train_policy": (
            "shuffled_selected_train_runs_until_raw_gpt2_target_met"
            if requested_num_val_docs is not None
            else "all_docs_from_shuffled_selected_train_runs"
        ),
        "train_runs": train_summaries,
        "val_runs": val_summaries,
    }
    sidecar_path = docs_jsonl.with_name(f"{docs_jsonl.stem}.source_manifest.json")
    sidecar_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def maybe_reuse_existing_docs_cache(
    *,
    docs_jsonl: Path,
    source_root: str,
    selection_mode: str,
    selection_seed: int,
    target_train_tokens: int,
    requested_train_shards: int | None,
    val_shards: int,
    requested_num_val_docs: int | None,
    max_docs_per_split: int | None,
) -> dict[str, Any] | None:
    sidecar_path = docs_jsonl.with_name(f"{docs_jsonl.stem}.source_manifest.json")
    if not docs_jsonl.is_file() or not sidecar_path.is_file():
        return None
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    expected = {
        "source_root": source_root,
        "selection_mode": selection_mode,
        "selection_seed": selection_seed,
        "target_train_tokens": target_train_tokens,
        "requested_train_shards": requested_train_shards,
        "selected_val_shards": val_shards,
        "requested_num_val_docs": requested_num_val_docs,
        "max_docs_per_split": max_docs_per_split,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(
                f"existing docs cache metadata mismatch for {key}: expected {value}, got {payload.get(key)}. "
                "Use --rebuild_docs_cache or a fresh --output_root."
            )
    return payload


def parse_vocab_sizes(value: str) -> list[int]:
    vocab_sizes = [int(piece) for piece in value.split(",") if piece]
    if not vocab_sizes:
        raise ValueError("--sp_vocab_sizes must specify at least one vocab size")
    return vocab_sizes


def write_tokenizer_config(output_root: Path, *, sp_vocab_sizes: list[int], skip_byte: bool) -> Path:
    payload = json.loads(DEMO_CONFIG.read_text(encoding="utf-8"))
    specs = payload["tokenizers"] if isinstance(payload, dict) else payload
    vocab_size_set = set(sp_vocab_sizes)
    selected_specs = []
    found_sp_vocab_sizes: set[int] = set()
    for spec in specs:
        if "vocab_size" in spec:
            vocab_size = int(spec["vocab_size"])
            if vocab_size in vocab_size_set:
                selected_specs.append(spec)
                found_sp_vocab_sizes.add(vocab_size)
        elif not skip_byte:
            selected_specs.append(spec)
    missing = vocab_size_set - found_sp_vocab_sizes
    if missing:
        raise ValueError(f"unknown SentencePiece vocab sizes requested: {sorted(missing)}")
    tokenizer_config_path = output_root / "tokenizer_config.export.json"
    tokenizer_config_path.write_text(json.dumps({"tokenizers": selected_specs}, indent=2) + "\n", encoding="utf-8")
    return tokenizer_config_path


def augment_manifest(output_root: Path, source_blobstore: dict[str, Any]) -> None:
    manifest_path = output_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_blobstore"] = source_blobstore
    manifest["shuffle_seed"] = int(source_blobstore["shuffle_seed"])
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export matched tokenizer datasets from raw FineWeb blobstore shards")
    parser.add_argument("--source_root", default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--target_train_tokens", type=int, default=DEFAULT_TARGET_TRAIN_TOKENS)
    parser.add_argument("--train_shards", type=int, default=None)
    parser.add_argument("--val_shards", type=int, default=1)
    parser.add_argument("--num_val_docs", type=int, default=None)
    parser.add_argument("--selection_mode", choices=("even", "first", "random"), default="even")
    parser.add_argument("--selection_seed", type=int, default=1337)
    parser.add_argument("--chunk_tokens", type=int, default=None)
    parser.add_argument("--sp_vocab_sizes", default=DEFAULT_SP_VOCAB_SIZES)
    parser.add_argument("--tokenizer_train_docs", type=int, default=None)
    parser.add_argument("--reuse_sp_model", action="append", default=[])
    parser.add_argument("--skip_byte", action="store_true")
    parser.add_argument("--docs_only", action="store_true")
    parser.add_argument("--max_docs_per_split", type=int, default=None)
    parser.add_argument("--rebuild_docs_cache", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.train_shards is not None and args.target_train_tokens is not None:
        if args.target_train_tokens != DEFAULT_TARGET_TRAIN_TOKENS:
            raise ValueError("pass either --train_shards or --target_train_tokens, not both")
    if args.val_shards <= 0:
        raise ValueError(f"--val_shards must be positive, got {args.val_shards}")
    if args.num_val_docs is not None and args.num_val_docs <= 0:
        raise ValueError(f"--num_val_docs must be positive, got {args.num_val_docs}")
    if args.train_shards is not None and args.train_shards <= 0:
        raise ValueError(f"--train_shards must be positive, got {args.train_shards}")
    if args.target_train_tokens <= 0:
        raise ValueError(f"--target_train_tokens must be positive, got {args.target_train_tokens}")

    output_root = Path(args.output_root).expanduser().resolve()
    docs_jsonl = output_root / "docs_selected.jsonl"
    source_shards = list_blobstore_shards(args.source_root)
    train_candidates = [shard for shard in source_shards if shard.split == "train"]
    val_candidates = [shard for shard in source_shards if shard.split == "val"]
    if args.val_shards > len(val_candidates):
        raise ValueError(f"requested {args.val_shards} val shards but only {len(val_candidates)} available")

    selected_val = (
        list(val_candidates)
        if args.num_val_docs is not None
        else select_shards(
            val_candidates,
            count=args.val_shards,
            selection_mode="first",
            selection_seed=args.selection_seed,
        )
    )
    selected_train = selected_train_shards(
        train_candidates,
        requested_train_shards=args.train_shards,
        target_train_tokens=args.target_train_tokens,
        selection_mode=args.selection_mode,
        selection_seed=args.selection_seed,
    )

    if args.rebuild_docs_cache:
        docs_meta = build_docs_cache_from_blobstore(
            source_root=args.source_root,
            docs_jsonl=docs_jsonl,
            selected_val=selected_val,
            selected_train=selected_train,
            selection_mode=args.selection_mode,
            selection_seed=args.selection_seed,
            target_train_tokens=args.target_train_tokens,
            requested_train_shards=args.train_shards,
            requested_num_val_docs=args.num_val_docs,
            max_docs_per_split=args.max_docs_per_split,
        )
    else:
        docs_meta = maybe_reuse_existing_docs_cache(
            docs_jsonl=docs_jsonl,
            source_root=args.source_root,
            selection_mode=args.selection_mode,
            selection_seed=args.selection_seed,
            target_train_tokens=args.target_train_tokens,
            requested_train_shards=args.train_shards,
            val_shards=len(selected_val),
            requested_num_val_docs=args.num_val_docs,
            max_docs_per_split=args.max_docs_per_split,
        )
        if docs_meta is None:
            docs_meta = build_docs_cache_from_blobstore(
                source_root=args.source_root,
                docs_jsonl=docs_jsonl,
                selected_val=selected_val,
                selected_train=selected_train,
                selection_mode=args.selection_mode,
                selection_seed=args.selection_seed,
                target_train_tokens=args.target_train_tokens,
                requested_train_shards=args.train_shards,
                requested_num_val_docs=args.num_val_docs,
                max_docs_per_split=args.max_docs_per_split,
            )

    if args.docs_only:
        return

    tokenizer_config = write_tokenizer_config(
        output_root,
        sp_vocab_sizes=parse_vocab_sizes(args.sp_vocab_sizes),
        skip_byte=args.skip_byte,
    )
    matched_args = build_matched_parser().parse_args(
        [
            "--tokenizer_config",
            str(tokenizer_config),
            "--trust_tokenizer_config_code",
            "--output_root",
            str(output_root),
            "--docs_jsonl",
            str(docs_jsonl),
            "--num_val_docs",
            str(int(docs_meta["docs_val"])),
            *(["--chunk_tokens", str(args.chunk_tokens)] if args.chunk_tokens is not None else []),
            *(["--tokenizer_train_docs", str(args.tokenizer_train_docs)] if args.tokenizer_train_docs is not None else []),
            *(["--skip_byte"] if args.skip_byte else []),
            *sum((["--reuse_sp_model", value] for value in args.reuse_sp_model), []),
        ]
    )
    run_matched_export(matched_args)
    augment_manifest(output_root, docs_meta)


if __name__ == "__main__":
    main()
