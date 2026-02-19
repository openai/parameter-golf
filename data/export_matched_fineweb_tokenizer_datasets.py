"""Export matched FineWeb datasets across tokenizer variants.

This script guarantees all exported datasets use the exact same raw document
sequence by caching docs once, then tokenizing that cache for each tokenizer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import sentencepiece as spm
from datasets import load_dataset
from tqdm import tqdm

from pure_byte_tokenizer import PureByteTokenizer, default_pure_byte_tokenizer


def write_datafile(filename: str, toks: np.ndarray) -> None:
    """Saves token data as a .bin file: 256 int32 header + uint16 payload."""
    assert len(toks) < 2**31, "token count too large"
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20240520
    header[1] = 1
    header[2] = len(toks)

    if toks.dtype != np.uint16:
        assert (0 <= toks).all() and (toks < 2**16).all(), "token dictionary too large for uint16"
        toks = toks.astype(np.uint16)

    with open(filename, "wb") as f:
        f.write(header.tobytes())
        f.write(toks.tobytes())


def next_multiple_of_n(v: int, n: int) -> int:
    return ((v + n - 1) // n) * n


def sanitize_for_sentencepiece(text: str) -> str:
    return text.replace("\x00", " ").strip()


def parse_sp_vocab_sizes(raw: str) -> list[int]:
    out: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        val = int(part)
        if val <= 0:
            raise ValueError(f"Invalid vocab size: {val}")
        out.append(val)
    if not out:
        raise ValueError("At least one vocab size is required")
    return out


def iter_docs_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            yield row["text"]


def iter_docs_for_sentencepiece(path: Path):
    for text in iter_docs_jsonl(path):
        clean = sanitize_for_sentencepiece(text)
        if clean:
            yield clean


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def count_docs(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def build_docs_cache(
    out_path: Path,
    *,
    version: str,
    num_docs: int,
    shuffle_seed: int | None,
    dataset_revision: str | None,
) -> dict:
    remote_name = "sample-10BT" if version == "10B" else "sample-100BT"
    print(
        f"Loading dataset HuggingFaceFW/fineweb name={remote_name}"
        + (f" revision={dataset_revision}" if dataset_revision else "")
    )
    ds = load_dataset(
        "HuggingFaceFW/fineweb",
        name=remote_name,
        split="train",
        revision=dataset_revision,
    )
    if shuffle_seed is not None:
        print(f"Shuffling dataset with seed={shuffle_seed}")
        ds = ds.shuffle(seed=shuffle_seed)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out_path.open("w", encoding="utf-8") as f:
        pbar = tqdm(total=num_docs, unit="docs", desc="Caching docs")
        for row in ds:
            text = row.get("text", "")
            if not isinstance(text, str):
                text = "" if text is None else str(text)
            f.write(json.dumps({"text": text}, ensure_ascii=False))
            f.write("\n")
            written += 1
            pbar.update(1)
            if written >= num_docs:
                break
        pbar.close()

    if written < num_docs:
        raise RuntimeError(f"Requested {num_docs} docs, only found {written}")

    return {
        "remote_name": remote_name,
        "num_docs": written,
        "docs_sha256": compute_sha256(out_path),
        "dataset_fingerprint": getattr(ds, "_fingerprint", None),
    }


def train_sentencepiece_from_docs(
    docs_jsonl: Path,
    *,
    vocab_size: int,
    model_prefix: Path,
) -> Path:
    model_prefix.parent.mkdir(parents=True, exist_ok=True)
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter_docs_for_sentencepiece(docs_jsonl),
        model_prefix=str(model_prefix),
        model_type="bpe",
        vocab_size=vocab_size,
        character_coverage=0.999,
        byte_fallback=True,
        split_digits=True,
        normalization_rule_name="nmt_nfkc",
        add_dummy_prefix=False,
        pad_id=0,
        bos_id=1,
        eos_id=2,
        unk_id=3,
        hard_vocab_limit=False,
    )
    return model_prefix.with_suffix(".model")


@dataclass
class ExportStats:
    docs_total: int = 0
    docs_val: int = 0
    docs_train: int = 0
    files_total: int = 0
    files_val: int = 0
    files_train: int = 0
    tokens_total: int = 0
    tokens_val: int = 0
    tokens_train: int = 0


def export_shards_from_docs(
    docs_jsonl: Path,
    *,
    output_dir: Path,
    shard_size: int,
    num_val_docs: int,
    bos_id: int,
    eos_id: int,
    append_eos: bool,
    encode_fn: Callable[[str], list[int]],
) -> ExportStats:
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = ExportStats()
    shard_buf = np.empty((shard_size,), dtype=np.uint16)
    shard_fill = 0
    shard_index = 0
    current_split = "val"

    def flush_current() -> None:
        nonlocal shard_fill, shard_index
        if shard_fill == 0:
            return
        filename = output_dir / f"fineweb_{current_split}_{shard_index:06d}.bin"
        write_datafile(str(filename), shard_buf[:shard_fill])
        stats.files_total += 1
        if current_split == "val":
            stats.files_val += 1
        else:
            stats.files_train += 1
        shard_index += 1
        shard_fill = 0

    pbar = tqdm(total=count_docs(docs_jsonl), unit="docs", desc=f"Tokenizing {output_dir.name}")
    for doc_idx, text in enumerate(iter_docs_jsonl(docs_jsonl)):
        target_split = "val" if doc_idx < num_val_docs else "train"
        if target_split != current_split:
            flush_current()
            current_split = target_split

        stats.docs_total += 1
        if target_split == "val":
            stats.docs_val += 1
        else:
            stats.docs_train += 1

        token_ids = [bos_id]
        token_ids.extend(encode_fn(text))
        if append_eos:
            token_ids.append(eos_id)
        toks = np.asarray(token_ids, dtype=np.int32)
        assert (0 <= toks).all() and (toks < 2**16).all(), "token dictionary too large for uint16"
        toks = toks.astype(np.uint16)

        stats.tokens_total += len(toks)
        if target_split == "val":
            stats.tokens_val += len(toks)
        else:
            stats.tokens_train += len(toks)

        pos = 0
        while pos < len(toks):
            space = shard_size - shard_fill
            take = min(space, len(toks) - pos)
            shard_buf[shard_fill : shard_fill + take] = toks[pos : pos + take]
            shard_fill += take
            pos += take
            if shard_fill == shard_size:
                flush_current()
        pbar.update(1)
    pbar.close()

    flush_current()
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Export matched FineWeb datasets across tokenizer sizes")
    parser.add_argument("-v", "--version", type=str, default="10B", choices=["10B", "100B"])
    parser.add_argument("--num_docs", type=int, required=True, help="Number of docs to include in shared cache")
    parser.add_argument(
        "--num_val_docs",
        type=int,
        default=50_000,
        help="Number of docs reserved for validation (same docs across all tokenizer variants)",
    )
    parser.add_argument("-s", "--shard_size", type=int, default=10**8)
    parser.add_argument(
        "--sp_vocab_sizes",
        type=str,
        default="512,1024,2048,4096",
        help="Comma-separated SentencePiece vocab sizes",
    )
    parser.add_argument("--shuffle_seed", type=int, default=None, help="Optional deterministic shuffle seed")
    parser.add_argument(
        "--dataset_revision",
        type=str,
        default=None,
        help="Optional HF dataset revision to pin for reproducibility",
    )
    parser.add_argument(
        "--append_eos",
        action="store_true",
        help="Append eos_id to each document for all exports (default: disabled)",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
        help="Root output directory. Defaults to data/matched_<version>_docs<num_docs>_<seed|ordered>",
    )
    parser.add_argument(
        "--docs_jsonl",
        type=str,
        default=None,
        help="Optional existing docs cache JSONL. If provided, dataset loading is skipped.",
    )
    parser.add_argument(
        "--rebuild_docs_cache",
        action="store_true",
        help="If docs cache exists at output_root, rebuild it from HF.",
    )
    args = parser.parse_args()

    if args.num_docs <= 0:
        raise ValueError("--num_docs must be > 0")
    if args.num_val_docs <= 0:
        raise ValueError("--num_val_docs must be > 0")
    if args.num_val_docs >= args.num_docs:
        raise ValueError("--num_val_docs must be < --num_docs")

    sp_vocab_sizes = parse_sp_vocab_sizes(args.sp_vocab_sizes)
    seed_tag = f"seed{args.shuffle_seed}" if args.shuffle_seed is not None else "ordered"
    if args.output_root is None:
        output_root = Path(os.path.dirname(__file__)) / f"matched_{args.version}_docs{args.num_docs}_{seed_tag}"
    else:
        output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    tokenizers_dir = output_root / "tokenizers"
    datasets_dir = output_root / "datasets"
    tokenizers_dir.mkdir(parents=True, exist_ok=True)
    datasets_dir.mkdir(parents=True, exist_ok=True)

    if args.docs_jsonl is not None:
        docs_jsonl = Path(args.docs_jsonl)
        if not docs_jsonl.is_file():
            raise FileNotFoundError(f"Missing docs cache: {docs_jsonl}")
        docs_meta = {
            "remote_name": "external_cache",
            "num_docs": count_docs(docs_jsonl),
            "docs_sha256": compute_sha256(docs_jsonl),
            "dataset_fingerprint": None,
        }
        if docs_meta["num_docs"] < args.num_docs:
            raise ValueError(
                f"docs_jsonl has {docs_meta['num_docs']} docs but --num_docs={args.num_docs}. "
                "Provide a bigger docs cache or lower --num_docs."
            )
    else:
        docs_jsonl = output_root / "docs_selected.jsonl"
        if docs_jsonl.exists() and not args.rebuild_docs_cache:
            print(f"Reusing existing docs cache: {docs_jsonl}")
            docs_meta = {
                "remote_name": "cached_local",
                "num_docs": count_docs(docs_jsonl),
                "docs_sha256": compute_sha256(docs_jsonl),
                "dataset_fingerprint": None,
            }
            if docs_meta["num_docs"] < args.num_docs:
                raise ValueError(
                    f"Existing docs cache has {docs_meta['num_docs']} docs, need at least {args.num_docs}. "
                    "Delete cache or pass --rebuild_docs_cache."
                )
        else:
            docs_meta = build_docs_cache(
                docs_jsonl,
                version=args.version,
                num_docs=args.num_docs,
                shuffle_seed=args.shuffle_seed,
                dataset_revision=args.dataset_revision,
            )

    # If docs cache is larger than num_docs, truncate view deterministically by rewriting.
    # This keeps all downstream exports exactly aligned to the same first N cached docs.
    if docs_meta["num_docs"] != args.num_docs:
        truncated = output_root / f"docs_selected_{args.num_docs}.jsonl"
        print(f"Truncating docs cache to first {args.num_docs} docs: {truncated}")
        with docs_jsonl.open("r", encoding="utf-8") as src, truncated.open("w", encoding="utf-8") as dst:
            for i, line in enumerate(src):
                if i >= args.num_docs:
                    break
                dst.write(line)
        docs_jsonl = truncated
        docs_meta["num_docs"] = args.num_docs
        docs_meta["docs_sha256"] = compute_sha256(docs_jsonl)

    byte_tok: PureByteTokenizer = default_pure_byte_tokenizer()
    byte_json = tokenizers_dir / "fineweb_pure_byte_260.json"
    byte_tok.save_json(byte_json)
    print(f"Wrote byte tokenizer: {byte_json}")

    sp_models: dict[int, Path] = {}
    for vocab_size in sp_vocab_sizes:
        model_prefix = tokenizers_dir / f"fineweb_{vocab_size}_bpe"
        model_path = model_prefix.with_suffix(".model")
        vocab_path = model_prefix.with_suffix(".vocab")
        if model_path.exists() and vocab_path.exists():
            print(f"Reusing existing SentencePiece model: {model_path}")
        else:
            print(f"Training SentencePiece vocab={vocab_size}")
            train_sentencepiece_from_docs(
                docs_jsonl,
                vocab_size=vocab_size,
                model_prefix=model_prefix,
            )
        sp_models[vocab_size] = model_path

    manifest = {
        "version": args.version,
        "num_docs": args.num_docs,
        "num_val_docs": args.num_val_docs,
        "shuffle_seed": args.shuffle_seed,
        "dataset_revision": args.dataset_revision,
        "shard_size": args.shard_size,
        "append_eos": args.append_eos,
        "docs_jsonl": str(docs_jsonl),
        "docs_meta": docs_meta,
        "tokenizers": [],
        "datasets": [],
    }

    # Export byte dataset
    byte_dataset_name = f"fineweb{args.version}_byte260"
    byte_output_dir = datasets_dir / byte_dataset_name
    print(f"Exporting dataset: {byte_dataset_name}")
    byte_stats = export_shards_from_docs(
        docs_jsonl,
        output_dir=byte_output_dir,
        shard_size=args.shard_size,
        num_val_docs=args.num_val_docs,
        bos_id=byte_tok.bos_id,
        eos_id=byte_tok.eos_id,
        append_eos=args.append_eos,
        encode_fn=byte_tok.encode,
    )
    byte_reco_bigram = next_multiple_of_n(byte_tok.vocab_size, 128) * 5
    manifest["tokenizers"].append(
        {
            "name": "pure_byte_260",
            "kind": "byte",
            "path": str(byte_json),
            "vocab_size": byte_tok.vocab_size,
            "bos_id": byte_tok.bos_id,
            "eos_id": byte_tok.eos_id,
            "pad_id": byte_tok.pad_id,
            "unk_id": byte_tok.unk_id,
            "recommended_bigram_vocab_size": byte_reco_bigram,
        }
    )
    manifest["datasets"].append(
        {
            "name": byte_dataset_name,
            "path": str(byte_output_dir),
            "train_glob": str(byte_output_dir / "fineweb_train_*.bin"),
            "val_glob": str(byte_output_dir / "fineweb_val_*.bin"),
            "vocab_size": byte_tok.vocab_size,
            "bos_id": byte_tok.bos_id,
            "recommended_bigram_vocab_size": byte_reco_bigram,
            "stats": byte_stats.__dict__,
        }
    )

    # Export SentencePiece datasets
    for vocab_size in sp_vocab_sizes:
        model_path = sp_models[vocab_size]
        sp_tok = spm.SentencePieceProcessor(model_file=str(model_path))
        bos_id = int(sp_tok.bos_id())
        eos_id = int(sp_tok.eos_id())
        vocab_actual = int(sp_tok.vocab_size())
        dataset_name = f"fineweb{args.version}_sp{vocab_size}"
        output_dir = datasets_dir / dataset_name

        print(f"Exporting dataset: {dataset_name}")
        stats = export_shards_from_docs(
            docs_jsonl,
            output_dir=output_dir,
            shard_size=args.shard_size,
            num_val_docs=args.num_val_docs,
            bos_id=bos_id,
            eos_id=eos_id,
            append_eos=args.append_eos,
            encode_fn=lambda text, tok=sp_tok: tok.encode(text, out_type=int),
        )
        reco_bigram = next_multiple_of_n(vocab_actual, 128) * 5
        manifest["tokenizers"].append(
            {
                "name": f"sp_bpe_{vocab_size}",
                "kind": "sentencepiece_bpe",
                "model_path": str(model_path),
                "vocab_path": str(model_path.with_suffix(".vocab")),
                "vocab_size": vocab_actual,
                "bos_id": bos_id,
                "eos_id": eos_id,
                "recommended_bigram_vocab_size": reco_bigram,
            }
        )
        manifest["datasets"].append(
            {
                "name": dataset_name,
                "path": str(output_dir),
                "train_glob": str(output_dir / "fineweb_train_*.bin"),
                "val_glob": str(output_dir / "fineweb_val_*.bin"),
                "vocab_size": vocab_actual,
                "bos_id": bos_id,
                "recommended_bigram_vocab_size": reco_bigram,
                "stats": stats.__dict__,
            }
        )

    manifest_path = output_root / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=False)
        f.write("\n")

    print(f"\nDone. Manifest: {manifest_path}")
    print("Datasets:")
    for ds in manifest["datasets"]:
        print(
            f"- {ds['name']}: "
            f"vocab_size={ds['vocab_size']} "
            f"bos_id={ds['bos_id']} "
            f"recommended BIGRAM_VOCAB_SIZE={ds['recommended_bigram_vocab_size']}"
        )
        print(f"  TRAIN_FILES={ds['train_glob']}")
        print(f"  VAL_FILES={ds['val_glob']}")


if __name__ == "__main__":
    main()

