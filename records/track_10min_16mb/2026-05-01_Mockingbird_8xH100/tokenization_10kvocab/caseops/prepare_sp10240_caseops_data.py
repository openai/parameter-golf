#!/usr/bin/env python3
"""Build SP10240 CaseOps tokenizer/shards with validation byte sidecars.

This is a local data-build lane. It does not launch training.

Defaults follow the PR1855/PR1797 CaseOps data shape:
- lossless_caps_caseops_v1 transform from the PR1855 source lane
- reserved SentencePiece user symbols U+E001..U+E004
- uint16 header-prefixed shards
- BOS per document
- validation byte sidecars named fineweb_val_bytes_*.bin
- 10,000,000 tokens per shard unless overridden
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pathlib
import sys
import time
from array import array
from typing import Any, Callable, Iterable

import numpy as np
import sentencepiece as spm


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
PR1855_SOURCE_DIR = REPO_ROOT / "legs" / "2026-04-30_pr1855_sp8192_lqer_smeargate_repro_8x"

SHARD_MAGIC = 20240520
SHARD_VERSION = 1
DEFAULT_SHARD_TOKENS = 10_000_000
DEFAULT_VAL_DOCS = 50_000
BOS_ID = 1
VOCAB_SIZE = 10_240
TOKENIZER_BASENAME = "fineweb_10240_bpe_lossless_caps_caseops_v1_reserved"
DATASET_NAME = "fineweb10B_sp10240_lossless_caps_caseops_v1_reserved"
CASEOPS_SYMBOLS = [chr(0xE001), chr(0xE002), chr(0xE003), chr(0xE004)]


def _sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json_if_exists(path: pathlib.Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _load_lossless_caps(source_dir: pathlib.Path):
    module_path = source_dir / "lossless_caps.py"
    if not module_path.is_file():
        raise FileNotFoundError(module_path)
    spec = importlib.util.spec_from_file_location("caseops_lossless_caps", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, module_path


def _write_shard(out_path: pathlib.Path, values: array) -> None:
    if not values:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(values, dtype="<u2")
    header = np.zeros(256, dtype="<i4")
    header[0] = SHARD_MAGIC
    header[1] = SHARD_VERSION
    header[2] = int(arr.size)
    with out_path.open("wb") as fh:
        fh.write(header.tobytes())
        fh.write(arr.tobytes())


def _iter_docs(docs_path: pathlib.Path) -> Iterable[str]:
    with docs_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            yield obj["text"] if isinstance(obj, dict) else obj


def _iter_caseops_training_text(
    docs_path: pathlib.Path,
    transform: Callable[[str], str],
    *,
    skip_docs: int,
    max_docs: int | None,
) -> Iterable[str]:
    yielded = 0
    for doc_index, text in enumerate(_iter_docs(docs_path)):
        if doc_index < skip_docs:
            continue
        text = text.replace("\x00", " ").strip()
        if not text:
            continue
        yield transform(text)
        yielded += 1
        if max_docs is not None and yielded >= max_docs:
            return


def _train_caseops_tokenizer(
    *,
    docs_path: pathlib.Path,
    model_prefix: pathlib.Path,
    transform: Callable[[str], str],
    skip_docs: int,
    max_docs: int | None,
) -> None:
    model_path = model_prefix.with_suffix(".model")
    vocab_path = model_prefix.with_suffix(".vocab")
    if model_path.exists() or vocab_path.exists():
        raise FileExistsError(f"refusing to overwrite existing tokenizer artifacts at {model_prefix}.*")
    model_prefix.parent.mkdir(parents=True, exist_ok=True)
    print(
        json.dumps(
            {
                "event": "train_tokenizer_start",
                "model_prefix": str(model_prefix),
                "vocab_size": VOCAB_SIZE,
                "tokenizer_skip_docs": skip_docs,
                "tokenizer_train_docs": max_docs,
                "user_defined_symbols_hex": [hex(ord(s)) for s in CASEOPS_SYMBOLS],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    spm.SentencePieceTrainer.train(
        sentence_iterator=_iter_caseops_training_text(
            docs_path,
            transform,
            skip_docs=skip_docs,
            max_docs=max_docs,
        ),
        model_prefix=str(model_prefix),
        model_type="bpe",
        vocab_size=VOCAB_SIZE,
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
        user_defined_symbols=CASEOPS_SYMBOLS,
    )


def _validate_tokenizer(sp: spm.SentencePieceProcessor, sp_path: pathlib.Path) -> None:
    if int(sp.vocab_size()) != VOCAB_SIZE:
        raise ValueError(f"{sp_path} vocab_size={sp.vocab_size()} != {VOCAB_SIZE}")
    for offset, symbol in enumerate(CASEOPS_SYMBOLS, start=4):
        token_id = int(sp.piece_to_id(symbol))
        if token_id != offset:
            raise ValueError(
                f"{sp_path} does not reserve CaseOps symbol {hex(ord(symbol))} at id {offset}; got {token_id}"
            )


def _encode_val_doc(
    sp: spm.SentencePieceProcessor,
    transformed_text: str,
    *,
    byte_counter: Callable[..., list[int]],
    transform_name: str,
) -> tuple[list[int], list[int]]:
    proto = sp.encode_as_immutable_proto(transformed_text)
    token_ids = [BOS_ID]
    token_ids.extend(int(piece.id) for piece in proto.pieces)
    byte_counts = [0]
    byte_counts.extend(
        int(v)
        for v in byte_counter(
            (piece.surface for piece in proto.pieces),
            text_transform_name=transform_name,
        )
    )
    if len(token_ids) != len(byte_counts):
        raise ValueError(f"token/byte sidecar length mismatch: {len(token_ids)} != {len(byte_counts)}")
    too_large = [v for v in byte_counts if v > 0xFFFF]
    if too_large:
        raise ValueError(f"byte sidecar value exceeds uint16: {too_large[0]}")
    return token_ids, byte_counts


def _append_uint16(buf: array, values: Iterable[int]) -> None:
    for value in values:
        if value < 0 or value > 0xFFFF:
            raise ValueError(f"value outside uint16 range: {value}")
        buf.append(int(value))


def _flush_full_shards(
    *,
    buf: array,
    out_dir: pathlib.Path,
    prefix: str,
    shard_index: int,
    shard_tokens: int,
) -> int:
    while len(buf) >= shard_tokens:
        shard = array("H", buf[:shard_tokens])
        _write_shard(out_dir / f"{prefix}_{shard_index:06d}.bin", shard)
        del buf[:shard_tokens]
        shard_index += 1
    return shard_index


def _flush_tail(*, buf: array, out_dir: pathlib.Path, prefix: str, shard_index: int) -> int:
    if buf:
        _write_shard(out_dir / f"{prefix}_{shard_index:06d}.bin", buf)
        del buf[:]
        shard_index += 1
    return shard_index


def _fail_if_existing_outputs(dataset_dir: pathlib.Path) -> None:
    patterns = ("fineweb_train_*.bin", "fineweb_val_*.bin", "fineweb_val_bytes_*.bin")
    existing: list[pathlib.Path] = []
    for pattern in patterns:
        existing.extend(sorted(dataset_dir.glob(pattern)))
    if existing:
        sample = ", ".join(str(p) for p in existing[:5])
        raise FileExistsError(f"refusing to overwrite existing shard outputs in {dataset_dir}; sample: {sample}")


def _write_manifest(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_shards(
    *,
    docs_path: pathlib.Path,
    out_root: pathlib.Path,
    sp_path: pathlib.Path,
    source_dir: pathlib.Path,
    val_docs: int,
    max_train_shards: int,
    shard_tokens: int,
    tokenizer_skip_docs: int,
    tokenizer_train_docs: int | None,
    trained_tokenizer: bool,
) -> pathlib.Path:
    caps, caps_path = _load_lossless_caps(source_dir)
    sp = spm.SentencePieceProcessor(model_file=str(sp_path))
    _validate_tokenizer(sp, sp_path)

    dataset_dir = out_root / "datasets" / DATASET_NAME
    _fail_if_existing_outputs(dataset_dir)
    tokenizers_dir = out_root / "tokenizers"
    tokenizers_dir.mkdir(parents=True, exist_ok=True)

    docs_sidecar = _load_json_if_exists(docs_path.with_name(f"{docs_path.stem}.source_manifest.json"))
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    manifest: dict[str, Any] = {
        "label": "new_experiment",
        "description": "SP10240 CaseOps tokenizer plus PR1855-style validation byte sidecars.",
        "started_at": started_at,
        "docs_path": str(docs_path),
        "docs_sidecar": docs_sidecar,
        "lossless_caps_path": str(caps_path),
        "lossless_caps_sha256": _sha256(caps_path),
        "output_root": str(out_root),
        "dataset_name": DATASET_NAME,
        "dataset_path": str(dataset_dir),
        "tokenizer_model": str(sp_path),
        "tokenizer_vocab": str(sp_path.with_suffix(".vocab")),
        "tokenizer_model_sha256": _sha256(sp_path),
        "tokenizer_vocab_sha256": _sha256(sp_path.with_suffix(".vocab")) if sp_path.with_suffix(".vocab").is_file() else None,
        "tokenizer_trained_in_this_run": trained_tokenizer,
        "tokenizer_training_spec": {
            "vocab_size": VOCAB_SIZE,
            "model_type": "bpe",
            "character_coverage": 0.999,
            "byte_fallback": True,
            "split_digits": True,
            "normalization_rule_name": "nmt_nfkc",
            "add_dummy_prefix": False,
            "pad_id": 0,
            "bos_id": 1,
            "eos_id": 2,
            "unk_id": 3,
            "hard_vocab_limit": False,
            "user_defined_symbols_hex": [hex(ord(s)) for s in CASEOPS_SYMBOLS],
            "tokenizer_skip_docs": tokenizer_skip_docs,
            "tokenizer_train_docs": tokenizer_train_docs,
        },
        "shard_spec": {
            "magic": SHARD_MAGIC,
            "version": SHARD_VERSION,
            "dtype": "uint16",
            "shard_tokens": shard_tokens,
            "val_docs": val_docs,
            "max_train_shards": max_train_shards,
            "bos_id": BOS_ID,
            "byte_sidecars": "validation_only",
        },
        "stats": {
            "docs_total": 0,
            "docs_val": 0,
            "docs_train": 0,
            "tokens_val": 0,
            "tokens_train": 0,
            "bytes_sidecar_tokens_val": 0,
            "files_val": 0,
            "files_val_bytes": 0,
            "files_train": 0,
        },
    }
    _write_manifest(out_root / "caseops_manifest.in_progress.json", manifest)

    val_buf_tokens = array("H")
    val_buf_bytes = array("H")
    train_buf = array("H")
    val_written = 0
    val_bytes_written = 0
    train_written = 0
    val_tail_flushed = False

    for text in _iter_docs(docs_path):
        doc_index = int(manifest["stats"]["docs_total"])
        transformed = caps.encode_lossless_caps_v2(text)
        if doc_index < val_docs:
            token_ids, byte_counts = _encode_val_doc(
                sp,
                transformed,
                byte_counter=caps.surface_piece_original_byte_counts,
                transform_name=caps.LOSSLESS_CAPS_CASEOPS_V1,
            )
            _append_uint16(val_buf_tokens, token_ids)
            _append_uint16(val_buf_bytes, byte_counts)
            manifest["stats"]["docs_val"] += 1
            manifest["stats"]["tokens_val"] += len(token_ids)
            manifest["stats"]["bytes_sidecar_tokens_val"] += len(byte_counts)
            new_val_written = _flush_full_shards(
                buf=val_buf_tokens,
                out_dir=dataset_dir,
                prefix="fineweb_val",
                shard_index=val_written,
                shard_tokens=shard_tokens,
            )
            new_val_bytes_written = _flush_full_shards(
                buf=val_buf_bytes,
                out_dir=dataset_dir,
                prefix="fineweb_val_bytes",
                shard_index=val_bytes_written,
                shard_tokens=shard_tokens,
            )
            val_written = new_val_written
            val_bytes_written = new_val_bytes_written
        else:
            if not val_tail_flushed:
                val_written = _flush_tail(
                    buf=val_buf_tokens,
                    out_dir=dataset_dir,
                    prefix="fineweb_val",
                    shard_index=val_written,
                )
                val_bytes_written = _flush_tail(
                    buf=val_buf_bytes,
                    out_dir=dataset_dir,
                    prefix="fineweb_val_bytes",
                    shard_index=val_bytes_written,
                )
                val_tail_flushed = True
            token_ids = [BOS_ID]
            token_ids.extend(int(v) for v in sp.encode(transformed, out_type=int))
            _append_uint16(train_buf, token_ids)
            manifest["stats"]["docs_train"] += 1
            manifest["stats"]["tokens_train"] += len(token_ids)
            while len(train_buf) >= shard_tokens:
                shard = array("H", train_buf[:shard_tokens])
                _write_shard(dataset_dir / f"fineweb_train_{train_written:06d}.bin", shard)
                del train_buf[:shard_tokens]
                train_written += 1
                if max_train_shards and train_written >= max_train_shards:
                    manifest["stats"]["docs_total"] = doc_index + 1
                    manifest["stats"]["files_val"] = val_written
                    manifest["stats"]["files_val_bytes"] = val_bytes_written
                    manifest["stats"]["files_train"] = train_written
                    manifest["stopped_reason"] = "max_train_shards"
                    manifest["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
                    _write_manifest(out_root / "caseops_manifest.json", manifest)
                    print(json.dumps({"event": "done", "stats": manifest["stats"]}, sort_keys=True), flush=True)
                    return out_root / "caseops_manifest.json"

        manifest["stats"]["docs_total"] = doc_index + 1
        if manifest["stats"]["docs_total"] % 10_000 == 0:
            manifest["stats"]["files_val"] = val_written
            manifest["stats"]["files_val_bytes"] = val_bytes_written
            manifest["stats"]["files_train"] = train_written
            print(json.dumps({"event": "progress", "stats": manifest["stats"]}, sort_keys=True), flush=True)

    if not val_tail_flushed:
        val_written = _flush_tail(
            buf=val_buf_tokens,
            out_dir=dataset_dir,
            prefix="fineweb_val",
            shard_index=val_written,
        )
        val_bytes_written = _flush_tail(
            buf=val_buf_bytes,
            out_dir=dataset_dir,
            prefix="fineweb_val_bytes",
            shard_index=val_bytes_written,
        )
    if train_buf:
        _write_shard(dataset_dir / f"fineweb_train_{train_written:06d}.bin", train_buf)
        train_written += 1

    manifest["stats"]["files_val"] = val_written
    manifest["stats"]["files_val_bytes"] = val_bytes_written
    manifest["stats"]["files_train"] = train_written
    manifest["stopped_reason"] = "eof"
    manifest["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    _write_manifest(out_root / "caseops_manifest.json", manifest)
    print(json.dumps({"event": "done", "stats": manifest["stats"]}, sort_keys=True), flush=True)
    return out_root / "caseops_manifest.json"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--docs", required=True, type=pathlib.Path, help="Path to docs_selected.jsonl")
    ap.add_argument("--out", required=True, type=pathlib.Path, help="Output root containing tokenizers/ and datasets/")
    ap.add_argument("--source-dir", type=pathlib.Path, default=PR1855_SOURCE_DIR, help="Directory containing lossless_caps.py")
    ap.add_argument("--sp", type=pathlib.Path, default=None, help="CaseOps SP10240 model path. Defaults under --out/tokenizers.")
    ap.add_argument("--train-tokenizer", action="store_true", help="Train the SP10240 CaseOps tokenizer if --sp is missing")
    ap.add_argument("--tokenizer-skip-docs", type=int, default=DEFAULT_VAL_DOCS)
    ap.add_argument("--tokenizer-train-docs", type=int, default=None, help="Optional count after --tokenizer-skip-docs")
    ap.add_argument("--val-docs", type=int, default=DEFAULT_VAL_DOCS)
    ap.add_argument("--max-train-shards", type=int, default=0, help="0 means run until EOF")
    ap.add_argument("--shard-tokens", type=int, default=DEFAULT_SHARD_TOKENS)
    return ap


def main() -> None:
    args = build_parser().parse_args()
    docs_path = args.docs.expanduser().resolve()
    out_root = args.out.expanduser().resolve()
    source_dir = args.source_dir.expanduser().resolve()
    sp_path = (
        args.sp.expanduser().resolve()
        if args.sp is not None
        else out_root / "tokenizers" / f"{TOKENIZER_BASENAME}.model"
    )

    if not docs_path.is_file():
        raise FileNotFoundError(docs_path)
    if args.val_docs < 0:
        raise ValueError("--val-docs must be non-negative")
    if args.max_train_shards < 0:
        raise ValueError("--max-train-shards must be non-negative")
    if args.shard_tokens <= 0:
        raise ValueError("--shard-tokens must be positive")

    caps, _ = _load_lossless_caps(source_dir)
    trained_tokenizer = False
    if not sp_path.is_file():
        if not args.train_tokenizer:
            raise FileNotFoundError(f"{sp_path}; rerun with --train-tokenizer to create it")
        _train_caseops_tokenizer(
            docs_path=docs_path,
            model_prefix=sp_path.with_suffix(""),
            transform=caps.encode_lossless_caps_v2,
            skip_docs=args.tokenizer_skip_docs,
            max_docs=args.tokenizer_train_docs,
        )
        trained_tokenizer = True

    manifest_path = build_shards(
        docs_path=docs_path,
        out_root=out_root,
        sp_path=sp_path,
        source_dir=source_dir,
        val_docs=args.val_docs,
        max_train_shards=args.max_train_shards,
        shard_tokens=args.shard_tokens,
        tokenizer_skip_docs=args.tokenizer_skip_docs,
        tokenizer_train_docs=args.tokenizer_train_docs,
        trained_tokenizer=trained_tokenizer,
    )
    print(f"manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
