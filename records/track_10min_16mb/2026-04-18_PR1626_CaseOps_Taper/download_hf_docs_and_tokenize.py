"""Download docs_selected.jsonl from Hugging Face and tokenize it locally.

This script is standalone. It does not import any local exporter or tokenizer
helpers. Tokenizer configs are JSON only and currently support the built-in
pure-byte and SentencePiece tokenizer definitions in `data/tokenizer_specs.json`.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
try:
    from lossless_caps import (
        IDENTITY,
        get_text_transform,
        get_text_transform_control_symbols,
        surface_piece_original_byte_counts,
    )
except ImportError:
    from data.lossless_caps import (
        IDENTITY,
        get_text_transform,
        get_text_transform_control_symbols,
        surface_piece_original_byte_counts,
    )


DOCS_FILENAME = "docs_selected.jsonl"
SIDECAR_FILENAME = "docs_selected.source_manifest.json"
VERSION = "10B"
NUM_VAL_DOCS = 50_000
SHARD_SIZE = 10**8
APPEND_EOS = False
DATAFILE_MAGIC = 20240520
DATAFILE_VERSION = 1
DEFAULT_REPO_ID = os.environ.get("MATCHED_FINEWEB_REPO_ID", "willdepueoai/parameter-golf")
DEFAULT_REMOTE_ROOT = os.environ.get("MATCHED_FINEWEB_REMOTE_ROOT_PREFIX", "datasets")
DEFAULT_CONFIG = Path(__file__).with_name("tokenizer_specs.json")
TOKENIZER_THREADS = max(1, int(os.environ.get("MATCHED_FINEWEB_TOKENIZER_THREADS", str(os.cpu_count() or 8))))
SP_BATCH_SIZE = max(1, int(os.environ.get("MATCHED_FINEWEB_SP_BATCH_SIZE", "1024")))


@dataclass(frozen=True)
class PureByteTokenizer:
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2
    unk_id: int = 3
    byte_offset: int = 4
    byte_count: int = 256

    @property
    def vocab_size(self) -> int:
        return self.byte_offset + self.byte_count

    def encode(self, text: str) -> np.ndarray:
        data = text.encode("utf-8", errors="replace")
        return np.frombuffer(data, dtype=np.uint8).astype(np.uint16, copy=False) + self.byte_offset

    def encode_batch(self, texts: list[str]) -> list[np.ndarray]:
        return [self.encode(text) for text in texts]

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "tokenizer_type": "pure_byte",
            "config": asdict(self),
            "vocab_size": self.vocab_size,
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def default_pure_byte_tokenizer() -> PureByteTokenizer:
    return PureByteTokenizer()


def docs_sidecar_path(docs_jsonl: Path) -> Path:
    return docs_jsonl.with_name(f"{docs_jsonl.stem}.source_manifest.json")


def maybe_load_docs_sidecar_meta(docs_jsonl: Path) -> dict[str, Any] | None:
    sidecar_path = docs_sidecar_path(docs_jsonl)
    if not sidecar_path.is_file():
        return None
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"docs sidecar must be a JSON object: {sidecar_path}")
    return payload


def copy_from_hf_cache(*, repo_id: str, remote_root: str, filename: str, destination: Path) -> bool:
    if destination.exists():
        return True
    remote_path = Path(remote_root) / filename if remote_root else Path(filename)
    try:
        cached_path = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=remote_path.name,
                subfolder=remote_path.parent.as_posix() if remote_path.parent != Path(".") else None,
                repo_type="dataset",
            )
        )
    except EntryNotFoundError:
        return False

    source = cached_path.resolve(strict=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
    return True


def iter_docs(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)["text"]


def count_docs(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def batched_docs_jsonl(path: Path, batch_size: int):
    batch: list[str] = []
    for text in iter_docs(path):
        batch.append(text)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def write_datafile(path: Path, toks: Any) -> None:
    if len(toks) >= 2**31:
        raise ValueError("token count too large")
    header = np.zeros(256, dtype="<i4")
    header[0] = DATAFILE_MAGIC
    header[1] = DATAFILE_VERSION
    header[2] = len(toks)
    toks = np.asarray(toks)
    if toks.dtype != np.uint16:
        if not ((0 <= toks).all() and (toks < 2**16).all()):
            raise ValueError("token dictionary too large for uint16")
        toks = toks.astype("<u2", copy=False)
    else:
        toks = toks.astype("<u2", copy=False)
    with path.open("wb") as f:
        f.write(header.tobytes())
        f.write(toks.tobytes())


def relativize_manifest_paths(value: Any, root: Path) -> Any:
    if isinstance(value, dict):
        return {k: relativize_manifest_paths(v, root) for k, v in value.items()}
    if isinstance(value, list):
        return [relativize_manifest_paths(v, root) for v in value]
    if isinstance(value, str):
        path = Path(value)
        if path.is_absolute():
            try:
                return path.relative_to(root).as_posix()
            except ValueError:
                return value
    return value


def parse_reuse_sp_models(values: list[str]) -> dict[int, Path]:
    reuse_models: dict[int, Path] = {}
    for value in values:
        vocab_size_str, model_path = value.split("=", 1)
        vocab_size = int(vocab_size_str)
        if vocab_size in reuse_models:
            raise ValueError(f"duplicate --reuse_sp_model for vocab_size={vocab_size}")
        reuse_models[vocab_size] = Path(model_path).expanduser().resolve()
    return reuse_models


def load_specs(config_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        specs = payload.get("tokenizer_specs", payload.get("tokenizers"))
    else:
        specs = payload
    if not isinstance(specs, list) or not specs:
        raise ValueError("tokenizer_config must define a non-empty list")
    if not all(isinstance(spec, dict) for spec in specs):
        raise ValueError("each tokenizer spec must be a JSON object")
    return [dict(spec) for spec in specs]


def tokenizer_kind(spec: dict[str, Any]) -> str:
    kind = spec.get("kind")
    if kind in {"byte", "pure_byte"}:
        return "byte"
    if kind in {"sentencepiece_bpe", "sentencepiece"}:
        return "sentencepiece_bpe"
    builder = str(spec.get("builder", ""))
    builder_name = builder.rsplit(":", 1)[-1]
    if builder_name == "build_pure_byte_tokenizer":
        return "byte"
    if builder_name == "build_sentencepiece_tokenizer":
        return "sentencepiece_bpe"
    if spec.get("dataset_suffix") == "byte260":
        return "byte"
    if "vocab_size" in spec:
        return "sentencepiece_bpe"
    raise ValueError(
        f"unsupported tokenizer spec {spec.get('name', '<unnamed>')!r}: "
        "expected a built-in pure-byte or sentencepiece builder"
    )


def write_tokenizer_config_export(output_root: Path, selected_specs: list[dict[str, Any]]) -> Path:
    path = output_root / "tokenizer_config.export.json"
    path.write_text(json.dumps({"tokenizers": selected_specs}, indent=2) + "\n", encoding="utf-8")
    return path


def _iter_sentencepiece_text(
    docs_jsonl: Path,
    *,
    max_docs: int | None = None,
    text_transform_name: str | None = None,
):
    text_transform = get_text_transform(text_transform_name)
    with docs_jsonl.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_docs is not None and i >= max_docs:
                break
            text = json.loads(line)["text"].replace("\x00", " ").strip()
            if text:
                yield text_transform(text)


def build_pure_byte_tokenizer(*, spec: dict[str, Any], docs_jsonl: Path, tokenizers_dir: Path) -> dict[str, Any]:
    del docs_jsonl
    text_transform_name = str(spec.get("text_transform") or IDENTITY)
    if text_transform_name != IDENTITY:
        raise ValueError(
            f"pure byte tokenizer does not support text_transform={text_transform_name!r}"
        )
    tok = default_pure_byte_tokenizer()
    path = tokenizers_dir / spec.get("filename", "fineweb_pure_byte_260.json")
    tok.save_json(path)
    return {
        "name": spec.get("name", "pure_byte_260"),
        "kind": "byte",
        "dataset_suffix": spec.get("dataset_suffix", "byte260"),
        "vocab_size": tok.vocab_size,
        "bos_id": tok.bos_id,
        "eos_id": tok.eos_id,
        "encode": tok.encode,
        "encode_batch": tok.encode_batch,
        "encode_with_original_byte_counts": lambda text, tok=tok: (
            tok.encode(text),
            np.ones((len(tok.encode(text)),), dtype=np.uint16),
        ),
        "text_transform": text_transform_name,
        "manifest": {
            "path": str(path),
            "pad_id": tok.pad_id,
            "unk_id": tok.unk_id,
            "text_transform": text_transform_name,
        },
    }


def build_sentencepiece_tokenizer(*, spec: dict[str, Any], docs_jsonl: Path, tokenizers_dir: Path) -> dict[str, Any]:
    try:
        import sentencepiece as spm
    except ImportError as exc:
        raise RuntimeError("sentencepiece is required for SentencePiece tokenizer exports") from exc

    text_transform_name = str(spec.get("text_transform") or IDENTITY)
    text_transform = get_text_transform(text_transform_name)
    vocab_size = int(spec["vocab_size"])
    prefix = tokenizers_dir / spec.get("model_prefix", f"fineweb_{vocab_size}_bpe")
    model_path = prefix.with_suffix(".model")
    vocab_path = prefix.with_suffix(".vocab")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for artifact in (model_path, vocab_path):
        if artifact.exists():
            artifact.unlink()

    reuse_model_path = spec.get("reuse_model_path")
    if reuse_model_path is not None:
        reuse_model_path = Path(reuse_model_path).expanduser().resolve()
        if not reuse_model_path.is_file():
            raise FileNotFoundError(reuse_model_path)
        shutil.copy2(reuse_model_path, model_path)
        reuse_vocab_path = reuse_model_path.with_suffix(".vocab")
        if reuse_vocab_path.is_file():
            shutil.copy2(reuse_vocab_path, vocab_path)
    else:
        trainer_overrides = dict(spec.get("trainer_overrides") or {})
        if bool(spec.get("reserve_text_transform_controls")):
            control_symbols = get_text_transform_control_symbols(text_transform_name)
            if control_symbols:
                existing = trainer_overrides.get("user_defined_symbols")
                merged_symbols: list[str] = []
                if isinstance(existing, str):
                    merged_symbols.extend(symbol for symbol in existing.split(",") if symbol)
                elif isinstance(existing, (list, tuple)):
                    merged_symbols.extend(str(symbol) for symbol in existing if str(symbol))
                elif existing is not None:
                    raise TypeError("trainer_overrides.user_defined_symbols must be a string or sequence")
                for symbol in control_symbols:
                    if symbol not in merged_symbols:
                        merged_symbols.append(symbol)
                trainer_overrides["user_defined_symbols"] = merged_symbols
        kwargs = {
            "sentence_iterator": _iter_sentencepiece_text(
                docs_jsonl,
                max_docs=None if spec.get("tokenizer_train_docs") is None else int(spec["tokenizer_train_docs"]),
                text_transform_name=text_transform_name,
            ),
            "model_prefix": str(prefix),
            "model_type": "bpe",
            "vocab_size": vocab_size,
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
        }
        kwargs.update(trainer_overrides)
        spm.SentencePieceTrainer.train(**kwargs)

    tok = spm.SentencePieceProcessor(model_file=str(model_path))

    def encode_with_original_byte_counts(
        text: str,
        *,
        tok=tok,
        transform=text_transform,
        text_transform_name=text_transform_name,
    ) -> tuple[np.ndarray, np.ndarray]:
        transformed = transform(text)
        proto = tok.encode_as_immutable_proto(transformed)
        token_ids = np.fromiter((piece.id for piece in proto.pieces), dtype=np.int32)
        byte_counts = np.asarray(
            surface_piece_original_byte_counts(
                (piece.surface for piece in proto.pieces),
                text_transform_name=text_transform_name,
            ),
            dtype=np.uint16,
        )
        if token_ids.shape[0] != byte_counts.shape[0]:
            raise ValueError("token id count and byte count length disagree")
        return token_ids, byte_counts

    return {
        "name": spec.get("name", f"sp_bpe_{vocab_size}"),
        "kind": "sentencepiece_bpe",
        "dataset_suffix": spec.get("dataset_suffix", f"sp{vocab_size}"),
        "vocab_size": int(tok.vocab_size()),
        "bos_id": int(tok.bos_id()),
        "eos_id": int(tok.eos_id()),
        "encode": lambda text, tok=tok, transform=text_transform: tok.encode(
            transform(text), out_type=int
        ),
        "encode_batch": lambda texts, tok=tok, transform=text_transform: tok.encode(
            [transform(text) for text in texts], out_type=int, num_threads=TOKENIZER_THREADS
        ),
        "encode_with_original_byte_counts": encode_with_original_byte_counts,
        "text_transform": text_transform_name,
        "manifest": {
            "model_path": str(model_path),
            "vocab_path": str(vocab_path),
            "text_transform": text_transform_name,
            "reserve_text_transform_controls": bool(spec.get("reserve_text_transform_controls")),
        },
    }


def export_shards(
    docs_jsonl: Path,
    tok: dict[str, Any],
    output_dir: Path,
    *,
    num_val_docs: int,
    shard_size: int,
    docs_total: int,
    max_train_shards: int | None = None,
) -> dict[str, int]:
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
    }
    buf = np.empty((shard_size,), dtype=np.uint16)
    val_byte_buf = np.empty((shard_size,), dtype=np.uint16)
    fill = 0
    split = "val"
    shards = {"val": 0, "train": 0}
    stop_after_flush = False

    def flush() -> None:
        nonlocal fill, stop_after_flush
        if fill == 0:
            return
        shard_index = shards[split]
        write_datafile(output_dir / f"fineweb_{split}_{shard_index:06d}.bin", buf[:fill])
        if split == "val":
            write_datafile(output_dir / f"fineweb_val_bytes_{shard_index:06d}.bin", val_byte_buf[:fill])
        stats["files_total"] += 1
        stats[f"files_{split}"] += 1
        shards[split] = shard_index + 1
        if split == "train" and max_train_shards is not None and shards["train"] >= max_train_shards:
            stop_after_flush = True
        fill = 0

    vocab_size = int(tok["vocab_size"])
    if vocab_size > 2**16:
        raise ValueError(f"vocab_size={vocab_size} is too large for uint16 shard storage")

    batch_encode = tok.get("encode_batch")
    encode_with_original_byte_counts = tok.get("encode_with_original_byte_counts")
    batch_size = SP_BATCH_SIZE if callable(batch_encode) else 1

    for texts in batched_docs_jsonl(docs_jsonl, batch_size):
        encoded_docs = batch_encode(texts) if callable(batch_encode) else [tok["encode"](text) for text in texts]
        byte_counts_docs: list[np.ndarray | None] = [None] * len(encoded_docs)
        if callable(encode_with_original_byte_counts):
            val_docs_remaining = max(0, num_val_docs - stats["docs_total"])
            if val_docs_remaining:
                for idx, text in enumerate(texts[:val_docs_remaining]):
                    _, byte_counts_arr = encode_with_original_byte_counts(text)
                    byte_counts_docs[idx] = byte_counts_arr

        for text, encoded, byte_counts_arr in zip(texts, encoded_docs, byte_counts_docs, strict=True):
                split_for_doc = "val" if stats["docs_total"] < num_val_docs else "train"
                if split_for_doc != split:
                    flush()
                    split = split_for_doc

                encoded_arr = np.asarray(encoded, dtype=np.int32)
                if byte_counts_arr is not None and byte_counts_arr.shape[0] != encoded_arr.shape[0]:
                    raise ValueError("encoded token count and original byte count length disagree")
                toks = np.empty((encoded_arr.size + 1 + int(APPEND_EOS),), dtype=np.int32)
                toks[0] = tok["bos_id"]
                toks[1 : 1 + encoded_arr.size] = encoded_arr
                if APPEND_EOS:
                    toks[-1] = tok["eos_id"]
                val_byte_counts = None
                if split == "val":
                    val_byte_counts = np.zeros((toks.size,), dtype=np.int32)
                    if byte_counts_arr is not None:
                        val_byte_counts[1 : 1 + encoded_arr.size] = byte_counts_arr.astype(np.int32, copy=False)
                    elif tok["kind"] == "byte":
                        val_byte_counts[1 : 1 + encoded_arr.size] = 1
                if not ((0 <= toks).all() and (toks < vocab_size).all()):
                    bad = int(toks[(toks < 0) | (toks >= vocab_size)][0])
                    raise ValueError(f"token id {bad} outside declared vocab_size={vocab_size}")
                toks = toks.astype("<u2", copy=False)
                if val_byte_counts is not None:
                    if not ((0 <= val_byte_counts).all() and (val_byte_counts < 2**16).all()):
                        raise ValueError("validation byte counts must fit in uint16")
                    val_byte_counts = val_byte_counts.astype("<u2", copy=False)

                stats["docs_total"] += 1
                stats[f"docs_{split}"] += 1
                stats["tokens_total"] += len(toks)
                stats[f"tokens_{split}"] += len(toks)

                pos = 0
                while pos < len(toks):
                    take = min(shard_size - fill, len(toks) - pos)
                    buf[fill : fill + take] = toks[pos : pos + take]
                    if split == "val":
                        val_byte_buf[fill : fill + take] = val_byte_counts[pos : pos + take]
                    fill += take
                    pos += take
                    if fill == shard_size:
                        flush()
                        if stop_after_flush:
                            break
                if stop_after_flush:
                    break

        if stats["docs_total"] and stats["docs_total"] % 100_000 == 0:
            print(f"{output_dir.name}: {stats['docs_total']}/{docs_total} docs", flush=True)
        if stop_after_flush:
            break

    flush()
    if max_train_shards is None and stats["docs_total"] != docs_total:
        raise ValueError(f"expected {docs_total} docs, exported {stats['docs_total']}")
    return stats


def build_tokenizers(
    *,
    specs: list[dict[str, Any]],
    docs_jsonl: Path,
    tokenizers_dir: Path,
    tokenizer_train_docs: int | None,
    skip_byte: bool,
    reuse_sp_models: dict[int, Path],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tokenizers: list[dict[str, Any]] = []
    selected_specs: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    seen_datasets: set[str] = set()

    for raw_spec in specs:
        spec = dict(raw_spec)
        kind = tokenizer_kind(spec)
        if skip_byte and kind == "byte":
            continue
        if kind == "sentencepiece_bpe":
            if tokenizer_train_docs is not None:
                spec["tokenizer_train_docs"] = int(tokenizer_train_docs)
            vocab_size = int(spec["vocab_size"])
            if vocab_size in reuse_sp_models:
                spec["reuse_model_path"] = str(reuse_sp_models[vocab_size])

        selected_specs.append(spec)
        built = (
            build_pure_byte_tokenizer(spec=spec, docs_jsonl=docs_jsonl, tokenizers_dir=tokenizers_dir)
            if kind == "byte"
            else build_sentencepiece_tokenizer(spec=spec, docs_jsonl=docs_jsonl, tokenizers_dir=tokenizers_dir)
        )
        name = str(built["name"])
        dataset_suffix = built.get("dataset_suffix")
        dataset_name = str(built.get("dataset_name", f"fineweb{VERSION}_{dataset_suffix}"))
        if name in seen_names:
            raise ValueError(f"duplicate tokenizer name: {name}")
        if dataset_name in seen_datasets:
            raise ValueError(f"duplicate dataset name: {dataset_name}")
        seen_names.add(name)
        seen_datasets.add(dataset_name)
        vocab_size = int(built["vocab_size"])
        recommended_bigram_vocab_size = int(
            built.get("recommended_bigram_vocab_size", ((vocab_size + 127) // 128) * 128 * 5)
        )
        tokenizers.append(
            {
                "name": name,
                "kind": str(built["kind"]),
                "dataset_name": dataset_name,
                "vocab_size": vocab_size,
                "bos_id": int(built["bos_id"]),
                "eos_id": int(built["eos_id"]),
                "encode": built["encode"],
                "encode_batch": built.get("encode_batch"),
                "encode_with_original_byte_counts": built.get("encode_with_original_byte_counts"),
                "recommended_bigram_vocab_size": recommended_bigram_vocab_size,
                "text_transform": str(built.get("text_transform", IDENTITY)),
                "manifest": {
                    "name": name,
                    "kind": str(built["kind"]),
                    "vocab_size": vocab_size,
                    "bos_id": int(built["bos_id"]),
                    "eos_id": int(built["eos_id"]),
                    "recommended_bigram_vocab_size": recommended_bigram_vocab_size,
                    "text_transform": str(built.get("text_transform", IDENTITY)),
                    "source_spec": spec,
                    **(built.get("manifest") or {}),
                },
            }
        )
    if not tokenizers:
        raise ValueError("tokenizer_config produced no tokenizers after filtering")
    return tokenizers, selected_specs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download docs_selected.jsonl from a Hugging Face dataset repo and tokenize it locally"
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help="Hugging Face dataset repo id, for example user/dataset",
    )
    parser.add_argument(
        "--remote-root",
        default=DEFAULT_REMOTE_ROOT,
        help="Optional subdirectory inside the dataset repo that contains docs_selected.jsonl",
    )
    parser.add_argument("--output-root", required=True, help="Directory where docs, tokenizers, shards, and manifest are written")
    parser.add_argument(
        "--tokenizer-config",
        default=str(DEFAULT_CONFIG),
        help="Local tokenizer config JSON. Defaults to data/tokenizer_specs.json.",
    )
    parser.add_argument(
        "--num-val-docs",
        type=int,
        default=None,
        help="Validation document count. Defaults to the downloaded sidecar when present, otherwise 50000.",
    )
    parser.add_argument("--chunk-tokens", type=int, default=SHARD_SIZE, help="Shard size in tokens.")
    parser.add_argument(
        "--max-train-shards",
        type=int,
        default=None,
        help="Stop export after writing this many training shards for each tokenizer.",
    )
    parser.add_argument(
        "--tokenizer-train-docs",
        type=int,
        default=None,
        help="Limit the number of docs used for tokenizer training.",
    )
    parser.add_argument("--skip-byte", action="store_true", help="Skip byte-tokenizer export.")
    parser.add_argument(
        "--reuse-sp-model",
        action="append",
        default=[],
        metavar="VOCAB=MODEL",
        help="Reuse an existing SentencePiece model for the given vocab size instead of retraining it.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.chunk_tokens <= 0:
        raise ValueError(f"--chunk_tokens must be positive, got {args.chunk_tokens}")

    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    tokenizers_dir = output_root / "tokenizers"
    datasets_dir = output_root / "datasets"
    tokenizers_dir.mkdir(parents=True, exist_ok=True)
    datasets_dir.mkdir(parents=True, exist_ok=True)

    docs_jsonl = output_root / DOCS_FILENAME
    sidecar = output_root / SIDECAR_FILENAME
    if not copy_from_hf_cache(
        repo_id=args.repo_id,
        remote_root=args.remote_root,
        filename=DOCS_FILENAME,
        destination=docs_jsonl,
    ):
        remote = f"{args.remote_root}/{DOCS_FILENAME}" if args.remote_root else DOCS_FILENAME
        raise FileNotFoundError(f"{remote} not found in Hugging Face dataset repo {args.repo_id}")
    if not copy_from_hf_cache(
        repo_id=args.repo_id,
        remote_root=args.remote_root,
        filename=SIDECAR_FILENAME,
        destination=sidecar,
    ):
        sidecar.unlink(missing_ok=True)

    docs_sidecar = maybe_load_docs_sidecar_meta(docs_jsonl)
    docs_total = int(docs_sidecar["num_docs"]) if docs_sidecar is not None and docs_sidecar.get("num_docs") is not None else count_docs(docs_jsonl)
    if args.num_val_docs is not None:
        num_val_docs = int(args.num_val_docs)
    elif docs_sidecar is not None and docs_sidecar.get("docs_val") is not None:
        num_val_docs = int(docs_sidecar["docs_val"])
    else:
        num_val_docs = NUM_VAL_DOCS
    if not (0 <= num_val_docs <= docs_total):
        raise ValueError(f"num_val_docs must be in [0, {docs_total}], got {num_val_docs}")

    specs = load_specs(Path(args.tokenizer_config).expanduser().resolve())
    reuse_sp_models = parse_reuse_sp_models(args.reuse_sp_model)
    tokenizers, selected_specs = build_tokenizers(
        specs=specs,
        docs_jsonl=docs_jsonl,
        tokenizers_dir=tokenizers_dir,
        tokenizer_train_docs=args.tokenizer_train_docs,
        skip_byte=args.skip_byte,
        reuse_sp_models=reuse_sp_models,
    )
    write_tokenizer_config_export(output_root, selected_specs)

    docs_meta = {
        "remote_repo_id": args.repo_id,
        "remote_root": args.remote_root,
        "num_docs": docs_total,
        "docs_sha256": None if docs_sidecar is None else docs_sidecar.get("docs_sha256"),
        "source_manifest": str(docs_sidecar_path(docs_jsonl)) if docs_sidecar is not None else None,
    }
    if docs_sidecar is not None:
        docs_meta["source_sidecar"] = docs_sidecar

    manifest = {
        "version": VERSION,
        "num_docs": docs_total,
        "num_val_docs": num_val_docs,
        "shuffle_seed": None if docs_sidecar is None else docs_sidecar.get("shuffle_seed"),
        "shard_size": int(args.chunk_tokens),
        "append_eos": APPEND_EOS,
        "docs_jsonl": str(docs_jsonl),
        "docs_meta": docs_meta,
        "tokenizer_specs": selected_specs,
        "tokenizers": [],
        "datasets": [],
    }

    for tok in tokenizers:
        output_dir = datasets_dir / tok["dataset_name"]
        print(f"Exporting dataset: {tok['dataset_name']}", flush=True)
        stats = export_shards(
            docs_jsonl,
            tok,
            output_dir,
            num_val_docs=num_val_docs,
            shard_size=int(args.chunk_tokens),
            docs_total=docs_total,
            max_train_shards=args.max_train_shards,
        )
        manifest["tokenizers"].append(tok["manifest"])
        manifest["datasets"].append(
            {
                "name": tok["dataset_name"],
                "tokenizer_name": tok["name"],
                "tokenizer_kind": tok["kind"],
                "path": str(output_dir),
                "train_glob": str(output_dir / "fineweb_train_*.bin"),
                "val_glob": str(output_dir / "fineweb_val_*.bin"),
                "vocab_size": tok["vocab_size"],
                "bos_id": tok["bos_id"],
                "eos_id": tok["eos_id"],
                "recommended_bigram_vocab_size": tok["recommended_bigram_vocab_size"],
                "val_bytes_glob": str(output_dir / "fineweb_val_bytes_*.bin"),
                "stats": stats,
            }
        )

    manifest = relativize_manifest_paths(manifest, output_root)
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Done. Manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
