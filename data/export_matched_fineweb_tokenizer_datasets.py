"""Export matched FineWeb datasets across tokenizer variants.

USER-EDITABLE:
- Pass `--tokenizer_config`, or edit `data/demo_tokenizer_specs.json`.
- Point each tokenizer spec at a Python builder in `data/demo_tokenizer_builders.py`
  or your own file/module.
- Custom tokenizer configs execute Python builder code. Only use them with trusted
  local configs, and pass `--trust_tokenizer_config_code` to opt in.

FIXED:
- The challenge dataset selection, split, shard format, and manifest fields below.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
from datasets import load_dataset
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent
VERSION = "10B"
REMOTE_NAME = "sample-10BT"
DATASET_REVISION = "9bb295ddab0e05d785b879661af7260fed5140fc"
NUM_DOCS = 2_000_000
NUM_VAL_DOCS = 50_000
SHARD_SIZE = 10**8
SHUFFLE_SEED = 1337
APPEND_EOS = False
OUTPUT_ROOT = ROOT / "challenge_fineweb"
DEMO_CONFIG = ROOT / "demo_tokenizer_specs.json"
EXPECTED_DOCS_SHA256 = "47812b882b6a11cf0f7cbdfeca77fb590785fbbda991db9599619633bfe9bca9"
SP_BATCH_SIZE = int(os.environ.get("MATCHED_FINEWEB_SP_BATCH_SIZE", "1024"))


def write_datafile(path: Path, toks: Any) -> None:
    if len(toks) >= 2**31:
        raise ValueError("token count too large")
    header = np.zeros(256, dtype="<i4")
    header[0] = 20240520
    header[1] = 1
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


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


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


def load_builder(ref: str, base_dir: Path):
    module_ref, fn_name = ref.split(":", 1)
    if module_ref.endswith(".py") or "/" in module_ref:
        module_path = (base_dir / module_ref).expanduser().resolve() if not Path(module_ref).is_absolute() else Path(module_ref)
        spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load builder module from {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_ref)
    fn = getattr(module, fn_name)
    if not callable(fn):
        raise TypeError(f"Builder reference is not callable: {ref}")
    return fn


def load_specs(config_path: str | None) -> list[dict[str, Any]]:
    config_path = Path(config_path or DEMO_CONFIG)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    specs = payload["tokenizers"] if isinstance(payload, dict) else payload
    if not isinstance(specs, list) or not specs:
        raise ValueError("tokenizer_config must define a non-empty list")
    for spec in specs:
        if not isinstance(spec, dict):
            raise ValueError("each tokenizer spec must be a JSON object")
        if not isinstance(spec.get("builder"), str) or not spec["builder"]:
            raise ValueError("each tokenizer spec needs a builder")
        spec["_base_dir"] = str(config_path.parent)
    return specs


def build_docs_cache(path: Path) -> dict[str, Any]:
    print(f"Loading dataset HuggingFaceFW/fineweb name={REMOTE_NAME} revision={DATASET_REVISION}")
    ds = load_dataset("HuggingFaceFW/fineweb", name=REMOTE_NAME, split="train", revision=DATASET_REVISION)
    ds = ds.shuffle(seed=SHUFFLE_SEED)
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open("w", encoding="utf-8") as f:
        pbar = tqdm(total=NUM_DOCS, unit="docs", desc="Caching docs")
        for row in ds:
            text = row.get("text", "")
            if not isinstance(text, str):
                text = "" if text is None else str(text)
            f.write(json.dumps({"text": text}, ensure_ascii=False))
            f.write("\n")
            written += 1
            pbar.update(1)
            if written == NUM_DOCS:
                break
        pbar.close()
    if written != NUM_DOCS:
        raise ValueError(f"Expected to cache {NUM_DOCS} docs, wrote {written}")
    return {
        "remote_name": REMOTE_NAME,
        "num_docs": written,
        "docs_sha256": sha256(path),
        "dataset_fingerprint": getattr(ds, "_fingerprint", None),
    }


def export_shards(
    docs_jsonl: Path,
    tok: dict[str, Any],
    output_dir: Path,
    *,
    num_val_docs: int,
    shard_size: int,
    docs_total_hint: int | None,
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
    fill = 0
    split = "val"
    shards = {"val": 0, "train": 0}

    def flush() -> None:
        nonlocal fill
        if fill == 0:
            return
        write_datafile(output_dir / f"fineweb_{split}_{shards[split]:06d}.bin", buf[:fill])
        stats["files_total"] += 1
        stats[f"files_{split}"] += 1
        shards[split] += 1
        fill = 0

    vocab_size = int(tok["vocab_size"])
    if vocab_size > 2**16:
        raise ValueError(f"vocab_size={vocab_size} is too large for uint16 shard storage")
    docs_total = docs_total_hint if docs_total_hint is not None else count_docs(docs_jsonl)
    batch_encode = tok.get("encode_batch")
    batch_size = SP_BATCH_SIZE if callable(batch_encode) else 1
    pbar = tqdm(total=docs_total, unit="docs", desc=f"Tokenizing {output_dir.name}")
    doc_index = 0
    for texts in batched_docs_jsonl(docs_jsonl, batch_size):
        encoded_docs = batch_encode(texts) if callable(batch_encode) else [tok["encode"](text) for text in texts]
        for text, encoded in zip(texts, encoded_docs, strict=True):
            del text
            next_split = "val" if doc_index < num_val_docs else "train"
            if next_split != split:
                flush()
                split = next_split

            encoded_arr = np.asarray(encoded, dtype=np.int32)
            toks = np.empty((encoded_arr.size + 1 + int(APPEND_EOS),), dtype=np.int32)
            toks[0] = tok["bos_id"]
            toks[1 : 1 + encoded_arr.size] = encoded_arr
            if APPEND_EOS:
                toks[-1] = tok["eos_id"]
            if not ((0 <= toks).all() and (toks < vocab_size).all()):
                bad = int(toks[(toks < 0) | (toks >= vocab_size)][0])
                raise ValueError(f"token id {bad} outside declared vocab_size={vocab_size}")
            toks = toks.astype("<u2", copy=False)

            stats["docs_total"] += 1
            stats[f"docs_{split}"] += 1
            stats["tokens_total"] += len(toks)
            stats[f"tokens_{split}"] += len(toks)

            pos = 0
            while pos < len(toks):
                take = min(shard_size - fill, len(toks) - pos)
                buf[fill : fill + take] = toks[pos : pos + take]
                fill += take
                pos += take
                if fill == shard_size:
                    flush()
            pbar.update(1)
            doc_index += 1
    pbar.close()
    flush()
    return stats


def parse_reuse_sp_models(values: list[str]) -> dict[int, Path]:
    reuse_models: dict[int, Path] = {}
    for value in values:
        vocab_size_str, model_path = value.split("=", 1)
        vocab_size = int(vocab_size_str)
        if vocab_size in reuse_models:
            raise ValueError(f"duplicate --reuse_sp_model for vocab_size={vocab_size}")
        reuse_models[vocab_size] = Path(model_path).expanduser().resolve()
    return reuse_models


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export matched FineWeb datasets across tokenizer variants")
    parser.add_argument("--tokenizer_config", type=str, default=None)
    parser.add_argument(
        "--trust_tokenizer_config_code",
        action="store_true",
        help="Allow a custom tokenizer config to import and execute its builder code. "
        "Required for non-bundled configs because the builder field is trusted code.",
    )
    parser.add_argument("--output_root", type=str, default=str(OUTPUT_ROOT))
    parser.add_argument("--docs_jsonl", type=str, default=None)
    parser.add_argument("--rebuild_docs_cache", action="store_true")
    parser.add_argument("--num_val_docs", type=int, default=None)
    parser.add_argument("--chunk_tokens", type=int, default=SHARD_SIZE)
    parser.add_argument("--tokenizer_train_docs", type=int, default=None)
    parser.add_argument("--skip_byte", action="store_true")
    parser.add_argument(
        "--reuse_sp_model",
        action="append",
        default=[],
        metavar="VOCAB=MODEL",
        help="Reuse an existing SentencePiece model for the given vocab size instead of retraining it.",
    )
    return parser


def run_export(args: argparse.Namespace) -> dict[str, Any]:
    if args.chunk_tokens <= 0:
        raise ValueError(f"--chunk_tokens must be positive, got {args.chunk_tokens}")

    config_path = Path(args.tokenizer_config).resolve() if args.tokenizer_config else DEMO_CONFIG.resolve()
    trusted_builder_code = config_path == DEMO_CONFIG.resolve() or args.trust_tokenizer_config_code
    if not trusted_builder_code:
        raise ValueError(
            "Custom tokenizer configs are trusted-code only because `builder` is imported and executed. "
            "Rerun with --trust_tokenizer_config_code only for a trusted local config."
        )

    output_root = Path(args.output_root).resolve()
    tokenizers_dir = output_root / "tokenizers"
    datasets_dir = output_root / "datasets"
    tokenizers_dir.mkdir(parents=True, exist_ok=True)
    datasets_dir.mkdir(parents=True, exist_ok=True)
    docs_jsonl = Path(args.docs_jsonl).resolve() if args.docs_jsonl else output_root / "docs_selected.jsonl"
    docs_sidecar = maybe_load_docs_sidecar_meta(docs_jsonl) if docs_jsonl.exists() else None
    reuse_sp_models = parse_reuse_sp_models(args.reuse_sp_model)

    if args.docs_jsonl and args.rebuild_docs_cache:
        raise ValueError("--rebuild_docs_cache conflicts with --docs_jsonl")

    build_cache = args.docs_jsonl is None and (args.rebuild_docs_cache or not docs_jsonl.exists())
    if build_cache:
        docs_meta = build_docs_cache(docs_jsonl)
    else:
        if not docs_jsonl.is_file():
            raise FileNotFoundError(docs_jsonl)
        docs_sidecar = maybe_load_docs_sidecar_meta(docs_jsonl)
        docs_total_hint = docs_sidecar.get("num_docs") if docs_sidecar is not None else None
        docs_meta = {
            "remote_name": "external_cache" if args.docs_jsonl else "cached_local",
            "num_docs": int(docs_total_hint) if docs_total_hint is not None else count_docs(docs_jsonl),
            "docs_sha256": None if args.docs_jsonl else sha256(docs_jsonl),
            "dataset_fingerprint": None,
        }
        if docs_sidecar is not None:
            docs_meta["source_manifest"] = str(docs_sidecar_path(docs_jsonl))
            if docs_sidecar.get("docs_sha256") is not None:
                docs_meta["docs_sha256"] = docs_sidecar["docs_sha256"]

    use_challenge_fairness_checks = args.docs_jsonl is None
    if use_challenge_fairness_checks and docs_meta["num_docs"] != NUM_DOCS:
        raise ValueError(f"docs cache must contain exactly {NUM_DOCS} docs, got {docs_meta['num_docs']}")
    if use_challenge_fairness_checks and EXPECTED_DOCS_SHA256 is not None:
        if docs_meta["docs_sha256"] != EXPECTED_DOCS_SHA256:
            raise ValueError(
                f"docs cache sha256 mismatch: expected {EXPECTED_DOCS_SHA256}, got {docs_meta['docs_sha256']}"
            )

    # Challenge-fairness rules:
    # - Users may change tokenizer specs only.
    # - Users must not change VERSION, DATASET_REVISION, NUM_DOCS, NUM_VAL_DOCS,
    #   SHUFFLE_SEED, APPEND_EOS, or SHARD_SIZE if they want comparable scores.
    # - docs_sha256 is the final check that two exports used the same raw docs.

    specs = load_specs(str(config_path))
    if args.skip_byte:
        specs = [spec for spec in specs if "byte" not in spec["builder"]]
    tokenizers: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    seen_datasets: set[str] = set()
    for spec in specs:
        spec = dict(spec)
        if args.tokenizer_train_docs is not None and "sentencepiece" in spec["builder"]:
            spec["tokenizer_train_docs"] = int(args.tokenizer_train_docs)
        if "vocab_size" in spec and int(spec["vocab_size"]) in reuse_sp_models:
            spec["reuse_model_path"] = str(reuse_sp_models[int(spec["vocab_size"])])
        built = load_builder(spec["builder"], Path(spec.pop("_base_dir")))(spec=spec, docs_jsonl=docs_jsonl, tokenizers_dir=tokenizers_dir)
        built = built if isinstance(built, list) else [built]
        for raw in built:
            name = str(raw["name"])
            dataset_suffix = raw.get("dataset_suffix")
            if dataset_suffix is None:
                dataset_suffix = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
            dataset_name = str(raw.get("dataset_name", f"fineweb{VERSION}_{dataset_suffix}"))
            vocab_size = int(raw["vocab_size"])
            bos_id = int(raw["bos_id"])
            eos_id = int(raw["eos_id"])
            encode = raw["encode"]
            encode_batch = raw.get("encode_batch")
            if name in seen_names:
                raise ValueError(f"duplicate tokenizer name: {name}")
            if dataset_name in seen_datasets:
                raise ValueError(f"duplicate dataset name: {dataset_name}")
            seen_names.add(name)
            seen_datasets.add(dataset_name)
            tokenizers.append(
                {
                    "name": name,
                    "kind": str(raw.get("kind", "custom")),
                    "dataset_name": dataset_name,
                    "vocab_size": vocab_size,
                    "bos_id": bos_id,
                    "eos_id": eos_id,
                    "encode": encode,
                    "encode_batch": encode_batch,
                    "recommended_bigram_vocab_size": int(
                        raw.get("recommended_bigram_vocab_size", ((vocab_size + 127) // 128) * 128 * 5)
                    ),
                    "manifest": {
                        "name": name,
                        "kind": str(raw.get("kind", "custom")),
                        "vocab_size": vocab_size,
                        "bos_id": bos_id,
                        "eos_id": eos_id,
                        "recommended_bigram_vocab_size": int(
                            raw.get("recommended_bigram_vocab_size", ((vocab_size + 127) // 128) * 128 * 5)
                        ),
                        "source_spec": spec,
                        **(raw.get("manifest") or {}),
                    },
                }
            )

    docs_total = int(docs_meta["num_docs"])
    docs_total_hint = docs_total
    if args.num_val_docs is not None:
        num_val_docs = int(args.num_val_docs)
    elif docs_sidecar is not None and docs_sidecar.get("docs_val") is not None:
        num_val_docs = int(docs_sidecar["docs_val"])
    else:
        num_val_docs = NUM_VAL_DOCS
    if not (0 <= num_val_docs <= docs_total):
        raise ValueError(f"num_val_docs must be in [0, {docs_total}], got {num_val_docs}")
    manifest_shuffle_seed = (
        int(docs_sidecar["shuffle_seed"])
        if docs_sidecar is not None and docs_sidecar.get("shuffle_seed") is not None
        else SHUFFLE_SEED
    )

    manifest = {
        "version": VERSION,
        "num_docs": docs_total,
        "num_val_docs": num_val_docs,
        "shuffle_seed": manifest_shuffle_seed,
        "dataset_revision": DATASET_REVISION,
        "shard_size": int(args.chunk_tokens),
        "append_eos": APPEND_EOS,
        "docs_jsonl": str(docs_jsonl),
        "docs_meta": docs_meta,
        "tokenizer_specs": specs,
        "tokenizers": [],
        "datasets": [],
    }

    for tok in tokenizers:
        output_dir = datasets_dir / tok["dataset_name"]
        print(f"Exporting dataset: {tok['dataset_name']}")
        stats = export_shards(
            docs_jsonl,
            tok,
            output_dir,
            num_val_docs=num_val_docs,
            shard_size=int(args.chunk_tokens),
            docs_total_hint=docs_total_hint,
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
                "stats": stats,
            }
        )

    manifest = relativize_manifest_paths(manifest, output_root)

    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"\nDone. Manifest: {manifest_path}")
    for ds in manifest["datasets"]:
        print(f"- {ds['name']}: vocab_size={ds['vocab_size']} bos_id={ds['bos_id']}")
        print(f"  TRAIN_FILES={ds['train_glob']}")
        print(f"  VAL_FILES={ds['val_glob']}")
    return manifest


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_export(args)


if __name__ == "__main__":
    main()
