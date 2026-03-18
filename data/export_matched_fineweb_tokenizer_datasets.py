"""Export matched FineWeb datasets across tokenizer variants.

USER-EDITABLE:
- Pass `--tokenizer_config`, or edit `data/demo_tokenizer_specs.json`.
- Point each tokenizer spec at a Python builder in `data/demo_tokenizer_builders.py`
  or your own file/module.

FIXED:
- The challenge dataset selection, split, shard format, and manifest fields below.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
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


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


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


def export_shards(docs_jsonl: Path, tok: dict[str, Any], output_dir: Path) -> dict[str, int]:
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
    buf = np.empty((SHARD_SIZE,), dtype=np.uint16)
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

    pbar = tqdm(total=count_docs(docs_jsonl), unit="docs", desc=f"Tokenizing {output_dir.name}")
    for i, text in enumerate(iter_docs(docs_jsonl)):
        next_split = "val" if i < NUM_VAL_DOCS else "train"
        if next_split != split:
            flush()
            split = next_split

        token_ids = [tok["bos_id"], *tok["encode"](text)]
        if APPEND_EOS:
            token_ids.append(tok["eos_id"])
        toks = np.asarray(token_ids, dtype=np.int32)
        vocab_size = int(tok["vocab_size"])
        if not ((0 <= toks).all() and (toks < vocab_size).all()):
            bad = int(toks[(toks < 0) | (toks >= vocab_size)][0])
            raise ValueError(f"token id {bad} outside declared vocab_size={vocab_size}")
        if vocab_size > 2**16:
            raise ValueError(f"vocab_size={vocab_size} is too large for uint16 shard storage")
        toks = toks.astype("<u2", copy=False)

        stats["docs_total"] += 1
        stats[f"docs_{split}"] += 1
        stats["tokens_total"] += len(toks)
        stats[f"tokens_{split}"] += len(toks)

        pos = 0
        while pos < len(toks):
            take = min(SHARD_SIZE - fill, len(toks) - pos)
            buf[fill : fill + take] = toks[pos : pos + take]
            fill += take
            pos += take
            if fill == SHARD_SIZE:
                flush()
        pbar.update(1)
    pbar.close()
    flush()
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Export matched FineWeb datasets across tokenizer variants")
    parser.add_argument("--tokenizer_config", type=str, default=None)
    parser.add_argument("--output_root", type=str, default=str(OUTPUT_ROOT))
    parser.add_argument("--docs_jsonl", type=str, default=None)
    parser.add_argument("--rebuild_docs_cache", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root).resolve()
    tokenizers_dir = output_root / "tokenizers"
    datasets_dir = output_root / "datasets"
    tokenizers_dir.mkdir(parents=True, exist_ok=True)
    datasets_dir.mkdir(parents=True, exist_ok=True)

    if args.docs_jsonl and args.rebuild_docs_cache:
        raise ValueError("--rebuild_docs_cache conflicts with --docs_jsonl")

    docs_jsonl = Path(args.docs_jsonl).resolve() if args.docs_jsonl else output_root / "docs_selected.jsonl"
    build_cache = args.docs_jsonl is None and (args.rebuild_docs_cache or not docs_jsonl.exists())
    if build_cache:
        docs_meta = build_docs_cache(docs_jsonl)
    else:
        if not docs_jsonl.is_file():
            raise FileNotFoundError(docs_jsonl)
        docs_meta = {
            "remote_name": "external_cache" if args.docs_jsonl else "cached_local",
            "num_docs": count_docs(docs_jsonl),
            "docs_sha256": sha256(docs_jsonl),
            "dataset_fingerprint": None,
        }
    if docs_meta["num_docs"] != NUM_DOCS:
        raise ValueError(f"docs cache must contain exactly {NUM_DOCS} docs, got {docs_meta['num_docs']}")
    if EXPECTED_DOCS_SHA256 is not None:
        if docs_meta["docs_sha256"] != EXPECTED_DOCS_SHA256:
            raise ValueError(
                f"docs cache sha256 mismatch: expected {EXPECTED_DOCS_SHA256}, got {docs_meta['docs_sha256']}"
            )

    # Challenge-fairness rules:
    # - Users may change tokenizer specs only.
    # - Users must not change VERSION, DATASET_REVISION, NUM_DOCS, NUM_VAL_DOCS,
    #   SHUFFLE_SEED, APPEND_EOS, or SHARD_SIZE if they want comparable scores.
    # - docs_sha256 is the final check that two exports used the same raw docs.

    specs = load_specs(args.tokenizer_config)
    tokenizers: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    seen_datasets: set[str] = set()
    for spec in specs:
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

    manifest = {
        "version": VERSION,
        "num_docs": NUM_DOCS,
        "num_val_docs": NUM_VAL_DOCS,
        "shuffle_seed": SHUFFLE_SEED,
        "dataset_revision": DATASET_REVISION,
        "shard_size": SHARD_SIZE,
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
        stats = export_shards(docs_jsonl, tok, output_dir)
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


if __name__ == "__main__":
    main()
