from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path


def load_export_module(repo_root: Path):
    module_path = repo_root / "data" / "download_hf_docs_and_tokenize.py"
    spec = importlib.util.spec_from_file_location("local_export_module_sp1024", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load export module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a local SP-1024 subset dataset from an existing selected-doc prefix")
    parser.add_argument("--docs-jsonl", required=True)
    parser.add_argument("--num-docs", type=int, default=120000)
    parser.add_argument("--num-val-docs", type=int, default=50000)
    parser.add_argument("--chunk-tokens", type=int, default=20000000)
    parser.add_argument("--dataset-name", default="fineweb10B_sp1024_local120k")
    parser.add_argument("--output-root", default="")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.num_docs <= args.num_val_docs:
        raise ValueError("--num-docs must be larger than --num-val-docs")

    repo_root = Path(__file__).resolve().parents[1]
    docs_jsonl = Path(args.docs_jsonl).expanduser().resolve()
    if not docs_jsonl.is_file():
        raise FileNotFoundError(docs_jsonl)

    output_root = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root
        else (repo_root / "data" / "sp1024_local_build").resolve()
    )
    output_root.mkdir(parents=True, exist_ok=True)
    tokenizers_dir = output_root / "tokenizers"
    datasets_dir = output_root / "datasets"
    tokenizers_dir.mkdir(parents=True, exist_ok=True)
    datasets_dir.mkdir(parents=True, exist_ok=True)

    export_module = load_export_module(repo_root)
    source_tokenizer = repo_root / "data" / "tokenizers" / "fineweb_1024_bpe.model"
    if not source_tokenizer.is_file():
        raise FileNotFoundError(source_tokenizer)

    spec = {
        "name": "sp_bpe_1024_local",
        "dataset_suffix": "sp1024_local120k",
        "vocab_size": 1024,
        "model_prefix": "fineweb_1024_bpe_local",
        "reuse_model_path": str(source_tokenizer),
    }
    tok = export_module.build_sentencepiece_tokenizer(
        spec=spec,
        docs_jsonl=docs_jsonl,
        tokenizers_dir=tokenizers_dir,
    )

    output_dir = datasets_dir / args.dataset_name
    stats = export_module.export_shards(
        docs_jsonl,
        tok,
        output_dir,
        num_val_docs=int(args.num_val_docs),
        shard_size=int(args.chunk_tokens),
        docs_total=int(args.num_docs),
    )

    manifest = {
        "version": "local_subset",
        "num_docs": int(args.num_docs),
        "num_val_docs": int(args.num_val_docs),
        "docs_jsonl": str(docs_jsonl),
        "tokenizers": [
            {
                "name": tok["name"],
                "kind": tok["kind"],
                "vocab_size": int(tok["vocab_size"]),
                "bos_id": int(tok["bos_id"]),
                "eos_id": int(tok["eos_id"]),
                "recommended_bigram_vocab_size": int(((int(tok["vocab_size"]) + 127) // 128) * 128 * 5),
                "source_spec": spec,
                **tok["manifest"],
            }
        ],
        "datasets": [
            {
                "name": args.dataset_name,
                "tokenizer_name": tok["name"],
                "tokenizer_kind": tok["kind"],
                "path": str(output_dir),
                "train_glob": str(output_dir / "fineweb_train_*.bin"),
                "val_glob": str(output_dir / "fineweb_val_*.bin"),
                "vocab_size": int(tok["vocab_size"]),
                "bos_id": int(tok["bos_id"]),
                "eos_id": int(tok["eos_id"]),
                "recommended_bigram_vocab_size": int(((int(tok["vocab_size"]) + 127) // 128) * 128 * 5),
                "stats": stats,
            }
        ],
    }
    manifest = export_module.relativize_manifest_paths(manifest, output_root)
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    target_dataset_dir = repo_root / "data" / "datasets" / args.dataset_name
    if target_dataset_dir.exists():
        shutil.rmtree(target_dataset_dir)
    shutil.copytree(output_dir, target_dataset_dir)

    print(f"dataset_dir:{target_dataset_dir}", flush=True)
    print(f"dataset_stats:{json.dumps(stats, sort_keys=True)}", flush=True)


if __name__ == "__main__":
    main()
