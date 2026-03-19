from __future__ import annotations

import argparse
import importlib.util
import json
import math
import shutil
import sys
import urllib.request
from pathlib import Path


HF_REPO_ID = "willdepueoai/parameter-golf"
HF_ROOT = "datasets"
DOCS_URL = f"https://huggingface.co/datasets/{HF_REPO_ID}/resolve/main/{HF_ROOT}/docs_selected.jsonl"
SIDECAR_URL = f"https://huggingface.co/datasets/{HF_REPO_ID}/resolve/main/{HF_ROOT}/docs_selected.source_manifest.json"


def load_export_module(repo_root: Path):
    module_path = repo_root / "data" / "download_hf_docs_and_tokenize.py"
    spec = importlib.util.spec_from_file_location("local_export_module", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load export module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def download_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def stream_doc_prefix(
    *,
    docs_url: str,
    tokenizer_docs_path: Path,
    export_docs_path: Path,
    tokenizer_train_docs: int,
    export_docs: int,
) -> None:
    max_docs = max(tokenizer_train_docs, export_docs)
    tokenizer_docs_path.parent.mkdir(parents=True, exist_ok=True)
    export_docs_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(docs_url, timeout=60) as response, tokenizer_docs_path.open("w", encoding="utf-8") as tok_out, export_docs_path.open("w", encoding="utf-8") as exp_out:
        for idx, raw_line in enumerate(response, start=1):
            line = raw_line.decode("utf-8")
            if idx <= tokenizer_train_docs:
                tok_out.write(line)
            if idx <= export_docs:
                exp_out.write(line)
            if idx % 50000 == 0:
                print(f"downloaded_docs:{idx}", flush=True)
            if idx >= max_docs:
                break


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a lightweight local SP-4096 subset export from the published selected-doc stream")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--tokenizer-train-docs", type=int, default=500000)
    parser.add_argument("--export-docs", type=int, default=120000)
    parser.add_argument("--num-val-docs", type=int, default=50000)
    parser.add_argument("--chunk-tokens", type=int, default=20000000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.export_docs <= args.num_val_docs:
        raise ValueError("--export-docs must be larger than --num-val-docs")

    repo_root = Path(__file__).resolve().parents[1]
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    tokenizers_dir = output_root / "tokenizers"
    datasets_dir = output_root / "datasets"
    tokenizers_dir.mkdir(parents=True, exist_ok=True)
    datasets_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_docs_jsonl = output_root / "docs_selected_tokenizer_train.jsonl"
    export_docs_jsonl = output_root / "docs_selected.jsonl"
    export_sidecar = output_root / "docs_selected.source_manifest.json"

    if not tokenizer_docs_jsonl.is_file() or not export_docs_jsonl.is_file():
        stream_doc_prefix(
            docs_url=DOCS_URL,
            tokenizer_docs_path=tokenizer_docs_jsonl,
            export_docs_path=export_docs_jsonl,
            tokenizer_train_docs=args.tokenizer_train_docs,
            export_docs=args.export_docs,
        )

    source_sidecar = download_json(SIDECAR_URL)
    subset_sidecar = {
        "source_repo_id": HF_REPO_ID,
        "source_remote_root": HF_ROOT,
        "source_num_docs": source_sidecar.get("num_docs"),
        "source_docs_val": source_sidecar.get("docs_val"),
        "num_docs": int(args.export_docs),
        "docs_val": int(args.num_val_docs),
        "docs_sha256": None,
        "subset_kind": "prefix",
        "tokenizer_train_docs": int(args.tokenizer_train_docs),
    }
    export_sidecar.write_text(json.dumps(subset_sidecar, indent=2) + "\n", encoding="utf-8")

    export_module = load_export_module(repo_root)
    spec = {
        "name": "sp_bpe_4096",
        "dataset_suffix": "sp4096_local",
        "vocab_size": 4096,
        "model_prefix": "fineweb_4096_bpe",
        "tokenizer_train_docs": int(args.tokenizer_train_docs),
    }
    tok = export_module.build_sentencepiece_tokenizer(
        spec=spec,
        docs_jsonl=tokenizer_docs_jsonl,
        tokenizers_dir=tokenizers_dir,
    )
    dataset_name = "fineweb10B_sp4096_local"
    output_dir = datasets_dir / dataset_name
    stats = export_module.export_shards(
        export_docs_jsonl,
        tok,
        output_dir,
        num_val_docs=int(args.num_val_docs),
        shard_size=int(args.chunk_tokens),
        docs_total=int(args.export_docs),
    )

    recommended_bigram_vocab_size = int(((int(tok["vocab_size"]) + 127) // 128) * 128 * 5)
    manifest = {
        "version": "local_subset",
        "num_docs": int(args.export_docs),
        "num_val_docs": int(args.num_val_docs),
        "tokenizer_train_docs": int(args.tokenizer_train_docs),
        "shard_size": int(args.chunk_tokens),
        "docs_jsonl": str(export_docs_jsonl),
        "tokenizers": [
            {
                "name": tok["name"],
                "kind": tok["kind"],
                "vocab_size": int(tok["vocab_size"]),
                "bos_id": int(tok["bos_id"]),
                "eos_id": int(tok["eos_id"]),
                "recommended_bigram_vocab_size": recommended_bigram_vocab_size,
                "source_spec": spec,
                **tok["manifest"],
            }
        ],
        "datasets": [
            {
                "name": dataset_name,
                "tokenizer_name": tok["name"],
                "tokenizer_kind": tok["kind"],
                "path": str(output_dir),
                "train_glob": str(output_dir / "fineweb_train_*.bin"),
                "val_glob": str(output_dir / "fineweb_val_*.bin"),
                "vocab_size": int(tok["vocab_size"]),
                "bos_id": int(tok["bos_id"]),
                "eos_id": int(tok["eos_id"]),
                "recommended_bigram_vocab_size": recommended_bigram_vocab_size,
                "stats": stats,
            }
        ],
    }
    manifest = export_module.relativize_manifest_paths(manifest, output_root)
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    target_tokenizer_dir = repo_root / "data" / "tokenizers"
    target_dataset_dir = repo_root / "data" / "datasets" / dataset_name
    target_tokenizer_dir.mkdir(parents=True, exist_ok=True)
    target_dataset_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(tokenizers_dir / "fineweb_4096_bpe.model", target_tokenizer_dir / "fineweb_4096_bpe.model")
    shutil.copy2(tokenizers_dir / "fineweb_4096_bpe.vocab", target_tokenizer_dir / "fineweb_4096_bpe.vocab")
    if target_dataset_dir.exists():
        shutil.rmtree(target_dataset_dir)
    shutil.copytree(output_dir, target_dataset_dir)

    print(f"tokenizer_model:{target_tokenizer_dir / 'fineweb_4096_bpe.model'}", flush=True)
    print(f"dataset_dir:{target_dataset_dir}", flush=True)
    print(f"dataset_stats:{json.dumps(stats, sort_keys=True)}", flush=True)


if __name__ == "__main__":
    main()
