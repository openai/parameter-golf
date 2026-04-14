from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

from huggingface_hub import hf_hub_download


REPO_ROOT = Path(__file__).resolve().parent
HF_REPO_ID = os.environ.get("MATCHED_FINEWEB_REPO_ID", "willdepueoai/parameter-golf")
REMOTE_ROOT_PREFIX = os.environ.get("MATCHED_FINEWEB_REMOTE_ROOT_PREFIX", "datasets")
HISTORICAL_BASELINE_REVISION = "1f2782522e6326a78ca5f1ed8edfb2eeeaf08d11"
DATASET_NAME = "fineweb10B_sp1024"
VOCAB_SIZE = 1024

ROOT_TRAIN_GPT = REPO_ROOT / "train_gpt.py"
RECORD_TRAIN_GPT = REPO_ROOT / "records/track_10min_16mb/2026-03-17_NaiveBaseline/train_gpt.py"
SNAPSHOT_ROOT = REPO_ROOT / "data" / "snapshots"


def _resolve_revision(value: str) -> str | None:
    return None if value in ("", "head", "current") else value


def _snapshot_label(hf_revision: str | None) -> str:
    return hf_revision or "head"


def _snapshot_root(hf_revision: str | None) -> Path:
    return SNAPSHOT_ROOT / _snapshot_label(hf_revision)


def _local_path_for_remote(relative_path: str, hf_revision: str | None) -> Path:
    snapshot_root = _snapshot_root(hf_revision)
    remote_path = Path(relative_path)
    if REMOTE_ROOT_PREFIX and remote_path.parts[:1] == (REMOTE_ROOT_PREFIX,):
        remote_path = remote_path.relative_to(REMOTE_ROOT_PREFIX)
    if remote_path.parts[:1] == ("datasets",):
        return snapshot_root.joinpath(*remote_path.parts)
    if remote_path.parts[:1] == ("tokenizers",):
        return snapshot_root.joinpath(*remote_path.parts)
    return snapshot_root / remote_path


def _download(relative_path: str, hf_revision: str | None) -> Path:
    destination = _local_path_for_remote(relative_path, hf_revision)
    if destination.exists():
        return destination

    remote_path = Path(relative_path)
    cached_path = Path(
        hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=remote_path.name,
            subfolder=remote_path.parent.as_posix() if remote_path.parent != Path(".") else None,
            repo_type="dataset",
            revision=hf_revision,
        )
    ).resolve(strict=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(cached_path, destination)
    except OSError:
        shutil.copy2(cached_path, destination)
    return destination


def _load_manifest(hf_revision: str | None) -> dict:
    manifest_path = _download(f"{REMOTE_ROOT_PREFIX}/manifest.json", hf_revision)
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _ensure_sp1024_data(
    *,
    train_shards: int,
    val_shards: int | None,
    hf_revision: str | None,
) -> tuple[Path, Path, int, dict]:
    manifest = _load_manifest(hf_revision)
    dataset_entry = next(x for x in manifest["datasets"] if x["name"] == DATASET_NAME)
    tokenizer_entry = next(x for x in manifest["tokenizers"] if x["name"] == dataset_entry["tokenizer_name"])

    max_train_shards = int(dataset_entry["stats"]["files_train"])
    max_val_shards = int(dataset_entry["stats"]["files_val"])
    if train_shards < 0 or train_shards > max_train_shards:
        raise ValueError(f"train_shards must be in [0, {max_train_shards}], got {train_shards}")
    val_shards_to_fetch = max_val_shards if val_shards is None else val_shards
    if val_shards_to_fetch <= 0 or val_shards_to_fetch > max_val_shards:
        raise ValueError(f"val_shards must be in [1, {max_val_shards}] or None, got {val_shards}")

    dataset_prefix = f"{REMOTE_ROOT_PREFIX}/datasets/{DATASET_NAME}"
    for i in range(val_shards_to_fetch):
        _download(f"{dataset_prefix}/fineweb_val_{i:06d}.bin", hf_revision)
    for i in range(train_shards):
        _download(f"{dataset_prefix}/fineweb_train_{i:06d}.bin", hf_revision)

    for key in ("model_path", "vocab_path", "path"):
        artifact = tokenizer_entry.get(key)
        if artifact:
            _download(f"{REMOTE_ROOT_PREFIX}/{artifact}", hf_revision)

    snapshot_root = _snapshot_root(hf_revision)
    tokenizer_model = snapshot_root / "tokenizers" / "fineweb_1024_bpe.model"
    dataset_dir = snapshot_root / "datasets" / DATASET_NAME
    return dataset_dir, tokenizer_model, val_shards_to_fetch, dataset_entry["stats"]


def _common_parser_defaults(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--hf-revision",
        default=HISTORICAL_BASELINE_REVISION,
        help="HF dataset revision to pin. Use 'head' for the current repo head.",
    )
    parser.add_argument(
        "--train-shards",
        type=int,
        default=25,
        help="Number of train shards to download/use.",
    )


def _cmd_prefetch(args: argparse.Namespace) -> None:
    hf_revision = _resolve_revision(args.hf_revision)
    dataset_dir, tokenizer_path, downloaded_val_shards, dataset_stats = _ensure_sp1024_data(
        train_shards=args.train_shards,
        val_shards=args.val_shards,
        hf_revision=hf_revision,
    )
    print(
        json.dumps(
            {
                "hf_repo_id": HF_REPO_ID,
                "hf_revision": _snapshot_label(hf_revision),
                "train_shards": args.train_shards,
                "val_shards": downloaded_val_shards,
                "dataset_dir": str(dataset_dir),
                "tokenizer_path": str(tokenizer_path),
                "manifest_stats": dataset_stats,
            },
            indent=2,
            sort_keys=True,
        )
    )


def _cmd_run(args: argparse.Namespace) -> None:
    hf_revision = _resolve_revision(args.hf_revision)
    dataset_dir, tokenizer_path, downloaded_val_shards, dataset_stats = _ensure_sp1024_data(
        train_shards=args.train_shards,
        val_shards=None,
        hf_revision=hf_revision,
    )
    if args.script_variant == "record":
        train_script = RECORD_TRAIN_GPT
    elif args.script_variant == "root":
        train_script = ROOT_TRAIN_GPT
    else:
        raise ValueError("script_variant must be 'root' or 'record'")

    run_id = args.run_id or f"runpod_baseline_seed{args.seed}"
    env = os.environ.copy()
    env.update(
        {
            "HF_HOME": str(REPO_ROOT / ".hf-cache"),
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "1",
            "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
            "NCCL_ASYNC_ERROR_HANDLING": "1",
            "NCCL_IB_DISABLE": "1",
            "RUN_ID": run_id,
            "SEED": str(args.seed),
            "DATA_PATH": str(dataset_dir),
            "TOKENIZER_PATH": str(tokenizer_path),
            "VOCAB_SIZE": str(VOCAB_SIZE),
            "ITERATIONS": str(args.iterations),
            "MAX_WALLCLOCK_SECONDS": str(args.max_wallclock_seconds),
            "TRAIN_LOG_EVERY": str(args.train_log_every),
            "VAL_LOSS_EVERY": str(args.val_loss_every),
        }
    )
    cmd = [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={args.nproc_per_node}",
        str(train_script),
    ]
    print(f"[runpod] Launching {' '.join(cmd)}")
    print(f"[runpod] SCRIPT_VARIANT={args.script_variant}")
    print(
        f"[runpod] REPO_ID={HF_REPO_ID} HF_REVISION={_snapshot_label(hf_revision)} "
        f"train_shards={args.train_shards} val_shards={downloaded_val_shards}"
    )
    print(
        "[runpod] MANIFEST "
        f"files_train={dataset_stats['files_train']} files_val={dataset_stats['files_val']} "
        f"tokens_train={dataset_stats['tokens_train']} tokens_val={dataset_stats['tokens_val']}"
    )
    print(f"[runpod] DATA_PATH={dataset_dir}")
    print(f"[runpod] TOKENIZER_PATH={tokenizer_path}")
    subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prefetch and run the historical baseline on a RunPod VM")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prefetch = subparsers.add_parser("prefetch", help="Download the pinned snapshot to local disk")
    _common_parser_defaults(prefetch)
    prefetch.add_argument(
        "--val-shards",
        type=int,
        default=1,
        help="Number of validation shards to download. Use 1 for the full fixed split on the historical baseline.",
    )
    prefetch.set_defaults(func=_cmd_prefetch)

    run = subparsers.add_parser("run", help="Run the baseline with torchrun on the local VM")
    _common_parser_defaults(run)
    run.add_argument("--script-variant", default="root", choices=("root", "record"))
    run.add_argument("--nproc-per-node", type=int, default=8)
    run.add_argument("--iterations", type=int, default=20000)
    run.add_argument("--max-wallclock-seconds", type=int, default=600)
    run.add_argument("--seed", type=int, default=1337)
    run.add_argument("--run-id", default="")
    run.add_argument("--train-log-every", type=int, default=50)
    run.add_argument("--val-loss-every", type=int, default=200)
    run.set_defaults(func=_cmd_run)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
