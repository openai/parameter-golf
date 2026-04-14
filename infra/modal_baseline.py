from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import modal
from huggingface_hub import hf_hub_download


APP_NAME = "parameter-golf-baseline"
REPO_ROOT = Path(__file__).resolve().parent
REPO_ID = os.environ.get("MATCHED_FINEWEB_REPO_ID", "willdepueoai/parameter-golf")
REMOTE_ROOT_PREFIX = os.environ.get("MATCHED_FINEWEB_REMOTE_ROOT_PREFIX", "datasets")
DATASET_NAME = "fineweb10B_sp1024"
VOCAB_SIZE = 1024
HISTORICAL_BASELINE_REVISION = "1f2782522e6326a78ca5f1ed8edfb2eeeaf08d11"
PYTHON_VERSION = "3.12"
TORCH_VERSION = "2.10.0+cu128"
TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu128"

CACHE_ROOT = Path("/cache")
REMOTE_CODE_DIR = Path("/root/code/parameter-golf")
ROOT_TRAIN_GPT_LOCAL = REPO_ROOT / "train_gpt.py"
BASELINE_RECORD_TRAIN_GPT_LOCAL = (
    REPO_ROOT / "records/track_10min_16mb/2026-03-17_NaiveBaseline/train_gpt.py"
)
REMOTE_ROOT_TRAIN_GPT = REMOTE_CODE_DIR / "train_gpt_root.py"
REMOTE_BASELINE_RECORD_TRAIN_GPT = REMOTE_CODE_DIR / "train_gpt_baseline_record.py"
SMOKE_GPU = "H100!"
BASELINE_GPU = "H100!:8"
SMOKE_MEMORY_MB = 16 * 1024
BASELINE_MEMORY_MB = 64 * 1024


def _snapshot_label(hf_revision: str | None) -> str:
    return hf_revision or "head"


def _snapshot_root(hf_revision: str | None) -> Path:
    return CACHE_ROOT / "snapshots" / _snapshot_label(hf_revision)


def _cache_path_for_remote(relative_path: str, hf_revision: str | None) -> Path:
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
    destination = _cache_path_for_remote(relative_path, hf_revision)
    if destination.exists():
        return destination

    remote_path = Path(relative_path)
    cached_path = Path(
        hf_hub_download(
            repo_id=REPO_ID,
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
    return snapshot_root / "datasets" / DATASET_NAME, tokenizer_model, val_shards_to_fetch, dataset_entry["stats"]


def _run_training(
    *,
    nproc_per_node: int,
    script_variant: str,
    hf_revision: str | None,
    train_shards: int,
    val_shards: int | None,
    iterations: int,
    max_wallclock_seconds: int,
    seed: int,
    run_id: str,
    train_log_every: int,
    val_loss_every: int,
) -> dict:
    modal_provider = os.environ.get("MODAL_CLOUD_PROVIDER", "")
    modal_region = os.environ.get("MODAL_REGION", "")
    os.environ.setdefault("HF_HOME", str(CACHE_ROOT / "hf"))
    volume.reload()
    dataset_dir, tokenizer_model, downloaded_val_shards, dataset_stats = _ensure_sp1024_data(
        train_shards,
        val_shards,
        hf_revision,
    )
    volume.commit()
    if script_variant == "record":
        remote_train_gpt = REMOTE_BASELINE_RECORD_TRAIN_GPT
    elif script_variant == "root":
        remote_train_gpt = REMOTE_ROOT_TRAIN_GPT
    else:
        raise ValueError("script_variant must be 'record' or 'root'")

    env = os.environ.copy()
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "1",
            "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
            "NCCL_ASYNC_ERROR_HANDLING": "1",
            "NCCL_IB_DISABLE": "1",
            "RUN_ID": run_id,
            "SEED": str(seed),
            "DATA_PATH": str(dataset_dir),
            "TOKENIZER_PATH": str(tokenizer_model),
            "VOCAB_SIZE": str(VOCAB_SIZE),
            "ITERATIONS": str(iterations),
            "MAX_WALLCLOCK_SECONDS": str(max_wallclock_seconds),
            "TRAIN_LOG_EVERY": str(train_log_every),
            "VAL_LOSS_EVERY": str(val_loss_every),
        }
    )

    cmd = [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={nproc_per_node}",
        str(remote_train_gpt),
    ]
    nvidia_smi_l = subprocess.run(
        ["nvidia-smi", "-L"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    nvidia_smi_topo = subprocess.run(
        ["nvidia-smi", "topo", "-m"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    print(f"[modal] Launching {' '.join(cmd)}")
    print(f"[modal] SCRIPT_VARIANT={script_variant}")
    print(f"[modal] MODAL_CLOUD_PROVIDER={modal_provider} MODAL_REGION={modal_region}")
    print(
        f"[modal] REPO_ID={REPO_ID} HF_REVISION={_snapshot_label(hf_revision)} "
        f"train_shards={train_shards} val_shards={downloaded_val_shards}"
    )
    print(
        "[modal] MANIFEST "
        f"files_train={dataset_stats['files_train']} files_val={dataset_stats['files_val']} "
        f"tokens_train={dataset_stats['tokens_train']} tokens_val={dataset_stats['tokens_val']}"
    )
    print(f"[modal] DATA_PATH={dataset_dir}")
    print(f"[modal] TOKENIZER_PATH={tokenizer_model}")
    if nvidia_smi_l.stdout:
        print("[modal] NVIDIA-SMI -L")
        print(nvidia_smi_l.stdout.rstrip())
    if nvidia_smi_topo.stdout:
        print("[modal] NVIDIA-SMI TOPO")
        print(nvidia_smi_topo.stdout.rstrip())
    subprocess.run(cmd, cwd=str(REMOTE_CODE_DIR), env=env, check=True)
    return {
        "run_id": run_id,
        "nproc_per_node": nproc_per_node,
        "script_variant": script_variant,
        "hf_revision": _snapshot_label(hf_revision),
        "train_shards": train_shards,
        "val_shards": downloaded_val_shards,
        "data_path": str(dataset_dir),
        "tokenizer_path": str(tokenizer_model),
    }


def _run_doctor(
    *,
    hf_revision: str | None,
    train_shards: int,
    val_shards: int | None,
) -> dict:
    import sentencepiece
    import torch
    import torch.distributed as dist

    os.environ.setdefault("HF_HOME", str(CACHE_ROOT / "hf"))
    volume.reload()
    dataset_dir, tokenizer_model, downloaded_val_shards, dataset_stats = _ensure_sp1024_data(
        train_shards,
        val_shards,
        hf_revision,
    )
    volume.commit()
    nvidia_smi = subprocess.run(
        ["nvidia-smi"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    torchrun_path = subprocess.run(
        ["which", "torchrun"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    code_listing = sorted(p.name for p in REMOTE_CODE_DIR.glob("*"))
    train_files = sorted(p.name for p in dataset_dir.glob("fineweb_train_*.bin"))
    val_files = sorted(p.name for p in dataset_dir.glob("fineweb_val_*.bin"))
    return {
        "python": sys.version,
        "modal_sdk": getattr(modal, "__version__", "unknown"),
        "torch": {
            "version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "device_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "nccl_available": dist.is_nccl_available(),
        },
        "sentencepiece": getattr(sentencepiece, "__version__", "unknown"),
        "paths": {
            "cache_root": str(CACHE_ROOT),
            "remote_code_dir": str(REMOTE_CODE_DIR),
            "dataset_dir": str(dataset_dir),
            "tokenizer_model": str(tokenizer_model),
        },
        "exists": {
            "remote_code_dir": REMOTE_CODE_DIR.exists(),
            "root_train_gpt": REMOTE_ROOT_TRAIN_GPT.exists(),
            "record_train_gpt": REMOTE_BASELINE_RECORD_TRAIN_GPT.exists(),
            "dataset_dir": dataset_dir.exists(),
            "tokenizer_model": tokenizer_model.exists(),
        },
        "code_listing": code_listing,
        "train_files_found": len(train_files),
        "val_files_found": len(val_files),
        "train_file_examples": train_files[:3],
        "val_file_examples": val_files[:3],
        "hf_revision": _snapshot_label(hf_revision),
        "downloaded_val_shards": downloaded_val_shards,
        "manifest_stats": dataset_stats,
        "env": {
            key: os.environ.get(key, "")
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "NVIDIA_VISIBLE_DEVICES",
                "HF_HOME",
                "RANK",
                "WORLD_SIZE",
                "LOCAL_RANK",
            )
        },
        "which_torchrun": {
            "returncode": torchrun_path.returncode,
            "stdout": torchrun_path.stdout.strip(),
            "stderr": torchrun_path.stderr.strip(),
        },
        "nvidia_smi": {
            "returncode": nvidia_smi.returncode,
            "stdout": nvidia_smi.stdout,
            "stderr": nvidia_smi.stderr,
        },
    }


prefetch_image = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .pip_install("huggingface-hub")
)
gpu_image = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .pip_install(f"torch=={TORCH_VERSION}", index_url=TORCH_INDEX_URL)
    .pip_install(
        "numpy",
        "tqdm",
        "sentencepiece",
        "huggingface-hub",
        "datasets",
        "tiktoken",
        "setuptools",
        "typing-extensions==4.15.0",
    )
    .add_local_file(str(ROOT_TRAIN_GPT_LOCAL), remote_path=str(REMOTE_ROOT_TRAIN_GPT), copy=True)
    .add_local_file(
        str(BASELINE_RECORD_TRAIN_GPT_LOCAL),
        remote_path=str(REMOTE_BASELINE_RECORD_TRAIN_GPT),
        copy=True,
    )
)
volume = modal.Volume.from_name("parameter-golf-fineweb-cache", create_if_missing=True)
app = modal.App(APP_NAME)


@app.function(
    image=prefetch_image,
    cpu=2,
    timeout=60 * 60,
    volumes={"/cache": volume},
)
def prefetch_sp1024_data(
    train_shards: int = 80,
    val_shards: int | None = None,
    hf_revision: str = "",
) -> dict:
    os.environ.setdefault("HF_HOME", str(CACHE_ROOT / "hf"))
    volume.reload()
    resolved_revision = hf_revision or None
    dataset_dir, tokenizer_model, downloaded_val_shards, dataset_stats = _ensure_sp1024_data(
        train_shards,
        val_shards,
        resolved_revision,
    )
    volume.commit()
    return {
        "hf_revision": _snapshot_label(resolved_revision),
        "train_shards": train_shards,
        "val_shards": downloaded_val_shards,
        "manifest_stats": dataset_stats,
        "data_path": str(dataset_dir),
        "tokenizer_path": str(tokenizer_model),
    }


@app.function(
    image=gpu_image,
    gpu=SMOKE_GPU,
    cpu=4,
    memory=SMOKE_MEMORY_MB,
    timeout=60 * 15,
    volumes={"/cache": volume},
)
def doctor(
    hf_revision: str = "",
    train_shards: int = 1,
    val_shards: int = 1,
) -> dict:
    return _run_doctor(
        hf_revision=hf_revision or None,
        train_shards=train_shards,
        val_shards=val_shards,
    )


@app.function(
    image=gpu_image,
    gpu=SMOKE_GPU,
    cpu=8,
    memory=SMOKE_MEMORY_MB,
    timeout=60 * 30,
    volumes={"/cache": volume},
)
def smoke_test(
    script_variant: str = "root",
    hf_revision: str = "",
    train_shards: int = 1,
    val_shards: int = 1,
    iterations: int = 400,
    max_wallclock_seconds: int = 120,
    seed: int = 1337,
    run_id: str = "modal_smoke_sp1024",
    train_log_every: int = 20,
    val_loss_every: int = 100,
) -> dict:
    return _run_training(
        nproc_per_node=1,
        script_variant=script_variant,
        hf_revision=hf_revision or None,
        train_shards=train_shards,
        val_shards=val_shards,
        iterations=iterations,
        max_wallclock_seconds=max_wallclock_seconds,
        seed=seed,
        run_id=run_id,
        train_log_every=train_log_every,
        val_loss_every=val_loss_every,
    )


@app.function(
    image=gpu_image,
    gpu=BASELINE_GPU,
    cpu=32,
    memory=BASELINE_MEMORY_MB,
    timeout=60 * 60 * 3,
    volumes={"/cache": volume},
)
def baseline_8x_h100(
    script_variant: str = "root",
    hf_revision: str = HISTORICAL_BASELINE_REVISION,
    train_shards: int = 25,
    iterations: int = 20000,
    max_wallclock_seconds: int = 600,
    seed: int = 1337,
    run_id: str = "modal_baseline_sp1024_8xh100",
    train_log_every: int = 50,
    val_loss_every: int = 200,
) -> dict:
    return _run_training(
        nproc_per_node=8,
        script_variant=script_variant,
        hf_revision=hf_revision or None,
        train_shards=train_shards,
        val_shards=None,
        iterations=iterations,
        max_wallclock_seconds=max_wallclock_seconds,
        seed=seed,
        run_id=run_id,
        train_log_every=train_log_every,
        val_loss_every=val_loss_every,
    )


@app.local_entrypoint()
def main(
    mode: str = "doctor",
    script_variant: str = "root",
    skip_prefetch: bool = False,
    hf_revision: str = "",
    train_shards: int = 0,
    val_shards: int = 0,
    iterations: int = 0,
    max_wallclock_seconds: int = 0,
    seed: int = 1337,
    run_id: str = "",
    train_log_every: int = 0,
    val_loss_every: int = -1,
) -> None:
    run_id = run_id or f"modal_{mode}_sp1024_seed{seed}"
    if mode == "doctor":
        train_shards = train_shards or 1
        val_shards = val_shards or 1
        if not skip_prefetch:
            print(
                json.dumps(
                    prefetch_sp1024_data.remote(
                        train_shards=train_shards,
                        val_shards=val_shards,
                        hf_revision=hf_revision,
                    ),
                    indent=2,
                    sort_keys=True,
                )
            )
        result = doctor.remote(
            hf_revision=hf_revision,
            train_shards=train_shards,
            val_shards=val_shards,
        )
    elif mode == "smoke":
        train_shards = train_shards or 1
        val_shards = val_shards or 1
        iterations = iterations or 400
        max_wallclock_seconds = max_wallclock_seconds or 120
        train_log_every = train_log_every or 20
        val_loss_every = 100 if val_loss_every < 0 else val_loss_every
        if not skip_prefetch:
            print(
                json.dumps(
                    prefetch_sp1024_data.remote(
                        train_shards=train_shards,
                        val_shards=val_shards,
                        hf_revision=hf_revision,
                    ),
                    indent=2,
                    sort_keys=True,
                )
            )
        result = smoke_test.remote(
            script_variant=script_variant,
            hf_revision=hf_revision,
            train_shards=train_shards,
            val_shards=val_shards,
            iterations=iterations,
            max_wallclock_seconds=max_wallclock_seconds,
            seed=seed,
            run_id=run_id,
            train_log_every=train_log_every,
            val_loss_every=val_loss_every,
        )
    elif mode == "bench":
        hf_revision = hf_revision or HISTORICAL_BASELINE_REVISION
        train_shards = train_shards or 25
        iterations = iterations or 20000
        max_wallclock_seconds = max_wallclock_seconds or 60
        train_log_every = train_log_every or 50
        val_loss_every = 0 if val_loss_every < 0 else val_loss_every
        if not skip_prefetch:
            print(
                json.dumps(
                    prefetch_sp1024_data.remote(
                        train_shards=train_shards,
                        val_shards=1,
                        hf_revision=hf_revision,
                    ),
                    indent=2,
                    sort_keys=True,
                )
            )
        result = baseline_8x_h100.remote(
            script_variant=script_variant,
            hf_revision=hf_revision,
            train_shards=train_shards,
            iterations=iterations,
            max_wallclock_seconds=max_wallclock_seconds,
            seed=seed,
            run_id=run_id,
            train_log_every=train_log_every,
            val_loss_every=val_loss_every,
        )
    elif mode == "baseline":
        hf_revision = hf_revision or HISTORICAL_BASELINE_REVISION
        train_shards = train_shards or 25
        iterations = iterations or 20000
        max_wallclock_seconds = max_wallclock_seconds or 600
        train_log_every = train_log_every or 50
        val_loss_every = 200 if val_loss_every < 0 else val_loss_every
        if not skip_prefetch:
            print(
                json.dumps(
                    prefetch_sp1024_data.remote(
                        train_shards=train_shards,
                        val_shards=None,
                        hf_revision=hf_revision,
                    ),
                    indent=2,
                    sort_keys=True,
                )
            )
        result = baseline_8x_h100.remote(
            script_variant=script_variant,
            hf_revision=hf_revision,
            train_shards=train_shards,
            iterations=iterations,
            max_wallclock_seconds=max_wallclock_seconds,
            seed=seed,
            run_id=run_id,
            train_log_every=train_log_every,
            val_loss_every=val_loss_every,
        )
    else:
        raise ValueError("mode must be 'doctor', 'smoke', 'bench' or 'baseline'")

    print(json.dumps(result, indent=2, sort_keys=True))
