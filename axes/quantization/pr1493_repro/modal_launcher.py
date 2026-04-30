"""Modal launcher for PR-1493 reproduction that saves a bundle instead of quantizing.

Mirrors infra/modal_pr1493.py but points at axes/quantization/pr1493_repro/train_save_bundle.py
and skips all post-quant plumbing. Bundle lands at /cache/runs/<run_id>/bundle/ on the
persistent Modal volume. Pull it down with `modal volume get` after the run.

Usage:
    modal run axes/quantization/pr1493_repro/modal_launcher.py --mode prefetch
    modal run axes/quantization/pr1493_repro/modal_launcher.py --mode train --seed 42

Environment:
    HF_ARTIFACT_REPO   default nprime06/parameter-golf-artifacts  (for post-run upload)
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import modal
from huggingface_hub import hf_hub_download


APP_NAME = "parameter-golf-pr1493-bundle"
_THIS = Path(__file__).resolve()
# Guard: when Modal copies this file into the container it lands at /root/modal_launcher.py
# (no parents[3]). REPO_ROOT is only needed locally to resolve LOCAL_SCRIPT.
REPO_ROOT = _THIS.parents[3] if len(_THIS.parents) >= 4 else _THIS.parent
REPO_ID = os.environ.get("MATCHED_FINEWEB_REPO_ID", "kevclark/parameter-golf")
REMOTE_ROOT_PREFIX = os.environ.get("MATCHED_FINEWEB_REMOTE_ROOT_PREFIX", "datasets")
DATASET_NAME = "fineweb10B_sp8192"
VOCAB_SIZE = 8192
PYTHON_VERSION = "3.12"
UBUNTU_IMAGE = "ubuntu:24.04"
TORCH_VERSION = "2.9.1+cu128"
TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu128"
FLASH_ATTN_FIND_LINKS = "https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/"

CACHE_ROOT = Path("/cache")
REMOTE_CODE_DIR = Path("/root/code/parameter-golf")
LOCAL_SCRIPT = REPO_ROOT / "axes/quantization/pr1493_repro/train_save_bundle.py"
REMOTE_SCRIPT = REMOTE_CODE_DIR / "axes/quantization/pr1493_repro/train_save_bundle.py"
LOCAL_QUANT_SCRIPT = REPO_ROOT / "axes/quantization/pr1493_repro/quantize_bundle.py"
REMOTE_QUANT_SCRIPT = REMOTE_CODE_DIR / "axes/quantization/pr1493_repro/quantize_bundle.py"

BUNDLE_GPU = "H100!:8"
QUANT_GPU = "H100!:1"
BUNDLE_CPU = int(os.environ.get("MODAL_CPU", "32"))
BUNDLE_MEMORY_MB = int(os.environ.get("MODAL_MEMORY_GB", "64")) * 1024


# ---------- dataset cache helpers (mirrored from infra/modal_pr1493.py) ----------

def _snapshot_label(hf_revision: str | None) -> str:
    return hf_revision or "head"


def _repo_cache_label() -> str:
    return REPO_ID.replace("/", "__")


def _snapshot_root(hf_revision: str | None) -> Path:
    return CACHE_ROOT / "snapshots" / _repo_cache_label() / _snapshot_label(hf_revision)


def _cache_path_for_remote(relative_path: str, hf_revision: str | None) -> Path:
    snapshot_root = _snapshot_root(hf_revision)
    remote_path = Path(relative_path)
    if REMOTE_ROOT_PREFIX and remote_path.parts[:1] == (REMOTE_ROOT_PREFIX,):
        remote_path = remote_path.relative_to(REMOTE_ROOT_PREFIX)
    if remote_path.parts[:1] in (("datasets",), ("tokenizers",)):
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


def _ensure_sp8192_data(train_shards: int, val_shards: int | None, hf_revision: str | None) -> tuple[Path, int, dict]:
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

    return _snapshot_root(hf_revision), val_shards_to_fetch, dataset_entry["stats"]


def _tail(text: str, max_chars: int = 24000) -> str:
    return text if len(text) <= max_chars else text[-max_chars:]


# ---------- Modal image + volume ----------

prefetch_image = (
    modal.Image.from_registry(UBUNTU_IMAGE, add_python=PYTHON_VERSION)
    .apt_install("build-essential", "ca-certificates")
    .pip_install("huggingface-hub==1.7.2", "hf_transfer==0.1.9")
)

gpu_image = (
    modal.Image.from_registry(UBUNTU_IMAGE, add_python=PYTHON_VERSION)
    .apt_install("build-essential", "ca-certificates")
    .pip_install(f"torch=={TORCH_VERSION}", index_url=TORCH_INDEX_URL)
    .pip_install(
        "brotli",
        "sentencepiece==0.2.1",
        "huggingface-hub==1.7.2",
        "hf_transfer==0.1.9",
        "numpy==2.4.3",
    )
    .run_commands(
        "python -m pip install --no-deps flash_attn_3 "
        f"--find-links {FLASH_ATTN_FIND_LINKS}"
    )
    .add_local_file(
        str(LOCAL_SCRIPT),
        remote_path=str(REMOTE_SCRIPT),
        copy=True,
    )
    .add_local_file(
        str(LOCAL_QUANT_SCRIPT),
        remote_path=str(REMOTE_QUANT_SCRIPT),
        copy=True,
    )
)

volume = modal.Volume.from_name("parameter-golf-fineweb-cache", create_if_missing=True)
app = modal.App(APP_NAME)


# ---------- Modal functions ----------

@app.function(
    image=prefetch_image,
    cpu=2,
    timeout=60 * 60,
    volumes={"/cache": volume},
)
def prefetch_sp8192_data(
    train_shards: int = 80,
    val_shards: int | None = None,
    hf_revision: str = "",
) -> str:
    os.environ.setdefault("HF_HOME", str(CACHE_ROOT / "hf"))
    volume.reload()
    data_root, downloaded_val_shards, dataset_stats = _ensure_sp8192_data(
        train_shards,
        val_shards,
        hf_revision or None,
    )
    volume.commit()
    return json.dumps({
        "hf_revision": _snapshot_label(hf_revision or None),
        "train_shards": train_shards,
        "val_shards": downloaded_val_shards,
        "manifest_stats": dataset_stats,
        "data_dir": str(data_root),
    }, indent=2, sort_keys=True)


def _run_bundle(
    *,
    hf_revision: str | None,
    train_shards: int,
    iterations: int,
    max_wallclock_seconds: int,
    seed: int,
    run_id: str,
    train_log_every: int,
    val_loss_every: int,
    qk_gain_init: float,
    wd_final: float = 0.095,
    wd_taper_start_frac: float = 1.0,
    cautious_wd: bool = False,
    num_layers: int = 11,
    parallel_residual_start: int = 7,
) -> dict:
    modal_provider = os.environ.get("MODAL_CLOUD_PROVIDER", "")
    modal_region = os.environ.get("MODAL_REGION", "")
    os.environ.setdefault("HF_HOME", str(CACHE_ROOT / "hf"))
    volume.reload()
    data_root, downloaded_val_shards, dataset_stats = _ensure_sp8192_data(
        train_shards, None, hf_revision,
    )
    run_dir = CACHE_ROOT / "runs" / run_id
    bundle_dir = run_dir / "bundle"
    run_dir.mkdir(parents=True, exist_ok=True)
    volume.commit()

    env = os.environ.copy()
    env.update({
        "PYTHONUNBUFFERED": "1",
        "OMP_NUM_THREADS": "1",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
        "NCCL_ASYNC_ERROR_HANDLING": "1",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "HF_XET_HIGH_PERFORMANCE": "1",
        "RUN_ID": run_id,
        "SEED": str(seed),
        "DATA_DIR": str(data_root),
        "VOCAB_SIZE": str(VOCAB_SIZE),
        "ITERATIONS": str(iterations),
        "MAX_WALLCLOCK_SECONDS": str(max_wallclock_seconds),
        "TRAIN_LOG_EVERY": str(train_log_every),
        "VAL_LOSS_EVERY": str(val_loss_every),
        "QK_GAIN_INIT": str(qk_gain_init),
        "BUNDLE_DIR": str(bundle_dir),
        # Disable eval-time adaptation — we only want clean training + bundle save
        "TTT_ENABLED": "0",
        "ETLB_ENABLED": "0",
        # WD taper (no-op unless overridden)
        "WD_FINAL": str(wd_final),
        "WD_TAPER_START_FRAC": str(wd_taper_start_frac),
        # Cautious WD (0 = off, 1 = on)
        "CAUTIOUS_WD": "1" if cautious_wd else "0",
        # Architecture overrides
        "NUM_LAYERS": str(num_layers),
        "XSA_LAST_N": str(num_layers),
        "PARALLEL_RESIDUAL_START": str(parallel_residual_start),
    })

    cmd = [
        "torchrun",
        "--standalone",
        "--nproc_per_node=8",
        str(REMOTE_SCRIPT),
    ]
    nvidia_smi_l = subprocess.run(
        ["nvidia-smi", "-L"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False,
    )
    launch_started = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=str(run_dir),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    launch_elapsed_seconds = time.perf_counter() - launch_started
    volume.commit()

    # Parse pre-quant eval line produced by timed_eval('pre-quantization post-ema', ...)
    # Format: "pre-quantization post-ema val_loss:X.XXXX val_bpb:X.XXXX eval_time:XXXms"
    preq_match = re.search(
        r"pre-quantization post-ema val_loss:(?P<vl>[0-9.]+) val_bpb:(?P<bpb>[0-9.]+) eval_time:(?P<et>\d+)ms",
        proc.stdout,
    )
    pre_quant_eval = None
    if preq_match:
        pre_quant_eval = {
            "val_loss": float(preq_match.group("vl")),
            "val_bpb": float(preq_match.group("bpb")),
            "eval_time_ms": int(preq_match.group("et")),
        }

    # Train log tail, last step
    train_matches = list(re.finditer(
        r"(?m)^(?P<step>\d+)/(?P<iterations>\d+) train_loss: (?P<train_loss>[0-9.]+) train_time: (?P<train_minutes>[0-9.]+)m tok/s: (?P<tok_per_sec>\d+)$",
        proc.stdout,
    ))
    latest_train = None
    if train_matches:
        last = train_matches[-1]
        latest_train = {
            "step": int(last.group("step")),
            "iterations": int(last.group("iterations")),
            "train_loss": float(last.group("train_loss")),
            "train_minutes": float(last.group("train_minutes")),
            "tok_per_sec": int(last.group("tok_per_sec")),
        }

    bundle_files = {}
    for name in ("ema_weights.pt", "hessians.pt", "template_sd.pt"):
        path = bundle_dir / name
        bundle_files[name] = {
            "exists": path.exists(),
            "bytes": path.stat().st_size if path.exists() else 0,
            "path": str(path),
        }

    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "bundle_dir": str(bundle_dir),
        "bundle_files": bundle_files,
        "hf_revision": _snapshot_label(hf_revision),
        "train_shards": train_shards,
        "val_shards": downloaded_val_shards,
        "data_dir": str(data_root),
        "modal_cloud_provider": modal_provider,
        "modal_region": modal_region,
        "requested_cpu": BUNDLE_CPU,
        "requested_memory_mb": BUNDLE_MEMORY_MB,
        "nvidia_smi_l": nvidia_smi_l.stdout.strip(),
        "manifest_stats": dataset_stats,
        "launch_elapsed_seconds": launch_elapsed_seconds,
        "pre_quant_eval": pre_quant_eval,
        "latest_train": latest_train,
        "stdout_tail": _tail(proc.stdout),
        "stderr_tail": _tail(proc.stderr),
    }


@app.function(
    image=gpu_image,
    gpu=BUNDLE_GPU,
    cpu=BUNDLE_CPU,
    memory=BUNDLE_MEMORY_MB,
    timeout=60 * 60 * 3,
    volumes={"/cache": volume},
)
def train_bundle_8x_h100(
    hf_revision: str = "",
    train_shards: int = 80,
    iterations: int = 20000,
    max_wallclock_seconds: int = 600,
    seed: int = 42,
    run_id: str = "pr1493_bundle_seed42",
    train_log_every: int = 500,
    val_loss_every: int = 4000,
    qk_gain_init: float = 5.25,
    wd_final: float = 0.095,
    wd_taper_start_frac: float = 1.0,
    cautious_wd: bool = False,
    num_layers: int = 11,
    parallel_residual_start: int = 7,
) -> str:
    return json.dumps(
        _run_bundle(
            hf_revision=hf_revision or None,
            train_shards=train_shards,
            iterations=iterations,
            max_wallclock_seconds=max_wallclock_seconds,
            seed=seed,
            run_id=run_id,
            train_log_every=train_log_every,
            val_loss_every=val_loss_every,
            qk_gain_init=qk_gain_init,
            wd_final=wd_final,
            wd_taper_start_frac=wd_taper_start_frac,
            cautious_wd=cautious_wd,
            num_layers=num_layers,
            parallel_residual_start=parallel_residual_start,
        ),
        indent=2, sort_keys=True,
    )


def _run_quantize(
    *,
    hf_revision: str | None,
    bundle_dir: str,
    run_id: str,
    matrix_bits: int,
    embed_bits: int,
    matrix_clip_sigmas: float,
    embed_clip_sigmas: float,
    quant_format: str = "uniform",
    prune_fraction: float = 0.0,
    prune_method: str = "magnitude",
    sparsity_threshold: float = 1.0,
    num_layers: int = 11,
    parallel_residual_start: int = 7,
) -> dict:
    os.environ.setdefault("HF_HOME", str(CACHE_ROOT / "hf"))
    volume.reload()
    data_root, _, dataset_stats = _ensure_sp8192_data(train_shards=0, val_shards=None, hf_revision=hf_revision)
    resolved_bundle = Path(bundle_dir)
    if not resolved_bundle.is_absolute():
        resolved_bundle = CACHE_ROOT / resolved_bundle
    if not resolved_bundle.exists():
        raise FileNotFoundError(f"Bundle dir not found: {resolved_bundle}")
    run_dir = CACHE_ROOT / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    volume.commit()

    env = os.environ.copy()
    env.update({
        "PYTHONUNBUFFERED": "1",
        "OMP_NUM_THREADS": "1",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
        "NCCL_ASYNC_ERROR_HANDLING": "1",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "RUN_ID": run_id,
        "DATA_DIR": str(data_root),
        "VOCAB_SIZE": str(VOCAB_SIZE),
        "BUNDLE_DIR": str(resolved_bundle),
        "VAL_LOSS_EVERY": "0",
        "TTT_ENABLED": "0",
        "ETLB_ENABLED": "0",
        "MATRIX_BITS": str(matrix_bits),
        "EMBED_BITS": str(embed_bits),
        "MATRIX_CLIP_SIGMAS": str(matrix_clip_sigmas),
        "EMBED_CLIP_SIGMAS": str(embed_clip_sigmas),
        "QUANT_FORMAT": quant_format,
        "PRUNE_FRACTION": str(prune_fraction),
        "PRUNE_METHOD": prune_method,
        "SPARSITY_THRESHOLD": str(sparsity_threshold),
        "NUM_LAYERS": str(num_layers),
        "PARALLEL_RESIDUAL_START": str(parallel_residual_start),
        "XSA_LAST_N": str(num_layers),
    })

    cmd = [
        "torchrun",
        "--standalone",
        "--nproc_per_node=1",
        str(REMOTE_QUANT_SCRIPT),
    ]
    launch_started = time.perf_counter()
    proc = subprocess.run(
        cmd, cwd=str(run_dir), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, check=False,
    )
    launch_elapsed_seconds = time.perf_counter() - launch_started
    volume.commit()

    # Parse eval lines. Pattern: "<label> val_loss:X.X val_bpb:X.X eval_time:Nms"
    eval_matches = list(re.finditer(
        r"(?m)^(?P<label>\S+) val_loss:(?P<vl>[0-9.]+) val_bpb:(?P<bpb>[0-9.]+) eval_time:(?P<et>\d+)ms",
        proc.stdout,
    ))
    eval_results = [
        {
            "label": m.group("label"),
            "val_loss": float(m.group("vl")),
            "val_bpb": float(m.group("bpb")),
            "eval_time_ms": int(m.group("et")),
        }
        for m in eval_matches
    ]
    artifact_match = re.search(
        r"quant_artifact_(?P<compressor>\w+): (?P<bytes>\d+) bytes",
        proc.stdout,
    )
    artifact = None
    if artifact_match:
        artifact = {
            "compressor": artifact_match.group("compressor"),
            "bytes": int(artifact_match.group("bytes")),
        }

    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "run_id": run_id,
        "bundle_dir": str(resolved_bundle),
        "config": {
            "matrix_bits": matrix_bits,
            "embed_bits": embed_bits,
            "matrix_clip_sigmas": matrix_clip_sigmas,
            "embed_clip_sigmas": embed_clip_sigmas,
        },
        "eval_results": eval_results,
        "artifact": artifact,
        "launch_elapsed_seconds": launch_elapsed_seconds,
        "stdout_tail": _tail(proc.stdout),
        "stderr_tail": _tail(proc.stderr),
    }


@app.function(
    image=gpu_image,
    gpu=QUANT_GPU,
    cpu=8,
    memory=32 * 1024,
    timeout=60 * 60,
    volumes={"/cache": volume},
)
def quantize_1x_h100(
    hf_revision: str = "",
    bundle_dir: str = "runs/pr1493_bundle_seed42/bundle",
    run_id: str = "pr1493_quantize_reference",
    matrix_bits: int = 6,
    embed_bits: int = 8,
    matrix_clip_sigmas: float = 12.85,
    embed_clip_sigmas: float = 20.0,
    quant_format: str = "uniform",
    prune_fraction: float = 0.0,
    prune_method: str = "magnitude",
    sparsity_threshold: float = 1.0,
    num_layers: int = 11,
    parallel_residual_start: int = 7,
) -> str:
    return json.dumps(
        _run_quantize(
            hf_revision=hf_revision or None,
            bundle_dir=bundle_dir,
            run_id=run_id,
            matrix_bits=matrix_bits,
            embed_bits=embed_bits,
            matrix_clip_sigmas=matrix_clip_sigmas,
            embed_clip_sigmas=embed_clip_sigmas,
            quant_format=quant_format,
            prune_fraction=prune_fraction,
            prune_method=prune_method,
            sparsity_threshold=sparsity_threshold,
            num_layers=num_layers,
            parallel_residual_start=parallel_residual_start,
        ),
        indent=2, sort_keys=True,
    )


@app.local_entrypoint()
def main(
    mode: str = "train",
    skip_prefetch: bool = False,
    hf_revision: str = "",
    train_shards: int = 80,
    val_shards: int = 0,
    iterations: int = 20000,
    max_wallclock_seconds: int = 600,
    seed: int = 42,
    run_id: str = "",
    train_log_every: int = 500,
    val_loss_every: int = 4000,
    qk_gain_init: float = 5.25,
    bundle_dir: str = "runs/pr1493_bundle_seed42/bundle",
    matrix_bits: int = 6,
    embed_bits: int = 8,
    matrix_clip_sigmas: float = 12.85,
    embed_clip_sigmas: float = 20.0,
    quant_format: str = "uniform",
    prune_fraction: float = 0.0,
    prune_method: str = "magnitude",
    sparsity_threshold: float = 1.0,
    wd_final: float = 0.095,
    wd_taper_start_frac: float = 1.0,
    cautious_wd: bool = False,
    num_layers: int = 11,
    parallel_residual_start: int = 7,
    write_result: str = "",
) -> None:
    """Entrypoints:
      --mode prefetch   Stage SP8192 dataset into the Modal volume cache.
      --mode train      Run training + bundle save on 8x H100.
      --mode quantize   Apply GPTQ + eval on a saved bundle (1x H100).
    """
    if mode == "prefetch":
        v = 0 if val_shards < 0 else val_shards
        result = prefetch_sp8192_data.remote(
            train_shards=train_shards,
            val_shards=None if v == 0 else v,
            hf_revision=hf_revision,
        )
    elif mode == "train":
        run_id = run_id or f"pr1493_bundle_seed{seed}"
        if not skip_prefetch:
            print(
                prefetch_sp8192_data.remote(
                    train_shards=train_shards,
                    val_shards=None,
                    hf_revision=hf_revision,
                )
            )
        result = train_bundle_8x_h100.remote(
            hf_revision=hf_revision,
            train_shards=train_shards,
            iterations=iterations,
            max_wallclock_seconds=max_wallclock_seconds,
            seed=seed,
            run_id=run_id,
            train_log_every=train_log_every,
            val_loss_every=val_loss_every,
            qk_gain_init=qk_gain_init,
            wd_final=wd_final,
            wd_taper_start_frac=wd_taper_start_frac,
            cautious_wd=cautious_wd,
            num_layers=num_layers,
            parallel_residual_start=parallel_residual_start,
        )
    elif mode == "quantize":
        run_id = run_id or f"pr1493_quantize_{Path(bundle_dir).name}"
        result = quantize_1x_h100.remote(
            hf_revision=hf_revision,
            bundle_dir=bundle_dir,
            run_id=run_id,
            matrix_bits=matrix_bits,
            embed_bits=embed_bits,
            matrix_clip_sigmas=matrix_clip_sigmas,
            embed_clip_sigmas=embed_clip_sigmas,
            quant_format=quant_format,
            prune_fraction=prune_fraction,
            prune_method=prune_method,
            sparsity_threshold=sparsity_threshold,
            num_layers=num_layers,
            parallel_residual_start=parallel_residual_start,
        )
    else:
        raise ValueError("mode must be 'prefetch', 'train', or 'quantize'")

    if write_result:
        Path(write_result).write_text(result, encoding="utf-8")
    print(result)
