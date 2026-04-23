"""
Modal launcher for parameter-golf training.

Usage examples:

1) Smoke run on your work branch (8xH100):
   modal run run_modal.py \
     --repo-url https://github.com/sachinnchaudhary/parameter-golf.git \
     --branch work/apr05-next \
     --script-path train_gpt.py \
     --variant sp1024 \
     --train-shards 1 \
     --iterations 300 \
     --max-wallclock-seconds 300 \
     --run-id modal_smoke_sp1024

2) SP8192 stack run (uses alt HF repo by default):
   modal run run_modal.py \
     --repo-url https://github.com/sachinnchaudhary/parameter-golf.git \
     --branch parent/apr05-sp8192 \
     --script-path records/track_10min_16mb/2026-04-05_SP8192_GPTQ-Embeddings_SDClip_Loop45x2/train_gpt_human.py \
     --variant sp8192 \
     --train-shards 8 \
     --iterations 20000 \
     --max-wallclock-seconds 600 \
     --run-id modal_sp8192_seed1337 \
     --seed 1337
"""

from __future__ import annotations

import glob
import os
import re
import shutil
import subprocess
from pathlib import Path

import modal


APP_NAME = "parameter-golf-train"
DATA_VOLUME_NAME = "parameter-golf-data"
CACHE_ROOT = "/cache/parameter-golf"
REPO_DIR = "/root/parameter-golf"


def _build_image() -> modal.Image:
    # FlashAttention 3 wheel from the record README.
    flash3_wheel = (
        "https://download.pytorch.org/whl/cu130/"
        "flash_attn_3-3.0.0-cp39-abi3-manylinux_2_28_x86_64.whl"
    )
    return (
        modal.Image.debian_slim(python_version="3.10")
        .apt_install("git", "curl", "ca-certificates")
        .run_commands("python -m pip install --upgrade pip")
        .run_commands("pip install torch --index-url https://download.pytorch.org/whl/cu130")
        .run_commands(f"pip install --no-cache-dir '{flash3_wheel}'")
        .pip_install(
            "numpy",
            "tqdm",
            "huggingface-hub",
            "kernels",
            "setuptools",
            "typing-extensions==4.15.0",
            "datasets",
            "tiktoken",
            "sentencepiece",
            "brotli",
        )
    )


app = modal.App(APP_NAME, image=_build_image())
data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)


def _run(cmd: list[str], *, cwd: str | None = None, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def _variant_to_vocab_size(variant: str) -> int:
    m = re.fullmatch(r"sp(\d+)", variant)
    if m:
        return int(m.group(1))
    if variant == "byte260":
        return 260
    raise ValueError(f"Unsupported variant: {variant!r}")


def _ensure_cache_symlinks() -> None:
    """
    Keep repo data scripts, but route heavy artifacts to persistent Modal volume.
    """
    repo_data = Path(REPO_DIR) / "data"
    cache_root = Path(CACHE_ROOT)
    cache_root.mkdir(parents=True, exist_ok=True)
    for name in ("datasets", "tokenizers", "manifest.json"):
        target = cache_root / name
        if name != "manifest.json":
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
        link_path = repo_data / name
        if link_path.exists() or link_path.is_symlink():
            if link_path.is_dir() and not link_path.is_symlink():
                shutil.rmtree(link_path)
            else:
                link_path.unlink()
        link_path.symlink_to(target, target_is_directory=(name != "manifest.json"))


def _persist_outputs(run_id: str) -> None:
    out_dir = Path(CACHE_ROOT) / "runs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    patterns = [
        "final_model.pt",
        "final_model.*.ptz",
        "submission*.json",
        "logs/*.txt",
    ]
    copied = 0
    for pattern in patterns:
        for path in glob.glob(str(Path(REPO_DIR) / pattern)):
            src = Path(path)
            dst = out_dir / src.name
            shutil.copy2(src, dst)
            copied += 1
    print(f"[modal] persisted {copied} output files to {out_dir}", flush=True)


@app.function(
    gpu="H100:8",
    cpu=32,
    memory=200_000,
    timeout=60 * 60 * 2,
    volumes={"/cache": data_volume},
)
def run_training(
    repo_url: str = "https://github.com/sachinnchaudhary/parameter-golf.git",
    branch: str = "work/apr05-next",
    script_path: str = "train_gpt.py",
    variant: str = "sp1024",
    train_shards: int = 1,
    hf_repo_id: str = "",
    run_id: str = "modal_run",
    seed: int = 1337,
    nproc_per_node: int = 8,
    iterations: int = 300,
    max_wallclock_seconds: float = 300.0,
    val_loss_every: int = 100,
    train_log_every: int = 50,
    recurrence_mode: str = "hard",
    recurrence_ramp_start_frac: float = 0.485,
    recurrence_ramp_mid_frac: float = 0.50,
    recurrence_ramp_end_frac: float = 0.515,
    enable_looping_at: float = 0.50,
) -> None:
    if hf_repo_id == "":
        hf_repo_id = "kevclark/parameter-golf" if variant == "sp8192" else "willdepueoai/parameter-golf"
    vocab_size = _variant_to_vocab_size(variant)

    if os.path.exists(REPO_DIR):
        shutil.rmtree(REPO_DIR)
    _run(["git", "clone", "--depth", "1", "--branch", branch, repo_url, REPO_DIR])
    _ensure_cache_symlinks()

    env = os.environ.copy()
    env["MATCHED_FINEWEB_REPO_ID"] = hf_repo_id
    _run(
        [
            "python",
            "data/cached_challenge_fineweb.py",
            "--variant",
            variant,
            "--train-shards",
            str(train_shards),
        ],
        cwd=REPO_DIR,
        env=env,
    )

    env.update(
        {
            "RUN_ID": run_id,
            "SEED": str(seed),
            "ITERATIONS": str(iterations),
            "MAX_WALLCLOCK_SECONDS": str(max_wallclock_seconds),
            "VAL_LOSS_EVERY": str(val_loss_every),
            "TRAIN_LOG_EVERY": str(train_log_every),
            # For scripts using DATA_DIR-style paths.
            "DATA_DIR": "./data/",
            # For scripts using explicit DATA_PATH/TOKENIZER_PATH.
            "DATA_PATH": f"./data/datasets/fineweb10B_{variant}",
            "TOKENIZER_PATH": f"./data/tokenizers/fineweb_{vocab_size}_bpe.model",
            "VOCAB_SIZE": str(vocab_size),
            "RECURRENCE_MODE": recurrence_mode,
            "RECURRENCE_RAMP_START_FRAC": str(recurrence_ramp_start_frac),
            "RECURRENCE_RAMP_MID_FRAC": str(recurrence_ramp_mid_frac),
            "RECURRENCE_RAMP_END_FRAC": str(recurrence_ramp_end_frac),
            "ENABLE_LOOPING_AT": str(enable_looping_at),
        }
    )

    _run(
        [
            "torchrun",
            "--standalone",
            f"--nproc_per_node={nproc_per_node}",
            script_path,
        ],
        cwd=REPO_DIR,
        env=env,
    )

    _persist_outputs(run_id)
    data_volume.commit()


@app.local_entrypoint()
def main(
    repo_url: str = "https://github.com/sachinnchaudhary/parameter-golf.git",
    branch: str = "work/apr05-next",
    script_path: str = "train_gpt.py",
    variant: str = "sp1024",
    train_shards: int = 1,
    hf_repo_id: str = "",
    run_id: str = "modal_run",
    seed: int = 1337,
    nproc_per_node: int = 8,
    iterations: int = 300,
    max_wallclock_seconds: float = 300.0,
    val_loss_every: int = 100,
    train_log_every: int = 50,
    recurrence_mode: str = "hard",
    recurrence_ramp_start_frac: float = 0.485,
    recurrence_ramp_mid_frac: float = 0.50,
    recurrence_ramp_end_frac: float = 0.515,
    enable_looping_at: float = 0.50,
) -> None:
    run_training.remote(
        repo_url=repo_url,
        branch=branch,
        script_path=script_path,
        variant=variant,
        train_shards=train_shards,
        hf_repo_id=hf_repo_id,
        run_id=run_id,
        seed=seed,
        nproc_per_node=nproc_per_node,
        iterations=iterations,
        max_wallclock_seconds=max_wallclock_seconds,
        val_loss_every=val_loss_every,
        train_log_every=train_log_every,
        recurrence_mode=recurrence_mode,
        recurrence_ramp_start_frac=recurrence_ramp_start_frac,
        recurrence_ramp_mid_frac=recurrence_ramp_mid_frac,
        recurrence_ramp_end_frac=recurrence_ramp_end_frac,
        enable_looping_at=enable_looping_at,
    )
