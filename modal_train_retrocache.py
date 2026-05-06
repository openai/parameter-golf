from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import modal


REPO_ROOT = Path(__file__).resolve().parent
REMOTE_REPO_STR = "/root/parameter-golf"
DATA_MOUNT_STR = "/mnt/parameter-golf-data"
HF_MOUNT_STR = "/mnt/hf-cache"


def _ignore_local(path: Path) -> bool:
    path = Path(path)
    if path.is_absolute():
        try:
            rel = path.relative_to(REPO_ROOT)
        except ValueError:
            return False
    else:
        rel = path
    parts = rel.parts
    if any(part in {".git", "__pycache__", ".pytest_cache", ".hypothesis", ".kiro"} for part in parts):
        return True
    if parts[:2] == ("data", "datasets"):
        return True
    if parts[:2] == ("data", "tokenizers"):
        return True
    if parts and parts[0] == "logs":
        return True
    return False


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install_from_requirements(str(REPO_ROOT / "requirements.txt"), gpu="H100")
    .pip_install("zstandard")
    .add_local_dir(str(REPO_ROOT), remote_path=REMOTE_REPO_STR, ignore=_ignore_local)
)

app = modal.App("parameter-golf-retrocache-train")
data_vol = modal.Volume.from_name("parameter-golf-data", create_if_missing=True)
hf_cache_vol = modal.Volume.from_name("parameter-golf-hf-cache", create_if_missing=True)


def _ensure_symlink(target: Path, link_path: Path) -> None:
    if link_path.is_symlink():
        if Path(os.readlink(link_path)) == target:
            return
        link_path.unlink()
    elif link_path.exists():
        if link_path.is_dir():
            shutil.rmtree(link_path)
        else:
            link_path.unlink()
    link_path.parent.mkdir(parents=True, exist_ok=True)
    link_path.symlink_to(target, target_is_directory=True)


def _stream_command(cmd: list[str], cwd: Path, env: dict[str, str]) -> None:
    print("$", " ".join(cmd), flush=True)
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
    rc = proc.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)


@app.function(
    image=image,
    gpu="H100!:8",
    cpu=32,
    memory=262144,
    timeout=60 * 60 * 4,
    volumes={
        DATA_MOUNT_STR: data_vol,
        HF_MOUNT_STR: hf_cache_vol,
    },
)
def run_training(
    run_id: str = "v38_baseline",
    cache_enabled: bool = False,
    train_shards: int = 80,
) -> None:
    repo = Path(REMOTE_REPO_STR)
    data_mount = Path(DATA_MOUNT_STR)
    hf_mount = Path(HF_MOUNT_STR)
    (data_mount / "datasets").mkdir(parents=True, exist_ok=True)
    (data_mount / "tokenizers").mkdir(parents=True, exist_ok=True)
    hf_mount.mkdir(parents=True, exist_ok=True)
    _ensure_symlink(data_mount / "datasets", repo / "data" / "datasets")
    _ensure_symlink(data_mount / "tokenizers", repo / "data" / "tokenizers")
    (repo / "logs").mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update(
        {
            "HF_HOME": str(hf_mount),
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )

    _stream_command(["nvidia-smi"], cwd=repo, env=env)
    _stream_command(
        [
            "python",
            "data/cached_challenge_fineweb.py",
            "--variant",
            "sp1024",
            "--train-shards",
            str(train_shards),
        ],
        cwd=repo,
        env=env,
    )

    train_env = env.copy()
    train_env.update(
        {
            "RUN_ID": run_id,
            "DATA_PATH": "./data/datasets/fineweb10B_sp1024",
            "TOKENIZER_PATH": "./data/tokenizers/fineweb_1024_bpe.model",
            "VOCAB_SIZE": "1024",
            "NUM_LAYERS": "11",
            "MLP_MULT": "3.0",
            "TRAIN_SEQ_LEN": "2048",
            "EVAL_SEQ_LEN": "2048",
            "TRAIN_BATCH_TOKENS": "786432",
            "BIGRAM_VOCAB_SIZE": "2048",
            "BIGRAM_DIM": "128",
            "XSA_LAST_N": "4",
            "ROPE_DIMS": "16",
            "LN_SCALE": "1",
            "VE_ENABLED": "1",
            "VE_DIM": "128",
            "VE_LAYERS": "9,10",
            "SWA_ENABLED": "1",
            "SWA_EVERY": "50",
            "ADAM_WD": "0.04",
            "MUON_WD": "0.04",
            "MATRIX_LR": "0.025",
            "SCALAR_LR": "0.025",
            "TIED_EMBED_LR": "0.035",
            "MUON_MOMENTUM": "0.99",
            "MUON_MOMENTUM_WARMUP_START": "0.92",
            "MUON_MOMENTUM_WARMUP_STEPS": "1500",
            "WARMDOWN_ITERS": "3000",
            "ITERATIONS": "9000",
            "MAX_WALLCLOCK_SECONDS": "600",
            "EVAL_STRIDE": "64",
            "CACHE_ENABLED": "1" if cache_enabled else "0",
            "CACHE_MAX_TOKENS": "32768",
            "CACHE_RECENT_TOKENS": "4096",
            "CACHE_OLD_STRIDE": "4",
            "CACHE_TOPK": "32",
            "CACHE_BETA": "24",
            "CACHE_LAMBDA_MAX": "0.35",
            "CACHE_WARMUP_TOKENS": "2048",
            "CACHE_KEY_SOURCE": "final_norm",
            "CACHE_RESET_ON_BOS": "0",
        }
    )

    _stream_command(
        [
            "torchrun",
            "--standalone",
            "--nproc_per_node=8",
            "records/track_10min_16mb/2026-03-21_v38_TightSWA_RetroCache/train_gpt.py",
        ],
        cwd=repo,
        env=train_env,
    )


@app.local_entrypoint()
def main(
    run_id: str = "v38_baseline",
    cache_enabled: bool = False,
    train_shards: int = 80,
) -> None:
    run_training.remote(
        run_id=run_id,
        cache_enabled=cache_enabled,
        train_shards=train_shards,
    )
