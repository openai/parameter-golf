from __future__ import annotations

import datetime as dt
import os
import shutil
import subprocess
import sys
from pathlib import Path

import modal

APP_NAME = "parameter-golf-int6-normuon-fa3"
LOCAL_FILE = Path(__file__).resolve()


def _infer_local_repo_root(script_path: Path) -> Path:
    parents = script_path.parents
    return parents[3] if len(parents) > 3 else script_path.parent


LOCAL_REPO_ROOT = _infer_local_repo_root(LOCAL_FILE)
REMOTE_REPO_ROOT = Path("/root/parameter-golf")
REMOTE_RECORD_DIR = REMOTE_REPO_ROOT / "records/track_10min_16mb/2026-03-20_Int6MLP3x_NorMuon_FA3"
REMOTE_TRAIN_FILE = REMOTE_RECORD_DIR / "train_gpt.py"
REMOTE_DATA_MOUNT = Path("/mnt/parameter-golf-data")
REMOTE_LOG_MOUNT = Path("/mnt/parameter-golf-logs")
DEFAULT_DATA_PATH = REMOTE_REPO_ROOT / "data/datasets/fineweb10B_sp1024"
DEFAULT_TOKENIZER_PATH = REMOTE_REPO_ROOT / "data/tokenizers/fineweb_1024_bpe.model"
DEFAULT_TRAIN_ENV = {
    "TRAIN_SEQ_LEN": "2048",
    "TRAIN_BATCH_TOKENS": "786432",
    "MATRIX_LR": "0.02",
    "SCALAR_LR": "0.02",
    "TIED_EMBED_LR": "0.03",
    "MUON_MOMENTUM": "0.99",
    "MUON_BETA2": "0.95",
    "MUON_MOMENTUM_WARMUP_START": "0.92",
    "MUON_MOMENTUM_WARMUP_STEPS": "1500",
    "WARMDOWN_ITERS": "3000",
    "GRAD_CLIP_NORM": "0.3",
    "MLP_HIDDEN": "1536",
    "EVAL_SEQ_LEN": "2048",
    "EVAL_STRIDE": "256",
    "SEED": "7",
}
FLASH_ATTN3_PREFETCH = (
    "from kernels import get_kernel; "
    "mod = get_kernel('kernels-community/flash-attn3', version=1); "
    "print(getattr(mod, '__file__', 'flash-attn3-prefetched'))"
)

data_volume = modal.Volume.from_name("parameter-golf-data", create_if_missing=True)
log_volume = modal.Volume.from_name("parameter-golf-logs", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements(str(LOCAL_REPO_ROOT / "requirements.txt"))
    .add_local_dir(
        str(LOCAL_REPO_ROOT),
        remote_path=str(REMOTE_REPO_ROOT),
        ignore=[
            ".git",
            ".git/**",
            "**/__pycache__",
            "**/__pycache__/**",
            "**/*.pyc",
            "**/*.log",
            "data/datasets",
            "data/datasets/**",
            "data/tokenizers",
            "data/tokenizers/**",
        ],
    )
)

app = modal.App(APP_NAME, image=image)


def _ensure_symlink(link_path: Path, target_path: Path) -> None:
    link_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.mkdir(parents=True, exist_ok=True)
    if link_path.is_symlink():
        if link_path.resolve() == target_path.resolve():
            return
        link_path.unlink()
    elif link_path.exists():
        if link_path.is_dir():
            shutil.rmtree(link_path)
        else:
            link_path.unlink()
    link_path.symlink_to(target_path, target_is_directory=True)


def _stream_subprocess(command: list[str], *, cwd: Path, env: dict[str, str], log_path: Path) -> None:
    with log_path.open("a", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_file.write(line)
            log_file.flush()
        return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, command)


@app.function(
    gpu="H100:8",
    cpu=16,
    timeout=60 * 60 * 2,
    volumes={
        str(REMOTE_DATA_MOUNT): data_volume,
        str(REMOTE_LOG_MOUNT): log_volume,
    },
)
def run_training(
    train_shards: int = 80,
    run_id: str = "",
    download_data: bool = True,
) -> str:
    _ensure_symlink(REMOTE_REPO_ROOT / "data/datasets", REMOTE_DATA_MOUNT / "datasets")
    _ensure_symlink(REMOTE_REPO_ROOT / "data/tokenizers", REMOTE_DATA_MOUNT / "tokenizers")

    effective_run_id = run_id or dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_dir = REMOTE_LOG_MOUNT / "int6_normuon_fa3"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"train_{effective_run_id}.log"

    env = os.environ.copy()
    env.update(DEFAULT_TRAIN_ENV)
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "DATA_PATH": str(DEFAULT_DATA_PATH),
            "TOKENIZER_PATH": str(DEFAULT_TOKENIZER_PATH),
            "RUN_ID": effective_run_id,
        }
    )

    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"run_id={effective_run_id}\n")
        log_file.write(f"train_file={REMOTE_TRAIN_FILE}\n")
        log_file.write(f"train_shards={train_shards}\n")
        log_file.write(f"download_data={download_data}\n")
        log_file.write(f"seed={env.get('SEED', '<unset>')}\n")

    try:
        if download_data:
            _stream_subprocess(
                [
                    "python3",
                    "data/cached_challenge_fineweb.py",
                    "--variant",
                    "sp1024",
                    "--train-shards",
                    str(train_shards),
                ],
                cwd=REMOTE_REPO_ROOT,
                env=env,
                log_path=log_path,
            )
            data_volume.commit()

        _stream_subprocess(
            ["python3", "-c", FLASH_ATTN3_PREFETCH],
            cwd=REMOTE_REPO_ROOT,
            env=env,
            log_path=log_path,
        )

        _stream_subprocess(
            [
                "python3",
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc_per_node=8",
                str(REMOTE_TRAIN_FILE),
            ],
            cwd=REMOTE_REPO_ROOT,
            env=env,
            log_path=log_path,
        )
    finally:
        log_volume.commit()
    return str(log_path)


@app.function(volumes={str(REMOTE_LOG_MOUNT): log_volume})
def fetch_log(log_path: str) -> bytes:
    requested_path = Path(log_path)
    resolved_path = requested_path if requested_path.is_absolute() else REMOTE_LOG_MOUNT / requested_path
    return resolved_path.read_bytes()


@app.local_entrypoint()
def main(
    train_shards: int = 80,
    run_id: str = "",
    skip_download: bool = False,
) -> None:
    log_path = run_training.remote(
        train_shards=train_shards,
        run_id=run_id,
        download_data=not skip_download,
    )
    print(log_path)
