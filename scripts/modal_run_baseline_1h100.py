from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import modal

APP_NAME = "parameter-golf-baseline-1h100"
REPO_REMOTE_PATH = "/workspace/parameter-golf"
RECORD_DIR = "records/track_10min_16mb/2026-03-19_WIP_PLACEHOLDER"
TRAIN_SCRIPT = f"{RECORD_DIR}/train_gpt.py"
TRAIN_LOG = f"{RECORD_DIR}/train.log"

app = modal.App(APP_NAME)

image = (
    modal.Image.from_registry("pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime")
    .apt_install("build-essential")
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
    )
    .add_local_dir(".", remote_path=REPO_REMOTE_PATH)
)


def _extract_metrics(log_text: str) -> dict[str, float | int | None]:
    bpb_match = re.search(r"final_int8_zlib_roundtrip_exact\\s+val_loss:([0-9.]+)\\s+val_bpb:([0-9.]+)", log_text)
    size_match = re.search(r"Total submission size int8\\+zlib:\\s*([0-9]+)\\s*bytes", log_text)
    steps_match = re.search(r"stopping_early: wallclock_cap .* step:([0-9]+)/([0-9]+)", log_text)

    out: dict[str, float | int | None] = {
        "val_loss": None,
        "val_bpb": None,
        "bytes_total_int8_zlib": None,
        "steps_done": None,
        "steps_target": None,
    }
    if bpb_match:
        out["val_loss"] = float(bpb_match.group(1))
        out["val_bpb"] = float(bpb_match.group(2))
    if size_match:
        out["bytes_total_int8_zlib"] = int(size_match.group(1))
    if steps_match:
        out["steps_done"] = int(steps_match.group(1))
        out["steps_target"] = int(steps_match.group(2))
    return out


@app.function(
    image=image,
    gpu="H100",
    timeout=60 * 60,
    cpu=8,
    memory=64 * 1024,
)
def run_baseline_one_h100(
    train_shards: int = 1,
    max_wallclock_seconds: int = 180,
    iterations: int = 600,
    val_loss_every: int = 200,
    train_log_every: int = 100,
    run_id: str = "modal_1h100_baseline_smoke",
) -> dict[str, object]:
    os.chdir(REPO_REMOTE_PATH)
    Path(RECORD_DIR).mkdir(parents=True, exist_ok=True)

    # Download challenge data/tokenizer cache (small shard count for smoke by default).
    subprocess.run(
        [
            "python",
            "data/cached_challenge_fineweb.py",
            "--variant",
            "sp1024",
            "--train-shards",
            str(train_shards),
        ],
        check=True,
    )

    env = os.environ.copy()
    env.update(
        {
            "RUN_ID": run_id,
            "DATA_PATH": "./data/datasets/fineweb10B_sp1024",
            "TOKENIZER_PATH": "./data/tokenizers/fineweb_1024_bpe.model",
            "VOCAB_SIZE": "1024",
            "MAX_WALLCLOCK_SECONDS": str(max_wallclock_seconds),
            "ITERATIONS": str(iterations),
            "VAL_LOSS_EVERY": str(val_loss_every),
            "TRAIN_LOG_EVERY": str(train_log_every),
            "CC": "gcc",
            "CXX": "g++",
        }
    )

    cmd = ["torchrun", "--standalone", "--nproc_per_node=1", TRAIN_SCRIPT]

    # Stream logs to Modal output and save a local copy in the record folder.
    lines: list[str] = []
    with open(TRAIN_LOG, "w", encoding="utf-8") as fout:
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            fout.write(line)
            lines.append(line)
        rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"Training command failed with exit code {rc}")

    full_log = "".join(lines)
    metrics = _extract_metrics(full_log)
    return {
        "run_id": run_id,
        "train_log_path": TRAIN_LOG,
        "train_shards": train_shards,
        "max_wallclock_seconds": max_wallclock_seconds,
        "iterations": iterations,
        "metrics": metrics,
        "log_tail": lines[-40:],
    }


@app.local_entrypoint()
def main(
    train_shards: int = 1,
    max_wallclock_seconds: int = 180,
    iterations: int = 600,
    val_loss_every: int = 200,
    train_log_every: int = 100,
    run_id: str = "modal_1h100_baseline_smoke",
):
    result = run_baseline_one_h100.remote(
        train_shards=train_shards,
        max_wallclock_seconds=max_wallclock_seconds,
        iterations=iterations,
        val_loss_every=val_loss_every,
        train_log_every=train_log_every,
        run_id=run_id,
    )
    print("\\n=== Modal run summary ===")
    print(result)
