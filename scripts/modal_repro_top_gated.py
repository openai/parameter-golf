from __future__ import annotations

import os
import re
import subprocess

import modal

APP_NAME = "parameter-golf-repro-top-gated-8h100"
REPO_REMOTE_PATH = "/workspace/parameter-golf"
TARGET_SCRIPT = "records/track_10min_16mb/2026-03-19_SlidingWindow_FP16Emb_10L_MuonWD_OvertoneInit/train_gpt.py"

STEP_RE = re.compile(r"step:(\d+)/(\d+).*step_avg:([0-9.]+)ms")
FINAL_RE = re.compile(r"final_int8_zlib_roundtrip_exact\s+val_loss:([0-9.]+)\s+val_bpb:([0-9.]+)")
SIZE_RE = re.compile(r"Total submission size int8\+zlib:\s*([0-9]+)\s*bytes")
STOP_RE = re.compile(r"stopping_early: wallclock_cap .* step:([0-9]+)/([0-9]+)")

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


@app.function(image=image, gpu="H100:8", timeout=90 * 60, cpu=32, memory=196608)
def run_top(
    run_id: str,
    max_wallclock_seconds: int = 120,
    enable_throughput_gate: bool = True,
    gate_step_avg_ms: float = 60.0,
    gate_check_step: int = 1000,
) -> dict[str, object]:
    os.chdir(REPO_REMOTE_PATH)

    subprocess.run(["python", "data/cached_challenge_fineweb.py", "--variant", "sp1024"], check=True)

    env = os.environ.copy()
    env.update(
        {
            "RUN_ID": run_id,
            "DATA_PATH": "./data/datasets/fineweb10B_sp1024",
            "TOKENIZER_PATH": "./data/tokenizers/fineweb_1024_bpe.model",
            "VOCAB_SIZE": "1024",
            "MAX_WALLCLOCK_SECONDS": str(max_wallclock_seconds),
            "NCCL_IB_DISABLE": "1",
            "CC": "gcc",
            "CXX": "g++",
        }
    )

    cmd = ["torchrun", "--standalone", "--nproc_per_node=8", TARGET_SCRIPT]
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

    lines: list[str] = []
    last_step = 0
    last_step_avg_ms = 0.0
    gate_checked = False

    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")
        lines.append(line)
        m = STEP_RE.search(line)
        if m:
            last_step = int(m.group(1))
            last_step_avg_ms = float(m.group(3))
            if enable_throughput_gate and (not gate_checked) and last_step >= gate_check_step:
                gate_checked = True
                if last_step_avg_ms > gate_step_avg_ms:
                    print(
                        f"THROUGHPUT_GATE_FAIL step={last_step} step_avg_ms={last_step_avg_ms:.2f} "
                        f"threshold_ms={gate_step_avg_ms:.2f}"
                    )
                    proc.terminate()
                    try:
                        proc.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    raise RuntimeError(
                        f"THROUGHPUT_GATE_FAIL step={last_step} step_avg_ms={last_step_avg_ms:.2f} "
                        f"threshold_ms={gate_step_avg_ms:.2f}"
                    )

    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"Training failed with exit code {rc}")

    if enable_throughput_gate and not gate_checked:
        raise RuntimeError(
            f"THROUGHPUT_GATE_INCONCLUSIVE did not reach step {gate_check_step} (last_step={last_step})"
        )

    full_log = "".join(lines)
    exact = FINAL_RE.search(full_log)
    size = SIZE_RE.search(full_log)
    stop = STOP_RE.search(full_log)

    out = {
        "run_id": run_id,
        "steps_seen": last_step,
        "step_avg_ms": last_step_avg_ms,
        "gate_enabled": enable_throughput_gate,
        "gate_step_avg_ms": gate_step_avg_ms,
        "gate_check_step": gate_check_step,
        "val_loss": float(exact.group(1)) if exact else None,
        "val_bpb": float(exact.group(2)) if exact else None,
        "bytes_total_int8_zlib": int(size.group(1)) if size else None,
        "steps_done": int(stop.group(1)) if stop else None,
        "steps_target": int(stop.group(2)) if stop else None,
    }

    print("\n=== REPRO SUMMARY ===")
    print(out)
    return out


@app.local_entrypoint()
def main(
    run_id: str = "modal_top_preflight",
    max_wallclock_seconds: int = 120,
    enable_throughput_gate: bool = True,
    gate_step_avg_ms: float = 60.0,
    gate_check_step: int = 1000,
):
    print(
        run_top.remote(
            run_id=run_id,
            max_wallclock_seconds=max_wallclock_seconds,
            enable_throughput_gate=enable_throughput_gate,
            gate_step_avg_ms=gate_step_avg_ms,
            gate_check_step=gate_check_step,
        )
    )
