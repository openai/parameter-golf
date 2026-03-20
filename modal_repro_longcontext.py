from __future__ import annotations

import json
import os
import re
import subprocess

import modal

APP_NAME = "parameter-golf-repro-longcontext-8h100"
REPO_REMOTE_PATH = "/workspace/parameter-golf"
TARGET_SCRIPT = "records/track_10min_16mb/2026-03-18_LongContextSeq2048/train_gpt.py"

STEP_RE = re.compile(r"step:(\d+)/(\d+).*step_avg:([0-9.]+)ms")
CONFIG_RE = re.compile(
    r"train_batch_tokens:(\d+)\s+train_seq_len:(\d+)\s+iterations:(\d+)\s+warmup_steps:(\d+)\s+"
    r"max_wallclock_seconds:([0-9.]+)"
)
FINAL_RE = re.compile(r"final_int8_zlib_roundtrip_exact\s+val_loss:([0-9.]+)\s+val_bpb:([0-9.]+)")
SIZE_RE = re.compile(r"Total submission size int8\+zlib:\s*([0-9]+)\s*bytes")
STOP_RE = re.compile(r"stopping_early: wallclock_cap .* step:([0-9]+)/([0-9]+)")

app = modal.App(APP_NAME)
image = (
    # The devel image includes the CUDA toolchain pieces that torch.compile / Triton
    # tend to expect on tuned boxes; the runtime image is more likely to underperform.
    modal.Image.from_registry("pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel")
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


def _run_checked(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    subprocess.run(cmd, check=True, env=env)


def _print_probe(command: list[str]) -> None:
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
    joined = " ".join(command)
    print(f"\n=== PROBE: {joined} ===")
    print(result.stdout.rstrip())


@app.function(image=image, gpu="H100:8", timeout=90 * 60, cpu=64, memory=196608)
def run(
    run_id: str = "modal_longcontext_8h100_repro",
    target_script: str = TARGET_SCRIPT,
    data_variant: str = "sp1024",
    max_wallclock_seconds: int = 600,
    expected_train_seq_len: int = 2048,
    expected_max_step_avg_ms: float = 60.0,
    gate_check_step: int = 1000,
    enable_throughput_gate: bool = False,
    nccl_ib_disable: int | None = None,
    extra_env_json: str = "{}",
) -> dict[str, object]:
    os.chdir(REPO_REMOTE_PATH)

    _run_checked(["python", "data/cached_challenge_fineweb.py", "--variant", data_variant])

    env = os.environ.copy()
    env.update(
        {
            "RUN_ID": run_id,
            "DATA_PATH": f"./data/datasets/fineweb10B_{data_variant}",
            "TOKENIZER_PATH": "./data/tokenizers/fineweb_1024_bpe.model",
            "VOCAB_SIZE": "1024",
            "MAX_WALLCLOCK_SECONDS": str(max_wallclock_seconds),
            "OMP_NUM_THREADS": "1",
            "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
            "CC": "gcc",
            "CXX": "g++",
        }
    )
    if nccl_ib_disable is None:
        env.pop("NCCL_IB_DISABLE", None)
    else:
        env["NCCL_IB_DISABLE"] = str(int(nccl_ib_disable))

    extra_env = json.loads(extra_env_json)
    if not isinstance(extra_env, dict):
        raise TypeError("extra_env_json must decode to a JSON object")
    env.update({str(k): str(v) for k, v in extra_env.items()})

    _print_probe(["python", "-c", "import torch; print(torch.__version__)"])
    _print_probe(["python", "-c", "import triton; print(triton.__version__)"])
    _print_probe(["bash", "-lc", "command -v ptxas || true"])
    _print_probe(["nvidia-smi", "topo", "-m"])
    _print_probe(["python", "-c", "import os; print(os.cpu_count())"])

    print("\n=== REPRO ENV ===")
    print(
        {
            key: env[key]
            for key in (
                "RUN_ID",
                "DATA_PATH",
                "TOKENIZER_PATH",
                "VOCAB_SIZE",
                "MAX_WALLCLOCK_SECONDS",
                "OMP_NUM_THREADS",
                "TORCH_NCCL_ASYNC_ERROR_HANDLING",
                "NCCL_IB_DISABLE",
                "CC",
                "CXX",
            )
            if key in env
        }
    )

    cmd = ["torchrun", "--standalone", "--nproc_per_node=8", target_script]
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

    lines: list[str] = []
    last_step = 0
    last_step_avg_ms = 0.0
    observed_train_seq_len: int | None = None
    gate_checked = False

    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")
        lines.append(line)

        step_match = STEP_RE.search(line)
        if step_match:
            last_step = int(step_match.group(1))
            last_step_avg_ms = float(step_match.group(3))
            if enable_throughput_gate and (not gate_checked) and last_step >= gate_check_step:
                gate_checked = True
                if last_step_avg_ms > expected_max_step_avg_ms:
                    proc.terminate()
                    try:
                        proc.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    raise RuntimeError(
                        "Throughput gate failed: "
                        f"step={last_step} step_avg_ms={last_step_avg_ms:.2f} "
                        f"threshold_ms={expected_max_step_avg_ms:.2f}"
                    )

        config_match = CONFIG_RE.search(line)
        if config_match:
            observed_train_seq_len = int(config_match.group(2))

    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"Training failed with exit code {rc}")

    if observed_train_seq_len is None:
        raise RuntimeError("Could not parse train_seq_len from training log")
    if observed_train_seq_len != expected_train_seq_len:
        raise RuntimeError(
            f"Unexpected TRAIN_SEQ_LEN in log: expected {expected_train_seq_len}, got {observed_train_seq_len}"
        )
    if enable_throughput_gate and not gate_checked:
        raise RuntimeError(f"Throughput gate was enabled but log never reached step {gate_check_step}")

    log = "".join(lines)
    exact = FINAL_RE.search(log)
    size = SIZE_RE.search(log)
    stop = STOP_RE.search(log)

    out = {
        "run_id": run_id,
        "target_script": target_script,
        "observed_train_seq_len": observed_train_seq_len,
        "last_step_seen": last_step,
        "last_step_avg_ms": last_step_avg_ms,
        "val_loss": float(exact.group(1)) if exact else None,
        "val_bpb": float(exact.group(2)) if exact else None,
        "bytes_total_int8_zlib": int(size.group(1)) if size else None,
        "steps_done": int(stop.group(1)) if stop else None,
        "steps_target": int(stop.group(2)) if stop else None,
        "nccl_ib_disable": env.get("NCCL_IB_DISABLE"),
    }
    print("\n=== REPRO SUMMARY ===")
    print(out)
    return out


@app.local_entrypoint()
def main(
    run_id: str = "modal_longcontext_8h100_repro",
    target_script: str = TARGET_SCRIPT,
    data_variant: str = "sp1024",
    max_wallclock_seconds: int = 600,
    expected_train_seq_len: int = 2048,
    expected_max_step_avg_ms: float = 60.0,
    gate_check_step: int = 1000,
    enable_throughput_gate: bool = False,
    nccl_ib_disable: int | None = None,
    extra_env_json: str = "{}",
) -> None:
    print(
        run.remote(
            run_id=run_id,
            target_script=target_script,
            data_variant=data_variant,
            max_wallclock_seconds=max_wallclock_seconds,
            expected_train_seq_len=expected_train_seq_len,
            expected_max_step_avg_ms=expected_max_step_avg_ms,
            gate_check_step=gate_check_step,
            enable_throughput_gate=enable_throughput_gate,
            nccl_ib_disable=nccl_ib_disable,
            extra_env_json=extra_env_json,
        )
    )
