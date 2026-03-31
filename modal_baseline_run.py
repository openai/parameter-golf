"""
Modal script to run the OFFICIAL BASELINE train_gpt.py on 8×H100 GPUs.
Purpose: measure whether Modal itself is slower than the reported 43ms/step.
"""

import modal
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENV_LOCAL = ROOT / ".env.local"


def load_env_local() -> dict[str, str]:
    values: dict[str, str] = {}
    if not ENV_LOCAL.exists():
        return values
    for line in ENV_LOCAL.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        values[k.strip()] = v.strip()
    return values


ENV_LOCAL_VALUES = load_env_local()
HF_TOKEN = os.environ.get("HF_TOKEN") or ENV_LOCAL_VALUES.get("HF_TOKEN")

# Build the container image — uses the STOCK train_gpt.py from repo root
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.5",
        "numpy",
        "sentencepiece",
        "huggingface_hub",
    )
    .env({"HF_TOKEN": HF_TOKEN} if HF_TOKEN else {})
    .add_local_file(
        "train_gpt.py",
        remote_path="/root/parameter-golf/train_gpt.py",
        copy=True,
    )
    .add_local_dir(
        "data",
        remote_path="/root/parameter-golf/data",
        copy=True,
    )
    .run_commands(
        "cd /root/parameter-golf && python3 data/cached_challenge_fineweb.py --variant sp1024"
    )
)

app = modal.App("parameter-golf-baseline", image=image)


@app.function(
    gpu="H100:8",
    timeout=30 * 60,
    memory=65536,
)
def run_baseline_training():
    """Run the stock baseline on 8×H100 and capture output."""

    work_dir = "/root/parameter-golf"

    # Verify GPUs
    result = subprocess.run(
        ["python3", "-c", "import torch; print(f'GPUs: {torch.cuda.device_count()}'); [print(f'  {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr)

    data_path = os.path.join(work_dir, "data", "datasets", "fineweb10B_sp1024")
    if not os.path.isdir(data_path):
        print(f"ERROR: Dataset not found at {data_path}")
        return

    print(f"Dataset ready: {len(os.listdir(data_path))} files")

    env = os.environ.copy()
    env.update({
        "RUN_ID": "baseline_modal_8xH100",
        "DATA_PATH": data_path,
        "TOKENIZER_PATH": os.path.join(work_dir, "data", "tokenizers", "fineweb_1024_bpe.model"),
        "VOCAB_SIZE": "1024",
        "MAX_WALLCLOCK_SECONDS": "600",
        "TRAIN_LOG_EVERY": "50",
        "VAL_LOSS_EVERY": "200",
        "NCCL_IB_DISABLE": "1",
    })

    train_script = os.path.join(work_dir, "train_gpt.py")
    cmd = [
        "torchrun",
        "--standalone",
        "--nproc_per_node=8",
        train_script,
    ]

    print(f"\n{'='*80}")
    print(f"BASELINE RUN: stock train_gpt.py, 8×H100, 600s wallclock")
    print(f"Expected step_avg from official: ~43ms/step")
    print(f"{'='*80}\n")

    log_lines = []
    start_time = time.time()

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        cwd=work_dir,
    )

    for line in process.stdout:
        line = line.rstrip()
        print(line)
        log_lines.append(line)

    process.wait()
    elapsed = time.time() - start_time

    log_text = "\n".join(log_lines)

    # Extract step_avg from last training step line
    last_step_avg = None
    last_step = None
    val_bpb = None
    for line in log_lines:
        if "step_avg:" in line and "train_loss:" in line:
            for part in line.split():
                if part.startswith("step_avg:"):
                    last_step_avg = part.split(":")[1]
                if part.startswith("step:"):
                    last_step = part.split(":")[1]
        if "final_int8_zlib_roundtrip_exact" in line and "val_bpb:" in line:
            for part in line.split():
                if part.startswith("val_bpb:"):
                    val_bpb = part.split(":")[1]

    print(f"\n{'='*80}")
    print("BASELINE RESULTS:")
    print(f"  Last step: {last_step}")
    print(f"  Step avg: {last_step_avg}")
    print(f"  Final val_bpb: {val_bpb}")
    print(f"  Exit code: {process.returncode}")
    print(f"  Total elapsed: {elapsed:.1f}s")
    print(f"{'='*80}\n")

    return {
        "log": log_text,
        "last_step": last_step,
        "step_avg": last_step_avg,
        "val_bpb": val_bpb,
        "exit_code": process.returncode,
        "elapsed_seconds": elapsed,
    }


@app.local_entrypoint()
def main():
    print("Launching BASELINE run on Modal (8×H100)...")
    print("Comparing Modal step time vs official 43ms/step baseline...")
    result = run_baseline_training.remote()

    if result is None:
        print("ERROR: No result returned")
        return

    # Save the log
    log_dir = "records/track_10min_16mb/2026-03-19_BaselineModal"
    os.makedirs(log_dir, exist_ok=True)

    with open(os.path.join(log_dir, "train.log"), "w") as f:
        f.write(result["log"])

    import shutil
    shutil.copy("train_gpt.py", os.path.join(log_dir, "train_gpt.py"))

    print(f"\n{'='*80}")
    print("BASELINE COMPARISON")
    print(f"  Official RunPod step_avg: ~43.54ms")
    print(f"  Modal step_avg:           {result['step_avg']}ms")
    print(f"  Last step:                {result['last_step']}")
    print(f"  Final val_bpb:            {result['val_bpb']}")
    print(f"  Saved to: {log_dir}/")
    print(f"{'='*80}")
