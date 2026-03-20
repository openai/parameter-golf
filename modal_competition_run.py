"""
Modal script to run the parameter-golf competition training on 8×H100 GPUs.
Captures the full train.log and prints all metrics for submission.
"""

import modal
import os
import subprocess
import sys
import time

# Build the container image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.5",
        "numpy",
        "sentencepiece",
        "huggingface_hub",
    )
    # Add our training script and data download script
    .add_local_file(
        "autoresearch/core_promotion/train_gpt.best.py",
        remote_path="/root/parameter-golf/train_gpt.py",
        copy=True,
    )
    .add_local_dir(
        "data",
        remote_path="/root/parameter-golf/data",
        copy=True,
    )
    # Download the dataset at image build time so it's cached
    .run_commands(
        "cd /root/parameter-golf && python3 data/cached_challenge_fineweb.py --variant sp1024"
    )
)

app = modal.App("parameter-golf-competition", image=image)


@app.function(
    gpu="H100:8",
    timeout=60 * 60,  # 60 min total (20k steps ~28 min + warmup + eval)
    memory=65536,  # 64 GB system RAM
)
def run_competition_training():
    """Run the full 8×H100 competition training and capture all output."""
    import json

    work_dir = "/root/parameter-golf"

    # Verify GPUs
    result = subprocess.run(
        ["python3", "-c", "import torch; print(f'GPUs: {torch.cuda.device_count()}'); [print(f'  {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr)

    # Verify dataset exists
    data_path = os.path.join(work_dir, "data", "datasets", "fineweb10B_sp1024")
    if os.path.isdir(data_path):
        files = os.listdir(data_path)
        print(f"Dataset ready: {len(files)} files in {data_path}")
    else:
        print(f"ERROR: Dataset not found at {data_path}")
        return

    # Build the torchrun command
    env = os.environ.copy()
    env.update({
        "RUN_ID": "full_20k_8xH100_run1",
        "DATA_PATH": data_path,
        "TOKENIZER_PATH": os.path.join(work_dir, "data", "tokenizers", "fineweb_1024_bpe.model"),
        "VOCAB_SIZE": "1024",
        "MAX_WALLCLOCK_SECONDS": "0",
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
    print(f"STARTING FULL 20K RUN: 8×H100, no wallclock cap")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")

    # Run training and capture ALL output
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
        print(line)  # Print to Modal logs
        log_lines.append(line)

    process.wait()
    elapsed = time.time() - start_time

    print(f"\n{'='*80}")
    print(f"TRAINING COMPLETE. Exit code: {process.returncode}. Elapsed: {elapsed:.1f}s")
    print(f"{'='*80}\n")

    # Write the full log
    log_text = "\n".join(log_lines)
    log_path = os.path.join(work_dir, "train.log")
    with open(log_path, "w") as f:
        f.write(log_text)

    # Extract key metrics from the log
    metrics = {}
    for line in log_lines:
        if "final_int8_zlib_roundtrip_exact" in line and "val_bpb" in line:
            # Parse: final_int8_zlib_roundtrip_exact val_bpb:X.XXXXX
            for part in line.split():
                if part.startswith("val_bpb:"):
                    metrics["val_bpb"] = float(part.split(":")[1])
        if "final_int8_zlib_roundtrip_exact" in line and "val_loss" in line:
            for part in line.split():
                if part.startswith("val_loss:"):
                    metrics["val_loss"] = float(part.split(":")[1])
        if "int8+zlib total_submission_bytes:" in line:
            for part in line.split():
                if part.startswith("total_submission_bytes:"):
                    metrics["bytes_total"] = int(part.split(":")[1])
        if "code_bytes:" in line and "int8+zlib" in line:
            for part in line.split():
                if part.startswith("code_bytes:"):
                    metrics["bytes_code"] = int(part.split(":")[1])

    print(f"\n{'='*80}")
    print("EXTRACTED METRICS:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"{'='*80}\n")

    # Return the full log and metrics
    return {
        "log": log_text,
        "metrics": metrics,
        "exit_code": process.returncode,
        "elapsed_seconds": elapsed,
    }


@app.local_entrypoint()
def main():
    """Local entrypoint — runs on your machine, dispatches to Modal."""
    import json

    print("Launching full 20k step run on Modal (8×H100, no wallclock cap)...")
    result = run_competition_training.remote()

    if result is None:
        print("ERROR: No result returned")
        return

    # Save the train log locally
    log_dir = "records/non_record/2026-03-19_CorePromotion13_full20k"
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, "train.log")
    with open(log_path, "w") as f:
        f.write(result["log"])
    print(f"Saved train.log to {log_path}")

    # Copy the training script
    import shutil
    shutil.copy(
        "autoresearch/core_promotion/train_gpt.best.py",
        os.path.join(log_dir, "train_gpt.py"),
    )
    print(f"Copied train_gpt.py to {log_dir}/")

    # Create submission.json
    metrics = result.get("metrics", {})
    submission = {
        "author": "Simon Marcus",
        "github_id": "simon",
        "name": "CorePromotion Exp13 — 12×448 Optimized",
        "blurb": "12-layer 448-dim transformer with optimized matrix_lr=0.08, warmdown_iters=500. Full 20,000 step run on 8×H100 (non-record, no wallclock cap).",
        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "val_loss": metrics.get("val_loss"),
        "val_bpb": metrics.get("val_bpb"),
        "bytes_total": metrics.get("bytes_total"),
        "bytes_code": metrics.get("bytes_code"),
    }
    sub_path = os.path.join(log_dir, "submission.json")
    with open(sub_path, "w") as f:
        json.dump(submission, f, indent=2)
    print(f"Saved submission.json to {sub_path}")

    # Print summary
    print(f"\n{'='*80}")
    print("COMPETITION RUN SUMMARY")
    print(f"  Exit code: {result['exit_code']}")
    print(f"  Elapsed: {result['elapsed_seconds']:.1f}s")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"  Output dir: {log_dir}/")
    print(f"{'='*80}")
