"""Modal app: run Trinity v5 (Pre-quant TTT + SLOT) on 8xH100 SXM.
Uses PyTorch 2.9 + Flash Attention (2.x or 3) to match PR #1329's performance.

Usage:
    modal run --detach modal/run_v4.py --seed 42
"""

import modal
import os
from pathlib import Path

app = modal.App("trinity-v5-parameter-golf")

# Use the official NVIDIA PyTorch 2.9 image that has CUDA runtime + PyTorch pre-installed.
# Based on nvcr.io/nvidia/pytorch images which come with FA3 support.
image = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel",
        add_python="3.11",
    )
    .apt_install("git", "build-essential", "wget")
    .run_commands(
        # Upgrade to torch 2.9.1+cu128 like PR #1329
        "pip install --upgrade pip",
        "pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "ninja",  # Required for flash-attn compilation
        "packaging",
        "wheel",
    )
    .run_commands(
        # flash-attn with TORCH_CUDA_ARCH_LIST set for H100 (sm_90)
        "TORCH_CUDA_ARCH_LIST='9.0' FLASH_ATTENTION_FORCE_BUILD=TRUE pip install flash-attn==2.7.4.post1 --no-build-isolation || pip install flash-attn==2.6.3 --no-build-isolation",
    )
    .pip_install(
        "sentencepiece",
        "huggingface-hub",
        "datasets",
        "tqdm",
        "numpy",
    )
    .run_commands(
        "git clone https://github.com/openai/parameter-golf.git /root/parameter-golf",
        "cd /root/parameter-golf && python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10",
    )
)

# Add train_gpt.py to image
LOCAL_TRAIN = str(Path(__file__).parent.parent / "records/track_10min_16mb/2026-04-02_Trinity_Hybrid_Ternary_GPTQ_XSA/train_gpt.py")
image = image.add_local_file(LOCAL_TRAIN, remote_path="/root/train_gpt.py")


@app.function(
    image=image,
    gpu="H100:8",
    timeout=3600,
)
def run_seed(seed: int):
    """Run a single seed of Trinity v5 and return the val_bpb."""
    import subprocess
    import shutil

    shutil.copy("/root/train_gpt.py", "/root/parameter-golf/train_gpt.py")

    env = os.environ.copy()
    env.update({
        "SEED": str(seed),
        "RUN_ID": f"trinity_v5_modal_seed{seed}",
        "TTT_ENABLED": "1",
        "TTT_LR": "0.001",
        "TTT_EPOCHS": "1",
        "TTT_CHUNK_TOKENS": "32768",
        "TTT_FREEZE_BLOCKS": "10",
        "TTT_BATCH_SEQS": "32",
        "SLOT_LR": "0.024",
        "SLOT_STEPS": "24",
        "SLOT_STRIDE": "64",
        "GPTQ_DAMP_FACTOR": "0.005",
        "GPTQ_CALIB_VAL": "1",
        "GPTQ_CALIB_BATCHES": "256",
        "QK_GAIN_INIT": "4.0",
        "MTP_NUM_HEADS": "2",
        "MTP_LOSS_WEIGHT": "0.1",
        "MAX_WALLCLOCK_SECONDS": "600",
    })

    result = subprocess.run(
        ["torchrun", "--standalone", "--nproc_per_node=8", "train_gpt.py"],
        cwd="/root/parameter-golf",
        env=env,
        capture_output=True,
        text=True,
    )

    log = result.stdout + result.stderr

    slot_bpb = None
    for line in log.splitlines():
        if "final_slot_exact" in line and "val_bpb:" in line:
            try:
                slot_bpb = float(line.split("val_bpb:")[-1].strip())
            except ValueError:
                pass

    return {
        "seed": seed,
        "slot_bpb": slot_bpb,
        "log_tail": log[-10000:],
    }


@app.local_entrypoint()
def main(seed: int = 42):
    print(f"Running Trinity v5 seed {seed} on Modal 8xH100 SXM...")
    result = run_seed.remote(seed)
    print(f"\n=== Seed {seed} done ===")
    print(f"SLOT BPB: {result['slot_bpb']}")
    print(f"\n=== Log tail ===\n{result['log_tail']}")
