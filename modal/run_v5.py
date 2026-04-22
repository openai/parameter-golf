"""Modal app: run Trinity v5 (3 bug fixes) on 8xH100 SXM.
Uses nvcr.io/nvidia/pytorch image which has pre-installed FA3 + CUDA 12.8 + PyTorch 2.9.

Usage:
    modal run --detach modal/run_v5.py --seed 42
"""

import modal
import os
from pathlib import Path

app = modal.App("trinity-v5-pgolf")

# Lightweight image: use Modal's debian_slim + install torch/flash-attn from pre-built wheels
# This is much faster than pulling 25GB nvcr image
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "wget", "build-essential")
    .pip_install(
        "torch==2.5.1",
        "torchvision",
        "torchaudio",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        # Flash Attention — use pre-built wheel for torch 2.5.1 + cu124 + python3.11
        "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl",
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
        "RUN_ID": f"trinity_v5_seed{seed}",
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

    return {"seed": seed, "slot_bpb": slot_bpb, "log_tail": log[-10000:]}


@app.local_entrypoint()
def main(seed: int = 42):
    print(f"Running Trinity v5 seed {seed} on Modal 8xH100 SXM...")
    result = run_seed.remote(seed)
    print(f"\n=== Seed {seed} done ===")
    print(f"SLOT BPB: {result['slot_bpb']}")
    print(f"\n=== Log tail ===\n{result['log_tail']}")
