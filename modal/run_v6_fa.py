"""Modal: Trinity v6 N-gram — WITH flash-attn on CUDA devel image.
Parallel attempt: if FA compiles, this will be 5x faster than SDPA fallback.

Usage: modal run --detach modal/run_v6_fa.py --seed 42
"""
import modal, os
from pathlib import Path

app = modal.App("trinity-v6-ngram-fa")

# CUDA devel image — has nvcc for flash-attn compilation
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "ninja-build")
    .pip_install(
        "torch==2.5.1",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install("packaging", "wheel", "setuptools")
    .run_commands(
        # Build flash-attn from source with H100 arch
        "MAX_JOBS=4 TORCH_CUDA_ARCH_LIST='9.0' pip install flash-attn==2.7.3 --no-build-isolation 2>&1 | tail -20",
    )
    .pip_install("sentencepiece", "huggingface-hub", "datasets", "tqdm", "numpy")
    .run_commands(
        "git clone https://github.com/openai/parameter-golf.git /root/pgolf",
        "cd /root/pgolf && python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10",
    )
)

LOCAL_TRAIN = str(Path(__file__).parent.parent / "records/track_10min_16mb/2026-04-02_Trinity_Hybrid_Ternary_GPTQ_XSA/train_gpt.py")
image = image.add_local_file(LOCAL_TRAIN, remote_path="/root/train_gpt.py")

@app.function(image=image, gpu="H100:8", timeout=3600)
def run_seed(seed: int):
    import subprocess, shutil, sys
    shutil.copy("/root/train_gpt.py", "/root/pgolf/train_gpt.py")

    # Smoke test
    smoke = subprocess.run(
        [sys.executable, "-c",
         "import torch; print(f'torch {torch.__version__}, cuda {torch.cuda.is_available()}, gpus {torch.cuda.device_count()}');"
         "try:\n from flash_attn import flash_attn_func; print('FA2 OK')\nexcept: print('FA2 MISSING');"
         "try:\n from flash_attn_interface import flash_attn_func; print('FA3 OK')\nexcept: print('FA3 MISSING')"],
        capture_output=True, text=True)
    print(f"SMOKE: {smoke.stdout.strip()}")
    if "MISSING" in smoke.stdout and "FA2 MISSING" in smoke.stdout:
        return {"seed": seed, "bpb": None, "log": f"FA install failed:\n{smoke.stderr[-3000:]}"}

    env = os.environ.copy()
    env.update({
        "SEED": str(seed), "RUN_ID": f"v6fa_s{seed}",
        "TTT_ENABLED": "1", "TTT_LR": "0.001", "TTT_EPOCHS": "1",
        "TTT_CHUNK_TOKENS": "32768", "TTT_FREEZE_BLOCKS": "10", "TTT_BATCH_SEQS": "32",
        "SLOT_LR": "0.432", "SLOT_STEPS": "24", "SLOT_STRIDE": "64",
        "SLOT_BETA1": "0.6", "SLOT_BETA2": "0.5", "SLOT_BATCH_SEQS": "128",
        "NGRAM_ENABLED": "1", "NGRAM_ORDER": "22", "NGRAM_BUCKETS": "4194304",
        "NGRAM_MIN_COUNT": "2", "NGRAM_MIN_TOKENS": "5000",
        "GPTQ_DAMP_FACTOR": "0.005", "GPTQ_CALIB_VAL": "1", "GPTQ_CALIB_BATCHES": "256",
        "QK_GAIN_INIT": "4.0", "MTP_NUM_HEADS": "2", "MTP_LOSS_WEIGHT": "0.1",
        "MAX_WALLCLOCK_SECONDS": "600",
    })
    r = subprocess.run(
        ["torchrun", "--standalone", "--nproc_per_node=8", "train_gpt.py"],
        cwd="/root/pgolf", env=env, capture_output=True, text=True,
    )
    log = r.stdout + r.stderr
    bpb = None
    for line in log.splitlines():
        if "final_slot_exact" in line and "val_bpb:" in line:
            try: bpb = float(line.split("val_bpb:")[-1].strip())
            except: pass
    return {"seed": seed, "bpb": bpb, "log": log[-15000:]}

@app.local_entrypoint()
def main(seed: int = 42):
    print(f"Running v6+FA seed {seed} on Modal 8xH100...")
    r = run_seed.remote(seed)
    print(f"\nSeed {seed}: BPB={r['bpb']}")
    print(f"\n{r['log']}")
