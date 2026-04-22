"""Modal: Trinity v6 N-gram Order-22 on 8xH100.
Simple image: torch 2.5.1 + flash-attn prebuilt wheel. No FA3 — our code has FA2 fallback.

Usage: modal run --detach modal/run_v6.py --seed 42
"""
import modal, os
from pathlib import Path

app = modal.App("trinity-v6-ngram")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch==2.5.1",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install("sentencepiece", "huggingface-hub", "datasets", "tqdm", "numpy")
    .run_commands(
        "git clone https://github.com/openai/parameter-golf.git /root/pgolf",
        "cd /root/pgolf && python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10",
    )
)

LOCAL_TRAIN = str(Path(__file__).parent.parent / "records/track_10min_16mb/2026-04-02_Trinity_Hybrid_Ternary_GPTQ_XSA/train_gpt.py")
image = image.add_local_file(LOCAL_TRAIN, remote_path="/root/train_gpt.py")

@app.function(image=image, gpu="H100:8", timeout=7200)  # 2 hours — SDPA fallback is slow
def run_seed(seed: int):
    import subprocess, shutil
    shutil.copy("/root/train_gpt.py", "/root/pgolf/train_gpt.py")
    env = os.environ.copy()
    env.update({
        "SEED": str(seed), "RUN_ID": f"v6_s{seed}",
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
    # First: quick smoke test — import check on 1 GPU
    import sys
    smoke = subprocess.run(
        [sys.executable, "-c", "import torch; print(f'torch {torch.__version__}, cuda {torch.cuda.is_available()}, gpus {torch.cuda.device_count()}'); import train_gpt; print('import OK')"],
        cwd="/root/pgolf", env=env, capture_output=True, text=True,
    )
    print(f"SMOKE: {smoke.stdout.strip()}")
    if smoke.returncode != 0:
        print(f"SMOKE ERROR: {smoke.stderr[-3000:]}")
        return {"seed": seed, "bpb": None, "log": f"SMOKE FAILED:\n{smoke.stderr[-5000:]}"}

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
    print(f"Running v6 seed {seed} on Modal 8xH100...")
    r = run_seed.remote(seed)
    print(f"\nSeed {seed}: BPB={r['bpb']}")
    print(f"\n{r['log']}")
