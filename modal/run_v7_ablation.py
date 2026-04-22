"""Modal: Trinity v7 Ablation Study — test each improvement independently.
Runs 5 configs on seed 42:
  A) v6 baseline (batch_seqs=32, no v7 features) — control
  B) v6 + fix slot_batch_seqs=128 only
  C) B + entropy skip (thresh=1.5)
  D) B + logistic mixing
  E) B + skip + logistic + APM (full v7)

Usage: modal run --detach modal/run_v7_ablation.py
"""
import modal, os
from pathlib import Path

app = modal.App("trinity-v7-ablation")

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

CONFIGS = {
    "A_v6_baseline": {"SLOT_BATCH_SEQS": "32", "NGRAM_SKIP_THRESH": "-1", "NGRAM_LOGISTIC_MIX": "0", "NGRAM_APM_ENABLED": "0"},
    "B_batch128": {"SLOT_BATCH_SEQS": "128", "NGRAM_SKIP_THRESH": "-1", "NGRAM_LOGISTIC_MIX": "0", "NGRAM_APM_ENABLED": "0"},
    "C_skip1.5": {"SLOT_BATCH_SEQS": "128", "NGRAM_SKIP_THRESH": "1.5", "NGRAM_LOGISTIC_MIX": "0", "NGRAM_APM_ENABLED": "0"},
    "D_logistic": {"SLOT_BATCH_SEQS": "128", "NGRAM_SKIP_THRESH": "-1", "NGRAM_LOGISTIC_MIX": "1", "NGRAM_APM_ENABLED": "0"},
    "E_full_v7": {"SLOT_BATCH_SEQS": "128", "NGRAM_SKIP_THRESH": "1.5", "NGRAM_LOGISTIC_MIX": "1", "NGRAM_APM_ENABLED": "1"},
}

@app.function(image=image, gpu="H100:4", timeout=7200)
def run_config(name: str, overrides: dict, seed: int = 42):
    import subprocess, shutil, sys
    shutil.copy("/root/train_gpt.py", "/root/pgolf/train_gpt.py")
    env = os.environ.copy()
    env.update({
        "SEED": str(seed), "RUN_ID": f"v7abl_{name}_s{seed}",
        "TTT_ENABLED": "1", "TTT_LR": "0.001", "TTT_EPOCHS": "1",
        "TTT_CHUNK_TOKENS": "32768", "TTT_FREEZE_BLOCKS": "10", "TTT_BATCH_SEQS": "32",
        "SLOT_LR": "0.432", "SLOT_STEPS": "24", "SLOT_STRIDE": "64",
        "SLOT_BETA1": "0.6", "SLOT_BETA2": "0.5",
        "NGRAM_ENABLED": "1", "NGRAM_ORDER": "22", "NGRAM_BUCKETS": "4194304",
        "NGRAM_MIN_COUNT": "2", "NGRAM_MIN_TOKENS": "5000",
        "NGRAM_ALPHA_BASE": "0.20", "NGRAM_ALPHA_RANGE": "0.55", "NGRAM_ALPHA_CENTER": "2.5",
        "NGRAM_APM_LR": "0.005",
        "GPTQ_DAMP_FACTOR": "0.005", "GPTQ_CALIB_VAL": "1", "GPTQ_CALIB_BATCHES": "256",
        "QK_GAIN_INIT": "4.0", "MTP_NUM_HEADS": "2", "MTP_LOSS_WEIGHT": "0.1",
        "MAX_WALLCLOCK_SECONDS": "600",
    })
    env.update(overrides)
    try:
        import torch
        nproc = torch.cuda.device_count()
    except:
        nproc = 4
    r = subprocess.run(
        ["torchrun", "--standalone", f"--nproc_per_node={nproc}", "train_gpt.py"],
        cwd="/root/pgolf", env=env, capture_output=True, text=True,
    )
    log = r.stdout + r.stderr
    bpb = None
    for line in log.splitlines():
        if "final_slot_exact" in line and "val_bpb:" in line:
            try: bpb = float(line.split("val_bpb:")[-1].strip())
            except: pass
    return {"name": name, "bpb": bpb, "log": log[-5000:]}

@app.local_entrypoint()
def main():
    print("=== Trinity v7 Ablation Study ===\n")
    # Launch all configs in parallel on separate machines
    futures = []
    for name, overrides in CONFIGS.items():
        print(f"  Launching {name}...")
        futures.append((name, run_config.spawn(name, overrides)))

    print(f"\n{len(futures)} configs running in parallel on Modal...\n")

    results = {}
    for name, future in futures:
        r = future.get()
        results[name] = r['bpb']
        print(f"  {name}: BPB = {r['bpb']}")

    print("\n=== ABLATION RESULTS ===")
    print(f"{'Config':<20} {'BPB':>10} {'vs baseline':>12}")
    baseline = results.get("A_v6_baseline")
    for name in CONFIGS:
        bpb = results.get(name)
        if bpb is not None and baseline is not None:
            delta = bpb - baseline
            print(f"  {name:<18} {bpb:>10.5f} {delta:>+12.5f}")
        else:
            print(f"  {name:<18} {'FAILED':>10}")
