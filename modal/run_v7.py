"""Modal: Trinity v7 — N-gram Entropy Skip + Logistic Mix + APM + slot_batch_seqs fix.
All v7 features controlled via env vars (disabled by default = pure v6 behavior).

Usage:
    modal run --detach modal/run_v7.py --seed 42
    modal run --detach modal/run_v7.py --seed 42 --skip-thresh 1.5 --logistic-mix --apm
"""
import modal, os
from pathlib import Path

app = modal.App("trinity-v7-ngram")

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

@app.function(image=image, gpu="H100", timeout=14400)  # 4 hours for SDPA eval
def run_seed(seed: int, skip_thresh: float = -1.0, logistic_mix: bool = False,
             apm: bool = False, slot_batch: int = 128, slot_steps: int = 24,
             ngram_buckets: int = 4194304, alpha_base: float = 0.20,
             alpha_range: float = 0.55, alpha_center: float = 2.5,
             legal: bool = False, legal_alpha: float = 0.10, legal_order: int = 4,
             slot_optimizer: str = "adamw", slot_phi_rank: bool = False,
             ngram_enabled: bool = True):
    import subprocess, shutil, sys
    shutil.copy("/root/train_gpt.py", "/root/pgolf/train_gpt.py")

    # Smoke test
    smoke = subprocess.run(
        [sys.executable, "-c",
         "import torch; print(f'torch {torch.__version__}, cuda {torch.cuda.is_available()}, gpus {torch.cuda.device_count()}')"],
        capture_output=True, text=True)
    print(f"SMOKE: {smoke.stdout.strip()}")

    env = os.environ.copy()
    env.update({
        "SEED": str(seed), "RUN_ID": f"v7_s{seed}",
        # TTT params (unchanged from v6)
        "TTT_ENABLED": "1", "TTT_LR": "0.001", "TTT_EPOCHS": "1",
        "TTT_CHUNK_TOKENS": "32768", "TTT_FREEZE_BLOCKS": "10", "TTT_BATCH_SEQS": "32",
        # SLOT params (unchanged, but batch_seqs now properly used!)
        "SLOT_LR": "0.432", "SLOT_STEPS": str(slot_steps), "SLOT_STRIDE": "64",
        "SLOT_BETA1": "0.6", "SLOT_BETA2": "0.5", "SLOT_BATCH_SEQS": str(slot_batch),
        # N-gram base params
        "NGRAM_ENABLED": "1" if ngram_enabled else "0", "NGRAM_ORDER": "22", "NGRAM_BUCKETS": str(ngram_buckets),
        "NGRAM_MIN_COUNT": "2", "NGRAM_MIN_TOKENS": "5000",
        # v7 NEW: configurable alpha
        "NGRAM_ALPHA_BASE": str(alpha_base),
        "NGRAM_ALPHA_RANGE": str(alpha_range),
        "NGRAM_ALPHA_CENTER": str(alpha_center),
        # v7 NEW: entropy skip
        "NGRAM_SKIP_THRESH": str(skip_thresh),
        # v7 NEW: logistic-domain mixing
        "NGRAM_LOGISTIC_MIX": "1" if logistic_mix else "0",
        # v7 NEW: APM post-processing
        "NGRAM_APM_ENABLED": "1" if apm else "0",
        "NGRAM_APM_LR": "0.005",
        # LEGAL N-gram (PR #1642 compliant)
        "NGRAM_LEGAL": "1" if legal else "0",
        "NGRAM_LEGAL_ALPHA": str(legal_alpha),
        "NGRAM_LEGAL_ORDER": str(legal_order),
        # Trinity experiments
        "SLOT_OPTIMIZER": slot_optimizer,  # adamw | lion
        "SLOT_PHI_RANK": "1" if slot_phi_rank else "0",
        # v7 NEW: FP16 embeddings + per-row GPTQ clip
        "EMBED_QUANT": "fp16",
        "GPTQ_PER_ROW_CLIP": "1",
        # Model / training params
        "GPTQ_DAMP_FACTOR": "0.005", "GPTQ_CALIB_VAL": "1", "GPTQ_CALIB_BATCHES": "256",
        "QK_GAIN_INIT": "4.0", "MTP_NUM_HEADS": "2", "MTP_LOSS_WEIGHT": "0.1",
        "MAX_WALLCLOCK_SECONDS": "600",
    })

    nproc = env.get("CUDA_VISIBLE_DEVICES", "0,1,2,3").count(",") + 1
    try:
        import torch
        nproc = torch.cuda.device_count()
    except:
        pass

    # Stream output live + save to file for later retrieval
    import sys
    log_path = "/tmp/train.log"
    bpb = None
    with open(log_path, "w") as logf:
        p = subprocess.Popen(
            ["torchrun", "--standalone", f"--nproc_per_node={nproc}", "train_gpt.py"],
            cwd="/root/pgolf", env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        for line in p.stdout:
            print(line, end="", flush=True)  # stream to Modal logs
            logf.write(line); logf.flush()
            if "final_slot_exact" in line and "val_bpb:" in line:
                try: bpb = float(line.split("val_bpb:")[-1].strip())
                except: pass
        p.wait()
    print(f"\n=== RESULT: seed={seed} bpb={bpb} ===", flush=True)
    with open(log_path) as f:
        log = f.read()
    # Save result to Modal Volume so it survives detach
    result_path = f"/tmp/result_seed{seed}.json"
    import json as _json
    with open(result_path, "w") as rf:
        _json.dump({"seed": seed, "bpb": bpb}, rf)
    print(f"Result saved to {result_path}", flush=True)
    return {"seed": seed, "bpb": bpb, "config": {
        "skip_thresh": skip_thresh, "logistic_mix": logistic_mix,
        "apm": apm, "slot_batch": slot_batch,
    }, "log": log[-15000:]}

@app.local_entrypoint()
def main(seed: int = 42, skip_thresh: float = -1.0,
         logistic_mix: bool = False, apm: bool = False,
         slot_batch: int = 128, slot_steps: int = 24,
         ngram_buckets: int = 4194304,
         legal: bool = False, legal_alpha: float = 0.10, legal_order: int = 4,
         slot_optimizer: str = "adamw", slot_phi_rank: bool = False,
         ngram_enabled: bool = True):
    feats = []
    if not ngram_enabled: feats.append("NO_NGRAM")
    if slot_optimizer != "adamw": feats.append(f"opt={slot_optimizer}")
    if slot_phi_rank: feats.append("phi_rank")
    if legal: feats.append(f"LEGAL@{legal_alpha}(ord={legal_order})")
    if skip_thresh > 0: feats.append(f"skip@{skip_thresh}")
    if logistic_mix: feats.append("logistic")
    if apm: feats.append("apm")
    if slot_steps != 24: feats.append(f"steps={slot_steps}")
    if ngram_buckets != 4194304: feats.append(f"bkt={ngram_buckets//1048576}M")
    feat_str = f" [{','.join(feats)}]" if feats else " [baseline]"
    print(f"Running v7{feat_str} seed {seed} on Modal...")
    r = run_seed.remote(seed, skip_thresh=skip_thresh, logistic_mix=logistic_mix,
                        apm=apm, slot_batch=slot_batch, slot_steps=slot_steps,
                        ngram_buckets=ngram_buckets,
                        legal=legal, legal_alpha=legal_alpha, legal_order=legal_order,
                        slot_optimizer=slot_optimizer, slot_phi_rank=slot_phi_rank,
                        ngram_enabled=ngram_enabled)
    print(f"\nSeed {seed}: BPB={r['bpb']}")
    print(f"Config: {r['config']}")
    print(f"\n{r['log']}")
