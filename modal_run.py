"""
Parameter Golf — Modal GPU runner
Runs train_gpt.py on a single H100 for quick dev iteration.
For final submission, use 8xH100 (change gpu="H100:8").
"""

import modal
import os

app = modal.App("parameter-golf")

# Build image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.10",
        "numpy",
        "sentencepiece",
        "huggingface-hub",
        "datasets",
        "tqdm",
        "zstandard",
        "setuptools",
        "typing-extensions==4.15.0",
    )
    .apt_install("git")
)

# Persistent volume for dataset (don't re-download every run)
vol = modal.Volume.from_name("parameter-golf-data", create_if_missing=True)

@app.function(
    image=image,
    gpu="H100",  # Single H100 for dev. Change to "H100:8" for submission.
    timeout=3600,  # 60 min (generous for data download + train + eval)
    volumes={"/data": vol},
)
def train(run_id: str = "modal_dev", wallclock: int = 600, nproc: int = 1, train_shards: int = 3):
    import subprocess
    import shutil
    import sys

    # Unbuffer stdout for live logs
    sys.stdout.reconfigure(line_buffering=True)

    # Clone our fork
    print("Cloning repo...", flush=True)
    subprocess.run(["git", "clone", "https://github.com/keshav55/parameter-golf.git", "/workspace"], check=True)
    os.chdir("/workspace")

    # Copy our modified train script
    shutil.copy("atris/experiments/v1_train_gpt.py", "train_gpt.py")

    # Download dataset (use volume for caching)
    data_dir = "/data/datasets/fineweb10B_sp1024"
    tok_dir = "/data/tokenizers"
    local_data = "./data/datasets/fineweb10B_sp1024"
    local_tok = "./data/tokenizers"

    if os.path.exists(f"{data_dir}/fineweb_val_000000.bin"):
        print("Dataset found in volume, symlinking...", flush=True)
        os.makedirs("./data/datasets", exist_ok=True)
        os.makedirs("./data/tokenizers", exist_ok=True)
        os.symlink(data_dir, local_data)
        for f in os.listdir(tok_dir):
            os.symlink(f"{tok_dir}/{f}", f"{local_tok}/{f}")
    else:
        print(f"Downloading dataset ({train_shards} train shards)...", flush=True)
        subprocess.run([
            "python3", "data/cached_challenge_fineweb.py",
            "--variant", "sp1024", "--train-shards", str(train_shards)
        ], check=True)
        # Cache to volume for next run
        print("Caching to volume...", flush=True)
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(tok_dir, exist_ok=True)
        for f in os.listdir(local_data):
            shutil.copy2(f"{local_data}/{f}", f"{data_dir}/{f}")
        for f in os.listdir(local_tok):
            shutil.copy2(f"{local_tok}/{f}", f"{tok_dir}/{f}")
        vol.commit()
        print("Dataset cached.", flush=True)

    # Run training
    env = os.environ.copy()
    env.update({
        "RUN_ID": run_id,
        "MAX_WALLCLOCK_SECONDS": str(wallclock),
        "VAL_LOSS_EVERY": "0",  # skip periodic val (save time)
        "TRAIN_LOG_EVERY": "50",
        "NCCL_IB_DISABLE": "1",
        # Dev-friendly: override heavy defaults for 1-GPU
        "MLP_MULT": "2",           # 2x not 3x (faster, fits easily)
        "TRAIN_SEQ_LEN": "1024",   # 1024 not 2048 (halves memory)
        "TRAIN_BATCH_TOKENS": "524288",  # smaller batch
        "EVAL_STRIDE": "0",       # disable sliding window (fast standard eval)
        "SWA_EVERY": "0",         # disable SWA (save time)
        "BIGRAM_BUCKETS": "0",    # disable BigramHash (save params)
        "SMEAR_GATE": "0",        # disable SmearGate
        "QAT_BITS": "0",          # disable QAT
        "PRUNE_PERCENT": "0",     # disable pruning
        "USE_ZSTD": "0",          # use zlib (no extra dep needed)
        "WARMUP_STEPS": "5",      # fewer warmup steps
    })

    cmd = [
        "torchrun", "--standalone", f"--nproc_per_node={nproc}",
        "train_gpt.py"
    ]

    print(f"\n{'='*80}")
    print(f"RUNNING: {' '.join(cmd)}")
    print(f"RUN_ID: {run_id}")
    print(f"WALLCLOCK: {wallclock}s, NPROC: {nproc}")
    print(f"{'='*80}\n")

    result = subprocess.run(cmd, env=env, capture_output=True, text=True)

    # Print output
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr[-2000:])

    # Extract key metrics
    output = result.stdout
    for line in output.split("\n"):
        if "final_int8_zlib_roundtrip" in line:
            print(f"\n{'='*80}")
            print(f"RESULT: {line}")
            print(f"{'='*80}")
        if "submission size" in line.lower() or "Total submission" in line:
            print(f"SIZE: {line}")

    return output


@app.local_entrypoint()
def main():
    # Lean dev run: 1xH100, baseline-like config, 5 min training
    # Goal: get a BPB score FAST, then iterate
    output = train.remote(
        run_id="atris_v8_lean",
        wallclock=300,  # 5 min training (enough for ~500 steps on 1 GPU)
        nproc=1,
        train_shards=3,
    )
    print("\n\nDone! Check the output above for val_bpb.")
