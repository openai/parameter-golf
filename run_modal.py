"""
Modal deployment script for parameter-golf training on 8xH100 SXM.

Usage:
    Val-only run:     modal run run_modal.py --seed 1337
    Standard run:     modal run run_modal.py --mode standard --seed 7

This will:
    - Download the FineWeb dataset inside Modal (cached in a Volume)
    - Run torchrun with 8xH100 GPUs for 10 minutes
    - Download train.log and model artifacts to your local machine
"""

import modal

app = modal.App("parameter-golf")

data_vol = modal.Volume.from_name("parameter-golf-data", create_if_missing=True)
output_vol = modal.Volume.from_name("parameter-golf-output", create_if_missing=True)

TRAIN_SCRIPT = "records/track_10min_16mb/2026-03-19_CombinedOptimal/train_gpt.py"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.10",
        "numpy",
        "sentencepiece",
        "huggingface-hub",
        "tqdm",
        "setuptools",
        "typing-extensions==4.15.0",
        "zstandard",
        extra_options="--extra-index-url https://download.pytorch.org/whl/cu124",
    )
    .add_local_file("data/cached_challenge_fineweb.py", "/root/data/cached_challenge_fineweb.py")
    .add_local_file(TRAIN_SCRIPT, "/root/train_gpt.py")
)


@app.function(
    image=image,
    gpu="H100:8",
    timeout=45 * 60,
    volumes={
        "/data": data_vol,
        "/output": output_vol,
    },
)
def train(mode: str = "valonly", seed: int = 1337, tag: str = ""):
    import os
    import shutil
    import subprocess
    import sys

    os.chdir("/root")

    data_base = "/data/datasets/fineweb10B_sp1024"
    tokenizer_dir = "/data/tokenizers"
    val_shard = f"{data_base}/fineweb_val_000000.bin"
    is_standard = mode == "standard"
    train_shards = "80" if is_standard else "1"
    run_id_base = f"standard_optimal_v6_s{seed}" if is_standard else f"combined_optimal_v6_s{seed}"
    run_id = f"{run_id_base}_{tag}" if tag else run_id_base

    # ----------------------------------------------------------------
    # Step 1: Download data
    # ----------------------------------------------------------------
    need_download = not os.path.exists(val_shard)
    if is_standard:
        need_download = need_download or not os.path.exists(f"{data_base}/fineweb_train_000079.bin")

    if need_download:
        print(f"=== Downloading FineWeb data ({train_shards} train shards) ===", flush=True)
        subprocess.run(
            [
                sys.executable,
                "/root/data/cached_challenge_fineweb.py",
                "--variant", "sp1024",
                "--train-shards", train_shards,
            ],
            check=True,
            env={**os.environ, "PYTHONPATH": "/root"},
            cwd="/root",
        )
        local_ds = "/root/data/datasets/fineweb10B_sp1024"
        local_tok = "/root/data/tokenizers"
        os.makedirs(data_base, exist_ok=True)
        os.makedirs(tokenizer_dir, exist_ok=True)
        for f in os.listdir(local_ds):
            src = os.path.join(local_ds, f)
            dst = os.path.join(data_base, f)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
        for f in os.listdir(local_tok):
            src = os.path.join(local_tok, f)
            dst = os.path.join(tokenizer_dir, f)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
        data_vol.commit()
        print("=== Data cached to volume ===", flush=True)
    else:
        print("=== Data already cached ===", flush=True)

    # ----------------------------------------------------------------
    # Step 2: Determine data path
    # ----------------------------------------------------------------
    if is_standard:
        data_path = data_base
    else:
        valonly_dir = "/data/datasets/fineweb10B_sp1024_valonly"
        os.makedirs(valonly_dir, exist_ok=True)
        valonly_train = f"{valonly_dir}/fineweb_train_000000.bin"
        valonly_val = f"{valonly_dir}/fineweb_val_000000.bin"
        if not os.path.exists(valonly_train):
            shutil.copy2(val_shard, valonly_train)
        if not os.path.exists(valonly_val):
            shutil.copy2(val_shard, valonly_val)
        data_vol.commit()
        data_path = valonly_dir

    # ----------------------------------------------------------------
    # Step 3: Run training with torchrun on 8xH100
    # ----------------------------------------------------------------
    print(f"=== Starting training (mode={mode}, data={data_path}) ===", flush=True)

    env = {
        **os.environ,
        "RUN_ID": run_id,
        "SEED": str(seed),
        "DATA_PATH": data_path,
        "TOKENIZER_PATH": f"{tokenizer_dir}/fineweb_1024_bpe.model",
        "VOCAB_SIZE": "1024",
        "MAX_WALLCLOCK_SECONDS": "600",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }
    if is_standard:
        # Standard-track profile: match top-recipe dynamics (no STE, mixed int6/int8, stronger SWA).
        env.update(
            {
                "VAL_LOSS_EVERY": "500",
                "TRAIN_LOG_EVERY": "100",
                "SWA_ENABLED": "1",
                "SWA_EVERY": "50",
                "SWA_START_FRAC": "0.5",
                "STE_QAT_ENABLED": "0",
                "MIXED_QUANT_INT6_CATS": "mlp,attn",
                "FP16_PASSTHROUGH_PATTERNS": "tok_emb,blocks.8.attn.c_k",
                "MUON_WD": "0.04",
            }
        )
    else:
        # Val-only profile: keep aggressive memorization settings.
        env.update(
            {
                "VAL_LOSS_EVERY": "1000",
                "TRAIN_LOG_EVERY": "200",
                "SWA_ENABLED": "0",
                "STE_QAT_ENABLED": "1",
                "MIXED_QUANT_INT6_CATS": "mlp,attn,other",
                "FP16_PASSTHROUGH_PATTERNS": "tok_emb",
                "MUON_WD": "0.02",
            }
        )

    result = subprocess.run(
        [
            sys.executable, "-m", "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=8",
            "/root/train_gpt.py",
        ],
        env=env,
        cwd="/root",
        stdout=sys.stdout,
        stderr=sys.stderr,
    )

    print(f"\n=== Training finished with exit code {result.returncode} ===", flush=True)

    # ----------------------------------------------------------------
    # Step 4: Copy outputs to the output volume
    # ----------------------------------------------------------------
    output_base = f"/output/{run_id}"
    os.makedirs(output_base, exist_ok=True)

    for fname in ["final_model.pt", "final_model.int8.ptz"]:
        src = f"/root/{fname}"
        if os.path.exists(src):
            shutil.copy2(src, f"{output_base}/{fname}")
            print(f"  Saved {fname} ({os.path.getsize(src)} bytes)", flush=True)

    log_dir = "/root/logs"
    if os.path.isdir(log_dir):
        for fname in os.listdir(log_dir):
            src = os.path.join(log_dir, fname)
            shutil.copy2(src, f"{output_base}/{fname}")
            print(f"  Saved log: {fname}", flush=True)

    output_vol.commit()
    print(f"\n=== All outputs saved to volume 'parameter-golf-output' at /{run_id}/ ===")


@app.function(
    image=modal.Image.debian_slim(),
    volumes={"/output": output_vol},
)
def download_results(run_id: str = "combined_optimal_v6_s1337"):
    import os

    output_base = f"/output/{run_id}"
    if not os.path.isdir(output_base):
        print(f"No results found for run_id={run_id}.")
        return

    for fname in sorted(os.listdir(output_base)):
        fpath = os.path.join(output_base, fname)
        size = os.path.getsize(fpath)
        print(f"  {fname}: {size:,} bytes")

        if fname.endswith(".txt"):
            print(f"\n--- {fname} contents (last 30 lines) ---")
            with open(fpath) as f:
                lines = f.readlines()
            for line in lines[-30:]:
                print(line, end="")
            print(f"\n--- end {fname} ---\n")


@app.local_entrypoint()
def main(mode: str = "valonly", seed: int = 1337, tag: str = ""):
    run_id_base = f"standard_optimal_v6_s{seed}" if mode == "standard" else f"combined_optimal_v6_s{seed}"
    run_id = f"{run_id_base}_{tag}" if tag else run_id_base
    print(f"Launching training on Modal 8xH100 SXM (mode={mode}, seed={seed}, tag={tag})...")
    print("This will take ~15 minutes (10 min train + ~5 min eval + overhead)\n")
    train.remote(mode=mode, seed=seed, tag=tag)
    print("\n=== Fetching results ===\n")
    download_results.remote(run_id=run_id)
    print(f"\nTo download files locally:")
    print(f"  modal volume get parameter-golf-output {run_id}/{run_id}.txt ./train.log")
    print(f"  modal volume get parameter-golf-output {run_id}/final_model.int8.ptz ./final_model.int8.ptz")
