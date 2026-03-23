"""Parameter Golf — 8xH100 submission run on Modal."""
import modal
import os

app = modal.App("parameter-golf-8gpu")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.10", "numpy", "sentencepiece", "huggingface-hub", "datasets", "tqdm", "zstandard", "setuptools", "typing-extensions==4.15.0")
    .apt_install("git")
)

vol = modal.Volume.from_name("parameter-golf-data", create_if_missing=True)

@app.function(image=image, gpu="H100:8", timeout=7200, volumes={"/data": vol})
def train_8gpu(run_id: str = "atris_v8_8gpu", wallclock: int = 600):
    import subprocess, shutil, sys
    sys.stdout.reconfigure(line_buffering=True)

    print("Cloning repo...", flush=True)
    subprocess.run(["git", "clone", "https://github.com/keshav55/parameter-golf.git", "/workspace"], check=True)
    os.chdir("/workspace")
    shutil.copy("atris/experiments/v1_train_gpt.py", "train_gpt.py")

    # Symlink cached dataset
    data_dir, tok_dir = "/data/datasets/fineweb10B_sp1024", "/data/tokenizers"
    if os.path.exists(f"{data_dir}/fineweb_val_000000.bin"):
        print("Dataset found in volume", flush=True)
        os.makedirs("./data/datasets", exist_ok=True)
        os.makedirs("./data/tokenizers", exist_ok=True)
        os.symlink(data_dir, "./data/datasets/fineweb10B_sp1024")
        for f in os.listdir(tok_dir):
            os.symlink(f"{tok_dir}/{f}", f"./data/tokenizers/{f}")
    else:
        print("Downloading dataset (10 shards)...", flush=True)
        subprocess.run(["python3", "data/cached_challenge_fineweb.py", "--variant", "sp1024", "--train-shards", "10"], check=True)
        os.makedirs(data_dir, exist_ok=True); os.makedirs(tok_dir, exist_ok=True)
        for f in os.listdir("./data/datasets/fineweb10B_sp1024"): shutil.copy2(f"./data/datasets/fineweb10B_sp1024/{f}", f"{data_dir}/{f}")
        for f in os.listdir("./data/tokenizers"): shutil.copy2(f"./data/tokenizers/{f}", f"{tok_dir}/{f}")
        vol.commit()

    env = os.environ.copy()
    env.update({
        "RUN_ID": run_id, "MAX_WALLCLOCK_SECONDS": str(wallclock),
        "VAL_LOSS_EVERY": "0",   # skip periodic val (final eval uses sliding window, ~2 min)
        "TRAIN_LOG_EVERY": "50",
        "NCCL_IB_DISABLE": "1",
        "WARMUP_STEPS": "5",
        # EVAL_STRIDE defaults to 64 (sliding window) — only runs at final eval
        # USE_ZSTD defaults to 1 — zstandard is in pip_install
    })

    cmd = ["torchrun", "--standalone", "--nproc_per_node=8", "train_gpt.py"]
    print(f"\nRUNNING: {' '.join(cmd)}\nRUN_ID: {run_id}, WALLCLOCK: {wallclock}s, 8xH100\n", flush=True)

    # Stream stdout directly (no buffering) so Modal logs show live progress
    proc = subprocess.Popen(cmd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)
    output_lines = []
    for line in proc.stdout:
        line = line.rstrip()
        print(line, flush=True)
        output_lines.append(line)
    proc.wait()
    output = "\n".join(output_lines)
    for line in output_lines:
        if "final_int8_zlib_roundtrip" in line or "submission size" in line.lower():
            print(f"\n{'='*60}\n{line}\n{'='*60}", flush=True)
    return output

@app.local_entrypoint()
def main():
    output = train_8gpu.remote(run_id="atris_v8_submission", wallclock=600)
    print("\nDone!")
