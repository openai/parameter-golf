"""Run sliding window attention training on 8xH100 via Modal."""
import os
import modal

app = modal.App("parameter-golf-swa")

data_vol = modal.Volume.from_name("parameter-golf-data", create_if_missing=True)
_hf_token = os.environ.get("HF_TOKEN", "")
if not _hf_token:
    _hf_token_path = os.path.expanduser("~/.cache/huggingface/token")
    if os.path.exists(_hf_token_path):
        _hf_token = open(_hf_token_path).read().strip()
hf_secret = modal.Secret.from_dict({"HF_TOKEN": _hf_token})

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04",
        add_python="3.12",
    )
    .apt_install("git")
    .pip_install(
        "torch==2.9.1",
        "numpy",
        "sentencepiece",
        "huggingface-hub",
        "datasets",
        "tqdm",
        extra_options="--extra-index-url https://download.pytorch.org/whl/cu128",
    )
    .pip_install("psutil", "packaging", "ninja", "wheel", "setuptools")
    .run_commands(
        # FA3 standalone for H100 — exact same setup as current #1 submission
        "pip install flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291",
    )
    .add_local_file("train_gpt_swa.py", "/opt/train_gpt_swa.py")
    .add_local_file("train_gpt_original_1.py", "/opt/train_gpt_original.py")
)


@app.function(
    image=image,
    gpu="H100:8",
    timeout=5400,
    volumes={"/data": data_vol},
    secrets=[hf_secret],
)
def train():
    import subprocess
    import os
    import shutil

    os.makedirs("/workspace/parameter-golf", exist_ok=True)
    os.chdir("/workspace/parameter-golf")

    # Copy scripts from image
    shutil.copy2("/opt/train_gpt_swa.py", "train_gpt_swa.py")
    shutil.copy2("/opt/train_gpt_original.py", "train_gpt_original.py")

    # Dataset
    dataset_vol = "/data/fineweb10B_sp1024"
    dataset_local = "./data/datasets/fineweb10B_sp1024"
    tokenizer_vol = "/data/tokenizers"
    tokenizer_local = "./data/tokenizers"

    if not os.path.exists(f"{dataset_vol}/fineweb_train_000000.bin"):
        print("Downloading dataset...", flush=True)
        subprocess.run(
            ["git", "clone", "https://github.com/Itssshikhar/parameter-golf.git", "/tmp/repo"],
            check=True,
        )
        shutil.copytree("/tmp/repo/data", "./data", dirs_exist_ok=True)
        subprocess.run(
            ["python3", "data/cached_challenge_fineweb.py", "--variant", "sp1024"],
            check=True,
        )
        os.makedirs(dataset_vol, exist_ok=True)
        os.makedirs(tokenizer_vol, exist_ok=True)
        for f in os.listdir(dataset_local):
            shutil.copy2(f"{dataset_local}/{f}", f"{dataset_vol}/{f}")
        for f in os.listdir(tokenizer_local):
            shutil.copy2(f"{tokenizer_local}/{f}", f"{tokenizer_vol}/{f}")
        data_vol.commit()
        print("Dataset saved to volume.", flush=True)
    else:
        print("Dataset found in volume.", flush=True)
        os.makedirs("./data/datasets", exist_ok=True)
        os.makedirs("./data", exist_ok=True)
        if os.path.exists(dataset_local):
            shutil.rmtree(dataset_local)
        os.symlink(dataset_vol, dataset_local)
        if os.path.exists(tokenizer_local):
            shutil.rmtree(tokenizer_local)
        os.symlink(tokenizer_vol, tokenizer_local)

    train_files = [f for f in os.listdir(dataset_local) if "train" in f]
    val_files = [f for f in os.listdir(dataset_local) if "val" in f]
    print(f"Dataset: {len(train_files)} train shards, {len(val_files)} val shards", flush=True)

    baseline_lines = ["(skipped — see previous run)"]

    env = {
        **os.environ,
        "RUN_ID": "seq4096_qk25_pko_sag",
        "TRAIN_SEQ_LEN": "4096",
        "EVAL_SEQ_LEN": "4096",
        "SWA_WINDOW_SIZE": "256",
        "SWA_FULL_ATTN_LAYERS": "5",
        # QK_GAIN_INIT defaults to 2.5 in code
        "PARTIAL_KEY_OFFSET": "1",
        "SPARSE_ATTN_GATE": "1",
        "BIGRAM_VOCAB_SIZE": "3072",
        "BIGRAM_DIM": "112",
        "WARMDOWN_ITERS": "4000",
        "SWA_WINDOW_SIZE": "256",
        "SWA_FULL_ATTN_LAYERS": "5",
    }

    print("=== SWA TRAINING START ===", flush=True)
    proc = subprocess.Popen(
        ["torchrun", "--standalone", "--nproc_per_node=8", "train_gpt_swa.py"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    lines = []
    for line in proc.stdout:
        line = line.rstrip()
        print(line, flush=True)
        lines.append(line)

    proc.wait()
    print(f"=== TRAINING DONE (exit code {proc.returncode}) ===", flush=True)
    return "=== BASELINE LOG ===\n" + "\n".join(baseline_lines) + "\n=== SWA LOG ===\n" + "\n".join(lines)


@app.local_entrypoint()
def main():
    log = train.remote()
    with open("swa_run.log", "w") as f:
        f.write(log)
    print(f"\nLog saved to swa_run.log")
