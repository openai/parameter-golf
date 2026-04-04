"""Run inline Gram NS training on 8xH100 via Modal.

Only overrides env vars that differ from code defaults.
Streams output line-by-line so we see training progress in real-time.
"""
import os
import modal
import subprocess
import sys

app = modal.App("parameter-golf-gram-ns")

data_vol = modal.Volume.from_name("parameter-golf-data", create_if_missing=True)

hf_secret = modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.6.3-devel-ubuntu22.04",
        add_python="3.12",
    )
    .apt_install("git")
    .pip_install(
        "torch==2.6.0",
        "numpy",
        "sentencepiece",
        "huggingface-hub",
        "datasets",
        "tqdm",
        extra_options="--extra-index-url https://download.pytorch.org/whl/cu126",
    )
    .pip_install("psutil", "packaging", "ninja")
    .pip_install(
        "flash-attn",
        extra_options="--no-build-isolation",
    )
)


@app.function(
    image=image,
    gpu="H100:8",
    timeout=1800,
    volumes={"/data": data_vol},
    secrets=[hf_secret],
)
def train():
    import os
    import shutil

    # Clone repo
    if os.path.exists("/workspace/parameter-golf"):
        shutil.rmtree("/workspace/parameter-golf")
    subprocess.run(
        ["git", "clone", "--branch", "shikhar",
         "https://github.com/Itssshikhar/parameter-golf.git",
         "/workspace/parameter-golf"],
        check=True,
    )
    os.chdir("/workspace/parameter-golf")

    # Dataset: use persistent volume
    dataset_vol = "/data/fineweb10B_sp1024"
    dataset_local = "./data/datasets/fineweb10B_sp1024"
    tokenizer_vol = "/data/tokenizers"
    tokenizer_local = "./data/tokenizers"

    if not os.path.exists(f"{dataset_vol}/fineweb_train_000000.bin"):
        print("Downloading dataset...")
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
        print("Dataset saved to volume.")
    else:
        print("Dataset found in volume.")
        os.makedirs("./data/datasets", exist_ok=True)
        os.makedirs("./data", exist_ok=True)
        if os.path.exists(dataset_local):
            shutil.rmtree(dataset_local)
        os.symlink(dataset_vol, dataset_local)
        if os.path.exists(tokenizer_local):
            shutil.rmtree(tokenizer_local)
        os.symlink(tokenizer_vol, tokenizer_local)

    # Only override what differs from code defaults
    env = {
        **os.environ,
        "RUN_ID": "gram_ns_inline",
        "BIGRAM_VOCAB_SIZE": "1536",  # code default is 2048, #1 submission uses 1536
    }

    # Stream output line-by-line
    print("=== TRAINING START ===", flush=True)
    proc = subprocess.Popen(
        ["torchrun", "--standalone", "--nproc_per_node=8", "train_gpt_gram_ns.py"],
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

    return "\n".join(lines)


@app.local_entrypoint()
def main():
    log = train.remote()
    with open("gram_ns_inline_run.log", "w") as f:
        f.write(log)
    print(f"\nLog saved to gram_ns_inline_run.log")
