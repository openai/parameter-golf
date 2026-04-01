"""
Modal launcher for parameter-golf training on 8x H100 SXM GPUs.

Usage:
    modal run run_modal.py
"""

import modal

app = modal.App("parameter-golf")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11"
    )
    .apt_install("git")
    .pip_install(
        "torch",
        "numpy",
        "tqdm",
        "huggingface-hub",
        "datasets",
        "sentencepiece",
        "tiktoken",
        "kernels",
        "setuptools",
        "typing-extensions==4.15.0",
    )
    .run_commands(
        "cd /root && git clone -b evan https://github.com/evnkm/parameter-golf.git",
    )
)

data_vol = modal.Volume.from_name("parameter-golf-data", create_if_missing=True)


@app.function(
    image=image,
    gpu="H100:8",
    timeout=30 * 60,
    volumes={"/data": data_vol},
)
def train():
    import os
    import shutil
    import subprocess
    import sys

    os.chdir("/root/parameter-golf")

    datasets_cache = "/data/datasets/fineweb10B_sp1024"
    tokenizers_cache = "/data/tokenizers"
    tokenizer_path = os.path.join(tokenizers_cache, "fineweb_1024_bpe.model")

    if not os.path.isdir(datasets_cache):
        print(">> Downloading FineWeb data (first run — will be cached in volume)...")
        subprocess.run(
            ["python3", "data/cached_challenge_fineweb.py", "--variant", "sp1024"],
            check=True,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        os.makedirs("/data/datasets", exist_ok=True)
        os.makedirs(tokenizers_cache, exist_ok=True)
        shutil.copytree("data/datasets/fineweb10B_sp1024", datasets_cache)
        for f in os.listdir("data/tokenizers"):
            shutil.copy2(f"data/tokenizers/{f}", tokenizers_cache)
        data_vol.commit()
        print(">> Data cached to persistent volume.")
    else:
        print(">> Using cached data from volume.")

    env = os.environ.copy()
    env["DATA_PATH"] = datasets_cache
    env["TOKENIZER_PATH"] = tokenizer_path

    print(">> Starting training on 8x H100...")
    result = subprocess.run(
        ["torchrun", "--standalone", "--nproc-per-node=8", "train_gpt_shared.py"],
        stdout=sys.stdout,
        stderr=sys.stderr,
        env=env,
    )
    sys.exit(result.returncode)


@app.local_entrypoint()
def main():
    train.remote()
