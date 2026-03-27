import modal
import os

app = modal.App("parameter-golf")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch",
        "numpy",
        "tiktoken",
        "sentencepiece",
        "huggingface_hub",
        "zstandard",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
)

volume = modal.Volume.from_name("parameter-golf-data", create_if_missing=True)

@app.function(
    image=image,
    gpu="H100",
    timeout=900,
    volumes={"/data": volume},
)
def train():
    import subprocess
    import sys
    import os

    # Clone repo
    subprocess.run([
        "git", "clone", "--branch", "int4-mlp-11l-qat",
        "https://github.com/Bharath-970/parameter-golf.git",
        "/repo"
    ], check=True)
    # Pull latest changes from current branch
    subprocess.run(["git", "-C", "/repo", "pull"], check=True)

    os.chdir("/repo")

    # Download data (tokenizer + val shards + 5 train shards)
    subprocess.run([
        sys.executable, "data/cached_challenge_fineweb.py",
        "--variant", "sp1024", "--train-shards", "5"
    ], check=True, env={**os.environ, "HOME": "/root"})

    # Run training
    result = subprocess.run([
        "torchrun", "--standalone", "--nproc_per_node=1",
        "records/track_10min_16mb/2026-03-25_16L_XSAall_GPTQ_EMA_PartialRoPE_TTT/train_gpt.py"
    ], capture_output=False, env={**os.environ, "HOME": "/root"})

    return result.returncode


@app.local_entrypoint()
def main():
    returncode = train.remote()
    print(f"Training finished with return code: {returncode}")
