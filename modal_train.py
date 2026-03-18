# modal launcher for parameter-golf training.
#
# usage:
#     # single h100 smoke test
#     modal run modal_train.py
#
#     # 8xh100 full run
#     modal run modal_train.py --gpu-count 8
#
#     # custom env vars
#     modal run modal_train.py --gpu-count 8 --env "ITERATIONS=5000" --env "VAL_LOSS_EVERY=200"

import modal

app = modal.App("parameter-golf")

# pre-built image with all dependencies + data cached
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy",
        "tqdm",
        "torch==2.10",
        "huggingface-hub",
        "setuptools",
        "typing-extensions==4.15.0",
        "datasets",
        "tiktoken",
        "sentencepiece",
    )
    .apt_install("git")
    .run_commands(
        "git clone https://github.com/openai/parameter-golf.git /opt/parameter-golf",
        "cd /opt/parameter-golf && python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80",
    )
)


@app.function(
    image=image,
    gpu="H100",
    timeout=1200,
)
def train(env_overrides: dict[str, str] | None = None):
    """single h100 training"""
    import os
    import subprocess

    os.chdir("/opt/parameter-golf")

    env = os.environ.copy()
    env.update({
        "DATA_PATH": "./data/datasets/fineweb10B_sp1024",
        "TOKENIZER_PATH": "./data/tokenizers/fineweb_1024_bpe.model",
        "VOCAB_SIZE": "1024",
        "RUN_ID": "modal_baseline",
    })
    if env_overrides:
        env.update(env_overrides)

    result = subprocess.run(
        ["torchrun", "--standalone", "--nproc_per_node=1", "train_gpt.py"],
        env=env,
        capture_output=False,
    )
    return result.returncode


@app.function(
    image=image,
    gpu="H100:8",
    timeout=1200,
)
def train_8gpu(env_overrides: dict[str, str] | None = None):
    """8xh100 training (leaderboard config)"""
    import os
    import subprocess

    os.chdir("/opt/parameter-golf")

    env = os.environ.copy()
    env.update({
        "DATA_PATH": "./data/datasets/fineweb10B_sp1024",
        "TOKENIZER_PATH": "./data/tokenizers/fineweb_1024_bpe.model",
        "VOCAB_SIZE": "1024",
        "RUN_ID": "modal_8gpu",
    })
    if env_overrides:
        env.update(env_overrides)

    result = subprocess.run(
        ["torchrun", "--standalone", "--nproc_per_node=8", "train_gpt.py"],
        env=env,
        capture_output=False,
    )
    return result.returncode


@app.local_entrypoint()
def main(
    gpu_count: int = 1,
    env: str = "",
):
    env_overrides = {}
    if env:
        for e in env.split(","):
            k, v = e.split("=", 1)
            env_overrides[k] = v

    if gpu_count == 8:
        print("launching 8xh100 training...")
        rc = train_8gpu.remote(env_overrides or None)
    else:
        print("launching 1xh100 training...")
        rc = train.remote(env_overrides or None)

    print(f"training finished with exit code: {rc}")
