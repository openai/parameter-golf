# modal launcher for parameter-golf autoresearch.
#
# usage:
#     modal run modal_train.py
#
# custom env vars:
#     modal run modal_train.py --env "ITERATIONS=5000,VAL_LOSS_EVERY=200"

import modal

app = modal.App("parameter-golf")

# base image with deps + cached data + local train_gpt.py mounted
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
    # mount local train_gpt.py so agent edits get picked up each run
    .add_local_file("train_gpt.py", "/opt/parameter-golf/train_gpt.py")
)


@app.function(
    image=image,
    gpu="H100:8",
    timeout=1200,
)
def train(env_overrides: dict[str, str] | None = None):
    """8xh100 training"""
    import os
    import subprocess

    os.chdir("/opt/parameter-golf")

    env = os.environ.copy()
    env.update({
        "DATA_PATH": "./data/datasets/fineweb10B_sp1024",
        "TOKENIZER_PATH": "./data/tokenizers/fineweb_1024_bpe.model",
        "VOCAB_SIZE": "1024",
        "RUN_ID": "modal_run",
    })
    if env_overrides:
        env.update(env_overrides)

    result = subprocess.run(
        ["torchrun", "--standalone", "--nproc_per_node=8", "train_gpt.py"],
        env=env,
    )
    return result.returncode


@app.local_entrypoint()
def main(
    env: str = "",
):
    env_overrides = {}
    if env:
        for e in env.split(","):
            k, v = e.split("=", 1)
            env_overrides[k] = v

    print("launching 8xh100 training...")
    rc = train.remote(env_overrides or None)
    print(f"training finished with exit code: {rc}")
