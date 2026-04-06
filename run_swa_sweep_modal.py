"""Sweep SWA full-attention layer count: 6, 7, 8 full layers.

Finding the sweet spot between SWA speed savings and sliding eval benefit.
"""
import os
import modal

app = modal.App("parameter-golf-swa-sweep2")

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
        "pip install flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291",
    )
    .add_local_file("train_gpt_swa.py", "/opt/train_gpt_swa.py")
)


@app.function(
    image=image,
    gpu="H100:8",
    timeout=10800,
    volumes={"/data": data_vol},
    secrets=[hf_secret],
)
def sweep():
    import subprocess
    import os
    import shutil

    os.makedirs("/workspace/parameter-golf", exist_ok=True)
    os.chdir("/workspace/parameter-golf")
    shutil.copy2("/opt/train_gpt_swa.py", "train_gpt_swa.py")

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

    configs = [
        {
            "name": "exp4_w256_full6",
            "desc": "Exp 4: window=256, 6 full attn layers (5 SWA), QAT@0.15",
            "env": {
                "RUN_ID": "exp4_w256_full6",
                "SWA_WINDOW_SIZE": "256",
                "SWA_FULL_ATTN_LAYERS": "6",
            },
        },
        {
            "name": "exp5_w256_full7",
            "desc": "Exp 5: window=256, 7 full attn layers (4 SWA), QAT@0.15",
            "env": {
                "RUN_ID": "exp5_w256_full7",
                "SWA_WINDOW_SIZE": "256",
                "SWA_FULL_ATTN_LAYERS": "7",
            },
        },
        {
            "name": "exp6_w256_full8",
            "desc": "Exp 6: window=256, 8 full attn layers (3 SWA), QAT@0.15",
            "env": {
                "RUN_ID": "exp6_w256_full8",
                "SWA_WINDOW_SIZE": "256",
                "SWA_FULL_ATTN_LAYERS": "8",
            },
        },
    ]

    common_env = {
        "BIGRAM_VOCAB_SIZE": "3072",
        "BIGRAM_DIM": "112",
        "WARMDOWN_ITERS": "4000",
        "SEED": "1337",
    }

    all_results = []

    for cfg in configs:
        print(f"\n{'='*70}", flush=True)
        print(f"  {cfg['desc']}", flush=True)
        print(f"{'='*70}", flush=True)

        env = {**os.environ, **common_env, **cfg["env"]}

        cache_dir = os.path.expanduser("~/.cache/torch_extensions")
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir, ignore_errors=True)

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
        print(f"=== {cfg['name']} DONE (exit code {proc.returncode}) ===", flush=True)
        all_results.append(f"=== {cfg['name']} ===\n" + "\n".join(lines))

    print(f"\n{'='*70}", flush=True)
    print("SWEEP 2 SUMMARY", flush=True)
    print(f"{'='*70}", flush=True)
    for result in all_results:
        for line in result.split("\n"):
            if any(k in line for k in ["step_avg", "post_ema", "final_int6_roundtrip ", "final_int6_sliding_window ", "=== exp"]):
                print(f"  {line.strip()}", flush=True)

    return "\n\n".join(all_results)


@app.local_entrypoint()
def main():
    log = sweep.remote()
    with open("swa_sweep.log", "w") as f:
        f.write(log)
    print(f"\nLog saved to swa_sweep.log")
