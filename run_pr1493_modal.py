"""Run PR1493-derived experiments on 8xH100 via Modal.

Each experiment enables exactly one env-gated change in train_pr1493.py unless
the "stacked" preset is selected. Keep one-at-a-time logs for attribution.
"""
import os
import re
import time
import modal


app = modal.App("parameter-golf-pr1493-priority")
data_vol = modal.Volume.from_name("parameter-golf-data", create_if_missing=True)

_hf_token = os.environ.get("HF_TOKEN", "")
if not _hf_token:
    _hf_token_path = os.path.expanduser("~/.cache/huggingface/token")
    if os.path.exists(_hf_token_path):
        with open(_hf_token_path, encoding="utf-8") as f:
            _hf_token = f.read().strip()
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
    .pip_install("psutil", "packaging", "ninja", "wheel", "setuptools", "brotli")
    .run_commands(
        "pip install flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291",
    )
    .add_local_file("train_pr1493.py", "/opt/train_pr1493.py")
    .add_local_file("data/cached_challenge_fineweb.py", "/opt/data/cached_challenge_fineweb.py")
)


EXPERIMENTS = {
    "baseline_ttt": {},
    "docshuffle": {"DOC_SHUFFLE_ENABLED": "1"},
    "wd": {"WD_SCHEDULE_ENABLED": "1"},
    "iha": {"IHA_ENABLED": "1"},
    "mtp": {"MTP_WEIGHT": "0.10", "MTP_STEPS": "1"},
    "evalloop3": {"EVAL_NUM_LOOPS": "3"},
    "stacked": {
        "DOC_SHUFFLE_ENABLED": "1",
        "WD_SCHEDULE_ENABLED": "1",
        "IHA_ENABLED": "1",
        "MTP_WEIGHT": "0.10",
        "MTP_STEPS": "1",
    },
}


def _summarize(log: str) -> dict[str, str]:
    summary = {}
    patterns = {
        "pre": r"pre-quantization post-ema val_loss:[^ ]+ val_bpb:([0-9.]+)",
        "quant": r"quantized val_loss:[^ ]+ val_bpb:([0-9.]+)",
        "sliding": r"quantized_sliding_window val_loss:[^ ]+ val_bpb:([0-9.]+)",
        "ttt": r"quantized_ttt val_loss:[^ ]+ val_bpb:([0-9.]+)",
        "size": r"Total submission size quantized\+brotli: ([0-9]+) bytes",
        "steps": r"stopping_early: wallclock_cap train_time: [^ ]+ step: ([0-9]+)/",
    }
    for key, pattern in patterns.items():
        matches = re.findall(pattern, log)
        if matches:
            summary[key] = matches[-1]
    return summary


@app.function(
    image=image,
    gpu="H100:8",
    timeout=7200,
    volumes={"/data": data_vol},
    secrets=[hf_secret],
)
def train_one(experiment: str, seed: int = 42) -> tuple[str, dict[str, str]]:
    import os
    import shutil
    import subprocess

    if experiment not in EXPERIMENTS:
        raise ValueError(f"unknown experiment {experiment!r}; choose one of {sorted(EXPERIMENTS)}")

    os.makedirs("/workspace/parameter-golf", exist_ok=True)
    os.chdir("/workspace/parameter-golf")
    os.makedirs("data", exist_ok=True)
    os.makedirs("data/datasets", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    shutil.copy2("/opt/train_pr1493.py", "train_pr1493.py")
    shutil.copytree("/opt/data", "data", dirs_exist_ok=True)

    dataset_vol = "/data/fineweb10B_sp8192"
    dataset_local = "./data/datasets/fineweb10B_sp8192"
    tokenizer_vol = "/data/tokenizers"
    tokenizer_local = "./data/tokenizers"

    needs_data = not os.path.exists(f"{dataset_vol}/fineweb_train_000000.bin")
    needs_tokenizer = not os.path.exists(f"{tokenizer_vol}/fineweb_8192_bpe.model")
    if needs_data or needs_tokenizer:
        print("Downloading SP8192 dataset into Modal volume...", flush=True)
        env_dl = {**os.environ, "MATCHED_FINEWEB_REPO_ID": "kevclark/parameter-golf"}
        subprocess.run(
            ["python3", "data/cached_challenge_fineweb.py", "--variant", "sp8192", "--train-shards", "128"],
            env=env_dl,
            check=True,
        )
        os.makedirs(dataset_vol, exist_ok=True)
        os.makedirs(tokenizer_vol, exist_ok=True)
        for name in os.listdir(dataset_local):
            shutil.copy2(os.path.join(dataset_local, name), os.path.join(dataset_vol, name))
        for name in os.listdir(tokenizer_local):
            shutil.copy2(os.path.join(tokenizer_local, name), os.path.join(tokenizer_vol, name))
        data_vol.commit()
    else:
        print("SP8192 dataset found in Modal volume.", flush=True)
        if os.path.exists(dataset_local):
            shutil.rmtree(dataset_local)
        os.symlink(dataset_vol, dataset_local)
        if os.path.exists(tokenizer_local):
            shutil.rmtree(tokenizer_local)
        os.symlink(tokenizer_vol, tokenizer_local)

    train_files = [name for name in os.listdir(dataset_local) if "train" in name]
    val_files = [name for name in os.listdir(dataset_local) if "val" in name]
    print(f"Dataset: {len(train_files)} train shards, {len(val_files)} val shards", flush=True)

    run_id = f"pr1493_{experiment}_s{seed}_{int(time.time())}"
    env = {
        **os.environ,
        "RUN_ID": run_id,
        "SEED": str(seed),
        "QK_GAIN_INIT": "5.25",
        "TTT_ENABLED": "1",
        "TTT_LR": "0.007",
        "TTT_EPOCHS": "5",
        **EXPERIMENTS[experiment],
    }

    print(f"=== TRAIN experiment={experiment} seed={seed} run_id={run_id} ===", flush=True)
    print(f"Enabled env: {EXPERIMENTS[experiment]}", flush=True)
    proc = subprocess.Popen(
        ["torchrun", "--standalone", "--nproc_per_node=8", "train_pr1493.py"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    lines = []
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip()
        print(line, flush=True)
        lines.append(line)
    proc.wait()

    log = "\n".join(lines)
    print(f"=== DONE experiment={experiment} exit={proc.returncode} ===", flush=True)
    if proc.returncode != 0:
        raise RuntimeError(f"training failed with exit code {proc.returncode}")
    return log, _summarize(log)


@app.local_entrypoint()
def main(experiment: str = "docshuffle", seed: int = 42):
    if experiment not in EXPERIMENTS:
        raise ValueError(f"unknown experiment {experiment!r}; choose one of {sorted(EXPERIMENTS)}")
    log, summary = train_one.remote(experiment=experiment, seed=seed)
    logfile = f"modal_pr1493_{experiment}_s{seed}.log"
    with open(logfile, "w", encoding="utf-8") as f:
        f.write(log)
    print(f"log -> {logfile}")
    print(f"summary -> {summary}")
