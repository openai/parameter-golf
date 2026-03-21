"""
Modal script to run parameter-golf training on GPU.

Run from anywhere — paths are relative to this file's location.

Usage:
    modal run records/track_10min_16mb/2026-03-21_Standard10L_Int6_BigramHash/modal_run.py
    modal run records/.../modal_run.py --minutes 10 --use-h100
    modal run records/.../modal_run.py --minutes 10 --use-8xh100
    modal run records/.../modal_run.py --baseline

Cost estimates (Modal pricing):
    A100-80GB 3 min  ~ $0.14
    A100-80GB 10 min ~ $0.47
    H100      10 min ~ $0.66
    8xH100    10 min ~ $5.28
"""

from pathlib import Path
import modal

app = modal.App("parameter-golf")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "numpy",
        "tqdm",
        "torch==2.10",
        "huggingface-hub",
        "kernels",
        "setuptools",
        "typing-extensions==4.15.0",
        "datasets",
        "tiktoken",
        "sentencepiece",
    )
    .run_commands(
        "git clone https://github.com/openai/parameter-golf.git /root/parameter-golf",
        "cd /root/parameter-golf && python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10",
    )
)

vol = modal.Volume.from_name("pgolf-results", create_if_missing=True)

# Path is resolved relative to this file so modal run works from any directory.
_here = Path(__file__).parent
image = image.add_local_file(
    local_path=str(_here / "train_gpt.py"),
    remote_path="/root/parameter-golf/our_train_gpt.py",
)


def _run_training(max_minutes: float, run_id: str, use_baseline: bool, num_gpus: int):
    """Shared training logic."""
    import subprocess, shutil, os

    workdir = "/root/parameter-golf"
    script = "train_gpt.py" if use_baseline else "our_train_gpt.py"

    env = {
        **os.environ,
        "RUN_ID": run_id,
        "DATA_PATH": f"{workdir}/data/datasets/fineweb10B_sp1024/",
        "TOKENIZER_PATH": f"{workdir}/data/tokenizers/fineweb_1024_bpe.model",
        "VOCAB_SIZE": "1024",
        "MAX_WALLCLOCK_SECONDS": str(max_minutes * 60),
        "TRAIN_LOG_EVERY": "50",
        "VAL_LOSS_EVERY": "0",
        "TRAIN_BATCH_TOKENS": "524288",
    }

    print(f"\n{'='*60}")
    print(f"Training: script={script} GPUs={num_gpus} max_time={max_minutes}min")
    print(f"run_id={run_id} baseline={use_baseline}")
    print(f"{'='*60}\n")

    result = subprocess.run(
        ["torchrun", "--standalone", f"--nproc_per_node={num_gpus}", script],
        cwd=workdir,
        env=env,
    )

    # Persist results to volume
    logs_dir = f"{workdir}/logs"
    if os.path.isdir(logs_dir):
        shutil.copytree(logs_dir, f"/results/{run_id}_logs", dirs_exist_ok=True)
    for fname in ["final_model.pt", "final_model.int8.ptz"]:
        src = os.path.join(workdir, fname)
        if os.path.isfile(src):
            shutil.copy2(src, f"/results/{run_id}_{fname}")

    print(f"\nExit code: {result.returncode}")
    return result.returncode


@app.function(image=image, gpu="A100-80GB", timeout=1800, volumes={"/results": vol})
def train_a100(max_minutes: float = 3.0, run_id: str = "ours_a100", use_baseline: bool = False):
    return _run_training(max_minutes, run_id, use_baseline, num_gpus=1)


@app.function(image=image, gpu="H100", timeout=1800, volumes={"/results": vol})
def train_h100(max_minutes: float = 10.0, run_id: str = "ours_h100", use_baseline: bool = False):
    return _run_training(max_minutes, run_id, use_baseline, num_gpus=1)


@app.function(image=image, gpu="H100:8", timeout=1800, volumes={"/results": vol})
def train_8xh100(max_minutes: float = 10.0, run_id: str = "ours_8xh100", use_baseline: bool = False):
    return _run_training(max_minutes, run_id, use_baseline, num_gpus=8)


@app.local_entrypoint()
def main(
    minutes: float = 3.0,
    use_h100: bool = False,
    use_8xh100: bool = False,
    baseline: bool = False,
    run_id: str = "",
):
    tag = "baseline" if baseline else "ours"

    if use_8xh100:
        rid = run_id or f"{tag}_8xh100"
        cost = minutes * 0.528
        print(f"🚀 Launching on 8xH100 for {minutes} min (est. ${cost:.2f})")
        exit_code = train_8xh100.remote(max_minutes=minutes, run_id=rid, use_baseline=baseline)
    elif use_h100:
        rid = run_id or f"{tag}_h100"
        cost = minutes * 0.066
        print(f"🚀 Launching on 1xH100 for {minutes} min (est. ${cost:.2f})")
        exit_code = train_h100.remote(max_minutes=minutes, run_id=rid, use_baseline=baseline)
    else:
        rid = run_id or f"{tag}_a100"
        cost = minutes * 0.047
        print(f"🚀 Launching on A100-80GB for {minutes} min (est. ${cost:.2f})")
        exit_code = train_a100.remote(max_minutes=minutes, run_id=rid, use_baseline=baseline)

    if exit_code == 0:
        print(f"\n✅ Training completed! Results saved to volume 'pgolf-results'")
    else:
        print(f"\n❌ Training failed (exit code {exit_code})")
