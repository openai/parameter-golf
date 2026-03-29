from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import modal

APP_NAME = os.environ.get("MODAL_APP_NAME", "parameter-golf")
VOLUME_NAME = os.environ.get("MODAL_VOLUME_NAME", "parameter-golf-data")
HF_SECRET_NAME = os.environ.get("MODAL_HF_SECRET_NAME", "huggingface-token")
PROJECT_PATH = "/workspace/openai_parameter_golf"
DATA_ROOT = "/data"
EXPERIMENTS_ROOT = f"{DATA_ROOT}/experiments"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch",
        "numpy",
        "sentencepiece",
        "huggingface-hub",
        "datasets",
        "tqdm",
        "zstandard",
    )
    .add_local_dir(".", remote_path=PROJECT_PATH)
)

image_competitive = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch",
        "numpy",
        "sentencepiece",
        "huggingface-hub",
        "datasets",
        "tqdm",
        "zstandard",
    )
    .add_local_dir(".", remote_path=PROJECT_PATH)
)


def run_cmd(cmd: list[str], *, cwd: str, env: dict | None = None) -> str:
    print(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, check=False, env=env)
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="")
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(cmd)}")
    return result.stdout


def next_run_number() -> int:
    exp_dir = Path(EXPERIMENTS_ROOT)
    if not exp_dir.exists():
        return 1
    existing = [
        int(d.name.split("_", 1)[1])
        for d in exp_dir.iterdir()
        if d.is_dir() and d.name.startswith("run_") and d.name.split("_", 1)[1].isdigit()
    ]
    return max(existing, default=0) + 1


@app.function(
    image=image,
    timeout=2 * 60 * 60,
    volumes={DATA_ROOT: volume},
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
)
def download_data(variant: str = "sp1024", train_shards: int = 80) -> None:
    os.environ.update({
        "HF_HOME": f"{DATA_ROOT}/hf-home",
        "HUGGINGFACE_HUB_CACHE": f"{DATA_ROOT}/hf-home/hub",
    })

    dataset_dir = Path(DATA_ROOT) / "datasets"
    tokenizer_dir = Path(DATA_ROOT) / "tokenizers"
    dataset_name = f"fineweb10B_{variant}"
    target_dataset = dataset_dir / dataset_name
    target_tokenizer = tokenizer_dir / "fineweb_1024_bpe.model"

    if target_dataset.exists() and target_tokenizer.exists():
        n_train = len(list(target_dataset.glob("fineweb_train_*.bin")))
        n_val = len(list(target_dataset.glob("fineweb_val_*.bin")))
        if n_train >= train_shards and n_val > 0:
            print(f"Dataset already cached: {n_train} train shards, {n_val} val shards. Skipping download.")
            return

    dataset_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_dir.mkdir(parents=True, exist_ok=True)

    run_cmd(
        ["python", "data/cached_challenge_fineweb.py", "--variant", variant, "--train-shards", str(train_shards)],
        cwd=PROJECT_PATH,
    )

    src_data = Path(PROJECT_PATH) / "data"
    for dirname in ("datasets", "tokenizers"):
        src = src_data / dirname
        dst = Path(DATA_ROOT) / dirname
        if src.exists():
            shutil.copytree(src, dst, dirs_exist_ok=True)
    for filename in ("manifest.json",):
        src_file = src_data / filename
        if src_file.exists():
            shutil.copy2(src_file, Path(DATA_ROOT) / filename)

    volume.commit()
    print(f"Download complete. Data persisted in Volume '{VOLUME_NAME}'.")


@app.function(
    image=image,
    timeout=15 * 60,
    volumes={DATA_ROOT: volume},
)
def cleanup_runs() -> None:
    """Remove old experiment runs, logs, and temp files. Keeps datasets/tokenizers intact."""
    for name in ("experiments", "logs"):
        target = Path(DATA_ROOT) / name
        if target.is_dir():
            shutil.rmtree(target, ignore_errors=True)
            print(f"removed dir: {target}")

    for f in Path(PROJECT_PATH).glob("final_model*"):
        f.unlink(missing_ok=True)
        print(f"removed: {f}")

    volume.commit()
    print(f"Cleanup complete. Datasets/tokenizers preserved in Volume '{VOLUME_NAME}'.")


def _run_training(
    variant: str,
    iterations: int,
    max_wallclock_seconds: int,
    nproc_per_node: int,
    val_loss_every: int,
    train_log_every: int,
) -> dict:
    """Shared training logic used by both 1-GPU and 8-GPU functions."""
    run_num = next_run_number()
    run_dir = Path(EXPERIMENTS_ROOT) / f"run_{run_num}"
    run_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"run_{run_num}"

    dataset_name = f"fineweb10B_{variant}"
    data_path = f"{DATA_ROOT}/datasets/{dataset_name}"
    tokenizer_path = f"{DATA_ROOT}/tokenizers/fineweb_1024_bpe.model"

    env = os.environ.copy()
    env.update({
        "RUN_ID": run_id,
        "DATA_PATH": data_path,
        "TOKENIZER_PATH": tokenizer_path,
        "VOCAB_SIZE": "1024",
        "ITERATIONS": str(iterations),
        "MAX_WALLCLOCK_SECONDS": str(max_wallclock_seconds),
        "VAL_LOSS_EVERY": str(val_loss_every),
        "TRAIN_LOG_EVERY": str(train_log_every),
        "NCCL_IB_DISABLE": "1",
    })

    cmd = ["torchrun", "--standalone", f"--nproc_per_node={nproc_per_node}", "train_gpt.py"]
    print(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=PROJECT_PATH, text=True, capture_output=True, check=False, env=env)
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="")

    log_file = run_dir / "train_log.txt"
    log_file.write_text((result.stdout or "") + "\n" + (result.stderr or ""), encoding="utf-8")

    artifacts = {
        "final_model.pt": "full_weights.pt",
        "final_model.int8.ptz": "quantized_weights.int8.ptz",
    }
    for src_name, dst_name in artifacts.items():
        src = Path(PROJECT_PATH) / src_name
        if src.exists():
            shutil.copy2(src, run_dir / dst_name)
            print(f"saved: {run_dir / dst_name} ({src.stat().st_size} bytes)")

    log_src = Path(PROJECT_PATH) / "logs" / f"{run_id}.txt"
    if log_src.exists():
        shutil.copy2(log_src, run_dir / "training_log.txt")

    code_src = Path(PROJECT_PATH) / "train_gpt.py"
    if code_src.exists():
        shutil.copy2(code_src, run_dir / "train_gpt.py")

    summary = _parse_results(result.stdout or "")
    summary["run_id"] = run_id
    summary["run_num"] = run_num
    summary["variant"] = variant
    summary["iterations_requested"] = iterations
    summary["max_wallclock_seconds"] = max_wallclock_seconds
    summary["nproc_per_node"] = nproc_per_node
    summary["gpu_type"] = "H100"
    summary["gpu_count"] = nproc_per_node
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\n=== Run Summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    volume.commit()
    print(f"\nAll artifacts saved to Volume '{VOLUME_NAME}' at /data/experiments/{run_id}/")

    if result.returncode != 0:
        raise RuntimeError(f"Training failed with exit code {result.returncode}")

    return summary


@app.function(
    image=image,
    gpu="H100",
    timeout=2 * 60 * 60,
    volumes={DATA_ROOT: volume},
)
def train_1gpu(
    variant: str = "sp1024",
    iterations: int = 20000,
    max_wallclock_seconds: int = 600,
    val_loss_every: int = 1000,
    train_log_every: int = 200,
) -> dict:
    return _run_training(variant, iterations, max_wallclock_seconds, 1, val_loss_every, train_log_every)


@app.function(
    image=image,
    gpu="H100:8",
    timeout=2 * 60 * 60,
    volumes={DATA_ROOT: volume},
)
def train_8gpu(
    variant: str = "sp1024",
    iterations: int = 20000,
    max_wallclock_seconds: int = 600,
    val_loss_every: int = 200,
    train_log_every: int = 50,
) -> dict:
    return _run_training(variant, iterations, max_wallclock_seconds, 8, val_loss_every, train_log_every)


@app.function(
    image=image_competitive,
    gpu="H100:8",
    timeout=4 * 60 * 60,
    volumes={DATA_ROOT: volume},
)
def train_competitive(
    variant: str = "sp1024",
    iterations: int = 20000,
    max_wallclock_seconds: int = 600,
) -> dict:
    """Competitive training with the SOTA-derived train_gpt_competitive.py (all advanced techniques)."""
    run_num = next_run_number()
    run_dir = Path(EXPERIMENTS_ROOT) / f"run_{run_num}"
    run_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"run_{run_num}"

    dataset_name = f"fineweb10B_{variant}"
    data_path = f"{DATA_ROOT}/datasets/{dataset_name}"
    tokenizer_path = f"{DATA_ROOT}/tokenizers/fineweb_1024_bpe.model"

    env = os.environ.copy()
    env.update({
        "RUN_ID": run_id,
        "DATA_PATH": data_path,
        "TOKENIZER_PATH": tokenizer_path,
        "VOCAB_SIZE": "1024",
        "ITERATIONS": str(iterations),
        "MAX_WALLCLOCK_SECONDS": str(max_wallclock_seconds),
        "VAL_LOSS_EVERY": "200",
        "TRAIN_LOG_EVERY": "50",
        "NCCL_IB_DISABLE": "1",
        "TTT_ENABLED": "1",
        "TTT_LR": "0.002",
        "TTT_EPOCHS": "3",
        "TTT_CHUNK_TOKENS": "32768",
        "TTT_FREEZE_BLOCKS": "0",
        "VALUE_RESIDUAL": "0",
        "COOLDOWN_SHAPE": "linear",
    })

    cmd = ["torchrun", "--standalone", "--nproc_per_node=8", "train_gpt_competitive.py"]
    print(f"$ {' '.join(cmd)}", flush=True)
    log_file = run_dir / "train_log.txt"
    with open(log_file, "w") as logf:
        proc = subprocess.Popen(
            cmd, cwd=PROJECT_PATH, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        for line in proc.stdout:
            print(line, end="", flush=True)
            logf.write(line)
        proc.wait()

    class _R:
        returncode = proc.returncode
        stdout = log_file.read_text()
        stderr = ""
    result = _R()

    for pattern in ("final_model*", "*.ptz", "*.lzma"):
        for src in Path(PROJECT_PATH).glob(pattern):
            shutil.copy2(src, run_dir / src.name)
            print(f"saved: {run_dir / src.name} ({src.stat().st_size} bytes)")

    log_src = Path(PROJECT_PATH) / "logs" / f"{run_id}.txt"
    if log_src.exists():
        shutil.copy2(log_src, run_dir / "training_log.txt")

    code_src = Path(PROJECT_PATH) / "train_gpt_competitive.py"
    if code_src.exists():
        shutil.copy2(code_src, run_dir / "train_gpt.py")

    summary = _parse_results(result.stdout or "")
    summary["run_id"] = run_id
    summary["run_num"] = run_num
    summary["variant"] = variant
    summary["iterations_requested"] = iterations
    summary["max_wallclock_seconds"] = max_wallclock_seconds
    summary["nproc_per_node"] = 8
    summary["gpu_type"] = "H100"
    summary["gpu_count"] = 8
    summary["script"] = "train_gpt_competitive.py"
    summary["techniques"] = "LeakyReLU2,XSA4,EMA,PartialRoPE,LNScale,SmearGate,BigramHash,VE,VRL,Int6GPTQ,LZMA,1sqrtCooldown,TTT"
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\n=== Run Summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    volume.commit()
    print(f"\nAll artifacts saved to Volume '{VOLUME_NAME}' at /data/experiments/{run_id}/")

    if result.returncode != 0:
        raise RuntimeError(f"Training failed with exit code {result.returncode}")

    return summary


def _parse_results(stdout: str) -> dict:
    summary: dict = {}
    for line in stdout.splitlines():
        if "final_int8_zlib_roundtrip_exact" in line:
            for part in line.split():
                if part.startswith("val_loss:"):
                    summary["final_val_loss"] = part.split(":")[1]
                if part.startswith("val_bpb:"):
                    summary["final_val_bpb"] = part.split(":")[1]
        if "final_int8_zlib_roundtrip " in line and "exact" not in line:
            for part in line.split():
                if part.startswith("eval_time:"):
                    summary["eval_time_ms"] = part.split(":")[1]
        if "Total submission size int8+zlib:" in line:
            summary["artifact_bytes"] = line.split(":")[1].strip().split()[0]
        if "stopping_early:" in line:
            for part in line.split():
                if part.startswith("step:"):
                    summary["steps_completed"] = part.split(":")[1].split("/")[0]
        if "model_params:" in line:
            summary["model_params"] = line.split("model_params:")[1].strip().split()[0]
        if "peak memory allocated:" in line:
            summary["peak_memory"] = line.strip()
        if "train_batch_tokens:" in line:
            for part in line.split():
                if part.startswith("train_batch_tokens:"):
                    summary["train_batch_tokens"] = part.split(":")[1]
        if "world_size:" in line:
            for part in line.split():
                if part.startswith("world_size:"):
                    summary["world_size"] = part.split(":")[1]
                if part.startswith("grad_accum_steps:"):
                    summary["grad_accum_steps"] = part.split(":")[1]
    return summary


@app.function(
    image=image,
    timeout=15 * 60,
    volumes={DATA_ROOT: volume},
)
def list_experiments() -> list[dict]:
    exp_dir = Path(EXPERIMENTS_ROOT)
    if not exp_dir.exists():
        print("No experiments found.")
        return []
    results = []
    for d in sorted(exp_dir.iterdir()):
        if not d.is_dir() or not d.name.startswith("run_"):
            continue
        summary_file = d / "summary.json"
        if summary_file.exists():
            summary = json.loads(summary_file.read_text(encoding="utf-8"))
        else:
            summary = {"run_id": d.name, "status": "no summary"}
        files = [f.name for f in d.iterdir() if f.is_file()]
        summary["saved_files"] = files
        results.append(summary)
        print(f"{d.name}: val_bpb={summary.get('final_val_bpb', '?')}, files={files}")
    return results


@app.function(
    image=image,
    timeout=30 * 60,
    volumes={DATA_ROOT: volume},
)
def get_experiment_files(run_name: str) -> dict[str, bytes]:
    run_dir = Path(EXPERIMENTS_ROOT) / run_name
    if not run_dir.exists():
        raise FileNotFoundError(f"Experiment {run_name} not found in volume")
    files = {}
    for f in run_dir.iterdir():
        if f.is_file():
            files[f.name] = f.read_bytes()
            print(f"  reading: {f.name} ({f.stat().st_size} bytes)")
    return files


@app.function(
    image=image,
    timeout=15 * 60,
    volumes={DATA_ROOT: volume},
)
def get_tokenizer() -> dict[str, bytes]:
    tok_dir = Path(DATA_ROOT) / "tokenizers"
    if not tok_dir.exists():
        raise FileNotFoundError("No tokenizers directory in volume")
    files = {}
    for f in tok_dir.iterdir():
        if f.is_file():
            files[f.name] = f.read_bytes()
            print(f"  reading: {f.name} ({f.stat().st_size} bytes)")
    return files


@app.local_entrypoint()
def main(
    action: str = "train",
    variant: str = "sp1024",
    train_shards: int = 80,
    iterations: int = 20000,
    max_wallclock_seconds: int = 600,
    gpu_count: int = 8,
    run_name: str = "",
) -> None:
    if action == "cleanup":
        cleanup_runs.remote()
        return

    if action == "download":
        download_data.remote(variant=variant, train_shards=train_shards)
        return

    if action == "train":
        download_data.remote(variant=variant, train_shards=train_shards)
        if gpu_count == 8:
            summary = train_8gpu.remote(
                variant=variant,
                iterations=iterations,
                max_wallclock_seconds=max_wallclock_seconds,
            )
        else:
            summary = train_1gpu.remote(
                variant=variant,
                iterations=iterations,
                max_wallclock_seconds=max_wallclock_seconds,
            )
        _save_locally(summary.get("run_id", "run_unknown"), summary)
        return

    if action == "competitive":
        download_data.remote(variant=variant, train_shards=train_shards)
        summary = train_competitive.remote(
            variant=variant,
            iterations=iterations,
            max_wallclock_seconds=max_wallclock_seconds,
        )
        _save_locally(summary.get("run_id", "run_unknown"), summary)
        return

    if action == "list":
        list_experiments.remote()
        return

    if action == "pull":
        if not run_name:
            raise ValueError("--run-name required for pull action (e.g. --run-name run_1)")
        files = get_experiment_files.remote(run_name)
        _save_locally(run_name, files_dict=files)
        return

    if action == "pull-tokenizer":
        tok_files = get_tokenizer.remote()
        tok_dir = Path("data/tokenizers")
        tok_dir.mkdir(parents=True, exist_ok=True)
        for name, data in tok_files.items():
            (tok_dir / name).write_bytes(data)
            print(f"Saved {tok_dir / name} ({len(data)} bytes)")
        return

    raise ValueError(f"Unknown action: {action}. Use 'cleanup', 'download', 'train', 'list', 'pull', or 'pull-tokenizer'.")


def _save_locally(run_name: str, summary: dict | None = None, files_dict: dict[str, bytes] | None = None) -> None:
    local_dir = Path("experiments") / run_name
    local_dir.mkdir(parents=True, exist_ok=True)

    if summary:
        (local_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"Saved summary to {local_dir / 'summary.json'}")

    if files_dict:
        for name, data in files_dict.items():
            (local_dir / name).write_bytes(data)
            print(f"Saved {local_dir / name} ({len(data)} bytes)")

    print(f"\nLocal artifacts: {local_dir}/")
