"""Single-node Modal launcher for the public Parameter Golf baseline stack."""

import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

import modal


ROOT = Path(__file__).resolve().parent


def dataset_dir_for_variant(name: str) -> str:
    if name == "byte260":
        return "fineweb10B_byte260"
    if name.startswith("sp") and name[2:].isdigit():
        return f"fineweb10B_{name}"
    raise ValueError(f"unsupported variant {name!r}; expected byte260 or sp<VOCAB_SIZE>")


APP_NAME = "parameter-golf-8xh100"
GPU_SPEC = os.environ.get("PGOLF_MODAL_GPU", "H100!:8")
RUNS_VOLUME_NAME = os.environ.get("PGOLF_MODAL_RUNS_VOLUME", "parameter-golf-runs")
DATASETS_VOLUME_NAME = os.environ.get("PGOLF_MODAL_DATASETS_VOLUME", "parameter-golf-datasets")
TOKENIZERS_VOLUME_NAME = os.environ.get("PGOLF_MODAL_TOKENIZERS_VOLUME", "parameter-golf-tokenizers")
HF_CACHE_VOLUME_NAME = os.environ.get("PGOLF_MODAL_HF_CACHE_VOLUME", "parameter-golf-hf-cache")
TRAIN_TIMEOUT_SECONDS = int(os.environ.get("PGOLF_MODAL_TIMEOUT_SECONDS", "14400"))
DOWNLOAD_TIMEOUT_SECONDS = int(os.environ.get("PGOLF_MODAL_DOWNLOAD_TIMEOUT_SECONDS", "7200"))
REMOTE_ROOT = Path("/root/parameter-golf")
RUNS_DIR = Path("/cache/runs")
DATASETS_DIR = REMOTE_ROOT / "data" / "datasets"
TOKENIZERS_DIR = REMOTE_ROOT / "data" / "tokenizers"
HF_CACHE_DIR = Path("/root/.cache/huggingface")
CUDA_OS_TAG = "ubuntu22.04"

COMMON_TRAIN_PACKAGES = (
    "numpy",
    "tqdm",
    "huggingface-hub",
    "datasets",
    "tiktoken",
    "sentencepiece",
    "typing-extensions",
    "setuptools",
)
DEFAULT_TRAIN_SHARDS = 80


def extract_cli_option(argv: list[str], name: str) -> str:
    flag = f"--{name}"
    prefix = f"{flag}="
    for index, arg in enumerate(argv):
        if arg == flag and index + 1 < len(argv):
            return argv[index + 1]
        if arg.startswith(prefix):
            return arg[len(prefix) :]
    return ""


def resolve_local_script_path(script: str) -> Path:
    script_path = Path(script).expanduser()
    if script_path.is_absolute():
        return script_path.resolve()
    return (ROOT / script_path).resolve()


def resolve_remote_script_path(script: str, local_script_path: Path) -> Path:
    script_path = Path(script)
    if script_path.is_absolute():
        try:
            relative = local_script_path.relative_to(ROOT)
        except ValueError:
            relative = Path(local_script_path.name)
    else:
        relative = script_path
    return REMOTE_ROOT / relative


REQUESTED_SCRIPT = os.environ.get("PGOLF_SCRIPT") or extract_cli_option(sys.argv, "script") or "train_gpt.py"
LOCAL_SCRIPT_PATH = resolve_local_script_path(REQUESTED_SCRIPT)
REMOTE_TRAIN_SCRIPT = resolve_remote_script_path(REQUESTED_SCRIPT, LOCAL_SCRIPT_PATH)

FORWARDED_ENV_KEYS = (
    "RUN_ID",
    "SEED",
    "DATA_PATH",
    "TOKENIZER_PATH",
    "VOCAB_SIZE",
    "ITERATIONS",
    "VAL_LOSS_EVERY",
    "VAL_BATCH_SIZE",
    "TRAIN_LOG_EVERY",
    "TRAIN_BATCH_TOKENS",
    "TRAIN_SEQ_LEN",
    "WARMDOWN_ITERS",
    "WARMUP_STEPS",
    "MAX_WALLCLOCK_SECONDS",
    "NUM_LAYERS",
    "MODEL_DIM",
    "NUM_HEADS",
    "NUM_KV_HEADS",
    "MLP_MULT",
    "TIE_EMBEDDINGS",
    "ROPE_BASE",
    "LOGIT_SOFTCAP",
    "QK_GAIN_INIT",
    "EMBED_LR",
    "HEAD_LR",
    "TIED_EMBED_LR",
    "TIED_EMBED_INIT_STD",
    "MATRIX_LR",
    "SCALAR_LR",
    "MUON_MOMENTUM",
    "MUON_BACKEND_STEPS",
    "MUON_MOMENTUM_WARMUP_START",
    "MUON_MOMENTUM_WARMUP_STEPS",
    "BETA1",
    "BETA2",
    "ADAM_EPS",
    "GRAD_CLIP_NORM",
    "MATCHED_FINEWEB_REPO_ID",
    "MATCHED_FINEWEB_REMOTE_ROOT_PREFIX",
    "HF_TOKEN",
    "HF_HUB_ENABLE_HF_TRANSFER",
)


def tokenizer_path_for_variant(variant: str) -> str:
    if variant == "byte260":
        raise ValueError("byte260 is not wired into this launcher yet")
    if not (variant.startswith("sp") and variant[2:].isdigit()):
        raise ValueError(f"Unsupported variant {variant!r}")
    return str(REMOTE_ROOT / "data" / "tokenizers" / f"fineweb_{variant[2:]}_bpe.model")


def data_path_for_variant(variant: str) -> str:
    return str(REMOTE_ROOT / "data" / "datasets" / dataset_dir_for_variant(variant))


image_base = modal.Image.debian_slim(python_version="3.12").run_commands("python -m pip install --upgrade pip")


def build_cuda_train_image(*, cuda_version: str, torch_version: str, index_url: str) -> modal.Image:
    cuda_tag = f"{cuda_version}-devel-{CUDA_OS_TAG}"
    return (
        modal.Image.from_registry(f"nvidia/cuda:{cuda_tag}", add_python="3.12")
        .apt_install(
            "git",
            "libibverbs-dev",
            "libibverbs1",
            "libhwloc15",
            "libnl-route-3-200",
        )
        .run_commands("python -m pip install --upgrade pip")
        .pip_install(
            f"torch=={torch_version}",
            index_url=index_url,
        )
        .pip_install(*COMMON_TRAIN_PACKAGES)
        # Mount only the requested training script; datasets, tokenizers, and outputs come from Modal Volumes.
        .add_local_file(str(LOCAL_SCRIPT_PATH), remote_path=str(REMOTE_TRAIN_SCRIPT))
    )

download_image = image_base.pip_install("huggingface-hub").add_local_file(
    str(ROOT / "data" / "cached_challenge_fineweb.py"),
    remote_path=str(REMOTE_ROOT / "data" / "cached_challenge_fineweb.py"),
)

train_image = build_cuda_train_image(
    cuda_version="12.8.0",
    torch_version="2.9.1",
    index_url="https://download.pytorch.org/whl/cu128",
)

app = modal.App(APP_NAME)
runs_volume = modal.Volume.from_name(RUNS_VOLUME_NAME, create_if_missing=True)
datasets_volume = modal.Volume.from_name(DATASETS_VOLUME_NAME, create_if_missing=True)
tokenizers_volume = modal.Volume.from_name(TOKENIZERS_VOLUME_NAME, create_if_missing=True)
hf_cache_volume = modal.Volume.from_name(HF_CACHE_VOLUME_NAME, create_if_missing=True)

DATA_VOLUMES = {
    str(DATASETS_DIR): datasets_volume,
    str(TOKENIZERS_DIR): tokenizers_volume,
    str(HF_CACHE_DIR): hf_cache_volume,
}

TRAIN_VOLUMES = {
    str(RUNS_DIR): runs_volume,
    **DATA_VOLUMES,
}


def prepare_mount_layout() -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    TOKENIZERS_DIR.mkdir(parents=True, exist_ok=True)
    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def prune_extra_train_shards(variant: str, train_shards: int) -> None:
    dataset_dir = DATASETS_DIR / dataset_dir_for_variant(variant)
    if not dataset_dir.is_dir():
        return
    prefix = "fineweb_train_"
    for train_file in dataset_dir.glob(f"{prefix}*.bin"):
        suffix = train_file.stem.removeprefix(prefix)
        if not suffix.isdigit():
            continue
        if int(suffix) >= train_shards:
            train_file.unlink()


def persist_run_outputs(run_id: str) -> Path:
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    logs_dir = REMOTE_ROOT / "logs"
    if logs_dir.is_dir():
        dst_logs = run_dir / "logs"
        dst_logs.mkdir(parents=True, exist_ok=True)
        run_log = logs_dir / f"{run_id}.txt"
        if run_log.is_file():
            shutil.copy2(run_log, dst_logs / run_log.name)

    for artifact_name in ("final_model.pt", "final_model.int8.ptz"):
        artifact = REMOTE_ROOT / artifact_name
        if artifact.is_file():
            shutil.copy2(artifact, run_dir / artifact_name)
    return run_dir


def resolve_remote_script(script: str) -> str:
    script_path = Path(script)
    if script_path.is_absolute():
        return str(script_path)
    resolved_local = resolve_local_script_path(script)
    if resolved_local == LOCAL_SCRIPT_PATH:
        return str(REMOTE_TRAIN_SCRIPT)
    raise ValueError(
        f"Script {script!r} was not mounted into the Modal image. "
        f"Rerun with --script {script!r} so the launcher can mount it."
    )


def build_remote_command(command: str) -> list[str]:
    if command:
        print(f"[modal] Running shell command on {GPU_SPEC}: {command}")
        return ["bash", "-lc", command]
    raise ValueError("command must be non-empty when using build_remote_command")


def launch_torchrun(script: str, nproc_per_node: int) -> None:
    # Invoke torchrun in-process so the Modal function owns the training lifecycle directly.
    from torch.distributed.run import parse_args, run

    resolved_script = resolve_remote_script(script)
    args = [
        f"--nproc-per-node={nproc_per_node}",
        resolved_script,
    ]
    print(f"[modal] Launching torchrun --standalone {' '.join(args)} on {GPU_SPEC}")
    run(parse_args(["--standalone", *args]))


@app.function(
    image=download_image,
    timeout=DOWNLOAD_TIMEOUT_SECONDS,
    cpu=4,
    volumes=DATA_VOLUMES,
)
def ensure_dataset(variant: str, train_shards: int) -> dict[str, str | int]:
    prepare_mount_layout()
    prune_extra_train_shards(variant, train_shards)
    cmd = [
        sys.executable,
        str(REMOTE_ROOT / "data" / "cached_challenge_fineweb.py"),
        "--variant",
        variant,
        "--train-shards",
        str(train_shards),
    ]
    subprocess.run(cmd, cwd=REMOTE_ROOT, check=True)
    datasets_volume.commit()
    tokenizers_volume.commit()
    hf_cache_volume.commit()
    return {
        "variant": variant,
        "train_shards": train_shards,
        "data_path": data_path_for_variant(variant),
        "tokenizer_path": tokenizer_path_for_variant(variant),
    }


def _run_job_impl(
    run_id: str,
    script: str,
    nproc_per_node: int,
    command: str,
    env_overrides: dict[str, str],
) -> dict[str, str | int]:
    runs_volume.reload()
    datasets_volume.reload()
    tokenizers_volume.reload()
    hf_cache_volume.reload()
    prepare_mount_layout()

    env = os.environ.copy()
    forwarded_env = dict(env_overrides)
    env.update(forwarded_env)
    env["RUN_ID"] = run_id

    print(f"[modal] Repo: {REMOTE_ROOT}")
    print(f"[modal] Run dir: {RUNS_DIR / run_id}")
    print(f"[modal] Forwarded env keys: {', '.join(sorted(forwarded_env)) or '(none)'}")
    started_at = time.time()
    os.environ.update(env)
    os.chdir(REMOTE_ROOT)
    if command:
        result = subprocess.run(build_remote_command(command), env=env, check=False)
    else:
        try:
            launch_torchrun(script, nproc_per_node)
            result = subprocess.CompletedProcess(args=["torchrun", script], returncode=0)
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else 1
            result = subprocess.CompletedProcess(args=["torchrun", script], returncode=code)
        except Exception as exc:
            raise RuntimeError(f"torchrun failed: {exc}") from None
    elapsed = time.time() - started_at
    print(f"[modal] Exit code: {result.returncode}")
    print(f"[modal] Elapsed seconds: {elapsed:.2f}")

    run_dir = persist_run_outputs(run_id)
    runs_volume.commit()
    if result.returncode != 0:
        raise RuntimeError(f"Remote command failed with exit code {result.returncode}")
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "exit_code": result.returncode,
        "elapsed_seconds": int(elapsed),
    }


@app.function(
    image=train_image,
    gpu=GPU_SPEC,
    timeout=TRAIN_TIMEOUT_SECONDS,
    cpu=16,
    volumes=TRAIN_VOLUMES,
)
def run_job(
    run_id: str,
    script: str,
    nproc_per_node: int,
    command: str,
    env_overrides: dict[str, str],
) -> dict[str, str | int]:
    return _run_job_impl(run_id, script, nproc_per_node, command, env_overrides)


def parse_extra_env(extra_env: str) -> list[str]:
    if not extra_env.strip():
        return []
    return [chunk.strip() for chunk in extra_env.split(",") if chunk.strip()]


@app.local_entrypoint()
def main(
    variant: str = os.environ.get("PGOLF_VARIANT", "sp1024"),
    train_shards: int = int(os.environ.get("PGOLF_TRAIN_SHARDS", str(DEFAULT_TRAIN_SHARDS))),
    script: str = os.environ.get("PGOLF_SCRIPT", "train_gpt.py"),
    run_id: str = "",
    nproc_per_node: int = int(os.environ.get("PGOLF_NPROC_PER_NODE", "8")),
    skip_dataset: bool = False,
    command: str = "",
    extra_env: str = "",
) -> None:
    resolved_run_id = run_id or os.environ.get("RUN_ID", f"modal-{uuid.uuid4().hex[:8]}")
    resolved_script = resolve_remote_script(script)
    env = {key: os.environ[key] for key in FORWARDED_ENV_KEYS if key in os.environ}
    for item in parse_extra_env(extra_env):
        if "=" not in item:
            raise ValueError(f"--extra-env expects KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        env[key] = value
    env.setdefault("RUN_ID", resolved_run_id)
    env.setdefault("DATA_PATH", data_path_for_variant(variant))
    env.setdefault("TOKENIZER_PATH", tokenizer_path_for_variant(variant))
    if variant.startswith("sp") and variant[2:].isdigit():
        env.setdefault("VOCAB_SIZE", variant[2:])

    if not skip_dataset:
        dataset_info = ensure_dataset.remote(variant, train_shards)
        print(f"[local] Dataset ready: {dataset_info}")

    result = run_job.remote(
        resolved_run_id,
        resolved_script,
        nproc_per_node,
        command.strip(),
        env,
    )
    print(f"[local] Run finished: {result}")
