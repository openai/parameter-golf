from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import uuid
from itertools import product
from pathlib import Path
from typing import Optional

import modal
from huggingface_hub import hf_hub_download

APP_NAME = "parameter-golf"
PYTHON_VERSION = "3.12"
REPO_ROOT = Path(__file__).resolve().parent
REMOTE_PROJECT_ROOT = Path("/root/project")
VOLUME_ROOT = Path("/vol")
HF_CACHE_ROOT = VOLUME_ROOT / "hf-cache"
RUNS_ROOT = VOLUME_ROOT / "runs"
REPO_ID = os.environ.get("MATCHED_FINEWEB_REPO_ID", "willdepueoai/parameter-golf")
REMOTE_ROOT_PREFIX = os.environ.get("MATCHED_FINEWEB_REMOTE_ROOT_PREFIX", "datasets")

app = modal.App(APP_NAME)
data_volume = modal.Volume.from_name("parameter-golf-data", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .apt_install("git")
    .uv_pip_install(
        "datasets>=4.8.2",
        "huggingface-hub>=1.7.1",
        "numpy>=2.4.3",
        "sentencepiece>=0.2.1",
        "torch==2.10.0",
        "tqdm>=4.67.3",
        "triton>=3.4.0",
        "typing-extensions==4.15.0",
    )
    .env(
        {
            "HF_HOME": str(HF_CACHE_ROOT),
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    .add_local_dir(str(REPO_ROOT), remote_path=str(REMOTE_PROJECT_ROOT))
)


def _dataset_dir_for_variant(variant: str) -> str:
    if variant == "byte260":
        return "fineweb10B_byte260"
    if variant.startswith("sp") and variant[2:].isdigit():
        return f"fineweb10B_{variant}"
    raise ValueError(f"Unsupported variant: {variant}")


def _expected_vocab_size(variant: str) -> int:
    if variant == "byte260":
        return 260
    if variant.startswith("sp") and variant[2:].isdigit():
        return int(variant[2:])
    raise ValueError(f"Unsupported variant: {variant}")


def _local_path_for_remote(relative_path: str) -> Path:
    remote_path = Path(relative_path)
    if REMOTE_ROOT_PREFIX and remote_path.parts[:1] == (REMOTE_ROOT_PREFIX,):
        remote_path = remote_path.relative_to(REMOTE_ROOT_PREFIX)
    if remote_path.parts[:1] == ("datasets",):
        return VOLUME_ROOT / "datasets" / Path(*remote_path.parts[1:])
    if remote_path.parts[:1] == ("tokenizers",):
        return VOLUME_ROOT / "tokenizers" / Path(*remote_path.parts[1:])
    return VOLUME_ROOT / remote_path


def _download_to_volume(relative_path: str) -> Path:
    destination = _local_path_for_remote(relative_path)
    if destination.exists():
        return destination
    remote_path = Path(relative_path)
    cached_path = Path(
        hf_hub_download(
            repo_id=REPO_ID,
            filename=remote_path.name,
            subfolder=remote_path.parent.as_posix() if remote_path.parent != Path(".") else None,
            repo_type="dataset",
        )
    )
    cached_source = cached_path.resolve(strict=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(cached_source, destination)
    except OSError:
        shutil.copy2(cached_source, destination)
    return destination


def _load_manifest() -> dict:
    path = _local_path_for_remote(f"{REMOTE_ROOT_PREFIX}/manifest.json")
    if not path.exists():
        _download_to_volume(f"{REMOTE_ROOT_PREFIX}/manifest.json")
    return json.loads(path.read_text(encoding="utf-8"))


def _artifact_paths_for_tokenizer(tokenizer_entry: dict) -> list[str]:
    out = [str(tokenizer_entry[k]) for k in ("model_path", "vocab_path", "path") if tokenizer_entry.get(k)]
    if not out:
        raise ValueError(f"Tokenizer entry has no artifacts: {tokenizer_entry}")
    return out


def _resolve_variant_paths(variant: str) -> tuple[Path, Path]:
    manifest = _load_manifest()
    dataset_name = _dataset_dir_for_variant(variant)
    dataset_entry = next((x for x in manifest.get("datasets", []) if x.get("name") == dataset_name), None)
    if dataset_entry is None:
        raise ValueError(f"Dataset {dataset_name} not found in manifest")
    tokenizer_name = dataset_entry.get("tokenizer_name")
    tokenizer_entry = next((x for x in manifest.get("tokenizers", []) if x.get("name") == tokenizer_name), None)
    if tokenizer_entry is None:
        raise ValueError(f"Tokenizer {tokenizer_name} not found in manifest")
    tokenizer_artifact = _artifact_paths_for_tokenizer(tokenizer_entry)[0]
    return VOLUME_ROOT / "datasets" / dataset_name, _local_path_for_remote(tokenizer_artifact)


def _parse_env_overrides(env_overrides: str) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for item in (part.strip() for part in env_overrides.split(",")):
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Expected KEY=VALUE override, got {item!r}")
        key, value = item.split("=", 1)
        overrides[key.strip()] = value.strip()
    return overrides


def _parse_csv_ints(values: str) -> list[int]:
    return [int(part.strip()) for part in values.split(",") if part.strip()]


def _build_sweep_envs(base_env_overrides: str, num_layers_values: str, model_dim_values: str, num_heads_values: str, num_kv_heads_values: str, mlp_mult_values: str) -> list[dict[str, str]]:
    base = _parse_env_overrides(base_env_overrides)
    configs: list[dict[str, str]] = []
    for num_layers, model_dim, num_heads, num_kv_heads, mlp_mult in product(
        _parse_csv_ints(num_layers_values),
        _parse_csv_ints(model_dim_values),
        _parse_csv_ints(num_heads_values),
        _parse_csv_ints(num_kv_heads_values),
        _parse_csv_ints(mlp_mult_values),
    ):
        if num_heads % num_kv_heads != 0:
            continue
        config = dict(base)
        config.update(
            {
                "NUM_LAYERS": str(num_layers),
                "MODEL_DIM": str(model_dim),
                "NUM_HEADS": str(num_heads),
                "NUM_KV_HEADS": str(num_kv_heads),
                "MLP_MULT": str(mlp_mult),
            }
        )
        configs.append(config)
    if not configs:
        raise ValueError("Sweep grid is empty")
    return configs


def _env_dict_to_string(env: dict[str, str]) -> str:
    return ",".join(f"{k}={v}" for k, v in sorted(env.items()))


def _extract_run_metrics(log_path: Path) -> dict:
    text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""

    def _last(pattern: str, key: str):
        found = None
        for match in re.finditer(pattern, text):
            found = match
        return found.group(key) if found else None

    val_loss = _last(r"final_int8_zlib_roundtrip\s+val_loss:(?P<val_loss>[-+0-9.]+)\s+val_bpb:(?P<val_bpb>[-+0-9.]+)", "val_loss")
    val_bpb = _last(r"final_int8_zlib_roundtrip\s+val_loss:(?P<val_loss>[-+0-9.]+)\s+val_bpb:(?P<val_bpb>[-+0-9.]+)", "val_bpb")
    submission_size = _last(r"Total submission size int8\+zlib:\s+(?P<bytes>\d+)\s+bytes", "bytes")
    compressed_size = _last(r"Serialized model int8\+zlib:\s+(?P<bytes>\d+)\s+bytes", "bytes")
    params = _last(r"model_params:(?P<params>\d+)", "params")
    return {
        "val_loss": float(val_loss) if val_loss is not None else None,
        "val_bpb": float(val_bpb) if val_bpb is not None else None,
        "submission_size_bytes": int(submission_size) if submission_size is not None else None,
        "compressed_model_bytes": int(compressed_size) if compressed_size is not None else None,
        "model_params": int(params) if params is not None else None,
    }


@app.function(image=image, volumes={str(VOLUME_ROOT): data_volume}, timeout=60 * 60)
def prepare_data(variant: str = "sp1024", train_shards: int = 80) -> dict:
    manifest = _load_manifest()
    dataset_name = _dataset_dir_for_variant(variant)
    dataset_entry = next((x for x in manifest.get("datasets", []) if x.get("name") == dataset_name), None)
    if dataset_entry is None:
        raise ValueError(f"Dataset {dataset_name} not found in manifest")
    max_train_shards = int((dataset_entry.get("stats") or {}).get("files_train"))
    val_shards = int((dataset_entry.get("stats") or {}).get("files_val"))
    if train_shards > max_train_shards:
        raise ValueError(f"{variant} only has {max_train_shards} training shards; requested {train_shards}")
    tokenizer_name = dataset_entry.get("tokenizer_name")
    tokenizer_entry = next((x for x in manifest.get("tokenizers", []) if x.get("name") == tokenizer_name), None)
    if tokenizer_entry is None:
        raise ValueError(f"Tokenizer {tokenizer_name} not found in manifest")
    dataset_prefix = f"{REMOTE_ROOT_PREFIX}/datasets/{dataset_name}"
    for i in range(val_shards):
        _download_to_volume(f"{dataset_prefix}/fineweb_val_{i:06d}.bin")
    for i in range(train_shards):
        _download_to_volume(f"{dataset_prefix}/fineweb_train_{i:06d}.bin")
    for artifact_path in _artifact_paths_for_tokenizer(tokenizer_entry):
        _download_to_volume(f"{REMOTE_ROOT_PREFIX}/{artifact_path}")
    data_volume.commit()
    dataset_path, tokenizer_path = _resolve_variant_paths(variant)
    return {"variant": variant, "train_shards": train_shards, "dataset_path": str(dataset_path), "tokenizer_path": str(tokenizer_path), "vocab_size": _expected_vocab_size(variant)}


def _run_training(nproc_per_node: int, variant: str, run_id: Optional[str], iterations: int, max_wallclock_seconds: float, val_loss_every: int, train_log_every: int, env_overrides: str) -> dict:
    dataset_path, tokenizer_path = _resolve_variant_paths(variant)
    run_id = run_id or f"modal_{variant}_{uuid.uuid4().hex[:8]}"
    run_dir = RUNS_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.log"
    env = os.environ.copy()
    env.update(
        {
            "RUN_ID": run_id,
            "DATA_PATH": str(dataset_path),
            "TOKENIZER_PATH": str(tokenizer_path),
            "VOCAB_SIZE": str(_expected_vocab_size(variant)),
            "ITERATIONS": str(iterations),
            "MAX_WALLCLOCK_SECONDS": str(max_wallclock_seconds),
            "VAL_LOSS_EVERY": str(val_loss_every),
            "TRAIN_LOG_EVERY": str(train_log_every),
            "PYTHONPATH": str(REMOTE_PROJECT_ROOT),
        }
    )
    env.update(_parse_env_overrides(env_overrides))
    cmd = ["torchrun", "--standalone", f"--nproc_per_node={nproc_per_node}", str(REMOTE_PROJECT_ROOT / "train_gpt.py")]
    with log_path.open("w", encoding="utf-8") as f:
        process = subprocess.Popen(cmd, cwd=str(REMOTE_PROJECT_ROOT), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            f.write(line)
        rc = process.wait()
    data_volume.commit()
    if rc != 0:
        raise RuntimeError(f"Training failed with exit code {rc}. Log: {log_path}")
    return {"run_id": run_id, "log_path": str(log_path), **_extract_run_metrics(log_path)}


@app.function(image=image, volumes={str(VOLUME_ROOT): data_volume}, gpu="H100", timeout=2 * 60 * 60, retries=modal.Retries(initial_delay=0.0, max_retries=1), single_use_containers=True)
def train_1xh100(variant: str = "sp1024", run_id: Optional[str] = None, iterations: int = 20_000, max_wallclock_seconds: float = 600.0, val_loss_every: int = 1000, train_log_every: int = 200, env_overrides: str = "") -> dict:
    return _run_training(1, variant, run_id, iterations, max_wallclock_seconds, val_loss_every, train_log_every, env_overrides)


@app.function(image=image, volumes={str(VOLUME_ROOT): data_volume}, gpu="H100:8", timeout=2 * 60 * 60, retries=modal.Retries(initial_delay=0.0, max_retries=1), single_use_containers=True)
def train_8xh100(variant: str = "sp1024", run_id: Optional[str] = None, iterations: int = 20_000, max_wallclock_seconds: float = 600.0, val_loss_every: int = 1000, train_log_every: int = 200, env_overrides: str = "") -> dict:
    return _run_training(8, variant, run_id, iterations, max_wallclock_seconds, val_loss_every, train_log_every, env_overrides)


@app.function(image=image, volumes={str(VOLUME_ROOT): data_volume}, timeout=24 * 60 * 60)
def sweep_jobs(variant: str = "sp1024", train_shards: int = 1, gpu_count: int = 1, sweep_id: str = "", iterations: int = 5000, max_wallclock_seconds: float = 600.0, val_loss_every: int = 500, train_log_every: int = 100, base_env_overrides: str = "", num_layers_values: str = "9,12", model_dim_values: str = "512,640", num_heads_values: str = "8,10", num_kv_heads_values: str = "4,5", mlp_mult_values: str = "2") -> dict:
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    prepare_data.remote(variant=variant, train_shards=train_shards)
    train_fn = train_8xh100 if gpu_count == 8 else train_1xh100
    if gpu_count not in (1, 8):
        raise ValueError("gpu_count must be 1 or 8")
    sweep_id = sweep_id or f"sweep_{variant}_{uuid.uuid4().hex[:8]}"
    calls: list[tuple[str, dict[str, str], object]] = []
    for idx, env in enumerate(_build_sweep_envs(base_env_overrides, num_layers_values, model_dim_values, num_heads_values, num_kv_heads_values, mlp_mult_values), start=1):
        run_id = f"{sweep_id}_{idx:03d}"
        call = train_fn.spawn(variant=variant, run_id=run_id, iterations=iterations, max_wallclock_seconds=max_wallclock_seconds, val_loss_every=val_loss_every, train_log_every=train_log_every, env_overrides=_env_dict_to_string(env))
        calls.append((run_id, env, call))
    results: list[dict] = []
    for run_id, env, call in calls:
        try:
            results.append({"status": "ok", "run_id": run_id, "env": env, **call.get()})
        except Exception as exc:
            results.append({"status": "error", "run_id": run_id, "env": env, "error": str(exc)})
    ranked = sorted([x for x in results if x.get("status") == "ok" and x.get("val_bpb") is not None], key=lambda x: (x["val_bpb"], x.get("submission_size_bytes") or 10**18))
    summary = {"sweep_id": sweep_id, "variant": variant, "gpu_count": gpu_count, "results": results, "ranked_results": ranked, "job_count": len(results), "best": ranked[0] if ranked else None}
    (RUNS_ROOT / f"{sweep_id}.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    data_volume.commit()
    return summary
