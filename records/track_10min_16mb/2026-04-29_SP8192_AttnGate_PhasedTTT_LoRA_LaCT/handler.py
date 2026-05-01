from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator

try:
    import runpod
except ImportError:  # pragma: no cover - local syntax check path only
    runpod = None


RECORD_DIR = Path(__file__).resolve().parent
REPO_ROOT = RECORD_DIR.parents[3]
RECORD_NAME = RECORD_DIR.name
DATA_SCRIPT = REPO_ROOT / "data" / "cached_challenge_fineweb.py"
RUNPOD_VOLUME_ROOT = Path(os.environ.get("RUNPOD_VOLUME_ROOT", "/runpod-volume"))
PERSIST_ROOT = Path(os.environ.get("RUNPOD_PERSIST_ROOT", RUNPOD_VOLUME_ROOT / "parameter-golf"))
DATA_ROOT = Path(os.environ.get("RUNPOD_DATA_ROOT", PERSIST_ROOT / "data"))
RESULTS_ROOT = Path(
    os.environ.get(
        "RUNPOD_RESULTS_ROOT",
        PERSIST_ROOT / "endpoint_results" / RECORD_NAME,
    )
)
LOCKS_ROOT = PERSIST_ROOT / "locks"
DEFAULT_ENDPOINT_TIMEOUT_SECONDS = int(os.environ.get("RUNPOD_ENDPOINT_TIMEOUT_SECONDS", "1400"))
DEFAULT_GPU_COUNT = int(os.environ.get("RUNPOD_GPU_COUNT_REQUIRED", "8"))
DEFAULT_PER_SEED_OVERHEAD_SECONDS = int(os.environ.get("RUNPOD_PER_SEED_OVERHEAD_SECONDS", "180"))
PRIMARY_EVAL_LABEL = "quantized_ttt_phased"
LATEST_VALID_RECORD = {
    "name": "2026-04-27_SP8192_LQER_SparseGate_BOSSmearFix_9HpStack_1.0611",
    "date": "2026-04-27",
    "ttt_bpb": 1.06108,
    "sliding_bpb": None,
    "artifact_bytes_mean": None,
}
COMPETITION_SEEDS = (42, 314, 999)
EVAL_RE = re.compile(
    r"^(?P<label>.+?) val_loss:(?P<loss>[-+0-9.eE]+) "
    r"val_bpb:(?P<bpb>[-+0-9.eE]+) eval_time:(?P<eval_ms>\d+)ms$"
)
ARTIFACT_RE = re.compile(
    r"^Total submission size quantized\+[A-Za-z0-9_+-]+: (?P<bytes>\d+) bytes$"
)

RECORD_PROFILE_ENV = {
    "VOCAB_SIZE": "8192",
    "NUM_LAYERS": "11",
    "MODEL_DIM": "512",
    "EMBEDDING_DIM": "512",
    "NUM_HEADS": "8",
    "NUM_KV_HEADS": "4",
    "QK_GAIN_INIT": "5.25",
    "NUM_LOOPS": "2",
    "LOOP_START": "3",
    "LOOP_END": "5",
    "ENABLE_LOOPING_AT": "0.35",
    "PARALLEL_RESIDUAL_START": "7",
    "GATED_ATTN_ENABLED": "1",
    "GATED_ATTN_INIT_STD": "0.01",
    "GATED_ATTN_QUANT_GATE": "1",
    "TTT_LORA_ENABLED": "0",
    "TTT_LORA_RANK": "96",
    "TTT_LORA_LR": "0.0001",
    "TTT_CHUNK_SIZE": "48",
    "TTT_BATCH_SIZE": "64",
    "TTT_GRAD_STEPS": "1",
    "TTT_WEIGHT_DECAY": "0.5",
    "TTT_BETA1": "0.0",
    "TTT_BETA2": "0.999",
    "TTT_K_LORA": "1",
    "TTT_MLP_LORA": "1",
    "TTT_O_LORA": "1",
    "TTT_OPTIMIZER": "adam",
    "PHASED_TTT_ENABLED": "1",
    "PHASED_TTT_PREFIX_DOCS": "2000",
    "PHASED_TTT_NUM_PHASES": "4",
    "GLOBAL_TTT_LR": "0.001",
    "GLOBAL_TTT_MOMENTUM": "0.9",
    "GLOBAL_TTT_EPOCHS": "1",
    "GLOBAL_TTT_CHUNK_TOKENS": "32768",
    "GLOBAL_TTT_BATCH_SEQS": "32",
    "GLOBAL_TTT_WARMUP_START_LR": "0.0",
    "GLOBAL_TTT_WARMUP_CHUNKS": "0",
    "GLOBAL_TTT_GRAD_CLIP": "1.0",
    "GLOBAL_TTT_RESPECT_DOC_BOUNDARIES": "1",
    "TTT_ENABLED": "1",
    "LACT_TTT_ENABLED": "0",
    "ARTIFACT_TARGET_BYTES": "16000000",
    "MLP_CLIP_SIGMAS": "12.0",
    "ATTN_CLIP_SIGMAS": "13.0",
    "LQER_ENABLED": "1",
    "LQER_RANK": "4",
    "LQER_TOP_K": "3",
    "LQER_FACTOR_BITS": "4",
    "LQER_ASYM_ENABLED": "1",
    "LQER_ASYM_GROUP": "64",
    "GPTQ_CALIBRATION_BATCHES": "64",
    "ITERATIONS": "20000",
    "WARMDOWN_FRAC": "0.72",
    "TRAIN_BATCH_TOKENS": "786432",
    "TRAIN_LOG_EVERY": "500",
    "VAL_LOSS_EVERY": "4000",
    "EMA_DECAY": "0.9965",
    "MUON_WD": "0.095",
    "PYTHONUNBUFFERED": "1",
}
RESERVED_ENV_KEYS = {"SEED", "RUN_ID", "DATA_DIR", "RANK", "LOCAL_RANK", "WORLD_SIZE"}


@contextmanager
def file_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def emit_line(emit: Callable[[str], str], prefix: str, line: str) -> str:
    return emit(f"{prefix}{line.rstrip()}")


def parse_bool(value: object, *, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        if value in (0, 1):
            return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"'{field_name}' must be a boolean-like value, got {value!r}")


def validate_worker() -> None:
    if not RUNPOD_VOLUME_ROOT.exists():
        raise FileNotFoundError(
            f"RunPod network volume is required but {RUNPOD_VOLUME_ROOT} does not exist. "
            "Attach a network volume to the endpoint so dataset, logs, and artifacts persist."
        )
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - container build issue
        raise RuntimeError("PyTorch is required in the worker image") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this endpoint worker")
    gpu_count = torch.cuda.device_count()
    if gpu_count != DEFAULT_GPU_COUNT:
        raise RuntimeError(
            f"This endpoint requires exactly {DEFAULT_GPU_COUNT} visible GPUs, got {gpu_count}. "
            "Set the endpoint to 8 GPUs per worker."
        )


def ensure_repo_data_symlinks() -> None:
    repo_data_dir = REPO_ROOT / "data"
    link_targets = {
        repo_data_dir / "datasets": DATA_ROOT / "datasets",
        repo_data_dir / "tokenizers": DATA_ROOT / "tokenizers",
    }
    for link_path, target_path in link_targets.items():
        target_path.mkdir(parents=True, exist_ok=True)
        if link_path.is_symlink():
            resolved = link_path.resolve()
            if resolved != target_path.resolve():
                raise RuntimeError(
                    f"{link_path} already points to {resolved}, expected {target_path}. "
                    "Refusing to repoint an existing symlink silently."
                )
            continue
        if link_path.exists():
            raise RuntimeError(
                f"{link_path} exists and is not a symlink. "
                "Remove or rename it before using the endpoint persistence path."
            )
        link_path.symlink_to(target_path, target_is_directory=True)


def coerce_seeds(job_input: dict, *, max_wallclock_seconds: int) -> tuple[list[int], str | None]:
    if "seed" in job_input and "seeds" in job_input:
        raise ValueError("Provide either 'seed' or 'seeds', not both")
    if "seed" in job_input:
        return [int(job_input["seed"])], None
    if "seeds" in job_input:
        seeds = [int(seed) for seed in job_input["seeds"]]
        if not seeds:
            raise ValueError("'seeds' must not be empty")
        return seeds, None

    required_timeout = len(COMPETITION_SEEDS) * (
        max_wallclock_seconds + DEFAULT_PER_SEED_OVERHEAD_SECONDS
    )
    if DEFAULT_ENDPOINT_TIMEOUT_SECONDS < required_timeout:
        return [COMPETITION_SEEDS[0]], (
            "Endpoint timeout is 1400s, which is too short for the required 3-seed sweep "
            f"{list(COMPETITION_SEEDS)} in one job. Defaulting to seed {COMPETITION_SEEDS[0]}; "
            "run three separate jobs with a shared run_group_id for competition-ready results."
        )
    return list(COMPETITION_SEEDS), None


def dataset_probe_paths(train_shards: int) -> tuple[Path, Path, Path]:
    if train_shards <= 0:
        raise ValueError(f"train_shards must be positive, got {train_shards}")
    dataset_dir = DATA_ROOT / "datasets" / "fineweb10B_sp8192"
    tokenizer_path = DATA_ROOT / "tokenizers" / "fineweb_8192_bpe.model"
    train_probe = dataset_dir / f"fineweb_train_{train_shards - 1:06d}.bin"
    val_probe = dataset_dir / "fineweb_val_000000.bin"
    return tokenizer_path, train_probe, val_probe


def stream_subprocess(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    emit: Callable[[str], str],
    prefix: str,
) -> tuple[int, list[str]]:
    process = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    captured: list[str] = []
    assert process.stdout is not None
    for raw_line in process.stdout:
        line = raw_line.rstrip("\n")
        captured.append(line)
        yield emit_line(emit, prefix, line)
    return_code = process.wait()
    return return_code, captured


def ensure_dataset(
    *,
    train_shards: int,
    allow_prepare: bool,
    matched_repo_id: str | None,
    emit: Callable[[str], str],
) -> Iterator[str]:
    tokenizer_path, train_probe, val_probe = dataset_probe_paths(train_shards)
    if tokenizer_path.is_file() and train_probe.is_file() and val_probe.is_file():
        yield emit(
            f"[dataset] ready variant=sp8192 train_shards={train_shards} root={DATA_ROOT}"
        )
        return

    if not allow_prepare:
        raise FileNotFoundError(
            "Dataset is missing on the network volume and prepare_dataset_if_missing=0. "
            f"Expected tokenizer={tokenizer_path}, train_probe={train_probe}, val_probe={val_probe}"
        )

    yield emit(
        f"[dataset] missing on network volume, preparing variant=sp8192 train_shards={train_shards}"
    )
    with file_lock(LOCKS_ROOT / "dataset_sp8192.lock"):
        tokenizer_path, train_probe, val_probe = dataset_probe_paths(train_shards)
        if tokenizer_path.is_file() and train_probe.is_file() and val_probe.is_file():
            yield emit("[dataset] another worker populated the dataset while waiting on the lock")
            return
        cmd = [
            sys.executable,
            str(DATA_SCRIPT),
            "--variant",
            "sp8192",
            "--train-shards",
            str(train_shards),
        ]
        env = os.environ.copy()
        if matched_repo_id:
            env["MATCHED_FINEWEB_REPO_ID"] = matched_repo_id
        return_code, _ = yield from stream_subprocess(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            emit=emit,
            prefix="[dataset] ",
        )
        if return_code != 0:
            raise RuntimeError(f"Dataset preparation failed with exit code {return_code}")
        tokenizer_path, train_probe, val_probe = dataset_probe_paths(train_shards)
        if not (tokenizer_path.is_file() and train_probe.is_file() and val_probe.is_file()):
            raise RuntimeError(
                "Dataset preparation completed without producing the expected files on the network volume"
            )
        yield emit(f"[dataset] prepared successfully at {DATA_ROOT}")


def build_run_env(
    *,
    seed: int,
    run_id: str,
    max_wallclock_seconds: int,
    gptq_reserve_seconds: int,
    extra_env: dict[str, str],
) -> dict[str, str]:
    env = os.environ.copy()
    env.update(RECORD_PROFILE_ENV)
    env["SEED"] = str(seed)
    env["RUN_ID"] = str(run_id)
    env["DATA_DIR"] = str(DATA_ROOT)
    env["MAX_WALLCLOCK_SECONDS"] = str(max_wallclock_seconds)
    env["GPTQ_RESERVE_SECONDS"] = str(gptq_reserve_seconds)
    for key, value in extra_env.items():
        if key in RESERVED_ENV_KEYS:
            raise ValueError(f"'{key}' is reserved and cannot be overridden from job input")
        env[key] = value
    return env


def parse_metrics(log_lines: list[str]) -> dict[str, object]:
    evals: dict[str, dict[str, float | int]] = {}
    artifact_total_bytes: int | None = None
    for line in log_lines:
        eval_match = EVAL_RE.match(line)
        if eval_match:
            evals[eval_match.group("label")] = {
                "val_loss": float(eval_match.group("loss")),
                "val_bpb": float(eval_match.group("bpb")),
                "eval_time_ms": int(eval_match.group("eval_ms")),
            }
            continue
        artifact_match = ARTIFACT_RE.match(line)
        if artifact_match:
            artifact_total_bytes = int(artifact_match.group("bytes"))
    if PRIMARY_EVAL_LABEL not in evals:
        raise RuntimeError(
            f"Primary metric {PRIMARY_EVAL_LABEL!r} was not found in the training log"
        )
    primary_bpb = float(evals[PRIMARY_EVAL_LABEL]["val_bpb"])
    return {
        "evals": evals,
        "primary_eval_label": PRIMARY_EVAL_LABEL,
        "primary_val_bpb": primary_bpb,
        "delta_vs_latest_valid_record_bpb": primary_bpb - LATEST_VALID_RECORD["ttt_bpb"],
        "improvement_vs_latest_valid_record_bpb": LATEST_VALID_RECORD["ttt_bpb"] - primary_bpb,
        "artifact_total_bytes": artifact_total_bytes,
        "artifact_delta_vs_latest_valid_record_mean": (
            None
            if artifact_total_bytes is None or LATEST_VALID_RECORD["artifact_bytes_mean"] is None
            else artifact_total_bytes - LATEST_VALID_RECORD["artifact_bytes_mean"]
        ),
        "latest_valid_record": LATEST_VALID_RECORD,
    }


def append_summary_index(summary_path: Path, payload: dict[str, object]) -> None:
    with file_lock(summary_path.with_suffix(".lock")):
        with summary_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True))
            handle.write("\n")


def run_seed(
    *,
    seed: int,
    run_group_id: str,
    run_id_prefix: str,
    job_id: str,
    max_wallclock_seconds: int,
    gptq_reserve_seconds: int,
    extra_env: dict[str, str],
    emit: Callable[[str], str],
) -> Iterator[str]:
    run_id = f"{run_id_prefix}_seed{seed}_{uuid.uuid4().hex[:8]}"
    work_parent = Path(tempfile.mkdtemp(prefix=f"{RECORD_NAME}_seed{seed}_"))
    work_dir = work_parent / RECORD_NAME
    shutil.copytree(RECORD_DIR, work_dir)

    group_root = RESULTS_ROOT / run_group_id / f"seed_{seed}"
    logs_root = group_root / "logs"
    artifacts_root = group_root / "artifacts"
    logs_root.mkdir(parents=True, exist_ok=True)
    artifacts_root.mkdir(parents=True, exist_ok=True)

    env = build_run_env(
        seed=seed,
        run_id=run_id,
        max_wallclock_seconds=max_wallclock_seconds,
        gptq_reserve_seconds=gptq_reserve_seconds,
        extra_env=extra_env,
    )
    hash_int = int(hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:4], 16)
    master_port = 29500 + (hash_int % 1000)
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        str(DEFAULT_GPU_COUNT),
        "--master_port",
        str(master_port),
        "train_gpt.py",
    ]

    yield emit(
        f"[seed={seed}] start run_id={run_id} max_wallclock_seconds={max_wallclock_seconds} "
        f"data_dir={DATA_ROOT} work_dir={work_dir}"
    )
    yield emit(f"[seed={seed}] command={' '.join(cmd)}")

    try:
        return_code, captured = yield from stream_subprocess(
            cmd,
            cwd=work_dir,
            env=env,
            emit=emit,
            prefix=f"[seed={seed}] ",
        )
        log_path = work_dir / "logs" / f"{run_id}.txt"
        quantized_path = work_dir / "final_model.int6.ptz"

        persisted_log_path = logs_root / f"{run_id}.txt"
        if log_path.is_file():
            shutil.copy2(log_path, persisted_log_path)
        else:
            persisted_log_path.write_text("\n".join(captured) + "\n", encoding="utf-8")

        if return_code != 0:
            raise RuntimeError(
                f"torchrun failed for seed {seed} with exit code {return_code}. "
                f"Persisted log: {persisted_log_path}"
            )
        if not quantized_path.is_file():
            raise FileNotFoundError(
                f"Expected artifact {quantized_path} was not produced by the training script"
            )

        artifact_dest = artifacts_root / quantized_path.name
        shutil.copy2(quantized_path, artifact_dest)

        metrics = parse_metrics(captured)
        metrics.update(
            {
                "job_id": job_id,
                "seed": seed,
                "run_group_id": run_group_id,
                "run_id": run_id,
                "log_path": str(persisted_log_path),
                "artifact_path": str(artifact_dest),
                "record_name": RECORD_NAME,
            }
        )

        summary_path = group_root / "summary.json"
        summary_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
        append_summary_index(RESULTS_ROOT / "summary_index.jsonl", metrics)

        yield emit(
            f"[seed={seed}] summary "
            + json.dumps(
                {
                    "seed": seed,
                    "primary_eval_label": metrics["primary_eval_label"],
                    "primary_val_bpb": metrics["primary_val_bpb"],
                    "improvement_vs_latest_valid_record_bpb": metrics[
                        "improvement_vs_latest_valid_record_bpb"
                    ],
                    "artifact_total_bytes": metrics["artifact_total_bytes"],
                    "artifact_path": metrics["artifact_path"],
                    "log_path": metrics["log_path"],
                },
                sort_keys=True,
            )
        )
    finally:
        shutil.rmtree(work_parent, ignore_errors=True)


def handler(job: dict) -> Iterator[str]:
    validate_worker()
    ensure_repo_data_symlinks()
    PERSIST_ROOT.mkdir(parents=True, exist_ok=True)
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    job_id = str(job.get("id") or uuid.uuid4())
    job_input = job.get("input") or {}
    if not isinstance(job_input, dict):
        raise ValueError("RunPod job input must be a JSON object")

    extra_env_raw = job_input.get("env_overrides") or {}
    if not isinstance(extra_env_raw, dict):
        raise ValueError("'env_overrides' must be an object when provided")
    extra_env = {str(key): str(value) for key, value in extra_env_raw.items()}
    max_wallclock_seconds = int(job_input.get("max_wallclock_seconds", 600))
    gptq_reserve_seconds = int(job_input.get("gptq_reserve_seconds", 12))
    train_shards = int(job_input.get("train_shards", 128))
    prepare_dataset_if_missing = parse_bool(
        job_input.get("prepare_dataset_if_missing", True),
        field_name="prepare_dataset_if_missing",
    )
    matched_repo_id = job_input.get("matched_fineweb_repo_id")
    run_group_id = str(job_input.get("run_group_id") or f"{RECORD_NAME}_{time.strftime('%Y%m%d_%H%M%S')}")
    run_id_prefix = str(job_input.get("run_id_prefix") or RECORD_NAME)
    seeds, seed_note = coerce_seeds(job_input, max_wallclock_seconds=max_wallclock_seconds)

    if len(seeds) * (max_wallclock_seconds + DEFAULT_PER_SEED_OVERHEAD_SECONDS) > DEFAULT_ENDPOINT_TIMEOUT_SECONDS:
        raise ValueError(
            f"Requested seeds {seeds} exceed the configured endpoint timeout of "
            f"{DEFAULT_ENDPOINT_TIMEOUT_SECONDS}s. Submit one seed per job or raise the endpoint timeout."
        )

    log_lines: list[str] = []

    def emit(message: str) -> str:
        line = message.rstrip()
        log_lines.append(line)
        return line

    yield emit(
        f"[handler] job_id={job_id} record={RECORD_NAME} run_group_id={run_group_id} "
        f"seeds={seeds} results_root={RESULTS_ROOT}"
    )
    yield emit(
        f"[handler] latest_valid_record={LATEST_VALID_RECORD['name']} "
        f"date={LATEST_VALID_RECORD['date']} ttt_bpb={LATEST_VALID_RECORD['ttt_bpb']} "
        f"sliding_bpb={LATEST_VALID_RECORD['sliding_bpb']} "
        f"artifact_bytes_mean={LATEST_VALID_RECORD['artifact_bytes_mean']}"
    )
    if seed_note:
        yield emit(f"[handler] note={seed_note}")
    yield emit(
        "[handler] competition requires three seeds; use the same run_group_id across "
        "seed 42, 314, and 999 jobs when running against the 1400s endpoint timeout."
    )

    yield from ensure_dataset(
        train_shards=train_shards,
        allow_prepare=prepare_dataset_if_missing,
        matched_repo_id=matched_repo_id,
        emit=emit,
    )

    for seed in seeds:
        yield from run_seed(
            seed=seed,
            run_group_id=run_group_id,
            run_id_prefix=run_id_prefix,
            job_id=job_id,
            max_wallclock_seconds=max_wallclock_seconds,
            gptq_reserve_seconds=gptq_reserve_seconds,
            extra_env=extra_env,
            emit=emit,
        )

    combined_log_path = RESULTS_ROOT / run_group_id / "combined_response_log.txt"
    combined_log_path.parent.mkdir(parents=True, exist_ok=True)
    emit(f"[handler] combined_log_path={combined_log_path}")
    combined_log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    yield log_lines[-1]


if __name__ == "__main__":
    if runpod is None:
        raise RuntimeError("The RunPod SDK is required to start this endpoint handler")
    runpod.serverless.start({"handler": handler, "return_aggregate_stream": True})
