#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import re
import shlex
import signal
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


OOM_PATTERNS = (
    "OutOfMemoryError",
    "CUDA out of memory",
)
BACKEND_ASSERT_PATTERNS = (
    "cannot extract sympy expressions",
    "BackendCompilerFailed",
    "torch._dynamo.exc.BackendCompilerFailed",
    "AssertionError:",
)
DDP_UNUSED_PATTERNS = (
    "Expected to have finished reduction in the prior iteration",
    "did not receive grad",
    "Parameter indices which did not receive grad",
)
TRACEBACK_PREFIX = "Traceback (most recent call last):"
STEP_RE = re.compile(r"^step:(\d+)")
AVG_RE = re.compile(r"avg:([0-9]+\.?[0-9]*)ms")
COMPILE_LINE_RE = re.compile(r"^compile:mode=")


@dataclass(frozen=True)
class HardwareInfo:
    gpu_names: list[str]
    gpu_count: int
    min_free_mb: int
    min_total_mb: int
    driver_version: str
    cuda_version: str
    nproc: int


@dataclass(frozen=True)
class ProbeConfig:
    compile_mode: str
    compile_target: str
    compile_max_modules: int
    train_batch_tokens: int
    compiler_warmup_batch_tokens: int
    ddp_find_unused_parameters: int
    stage: str


def parse_csv_lines(raw: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append([part.strip() for part in line.split(",")])
    return rows


def run_checked(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, check=True, text=True, capture_output=True)
    return proc.stdout


def detect_hardware(nproc: int, cuda_visible_devices: str | None) -> HardwareInfo:
    smi_cmd = ["nvidia-smi"]
    if cuda_visible_devices:
        smi_cmd.extend(["-i", cuda_visible_devices])
    query_cmd = smi_cmd + [
        "--query-gpu=name,memory.total,memory.free,driver_version",
        "--format=csv,noheader,nounits",
    ]
    rows = parse_csv_lines(run_checked(query_cmd))
    if not rows:
        raise RuntimeError("nvidia-smi returned no GPU rows")
    rows = rows[:nproc]
    gpu_names = [row[0] for row in rows]
    totals = [int(row[1]) for row in rows]
    frees = [int(row[2]) for row in rows]
    driver = rows[0][3]
    header = run_checked(["nvidia-smi"])
    match = re.search(r"CUDA Version:\s*([0-9.]+)", header)
    cuda_version = match.group(1) if match else "unknown"
    return HardwareInfo(
        gpu_names=gpu_names,
        gpu_count=len(rows),
        min_free_mb=min(frees),
        min_total_mb=min(totals),
        driver_version=driver,
        cuda_version=cuda_version,
        nproc=nproc,
    )


def relevant_model_shape(env: dict[str, str]) -> dict[str, str]:
    keys = (
        "NUM_LAYERS",
        "MODEL_DIM",
        "NUM_HEADS",
        "NUM_KV_HEADS",
        "EMBED_DIM",
        "MLP_MULT",
        "TRAIN_SEQ_LEN",
        "SKC_NUM_CAPSULES",
        "SKC_CAPSULE_DIM",
        "SKC_BLOCK_SIZE",
    )
    return {key: env[key] for key in keys if key in env}


def build_signature(hw: HardwareInfo, profile: str, env: dict[str, str]) -> dict[str, Any]:
    return {
        "profile": profile,
        "gpu_names": hw.gpu_names,
        "gpu_count": hw.gpu_count,
        "min_total_mb": hw.min_total_mb,
        "driver_version": hw.driver_version,
        "cuda_version": hw.cuda_version,
        "nproc": hw.nproc,
        "model_shape": relevant_model_shape(env),
    }


def signature_key(signature: dict[str, Any]) -> str:
    encoded = json.dumps(signature, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def default_train_batch_tokens(free_mb: int) -> int:
    if free_mb >= 70000:
        return 196608
    if free_mb >= 50000:
        return 131072
    if free_mb >= 35000:
        return 98304
    if free_mb >= 30000:
        return 65536
    if free_mb >= 24000:
        return 49152
    if free_mb >= 16000:
        return 32768
    return 32768


def default_warmup_batch_tokens(free_mb: int) -> int:
    if free_mb < 30000:
        return 8192
    if free_mb < 50000:
        return 16384
    return 0


def dedupe_preserve_order(values: list[int]) -> list[int]:
    seen: set[int] = set()
    ordered: list[int] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def parse_int_list(raw: str | None) -> list[int]:
    if not raw:
        return []
    parts = [p.strip() for p in raw.split(",")]
    return [int(part) for part in parts if part]


def auto_tune_quick(env: dict[str, str]) -> bool:
    return env.get("AUTO_TUNE_QUICK", "0") == "1"


def build_batch_candidates(heuristic: int, env: dict[str, str], profile: str) -> list[int]:
    explicit = parse_int_list(env.get("AUTO_TUNE_BATCH_CANDIDATES"))
    min_batch = int(env.get("AUTO_TUNE_MIN_BATCH_TOKENS", "8192"))
    max_batch = int(env.get("AUTO_TUNE_MAX_BATCH_TOKENS", str(max(heuristic, 16000))))
    if explicit:
        return [value for value in dedupe_preserve_order(explicit) if min_batch <= value <= max_batch]
    if auto_tune_quick(env):
        quick_values = [heuristic, min(10128, max_batch), min_batch]
        return [value for value in dedupe_preserve_order(quick_values) if min_batch <= value <= max_batch]

    floor_values = [16000, 10128, 8192]
    if profile == "h100_final":
        floor_values = [65536, 32768, 16000, 10128, 8192]
    values = [heuristic]
    candidate = heuristic
    while candidate > min_batch:
        candidate = max(candidate // 2, min_batch)
        values.append(candidate)
        if candidate == min_batch:
            break
    values.extend(floor_values)
    filtered = [value for value in dedupe_preserve_order(values) if min_batch <= value <= max_batch]
    return filtered or [min_batch]


def build_warmup_candidates(heuristic: int, train_batch: int, env: dict[str, str]) -> list[int]:
    if auto_tune_quick(env):
        values = [
            min(heuristic, train_batch),
            min(train_batch, 8192),
            0,
        ]
        filtered = [value for value in dedupe_preserve_order(values) if value <= max(train_batch, heuristic)]
        return filtered or [0]
    values = [
        heuristic,
        min(train_batch, 16384),
        min(train_batch, 8192),
        4096,
        2048,
        0,
    ]
    filtered = [value for value in dedupe_preserve_order(values) if value <= max(train_batch, heuristic)]
    return filtered or [0]


def compile_ladder(profile: str, env: dict[str, str] | None = None) -> list[tuple[str, str, int]]:
    env = env or {}
    if profile in {"diagnostic", "convergence"}:
        # For convergence diagnostics we intentionally avoid compile effects.
        return [("none", "full", 0)]
    if auto_tune_quick(env):
        return [
            ("none", "full", 0),
            ("reduce-overhead", "blocks", 1),
        ]
    return [
        ("max-autotune", "blocks", 4),
        ("max-autotune", "blocks", 2),
        ("max-autotune", "blocks", 1),
        ("reduce-overhead", "blocks", 4),
        ("reduce-overhead", "blocks", 2),
        ("reduce-overhead", "blocks", 1),
        ("none", "full", 0),
    ]


def classify_failure(log_text: str, timed_out: bool, step_lines: int, stage: str) -> tuple[str, str]:
    first_traceback = ""
    for line in log_text.splitlines():
        if TRACEBACK_PREFIX in line:
            first_traceback = TRACEBACK_PREFIX
            break
        if "Traceback" in line and not first_traceback:
            first_traceback = line.strip()
            break
    for pattern in OOM_PATTERNS:
        if pattern in log_text:
            return ("oom", pattern)
    if any(pattern in log_text for pattern in DDP_UNUSED_PATTERNS):
        return ("ddp_unused_grad", "ddp_unused_grad")
    if "cannot extract sympy expressions" in log_text:
        return ("compile_backend", "cannot extract sympy expressions")
    for pattern in BACKEND_ASSERT_PATTERNS:
        if pattern in log_text:
            return ("compile_backend", pattern)
    if timed_out and step_lines == 0 and stage == "train":
        return ("timeout_compile_bound", "timeout without train step")
    if timed_out and stage == "precompile":
        return ("timeout_compile_bound", "precompile timeout")
    if first_traceback:
        return ("unknown", first_traceback)
    lines = [line.strip() for line in log_text.splitlines() if line.strip()]
    return ("unknown", lines[-1] if lines else "no output")


class ProcessReader(threading.Thread):
    def __init__(self, pipe, sink: queue.Queue[str]):
        super().__init__(daemon=True)
        self.pipe = pipe
        self.sink = sink

    def run(self) -> None:
        try:
            for line in iter(self.pipe.readline, ""):
                self.sink.put(line)
        finally:
            self.pipe.close()


def terminate_process(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def run_probe(
    root: Path,
    env: dict[str, str],
    nproc: int,
    log_path: Path,
    timeout_seconds: int,
    min_success_steps: int,
    stage: str,
) -> dict[str, Any]:
    cmd = [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={nproc}",
        "train_gpt_verbose.py",
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    q: queue.Queue[str] = queue.Queue()
    reader = ProcessReader(proc.stdout, q)
    reader.start()

    lines: list[str] = []
    start = time.time()
    success_gate = False
    step_lines = 0
    last_step = 0
    last_avg_ms: float | None = None
    compile_summary = ""
    timed_out = False

    with log_path.open("w", encoding="utf-8") as handle:
        while True:
            if time.time() - start > timeout_seconds:
                timed_out = True
                terminate_process(proc)
                break
            try:
                line = q.get(timeout=0.25)
            except queue.Empty:
                if proc.poll() is not None:
                    break
                continue
            handle.write(line)
            handle.flush()
            lines.append(line)
            stripped = line.strip()
            step_match = STEP_RE.search(stripped)
            if step_match:
                step_lines += 1
                last_step = max(last_step, int(step_match.group(1)))
                avg_match = AVG_RE.search(stripped)
                if avg_match:
                    last_avg_ms = float(avg_match.group(1))
                if stage == "train" and last_step >= min_success_steps:
                    success_gate = True
                    terminate_process(proc)
                    break
            elif COMPILE_LINE_RE.search(stripped):
                compile_summary = stripped
            elif stage == "precompile" and "precompile_only:done" in stripped:
                success_gate = True
        while True:
            try:
                line = q.get_nowait()
            except queue.Empty:
                break
            handle.write(line)
            handle.flush()
            lines.append(line)

    rc = proc.poll()
    if rc is None:
        rc = proc.wait(timeout=5)
    elapsed = time.time() - start
    log_text = "".join(lines)
    return {
        "returncode": rc,
        "elapsed_s": round(elapsed, 3),
        "timed_out": timed_out,
        "success_gate": success_gate,
        "step_lines": step_lines,
        "last_step": last_step,
        "last_avg_ms": last_avg_ms,
        "compile_summary": compile_summary,
        "log_text": log_text,
        "log_path": str(log_path),
    }


def build_base_env(root: Path, run_id: str, env: dict[str, str], nproc: int) -> dict[str, str]:
    free_mb = int(env["AUTO_TUNE_FREE_MB_MIN"])
    total_mb = int(env["AUTO_TUNE_TOTAL_MB_MIN"])
    train_batch = env.get("TRAIN_BATCH_TOKENS", str(default_train_batch_tokens(free_mb)))
    warmup_batch = env.get("COMPILER_WARMUP_BATCH_TOKENS", str(default_warmup_batch_tokens(free_mb)))
    sliding = "128" if total_mb >= 50000 else "64"
    merged = os.environ.copy()
    merged.update(
        {
            "RUN_ID": run_id,
            "AUTO_TUNE_QUICK": env.get("AUTO_TUNE_QUICK", "0"),
            "PYTORCH_CUDA_ALLOC_CONF": env.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"),
            "DDP_FIND_UNUSED_PARAMETERS": env.get("DDP_FIND_UNUSED_PARAMETERS", "0"),
            "OMP_NUM_THREADS": env.get("OMP_NUM_THREADS", "8"),
            "DATA_PATH": env.get("DATA_PATH", "/workspace/data/datasets/fineweb10B_sp8192"),
            "TOKENIZER_PATH": env.get("TOKENIZER_PATH", "/workspace/data/tokenizers/fineweb_8192_bpe.model"),
            "COMPETITION_PROFILE": env.get("COMPETITION_PROFILE", "1"),
            "EXPORT_MODE": env.get("EXPORT_MODE", "competition_gptq"),
            "RUNTIME_PATH_POLICY": env.get("RUNTIME_PATH_POLICY", "strict"),
            "HARD_BUDGET_BYTES": env.get("HARD_BUDGET_BYTES", "16000000"),
            "HARD_BUDGET_ENFORCE": env.get("HARD_BUDGET_ENFORCE", "1"),
            "MAX_WALLCLOCK_SECONDS": env.get("MAX_WALLCLOCK_SECONDS", "570"),
            "ITERATIONS": env.get("ITERATIONS", "200000"),
            "TRAIN_BATCH_TOKENS": train_batch,
            "TRAIN_SEQ_LEN": env.get("TRAIN_SEQ_LEN", "1024"),
            "SLIDING_EVAL": env.get("SLIDING_EVAL", "1"),
            "SLIDING_BATCH_SIZE": env.get("SLIDING_BATCH_SIZE", sliding),
            "FINAL_EVAL_SEQUENTIAL_CARRY": env.get("FINAL_EVAL_SEQUENTIAL_CARRY", "1"),
            "TORCH_NCCL_TIMEOUT_SEC": env.get("TORCH_NCCL_TIMEOUT_SEC", "3600"),
            "COMPILE_MODE": env.get("COMPILE_MODE", "reduce-overhead"),
            "COMPILER_WARMUP_STEPS": env.get("COMPILER_WARMUP_STEPS", "1"),
            "SYNTHETIC_WARMUP": env.get("SYNTHETIC_WARMUP", "1"),
            "COMPILER_WARMUP_BATCH_TOKENS": warmup_batch,
            "INDUCTOR_DISABLE_CONSTANT_FOLDING": env.get("INDUCTOR_DISABLE_CONSTANT_FOLDING", "0"),
            "DIAGNOSTICS_ENABLED": env.get("DIAGNOSTICS_ENABLED", "0"),
            "TORCHINDUCTOR_FX_GRAPH_CACHE": env.get("TORCHINDUCTOR_FX_GRAPH_CACHE", "1"),
            "TORCHINDUCTOR_AUTOGRAD_CACHE": env.get("TORCHINDUCTOR_AUTOGRAD_CACHE", "1"),
            "COMPILE_SHAPE_PADDING": env.get("COMPILE_SHAPE_PADDING", "1"),
            "COMPILE_TRITON_CUDAGRAPHS": env.get("COMPILE_TRITON_CUDAGRAPHS", "1"),
            "TORCHINDUCTOR_CACHE_DIR": env.get("TORCHINDUCTOR_CACHE_DIR", str(root / "cache" / "torch")),
            "TRITON_CACHE_DIR": env.get("TRITON_CACHE_DIR", str(root / "cache" / "triton")),
            "TERNARY_THRESHOLD_SEARCH": env.get("TERNARY_THRESHOLD_SEARCH", "1"),
            "TERNARY_SCALE_SEARCH": env.get("TERNARY_SCALE_SEARCH", "1"),
            "EXPORT_ALIGNED_TRAIN": env.get("EXPORT_ALIGNED_TRAIN", "1"),
            "EXPORT_ALIGNED_TRAIN_START_FRACTION": env.get("EXPORT_ALIGNED_TRAIN_START_FRACTION", "0.75"),
            "EXPORT_PROXY_EVAL": env.get("EXPORT_PROXY_EVAL", "1"),
            "EXPORT_PROXY_EVERY": env.get("EXPORT_PROXY_EVERY", "250"),
            "EXPORT_PROXY_NUM_SEQS": env.get("EXPORT_PROXY_NUM_SEQS", "16"),
            "TERNARY_COMPRESS_BROTLI": env.get("TERNARY_COMPRESS_BROTLI", "1"),
            "ENGRAM_COMPETITION_ENABLED": env.get("ENGRAM_COMPETITION_ENABLED", "1"),
            "SKC_RECURRENT_CORE": env.get("SKC_RECURRENT_CORE", "1"),
            "ENGRAM_EXPORT_PRUNE_ENABLED": env.get("ENGRAM_EXPORT_PRUNE_ENABLED", "1"),
            "ENGRAM_EXPORT_KEEP_BIGRAM_RATIO": env.get("ENGRAM_EXPORT_KEEP_BIGRAM_RATIO", "0.45"),
            "ENGRAM_EXPORT_KEEP_TRIGRAM_RATIO": env.get("ENGRAM_EXPORT_KEEP_TRIGRAM_RATIO", "0.20"),
            "ENGRAM_EXPORT_KEEP_MIN_BUCKETS": env.get("ENGRAM_EXPORT_KEEP_MIN_BUCKETS", "256"),
            "ENGRAM_EXPORT_SCORE_ALPHA": env.get("ENGRAM_EXPORT_SCORE_ALPHA", "0.80"),
            "ENGRAM_EXPORT_TOKEN_BUDGET": env.get("ENGRAM_EXPORT_TOKEN_BUDGET", "131072"),
            "WALL_CLOCK_TIMEOUT": env.get("WALL_CLOCK_TIMEOUT", "570"),
            "NPROC": str(nproc),
        }
    )
    return merged


def load_cache(cache_path: Path, signature: dict[str, Any], force: bool) -> dict[str, Any] | None:
    if force or not cache_path.exists():
        return None
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if payload.get("signature") != signature:
        return None
    return payload


def write_cache(cache_path: Path, payload: dict[str, Any]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_env_file(path: Path, env_values: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for key, value in sorted(env_values.items()):
        lines.append(f"export {key}={shlex.quote(str(value))}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def attempt_candidate(
    root: Path,
    base_env: dict[str, str],
    nproc: int,
    logs_dir: Path,
    precompile_seconds: int,
    smoke_seconds: int,
    min_success_steps: int,
    compile_cfg: tuple[str, str, int],
    train_batch: int,
    warmup_candidates: list[int],
    try_find_unused: bool,
    run_prefix: str,
) -> tuple[dict[str, str] | None, list[dict[str, Any]]]:
    attempts: list[dict[str, Any]] = []
    compile_mode, compile_target, compile_max_modules = compile_cfg
    find_unused_values = [0]
    if try_find_unused:
        find_unused_values = [1, 0] if auto_tune_quick(base_env) else [0, 1]

    for ddp_find_unused in find_unused_values:
        selected_warmup = 0
        precompile_status = "ok"
        precompile_reason = ""
        if compile_mode != "none":
            precompile_ok = False
            for warmup in warmup_candidates:
                run_id = f"{run_prefix}_pc_{compile_mode}_{compile_max_modules}_b{train_batch}_w{warmup}_fu{ddp_find_unused}"
                log_path = logs_dir / f"{run_id}.log"
                probe_env = dict(base_env)
                probe_env.update(
                    {
                        "RUN_ID": run_id,
                        "PRECOMPILE_ONLY": "1",
                        "COMPILE_MODE": compile_mode,
                        "COMPILE_TARGET": compile_target,
                        "COMPILE_MAX_MODULES": str(compile_max_modules),
                        "TRAIN_BATCH_TOKENS": str(train_batch),
                        "COMPILER_WARMUP_BATCH_TOKENS": str(warmup),
                        "DDP_FIND_UNUSED_PARAMETERS": str(ddp_find_unused),
                    }
                )
                result = run_probe(
                    root=root,
                    env=probe_env,
                    nproc=nproc,
                    log_path=log_path,
                    timeout_seconds=precompile_seconds,
                    min_success_steps=min_success_steps,
                    stage="precompile",
                )
                failure_class, reason = classify_failure(
                    result["log_text"],
                    timed_out=result["timed_out"],
                    step_lines=result["step_lines"],
                    stage="precompile",
                )
                attempt = {
                    "stage": "precompile",
                    "run_id": run_id,
                    "compile_mode": compile_mode,
                    "compile_target": compile_target,
                    "compile_max_modules": compile_max_modules,
                    "train_batch_tokens": train_batch,
                    "compiler_warmup_batch_tokens": warmup,
                    "ddp_find_unused_parameters": ddp_find_unused,
                    "elapsed_s": result["elapsed_s"],
                    "returncode": result["returncode"],
                    "success": bool(result["success_gate"] and result["returncode"] == 0),
                    "failure_class": "ok" if result["success_gate"] and result["returncode"] == 0 else failure_class,
                    "first_failing_line": reason,
                    "last_successful_step": result["last_step"],
                    "last_avg_ms": result["last_avg_ms"],
                    "compile_summary": result["compile_summary"],
                    "log_path": result["log_path"],
                }
                attempts.append(attempt)
                if attempt["success"]:
                    precompile_ok = True
                    selected_warmup = warmup
                    break
                precompile_status = failure_class
                precompile_reason = reason
                if failure_class == "oom":
                    continue
                if failure_class in {"compile_backend", "timeout_compile_bound"}:
                    break
            if not precompile_ok:
                if precompile_status == "oom":
                    return (None, attempts)
                return (None, attempts)

        run_id = f"{run_prefix}_tr_{compile_mode}_{compile_max_modules}_b{train_batch}_fu{ddp_find_unused}"
        log_path = logs_dir / f"{run_id}.log"
        probe_env = dict(base_env)
        probe_env.update(
            {
                "RUN_ID": run_id,
                "PRECOMPILE_ONLY": "0",
                "COMPILE_MODE": compile_mode,
                "COMPILE_TARGET": compile_target,
                "COMPILE_MAX_MODULES": str(compile_max_modules),
                "TRAIN_BATCH_TOKENS": str(train_batch),
                "COMPILER_WARMUP_BATCH_TOKENS": str(selected_warmup),
                "DDP_FIND_UNUSED_PARAMETERS": str(ddp_find_unused),
                "VAL_LOSS_EVERY": "0",
                "SLIDING_EVAL": "0",
                "TRAIN_LOG_EVERY": "1",
            }
        )
        result = run_probe(
            root=root,
            env=probe_env,
            nproc=nproc,
            log_path=log_path,
            timeout_seconds=smoke_seconds,
            min_success_steps=min_success_steps,
            stage="train",
        )
        success = bool(result["success_gate"])
        failure_class, reason = classify_failure(
            result["log_text"],
            timed_out=result["timed_out"],
            step_lines=result["step_lines"],
            stage="train",
        )
        attempt = {
            "stage": "train",
            "run_id": run_id,
            "compile_mode": compile_mode,
            "compile_target": compile_target,
            "compile_max_modules": compile_max_modules,
            "train_batch_tokens": train_batch,
            "compiler_warmup_batch_tokens": selected_warmup,
            "ddp_find_unused_parameters": ddp_find_unused,
            "elapsed_s": result["elapsed_s"],
            "returncode": result["returncode"],
            "success": success,
            "failure_class": "ok" if success else failure_class,
            "first_failing_line": "" if success else reason,
            "last_successful_step": result["last_step"],
            "last_avg_ms": result["last_avg_ms"],
            "compile_summary": result["compile_summary"],
            "log_path": result["log_path"],
        }
        attempts.append(attempt)
        if success:
            selected_env = {
                "TRAIN_BATCH_TOKENS": str(train_batch),
                "COMPILER_WARMUP_BATCH_TOKENS": str(selected_warmup),
                "COMPILE_MODE": compile_mode,
                "COMPILE_TARGET": compile_target,
                "COMPILE_MAX_MODULES": str(compile_max_modules),
                "DDP_FIND_UNUSED_PARAMETERS": str(ddp_find_unused),
            }
            return (selected_env, attempts)
        if failure_class == "ddp_unused_grad" and ddp_find_unused == 0 and try_find_unused:
            continue
        return (None, attempts)
    return (None, attempts)


def tune_profile(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root).resolve()
    logs_dir = root / args.logs_dir
    logs_dir.mkdir(parents=True, exist_ok=True)

    hw = detect_hardware(args.nproc, os.environ.get("CUDA_VISIBLE_DEVICES"))
    env = os.environ.copy()
    env["AUTO_TUNE_FREE_MB_MIN"] = str(hw.min_free_mb)
    env["AUTO_TUNE_TOTAL_MB_MIN"] = str(hw.min_total_mb)
    signature = build_signature(hw, args.profile, env)
    sig_key = signature_key(signature)
    cache_dir = Path(env.get("AUTO_TUNE_CACHE_DIR", str(root / "logs" / "auto_tune_cache")))
    cache_path = cache_dir / f"{sig_key}.json"
    cached = load_cache(cache_path, signature, force=env.get("AUTO_TUNE_FORCE", "0") == "1")
    if cached:
        return cached

    heuristic_batch = int(env.get("TRAIN_BATCH_TOKENS", str(default_train_batch_tokens(hw.min_free_mb))))
    heuristic_warmup = int(env.get("COMPILER_WARMUP_BATCH_TOKENS", str(default_warmup_batch_tokens(hw.min_free_mb))))
    batch_candidates = build_batch_candidates(heuristic_batch, env, args.profile)
    quick_mode = auto_tune_quick(env)
    min_success_steps = int(env.get("AUTO_TUNE_MIN_SUCCESS_STEPS", "1" if quick_mode else "2"))
    precompile_seconds = int(env.get("AUTO_TUNE_MAX_PRECOMPILE_SECONDS", "60" if quick_mode else "240"))
    smoke_seconds = int(env.get("AUTO_TUNE_MAX_SMOKE_SECONDS", "20" if quick_mode else "90"))
    try_find_unused = env.get("AUTO_TUNE_TRY_FIND_UNUSED", "1") == "1"
    attempts: list[dict[str, Any]] = []
    selected_env: dict[str, str] | None = None

    base_env = build_base_env(root, args.run_id, env, args.nproc)
    for batch in batch_candidates:
        warmup_candidates = build_warmup_candidates(heuristic_warmup, batch, env)
        for compile_cfg in compile_ladder(args.profile, env):
            candidate_env, candidate_attempts = attempt_candidate(
                root=root,
                base_env=base_env,
                nproc=args.nproc,
                logs_dir=logs_dir,
                precompile_seconds=precompile_seconds,
                smoke_seconds=smoke_seconds,
                min_success_steps=min_success_steps,
                compile_cfg=compile_cfg,
                train_batch=batch,
                warmup_candidates=warmup_candidates,
                try_find_unused=try_find_unused,
                run_prefix=f"{args.run_id}_{sig_key}",
            )
            attempts.extend(candidate_attempts)
            if candidate_env:
                selected_env = candidate_env
                break
            failure_class = candidate_attempts[-1]["failure_class"] if candidate_attempts else "unknown"
            if failure_class == "oom":
                break
        if selected_env:
            break

    if not selected_env:
        selected_env = {
            "TRAIN_BATCH_TOKENS": str(batch_candidates[-1]),
            "COMPILER_WARMUP_BATCH_TOKENS": str(min(batch_candidates[-1], max(heuristic_warmup, 2048))),
            "COMPILE_MODE": "none",
            "COMPILE_TARGET": "full",
            "COMPILE_MAX_MODULES": "0",
            "DDP_FIND_UNUSED_PARAMETERS": "1" if args.profile in {"diagnostic", "convergence"} else "0",
        }

    payload = {
        "signature": signature,
        "hardware": asdict(hw),
        "selected_env": selected_env,
        "attempts": attempts,
        "heuristic_batch_tokens": heuristic_batch,
        "heuristic_warmup_batch_tokens": heuristic_warmup,
        "batch_candidates": batch_candidates,
        "cached": False,
    }
    write_cache(cache_path, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Hardware-aware launcher auto-tuner")
    parser.add_argument("--root", default=".")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--nproc", type=int, required=True)
    parser.add_argument("--profile", default=os.environ.get("AUTO_TUNE_PROFILE", "competition"))
    parser.add_argument("--logs-dir", default="logs/auto_tune")
    parser.add_argument("--emit-env-file", required=True)
    parser.add_argument("--emit-json-file", required=True)
    args = parser.parse_args()

    payload = tune_profile(args)
    env_file = Path(args.emit_env_file)
    json_file = Path(args.emit_json_file)
    write_env_file(env_file, payload["selected_env"])
    json_file.parent.mkdir(parents=True, exist_ok=True)
    json_file.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    hw = payload["hardware"]
    chosen = payload["selected_env"]
    print(
        "auto_tune:selected "
        f"gpu={hw['gpu_names'][0]}x{hw['gpu_count']} "
        f"batch={chosen['TRAIN_BATCH_TOKENS']} "
        f"warmup_batch={chosen['COMPILER_WARMUP_BATCH_TOKENS']} "
        f"compile={chosen['COMPILE_MODE']} "
        f"target={chosen['COMPILE_TARGET']} "
        f"max_modules={chosen['COMPILE_MAX_MODULES']} "
        f"ddp_find_unused={chosen['DDP_FIND_UNUSED_PARAMETERS']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
