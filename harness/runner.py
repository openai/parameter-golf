from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Any

from .code_mutations import materialize_script_for_run
from .common import RUNS_DIR, ensure_lab_layout, preferred_python, utc_now_iso, write_json
from .history import record_evidence_tier, record_is_planner_eligible
from .log_parser import parse_log
from .preflight import run_preflight


def _safe_log_name(run_id: str) -> list[Path]:
    return [Path(f"{run_id}.txt"), Path("logs") / f"{run_id}.txt"]


def _discover_log_path(run_dir: Path, run_id: str) -> Path | None:
    for rel in _safe_log_name(run_id):
        candidate = run_dir / rel
        if candidate.is_file():
            return candidate
    return None


def _artifact_paths(run_dir: Path, run_id: str, script_path: Path) -> tuple[Path | None, Path | None]:
    if script_path.name == "train_gpt_mlx.py":
        raw = run_dir / f"{run_id}_mlx_model.npz"
        quant = run_dir / f"{run_id}_mlx_model.int8.ptz"
    else:
        raw = run_dir / "final_model.pt"
        quant = run_dir / "final_model.int8.ptz"
    return (raw if raw.is_file() else None, quant if quant.is_file() else None)


def _build_command(spec: dict[str, Any], script_path: str | None = None) -> list[str]:
    launcher = spec["launcher"]
    script = script_path or spec["script"]
    if launcher == "python":
        return [preferred_python(), script]
    if launcher == "torchrun":
        nproc = str(spec.get("nproc_per_node", 1))
        return [preferred_python(), "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={nproc}", script]
    raise ValueError(f"Unsupported launcher {launcher!r}")


def _latest_mtime(paths: list[Path]) -> float | None:
    mtimes = [path.stat().st_mtime for path in paths if path.is_file()]
    return max(mtimes) if mtimes else None


def _script_code_bytes(script_path: str | Path) -> int | None:
    path = Path(script_path)
    if not path.is_file():
        return None
    return path.stat().st_size


def _base_metrics(*, code_bytes: int | None, code_bytes_delta: int | None = 0) -> dict[str, Any]:
    return {
        "final_roundtrip_val_bpb": None,
        "last_pre_quant_val_bpb": None,
        "best_val_bpb": None,
        "quant_gap_bpb": None,
        "serialized_model_int8_zlib_bytes": None,
        "total_submission_size_int8_zlib_bytes": None,
        "raw_model_bytes": None,
        "code_bytes": code_bytes,
        "code_bytes_delta": code_bytes_delta,
        "metrics_valid": False,
        "parse_warnings": [],
        "has_training_signal": False,
        "has_validation_signal": False,
        "has_final_roundtrip_signal": False,
        "has_dataset_signal": False,
        "log_signature": None,
    }


def run_experiment(spec: dict[str, Any], *, dry_run: bool = False) -> dict[str, Any]:
    ensure_lab_layout()
    run_dir = RUNS_DIR / spec["experiment_id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    spec = dict(spec)
    spec["run_dir"] = str(run_dir)

    stdout_path = run_dir / "stdout.txt"
    env = os.environ.copy()
    env.update({k: str(v) for k, v in spec["env"].items()})
    env.setdefault("RUN_ID", spec["experiment_id"])
    env.setdefault("PYTHONUNBUFFERED", "1")
    effective_env = {k: str(v) for k, v in env.items() if k in spec["env"] or k == "RUN_ID"}
    preflight = run_preflight(spec)
    prepared_script: dict[str, Any] | None = None
    command = _build_command(spec)
    write_json(run_dir / "spec.json", spec)
    write_json(run_dir / "env.json", effective_env)
    write_json(run_dir / "preflight.json", preflight)

    started_at = utc_now_iso()
    if dry_run:
        if not preflight["can_launch"] or not preflight.get("launch_policy_ok", True):
            blocked_reason = "preflight_failed" if not preflight["can_launch"] else "challenge_not_ready"
            metrics = _base_metrics(code_bytes=_script_code_bytes(spec["script"]))
            result = {
                "status": "blocked",
                "run_state": "blocked",
                "started_at": started_at,
                "completed_at": utc_now_iso(),
                "returncode": 2 if blocked_reason == "preflight_failed" else 3,
                "duration_sec": 0.0,
                "run_dir": str(run_dir),
                "stdout_path": str(stdout_path),
                "log_path": None,
                "command": " ".join(command),
                "materialized_script_path": None,
                "raw_model_path": None,
                "quant_model_path": None,
                "effective_env": effective_env,
                "comparability": preflight["comparability"],
                "failure_reason": blocked_reason,
                "timed_out": False,
                "failure_stage": "preflight",
                "train_started": False,
                "final_eval_seen": False,
                "metrics_valid": False,
                "planner_eligible": False,
                "evidence_tier": "invalid",
                "parse_warnings": [],
            }
            record = {"spec": spec, "result": result, "metrics": metrics, "preflight": preflight}
            write_json(run_dir / "summary.json", record)
            return record

        try:
            prepared_script = materialize_script_for_run(spec["script"], run_dir, spec.get("code_mutation"))
        except Exception as exc:
            metrics = _base_metrics(code_bytes=_script_code_bytes(spec["script"]))
            metrics["parse_warnings"] = [f"materialize_failed:{type(exc).__name__}"]
            result = {
                "status": "blocked",
                "run_state": "blocked",
                "started_at": started_at,
                "completed_at": utc_now_iso(),
                "returncode": 4,
                "duration_sec": 0.0,
                "run_dir": str(run_dir),
                "stdout_path": str(stdout_path),
                "log_path": None,
                "command": " ".join(command),
                "materialized_script_path": None,
                "raw_model_path": None,
                "quant_model_path": None,
                "effective_env": effective_env,
                "comparability": preflight["comparability"],
                "failure_reason": "code_materialization_failed",
                "timed_out": False,
                "failure_stage": "launch",
                "train_started": False,
                "final_eval_seen": False,
                "metrics_valid": False,
                "planner_eligible": False,
                "evidence_tier": "invalid",
                "parse_warnings": metrics["parse_warnings"],
            }
            record = {"spec": spec, "result": result, "metrics": metrics, "preflight": preflight}
            write_json(run_dir / "summary.json", record)
            return record
        spec["materialized_script"] = prepared_script["materialized_script_path"]
        spec["source_script_hash"] = prepared_script["source_hash"]
        if prepared_script.get("code_mutation") is not None:
            spec["code_mutation"] = prepared_script["code_mutation"]
        command = _build_command(spec, spec["materialized_script"])
        write_json(run_dir / "spec.json", spec)
        write_json(
            run_dir / "code_mutation.json",
            {
                "source_script": spec["script"],
                "materialized_script": spec["materialized_script"],
                "source_hash": prepared_script["source_hash"],
                "source_code_bytes": prepared_script["source_code_bytes"],
                "materialized_code_bytes": prepared_script["materialized_code_bytes"],
                "code_bytes_delta": prepared_script["code_bytes_delta"],
                "code_mutation": prepared_script.get("code_mutation"),
            },
        )
        metrics = _base_metrics(
            code_bytes=prepared_script["materialized_code_bytes"],
            code_bytes_delta=prepared_script["code_bytes_delta"],
        )
        result = {
            "status": "dry-run",
            "run_state": "dry_run",
            "started_at": started_at,
            "completed_at": utc_now_iso(),
            "returncode": 0,
            "duration_sec": 0.0,
            "run_dir": str(run_dir),
            "stdout_path": str(stdout_path),
            "log_path": None,
            "command": " ".join(command),
            "materialized_script_path": spec["materialized_script"],
            "raw_model_path": None,
            "quant_model_path": None,
            "effective_env": effective_env,
            "comparability": "dry-run",
            "failure_reason": None,
            "timed_out": False,
            "failure_stage": "none",
            "train_started": False,
            "final_eval_seen": False,
            "metrics_valid": False,
            "planner_eligible": False,
            "evidence_tier": "invalid",
            "parse_warnings": [],
        }
        record = {"spec": spec, "result": result, "metrics": metrics, "preflight": preflight}
        write_json(run_dir / "summary.json", record)
        return record

    if not preflight["can_launch"]:
        metrics = _base_metrics(code_bytes=_script_code_bytes(spec["script"]))
        result = {
            "status": "blocked",
            "run_state": "blocked",
            "started_at": started_at,
            "completed_at": utc_now_iso(),
            "returncode": 2,
            "duration_sec": 0.0,
            "run_dir": str(run_dir),
            "stdout_path": str(stdout_path),
            "log_path": None,
            "command": " ".join(command),
            "materialized_script_path": spec.get("materialized_script"),
            "raw_model_path": None,
            "quant_model_path": None,
            "effective_env": effective_env,
            "comparability": preflight["comparability"],
            "failure_reason": "preflight_failed",
            "timed_out": False,
            "failure_stage": "preflight",
            "train_started": False,
            "final_eval_seen": False,
            "metrics_valid": False,
            "planner_eligible": False,
            "evidence_tier": "invalid",
            "parse_warnings": [],
        }
        record = {"spec": spec, "result": result, "metrics": metrics, "preflight": preflight}
        write_json(run_dir / "summary.json", record)
        return record

    if not preflight.get("launch_policy_ok", True):
        metrics = _base_metrics(code_bytes=_script_code_bytes(spec["script"]))
        result = {
            "status": "blocked",
            "run_state": "blocked",
            "started_at": started_at,
            "completed_at": utc_now_iso(),
            "returncode": 3,
            "duration_sec": 0.0,
            "run_dir": str(run_dir),
            "stdout_path": str(stdout_path),
            "log_path": None,
            "command": " ".join(command),
            "materialized_script_path": spec.get("materialized_script"),
            "raw_model_path": None,
            "quant_model_path": None,
            "effective_env": effective_env,
            "comparability": preflight["comparability"],
            "failure_reason": "challenge_not_ready",
            "timed_out": False,
            "failure_stage": "preflight",
            "train_started": False,
            "final_eval_seen": False,
            "metrics_valid": False,
            "planner_eligible": False,
            "evidence_tier": "invalid",
            "parse_warnings": [],
        }
        record = {"spec": spec, "result": result, "metrics": metrics, "preflight": preflight}
        write_json(run_dir / "summary.json", record)
        return record

    try:
        prepared_script = materialize_script_for_run(spec["script"], run_dir, spec.get("code_mutation"))
    except Exception as exc:
        metrics = _base_metrics(code_bytes=_script_code_bytes(spec["script"]))
        metrics["parse_warnings"] = [f"materialize_failed:{type(exc).__name__}"]
        result = {
            "status": "blocked",
            "run_state": "blocked",
            "started_at": started_at,
            "completed_at": utc_now_iso(),
            "returncode": 4,
            "duration_sec": 0.0,
            "run_dir": str(run_dir),
            "stdout_path": str(stdout_path),
            "log_path": None,
            "command": " ".join(command),
            "materialized_script_path": None,
            "raw_model_path": None,
            "quant_model_path": None,
            "effective_env": effective_env,
            "comparability": preflight["comparability"],
            "failure_reason": "code_materialization_failed",
            "timed_out": False,
            "failure_stage": "launch",
            "train_started": False,
            "final_eval_seen": False,
            "metrics_valid": False,
            "planner_eligible": False,
            "evidence_tier": "invalid",
            "parse_warnings": metrics["parse_warnings"],
        }
        record = {"spec": spec, "result": result, "metrics": metrics, "preflight": preflight}
        write_json(run_dir / "summary.json", record)
        return record
    spec["materialized_script"] = prepared_script["materialized_script_path"]
    spec["source_script_hash"] = prepared_script["source_hash"]
    if prepared_script.get("code_mutation") is not None:
        spec["code_mutation"] = prepared_script["code_mutation"]
    command = _build_command(spec, spec["materialized_script"])
    write_json(run_dir / "spec.json", spec)
    write_json(
        run_dir / "code_mutation.json",
        {
            "source_script": spec["script"],
            "materialized_script": spec["materialized_script"],
            "source_hash": prepared_script["source_hash"],
            "source_code_bytes": prepared_script["source_code_bytes"],
            "materialized_code_bytes": prepared_script["materialized_code_bytes"],
            "code_bytes_delta": prepared_script["code_bytes_delta"],
            "code_mutation": prepared_script.get("code_mutation"),
        },
    )

    t0 = time.perf_counter()
    started_monotonic = time.monotonic()
    timed_out = False
    idle_killed = False
    failure_reason = None
    idle_timeout_seconds = spec.get("idle_timeout_seconds")
    with stdout_path.open("w", encoding="utf-8") as stdout_file:
        stdout_file.write(f"# started_at={started_at}\n")
        stdout_file.write(f"# command={' '.join(command)}\n")
        stdout_file.flush()
        process = subprocess.Popen(
            command,
            cwd=run_dir,
            env=env,
            stdout=stdout_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        last_output_at = time.time()
        timeout_seconds = spec.get("run_timeout_seconds")
        poll_interval = 5.0
        if timeout_seconds is not None:
            poll_interval = min(poll_interval, max(float(timeout_seconds) / 20.0, 0.25))
        if idle_timeout_seconds is not None:
            poll_interval = min(poll_interval, max(float(idle_timeout_seconds) / 5.0, 0.25))
        while True:
            returncode = process.poll()
            now = time.time()
            log_path = _discover_log_path(run_dir, effective_env["RUN_ID"])
            latest_output_mtime = _latest_mtime([stdout_path] + ([log_path] if log_path else []))
            if latest_output_mtime is not None:
                last_output_at = max(last_output_at, latest_output_mtime)

            if returncode is not None:
                process = subprocess.CompletedProcess(command, returncode=returncode)
                break

            if timeout_seconds is not None and (time.monotonic() - started_monotonic) >= float(timeout_seconds):
                timed_out = True
                failure_reason = "run_timeout"
                stdout_file.write(f"\n# harness_timeout_seconds={timeout_seconds}\n")
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                process = subprocess.CompletedProcess(command, returncode=124)
                break

            if idle_timeout_seconds is not None and (now - last_output_at) >= float(idle_timeout_seconds):
                idle_killed = True
                failure_reason = "no_progress_timeout"
                stdout_file.write(f"\n# harness_idle_timeout_seconds={idle_timeout_seconds}\n")
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                process = subprocess.CompletedProcess(command, returncode=125)
                break

            time.sleep(poll_interval)
    duration_sec = time.perf_counter() - t0

    script_path = Path(spec["materialized_script"])
    run_id = effective_env["RUN_ID"]
    log_path = _discover_log_path(run_dir, run_id)
    raw_model_path, quant_model_path = _artifact_paths(run_dir, run_id, script_path)
    metrics = parse_log(log_path or stdout_path)
    comparability = "full-comparable"
    if spec.get("track", "").startswith("local-"):
        comparability = "smoke-only"
    if metrics.get("subset_warning"):
        comparability = "subset-only"
    code_bytes = prepared_script["materialized_code_bytes"]
    metrics.setdefault("code_bytes", code_bytes)
    metrics.setdefault("code_bytes_delta", prepared_script["code_bytes_delta"])
    if metrics.get("serialized_model_int8_zlib_bytes") is None and quant_model_path is not None:
        metrics["serialized_model_int8_zlib_bytes"] = quant_model_path.stat().st_size
    if metrics.get("raw_model_bytes") is None and raw_model_path is not None:
        metrics["raw_model_bytes"] = raw_model_path.stat().st_size
    if metrics.get("total_submission_size_int8_zlib_bytes") is None and metrics.get("serialized_model_int8_zlib_bytes") is not None:
        metrics["total_submission_size_int8_zlib_bytes"] = int(metrics["serialized_model_int8_zlib_bytes"]) + code_bytes
    metrics["code_bytes"] = code_bytes

    train_started = bool(metrics.get("has_training_signal"))
    final_eval_seen = bool(metrics.get("has_final_roundtrip_signal"))
    metrics_valid = bool(metrics.get("metrics_valid"))
    parse_warnings = list(metrics.get("parse_warnings") or [])
    if process.returncode == 0:
        run_state = "completed" if metrics_valid else "completed_invalid_metrics"
        status = "completed" if metrics_valid else "failed"
        failure_stage = "none" if metrics_valid else "post_train_parse"
    else:
        run_state = "failed_mid_run" if train_started else "failed_pre_train"
        status = "failed"
        failure_stage = "during_train" if train_started else "before_train_log"
    result = {
        "status": status,
        "run_state": run_state,
        "started_at": started_at,
        "completed_at": utc_now_iso(),
        "returncode": process.returncode,
        "duration_sec": round(duration_sec, 3),
        "run_dir": str(run_dir),
        "stdout_path": str(stdout_path),
        "log_path": str(log_path) if log_path else None,
        "command": " ".join(command),
        "materialized_script_path": spec["materialized_script"],
        "raw_model_path": str(raw_model_path) if raw_model_path else None,
        "quant_model_path": str(quant_model_path) if quant_model_path else None,
        "effective_env": effective_env,
        "comparability": comparability,
        "timed_out": timed_out,
        "idle_killed": idle_killed,
        "failure_reason": failure_reason or ("trainer_failed" if process.returncode != 0 else None),
        "failure_stage": failure_stage,
        "train_started": train_started,
        "final_eval_seen": final_eval_seen,
        "metrics_valid": metrics_valid,
        "parse_warnings": parse_warnings,
    }
    record = {"spec": spec, "result": result, "metrics": metrics, "preflight": preflight}
    result["evidence_tier"] = record_evidence_tier(record)
    result["planner_eligible"] = record_is_planner_eligible(record)
    write_json(run_dir / "summary.json", record)
    return record
