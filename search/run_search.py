from __future__ import annotations

from argparse import ArgumentParser
from collections.abc import Iterable
from dataclasses import asdict
from datetime import datetime, timezone
import json
import re
import statistics
from pathlib import Path
import sys
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from search.config import SearchConfig, SearchSettings, load_search_config
from search.log_parser import ARTIFACT_LIMIT_BYTES, ParsedRunLog, StepValidation, parse_run_log
from search.protein_lite import ProteinLite
from search.runner import PreparedRun, build_prepared_run, launch_run

LIVE_STEP_RE = re.compile(
    r"^step:(?P<step>\d+)/(?P<total>\d+) val_loss:(?P<loss>[-+0-9.eEnNaA]+) "
    r"val_bpb:(?P<bpb>[-+0-9.eEnNaA]+) train_time:(?P<train_time>\d+)ms"
)
LIVE_ROUNDTRIP_RE = re.compile(
    r"^roundtrip_(?P<label>int(?P<bits>6|8)\S*) val_loss:(?P<loss>[-+0-9.eEnNaA]+) "
    r"val_bpb:(?P<bpb>[-+0-9.eEnNaA]+) eval_time:(?P<eval_time>\d+)ms"
)
LIVE_SLIDING_START_RE = re.compile(
    r"^sliding_window:start total_windows:(?P<total>\d+) "
    r"rank0_windows:(?P<rank0>\d+) world_size:(?P<world_size>\d+) "
    r"seq_len:(?P<seq_len>\d+) stride:(?P<stride>\d+) log_every:(?P<log_every>\d+)"
)
LIVE_SLIDING_PROGRESS_RE = re.compile(
    r"^sliding_window:progress rank0_windows:(?P<rank_done>\d+)/(?P<rank_total>\d+) "
    r"approx_global_windows:(?P<global_done>\d+)/(?P<global_total>\d+) "
    r"approx_pct:(?P<pct>[-+0-9.]+) scored_tokens:(?P<scored_tokens>\d+) "
    r"elapsed:(?P<elapsed>[-+0-9.]+)s windows_per_sec:(?P<wps>[-+0-9.]+) "
    r"scored_tokens_per_sec:(?P<tps>[-+0-9.]+)"
)
LIVE_SLIDING_FINAL_RE = re.compile(
    r"^sliding_window val_loss:(?P<loss>[-+0-9.eEnNaA]+) val_bpb:(?P<bpb>[-+0-9.eEnNaA]+) "
    r"seq_len:(?P<seq_len>\d+) stride:(?P<stride>\d+) eval_time:(?P<eval_time>\d+)ms"
)
LIVE_QUANT_SUMMARY_RE = re.compile(
    r"^quant_summary int8_bpb:(?P<int8_bpb>[-+0-9.eEnNaA]+) "
    r"int6_bpb:(?P<int6_bpb>[-+0-9.eEnNaA]+) int8_sz:(?P<int8_sz>\d+) int6_sz:(?P<int6_sz>\d+)"
)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def render_scalar(value: Any) -> Any:
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def proxy_gap(default_gap: float, previous_runs: Iterable[dict[str, Any]]) -> float:
    gaps = [
        run["final_int6_bpb"] - run["terminal_prequant_bpb"]
        for run in previous_runs
        if run.get("eligible") and run.get("final_int6_bpb") is not None and run.get("terminal_prequant_bpb") is not None
    ]
    return statistics.median(gaps) if gaps else default_gap


def build_run_observations(
    parsed: ParsedRunLog,
    *,
    params: dict[str, Any],
    run_id: str,
    run_index: int,
    prior_runs: list[dict[str, Any]],
    default_proxy_int6_gap: float,
) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    gap = proxy_gap(default_proxy_int6_gap, prior_runs)
    objective_bpb = parsed.objective_bpb()
    objective_source = parsed.objective_source()

    for step in parsed.step_validations:
        obs = {
            "run_id": run_id,
            "run_index": run_index,
            "params": {key: render_scalar(value) for key, value in params.items()},
            "step": step.step,
            "total_steps": step.total_steps,
            "train_time_ms": step.train_time_ms,
            "prequant_val_bpb": step.val_bpb,
            "score": -(step.val_bpb + gap),
            "target_bpb": step.val_bpb + gap,
            "source": "proxy_int6",
            "eligible": False,
        }
        observations.append(obs)

    if observations and objective_bpb is not None and objective_source is not None:
        final = observations[-1].copy()
        final["score"] = -objective_bpb
        final["target_bpb"] = objective_bpb
        final["source"] = objective_source
        final["eligible"] = not parsed.oversize_int6 and parsed.status == "completed"
        observations[-1] = final

    return observations


def summarize_run(
    *,
    prepared: PreparedRun,
    parsed: ParsedRunLog,
    resolved_log_path: Path | None,
    run_index: int,
    params: dict[str, Any],
    suggestion_info: dict[str, float],
    return_code: int,
    started_at: str,
    finished_at: str,
) -> dict[str, Any]:
    terminal = parsed.terminal_validation
    final_int6 = parsed.roundtrip_int6
    objective_bpb = parsed.objective_bpb()
    objective_source = parsed.objective_source()
    return {
        "run_id": prepared.run_id,
        "run_index": run_index,
        "status": parsed.status if return_code == 0 else "launch_failed",
        "return_code": return_code,
        "started_at": started_at,
        "finished_at": finished_at,
        "params": {key: render_scalar(value) for key, value in params.items()},
        "suggestion_info": suggestion_info,
        "command": prepared.display_command,
        "log_path": str(resolved_log_path or prepared.log_path),
        "stdout_path": str(prepared.stdout_path),
        "terminal_prequant_bpb": terminal.val_bpb if terminal else None,
        "final_train_time_ms": terminal.train_time_ms if terminal else None,
        "final_int6_bpb": final_int6.val_bpb if final_int6 else None,
        "objective_source": objective_source,
        "objective_bpb": objective_bpb,
        "sliding_window_bpb": parsed.sliding_window_int6.val_bpb if parsed.sliding_window_int6 else None,
        "sliding_window_seq_len": parsed.sliding_window_int6.seq_len if parsed.sliding_window_int6 else None,
        "sliding_window_stride": parsed.sliding_window_int6.stride if parsed.sliding_window_int6 else None,
        "int6_artifact_bytes": parsed.int6_artifact_bytes,
        "eligible": bool(objective_bpb is not None and not parsed.oversize_int6 and parsed.status == "completed" and return_code == 0),
        "failure_reason": parsed.failure_reason if return_code == 0 else f"launcher exited with {return_code}",
        "stopped_early": parsed.stopped_early,
        "has_nan": parsed.has_nan,
        "parsed_log": parsed.to_dict(),
    }


def resolve_run_log_path(prepared: PreparedRun, *, workdir: Path) -> Path | None:
    if prepared.log_path.exists():
        return prepared.log_path
    fallback = (workdir / "logs" / f"{prepared.run_id}.txt").resolve()
    if fallback.exists():
        return fallback
    return None


def restore_optimizer(optimizer: ProteinLite, observation_records: list[dict[str, Any]]) -> None:
    for record in observation_records:
        optimizer.observe(
            record["params"],
            score=float(record["score"]),
            cost=float(record["train_time_ms"]),
            metadata=record,
        )


def select_best_run(run_records: list[dict[str, Any]]) -> dict[str, Any] | None:
    eligible = [run for run in run_records if run.get("eligible")]
    if not eligible:
        return None
    return min(eligible, key=lambda record: record["objective_bpb"])


def write_best(best_record: dict[str, Any] | None, *, path: Path) -> None:
    if best_record is None:
        write_json(path, {"best": None, "updated_at": datetime.now(timezone.utc).isoformat()})
        return
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "best": {
            "run_id": best_record["run_id"],
            "run_index": best_record["run_index"],
            "objective_source": best_record["objective_source"],
            "objective_bpb": best_record["objective_bpb"],
            "final_int6_bpb": best_record["final_int6_bpb"],
            "terminal_prequant_bpb": best_record["terminal_prequant_bpb"],
            "final_train_time_ms": best_record["final_train_time_ms"],
            "int6_artifact_bytes": best_record["int6_artifact_bytes"],
            "params": best_record["params"],
            "log_path": best_record["log_path"],
            "stdout_path": best_record["stdout_path"],
            "rerun_command": best_record["command"],
        },
    }
    write_json(path, payload)


def initialize_live_status(
    *,
    prepared: PreparedRun,
    run_index: int,
    params: dict[str, Any],
    suggestion_info: dict[str, float],
) -> dict[str, Any]:
    return {
        "run_id": prepared.run_id,
        "run_index": run_index,
        "status": "running",
        "phase": "training",
        "params": {key: render_scalar(value) for key, value in params.items()},
        "suggestion_info": suggestion_info,
        "command": prepared.display_command,
        "log_path": str(prepared.log_path),
        "stdout_path": str(prepared.stdout_path),
        "status_path": str(prepared.status_path),
        "last_update_at": datetime.now(timezone.utc).isoformat(),
        "latest_step_validation": None,
        "latest_roundtrip": {},
        "quant_summary": None,
        "sliding_window": None,
        "last_line": None,
    }


def update_live_status_from_line(status: dict[str, Any], line: str) -> None:
    status["last_update_at"] = datetime.now(timezone.utc).isoformat()
    status["last_line"] = line

    if match := LIVE_STEP_RE.match(line):
        status["phase"] = "training"
        status["latest_step_validation"] = {
            "step": int(match.group("step")),
            "total_steps": int(match.group("total")),
            "val_loss": float(match.group("loss")),
            "val_bpb": float(match.group("bpb")),
            "train_time_ms": int(match.group("train_time")),
        }
        return

    if match := LIVE_ROUNDTRIP_RE.match(line):
        status["phase"] = "quant_eval"
        roundtrips = dict(status.get("latest_roundtrip") or {})
        roundtrips[match.group("label")] = {
            "bits": int(match.group("bits")),
            "val_loss": float(match.group("loss")),
            "val_bpb": float(match.group("bpb")),
            "eval_time_ms": int(match.group("eval_time")),
        }
        status["latest_roundtrip"] = roundtrips
        return

    if match := LIVE_QUANT_SUMMARY_RE.match(line):
        status["phase"] = "quant_eval"
        status["quant_summary"] = {
            "int8_bpb": float(match.group("int8_bpb")),
            "int6_bpb": float(match.group("int6_bpb")),
            "int8_sz": int(match.group("int8_sz")),
            "int6_sz": int(match.group("int6_sz")),
        }
        return

    if match := LIVE_SLIDING_START_RE.match(line):
        status["phase"] = "sliding_eval"
        status["sliding_window"] = {
            "status": "running",
            "total_windows": int(match.group("total")),
            "rank0_windows": int(match.group("rank0")),
            "world_size": int(match.group("world_size")),
            "seq_len": int(match.group("seq_len")),
            "stride": int(match.group("stride")),
            "log_every": int(match.group("log_every")),
            "progress": None,
            "final_metric": None,
        }
        return

    if match := LIVE_SLIDING_PROGRESS_RE.match(line):
        status["phase"] = "sliding_eval"
        sliding = dict(status.get("sliding_window") or {})
        progress = {
            "rank0_windows_done": int(match.group("rank_done")),
            "rank0_windows_total": int(match.group("rank_total")),
            "approx_global_windows_done": int(match.group("global_done")),
            "approx_global_windows_total": int(match.group("global_total")),
            "approx_pct": float(match.group("pct")),
            "scored_tokens": int(match.group("scored_tokens")),
            "elapsed_s": float(match.group("elapsed")),
            "windows_per_sec": float(match.group("wps")),
            "scored_tokens_per_sec": float(match.group("tps")),
        }
        remaining_windows = progress["approx_global_windows_total"] - progress["approx_global_windows_done"]
        if progress["windows_per_sec"] > 0:
            progress["eta_s"] = remaining_windows / progress["windows_per_sec"]
        sliding["status"] = "running"
        sliding["progress"] = progress
        status["sliding_window"] = sliding
        return

    if match := LIVE_SLIDING_FINAL_RE.match(line):
        status["phase"] = "completed"
        sliding = dict(status.get("sliding_window") or {})
        sliding["status"] = "completed"
        sliding["final_metric"] = {
            "val_loss": float(match.group("loss")),
            "val_bpb": float(match.group("bpb")),
            "seq_len": int(match.group("seq_len")),
            "stride": int(match.group("stride")),
            "eval_time_ms": int(match.group("eval_time")),
        }
        status["sliding_window"] = sliding


def write_live_status(status: dict[str, Any], *, current_path: Path, per_run_path: Path) -> None:
    write_json(current_path, status)
    write_json(per_run_path, status)


def make_optimizer(config: SearchConfig) -> ProteinLite:
    return ProteinLite(
        config.search_space,
        seed=config.search.seed,
        warm_start_suggestions=config.search.warm_start_suggestions,
        candidate_samples=config.search.candidate_samples,
        max_observations=config.search.max_observations,
        suggestions_per_center=config.search.suggestions_per_center,
        prune_pareto=config.search.prune_pareto,
        gp_alpha=config.search.gp_alpha,
        target_cost_ratios=config.search.target_cost_ratios,
    )


def ensure_output_root(config: SearchConfig) -> None:
    config.search.output_root.mkdir(parents=True, exist_ok=True)


def save_resolved_config(config: SearchConfig) -> None:
    payload = {
        "config_path": str(config.path),
        "runner": {
            "workdir": str(config.runner.workdir),
            "script_path": str(config.runner.script_path),
            "python_bin": str(config.runner.python_bin),
            "activate_script": str(config.runner.activate_script) if config.runner.activate_script else None,
            "gpus": config.runner.gpus,
            "logs_dir": str(config.runner.logs_dir),
        },
        "fixed_env": config.fixed_env,
        "search_space": {key: asdict(value) for key, value in config.search_space.items()},
        "search": {
            "seed": config.search.seed,
            "max_runs": config.search.max_runs,
            "warm_start_suggestions": config.search.warm_start_suggestions,
            "candidate_samples": config.search.candidate_samples,
            "max_observations": config.search.max_observations,
            "prune_pareto": config.search.prune_pareto,
            "suggestions_per_center": config.search.suggestions_per_center,
            "gp_alpha": config.search.gp_alpha,
            "default_proxy_int6_gap": config.search.default_proxy_int6_gap,
            "target_cost_ratios": list(config.search.target_cost_ratios),
            "output_root": str(config.search.output_root),
            "run_id_prefix": config.search.run_id_prefix,
        },
    }
    write_json(config.search.output_root / "resolved_config.json", payload)


def run_search(config: SearchConfig, *, max_runs: int | None, dry_run: bool) -> None:
    ensure_output_root(config)
    save_resolved_config(config)

    runs_path = config.search.output_root / "runs.jsonl"
    observations_path = config.search.output_root / "observations.jsonl"
    best_path = config.search.output_root / "best.json"
    current_run_path = config.search.output_root / "current_run.json"

    run_records = read_jsonl(runs_path)
    observation_records = read_jsonl(observations_path)
    optimizer = make_optimizer(config)
    restore_optimizer(optimizer, observation_records)

    target_runs = max_runs if max_runs is not None else config.search.max_runs
    for run_index in range(len(run_records), target_runs):
        params, suggestion_info = optimizer.suggest(run_index)
        run_id = f"{config.search.run_id_prefix}_{run_index:04d}"
        prepared = build_prepared_run(
            config.runner,
            config.fixed_env,
            params,
            run_id=run_id,
            output_root=config.search.output_root,
        )

        if dry_run:
            print(prepared.display_command)
            return

        print(f"[run {run_index}] launching {run_id}")
        live_status = initialize_live_status(
            prepared=prepared,
            run_index=run_index,
            params=params,
            suggestion_info=suggestion_info,
        )
        write_live_status(live_status, current_path=current_run_path, per_run_path=prepared.status_path)

        def on_output_line(line: str) -> None:
            update_live_status_from_line(live_status, line)
            write_live_status(live_status, current_path=current_run_path, per_run_path=prepared.status_path)

        completed = launch_run(
            prepared,
            workdir=config.runner.workdir,
            tee_to_stdout=True,
            on_output_line=on_output_line,
        )
        resolved_log_path = resolve_run_log_path(prepared, workdir=config.runner.workdir)
        if resolved_log_path is None:
            run_record = {
                "run_id": run_id,
                "run_index": run_index,
                "status": "launch_failed",
                "return_code": completed.return_code,
                "started_at": completed.started_at,
                "finished_at": completed.finished_at,
                "params": {key: render_scalar(value) for key, value in params.items()},
                "suggestion_info": suggestion_info,
                "command": prepared.display_command,
                "log_path": str(prepared.log_path),
                "stdout_path": str(prepared.stdout_path),
                "terminal_prequant_bpb": None,
                "final_train_time_ms": None,
                "final_int6_bpb": None,
                "int6_artifact_bytes": None,
                "eligible": False,
                "failure_reason": f"log file not found after exit {completed.return_code}",
                "stopped_early": False,
                "has_nan": False,
            }
            append_jsonl(runs_path, run_record)
            run_records.append(run_record)
            write_best(select_best_run(run_records), path=best_path)
            live_status["status"] = "launch_failed"
            live_status["phase"] = "completed"
            live_status["return_code"] = completed.return_code
            live_status["finished_at"] = completed.finished_at
            live_status["failure_reason"] = run_record["failure_reason"]
            write_live_status(live_status, current_path=current_run_path, per_run_path=prepared.status_path)
            continue

        parsed = parse_run_log(resolved_log_path)
        run_record = summarize_run(
            prepared=prepared,
            parsed=parsed,
            resolved_log_path=resolved_log_path,
            run_index=run_index,
            params=params,
            suggestion_info=suggestion_info,
            return_code=completed.return_code,
            started_at=completed.started_at,
            finished_at=completed.finished_at,
        )
        append_jsonl(runs_path, run_record)
        run_records.append(run_record)
        live_status["status"] = run_record["status"]
        live_status["phase"] = "completed"
        live_status["return_code"] = completed.return_code
        live_status["finished_at"] = completed.finished_at
        live_status["terminal_prequant_bpb"] = run_record["terminal_prequant_bpb"]
        live_status["final_int6_bpb"] = run_record["final_int6_bpb"]
        live_status["objective_source"] = run_record["objective_source"]
        live_status["objective_bpb"] = run_record["objective_bpb"]
        live_status["int6_artifact_bytes"] = run_record["int6_artifact_bytes"]
        live_status["eligible"] = run_record["eligible"]
        live_status["failure_reason"] = run_record["failure_reason"]
        write_live_status(live_status, current_path=current_run_path, per_run_path=prepared.status_path)

        observations = build_run_observations(
            parsed,
            params=params,
            run_id=run_id,
            run_index=run_index,
            prior_runs=run_records[:-1],
            default_proxy_int6_gap=config.search.default_proxy_int6_gap,
        )
        for record in observations:
            optimizer.observe(record["params"], score=record["score"], cost=record["train_time_ms"], metadata=record)
            append_jsonl(observations_path, record)
            observation_records.append(record)

        write_best(select_best_run(run_records), path=best_path)
        if run_record["final_int6_bpb"] is None:
            print(f"[run {run_index}] finished with no final int6 metric")
        else:
            status = "eligible" if run_record["eligible"] else run_record["status"]
            print(
                f"[run {run_index}] {status} "
                f"pre={run_record['terminal_prequant_bpb']:.4f} "
                f"obj={run_record['objective_bpb']:.4f}({run_record['objective_source']}) "
                f"int6={run_record['final_int6_bpb']:.4f} "
                f"size={run_record['int6_artifact_bytes']}"
            )


def main() -> None:
    parser = ArgumentParser(description="Run a standalone Parameter Golf search.")
    parser.add_argument("--config", required=True, help="YAML search config")
    parser.add_argument("--max-runs", type=int, default=None, help="Override total run count")
    parser.add_argument("--dry-run", action="store_true", help="Print the next command and exit")
    args = parser.parse_args()

    config = load_search_config(args.config)
    run_search(config, max_runs=args.max_runs, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
