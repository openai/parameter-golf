#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.presets import PRESETS, Preset, get_preset
from scripts.check_data import check_data
from scripts.check_env import check_env


RESULTS_ROOT = ROOT / "research" / "results"
RUNS_ROOT = RESULTS_ROOT / "runs"
INDEX_JSONL = RESULTS_ROOT / "index.jsonl"
INDEX_CSV = RESULTS_ROOT / "index.csv"
FINAL_BPB_MISSING = 10**9


TRAIN_LINE_RE = re.compile(
    r"^step:(?P<step>\d+/\d+) train_loss:(?P<train_loss>[-+0-9.eE]+) "
    r"train_time:(?P<train_time_ms>[-+0-9.eE]+)ms step_avg:(?P<step_avg_ms>[-+0-9.eE]+)ms"
    r"(?: tok_s:(?P<tok_s>[-+0-9.eE]+))?$"
)
VAL_LINE_RE = re.compile(
    r"^step:(?P<step>\d+/\d+) val_loss:(?P<val_loss>[-+0-9.eE]+) val_bpb:(?P<val_bpb>[-+0-9.eE]+) "
    r"train_time:(?P<train_time_ms>[-+0-9.eE]+)ms step_avg:(?P<step_avg_ms>[-+0-9.eE]+)ms$"
)
FINAL_LINE_RE = re.compile(
    r"^final_int8_zlib_roundtrip val_loss:(?P<val_loss>[-+0-9.eE]+) "
    r"val_bpb:(?P<val_bpb>[-+0-9.eE]+) eval_time:(?P<eval_time_ms>[-+0-9.eE]+)ms$"
)
FINAL_EXACT_LINE_RE = re.compile(
    r"^final_int8_zlib_roundtrip_exact val_loss:(?P<val_loss>[-+0-9.eE]+) "
    r"val_bpb:(?P<val_bpb>[-+0-9.eE]+)$"
)
SERIALIZED_MODEL_RE = re.compile(r"^Serialized model int8\+zlib: (?P<bytes>\d+) bytes ")
SERIALIZED_MODEL_MLX_RE = re.compile(r"^serialized_model_int8_zlib:(?P<bytes>\d+) bytes ")
TOTAL_SUBMISSION_RE = re.compile(r"^Total submission size int8\+zlib: (?P<bytes>\d+) bytes$")
CODE_SIZE_RE = re.compile(r"^Code size: (?P<bytes>\d+) bytes$")


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    slug = slug.strip("._-")
    return slug or "run"


def parse_assignments(items: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected KEY=VALUE override, got {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Override key cannot be empty: {item!r}")
        parsed[key] = value
    return parsed


def iso_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def timestamp_slug() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")


def build_command(preset: Preset, nproc_per_node: int | None) -> list[str]:
    entrypoint = str(ROOT / preset.entrypoint)
    if preset.launch_mode == "python":
        return [sys.executable, entrypoint]
    if preset.launch_mode == "torchrun":
        torchrun = shutil.which("torchrun") or "torchrun"
        return [torchrun, "--standalone", f"--nproc_per_node={nproc_per_node}", entrypoint]
    raise ValueError(f"Unsupported launch mode {preset.launch_mode!r}")


def git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def git_is_dirty() -> bool | None:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return bool(result.stdout.strip())


def entrypoint_code_bytes(entrypoint: str) -> int:
    return len((ROOT / entrypoint).read_bytes())


def to_number(value: str) -> float | int | str:
    if "/" in value:
        return value
    try:
        integer = int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value
    else:
        return integer


def parse_trainer_log(log_path: Path) -> dict[str, object]:
    metrics: dict[str, object] = {
        "last_train": None,
        "last_val": None,
        "final_roundtrip": None,
        "final_roundtrip_exact": None,
        "artifact_metrics": {},
    }
    if not log_path.is_file():
        return metrics

    artifact_metrics: dict[str, object] = {}
    for raw_line in log_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        match = TRAIN_LINE_RE.match(line)
        if match:
            metrics["last_train"] = {key: to_number(value) for key, value in match.groupdict().items() if value is not None}
            continue

        match = VAL_LINE_RE.match(line)
        if match:
            metrics["last_val"] = {key: to_number(value) for key, value in match.groupdict().items()}
            continue

        match = FINAL_LINE_RE.match(line)
        if match:
            metrics["final_roundtrip"] = {key: to_number(value) for key, value in match.groupdict().items()}
            continue

        match = FINAL_EXACT_LINE_RE.match(line)
        if match:
            metrics["final_roundtrip_exact"] = {key: to_number(value) for key, value in match.groupdict().items()}
            continue

        match = SERIALIZED_MODEL_RE.match(line)
        if match:
            artifact_metrics["compressed_model_bytes_logged"] = int(match.group("bytes"))
            continue

        match = SERIALIZED_MODEL_MLX_RE.match(line)
        if match:
            artifact_metrics["compressed_model_bytes_logged"] = int(match.group("bytes"))
            continue

        match = TOTAL_SUBMISSION_RE.match(line)
        if match:
            artifact_metrics["total_submission_bytes_logged"] = int(match.group("bytes"))
            continue

        match = CODE_SIZE_RE.match(line)
        if match:
            artifact_metrics["code_bytes_logged"] = int(match.group("bytes"))
            continue

    metrics["artifact_metrics"] = artifact_metrics
    return metrics


def collect_artifacts(run_dir: Path) -> dict[str, object]:
    artifacts: dict[str, object] = {}
    for path in sorted(run_dir.glob("*")):
        if not path.is_file():
            continue
        if path.suffix in {".pt", ".ptz", ".npz", ".txt", ".json"} or path.name.endswith(".csv"):
            artifacts[path.name] = {
                "path": str(path),
                "bytes": path.stat().st_size,
            }
    return artifacts


def final_bpb(metrics: dict[str, object]) -> float | None:
    exact = metrics.get("final_roundtrip_exact")
    if isinstance(exact, dict) and "val_bpb" in exact:
        return float(exact["val_bpb"])
    rounded = metrics.get("final_roundtrip")
    if isinstance(rounded, dict) and "val_bpb" in rounded:
        return float(rounded["val_bpb"])
    last_val = metrics.get("last_val")
    if isinstance(last_val, dict) and "val_bpb" in last_val:
        return float(last_val["val_bpb"])
    return None


def regenerate_csv_index(records: list[dict[str, object]]) -> None:
    rows = []
    for record in records:
        metrics = record.get("metrics") or {}
        artifact_metrics = metrics.get("artifact_metrics") or {}
        rows.append(
            {
                "timestamp": record.get("timestamp"),
                "run_name": record.get("run_name"),
                "preset": record.get("preset"),
                "target": record.get("target"),
                "status": record.get("status"),
                "exit_code": record.get("exit_code"),
                "final_val_bpb": final_bpb(metrics) or "",
                "final_val_loss": (
                    (metrics.get("final_roundtrip_exact") or {}).get("val_loss")
                    or (metrics.get("final_roundtrip") or {}).get("val_loss")
                    or ""
                ),
                "compressed_model_bytes": record.get("compressed_model_bytes") or artifact_metrics.get("compressed_model_bytes_logged") or "",
                "submission_budget_estimate_bytes": record.get("submission_budget_estimate_bytes") or artifact_metrics.get("total_submission_bytes_logged") or "",
                "wall_clock_seconds": record.get("wall_clock_seconds"),
                "git_commit": record.get("git_commit") or "",
                "run_dir": record.get("run_dir"),
                "train_log_path": record.get("train_log_path") or "",
                "result_path": record.get("result_path"),
            }
        )

    with INDEX_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp",
                "run_name",
                "preset",
                "target",
                "status",
                "exit_code",
                "final_val_bpb",
                "final_val_loss",
                "compressed_model_bytes",
                "submission_budget_estimate_bytes",
                "wall_clock_seconds",
                "git_commit",
                "run_dir",
                "train_log_path",
                "result_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def append_index(record: dict[str, object]) -> None:
    existing: list[dict[str, object]] = []
    if INDEX_JSONL.is_file():
        with INDEX_JSONL.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    existing.append(json.loads(line))
    existing.append(record)
    with INDEX_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
    regenerate_csv_index(existing)


def print_presets() -> None:
    for preset_name in sorted(PRESETS):
        preset = PRESETS[preset_name]
        print(f"{preset.name}: {preset.description}")
        for note in preset.notes:
            print(f"  - {note}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a named Parameter Golf preset and record structured metadata.")
    parser.add_argument("--preset", help="Preset name to run.")
    parser.add_argument("--run-name", help="Optional human-readable run name. Defaults to the preset name.")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="Override trainer env vars.")
    parser.add_argument("--nproc-per-node", type=int, help="Override torchrun process count for CUDA presets.")
    parser.add_argument("--skip-checks", action="store_true", help="Skip dependency and dataset preflight checks.")
    parser.add_argument("--dry-run", action="store_true", help="Resolve the command and write metadata without launching.")
    parser.add_argument("--list", action="store_true", help="List available presets.")
    args = parser.parse_args()

    if args.list:
        print_presets()
        return
    if not args.preset:
        parser.error("--preset is required unless --list is used")

    preset = get_preset(args.preset)
    if preset.launch_mode != "torchrun" and args.nproc_per_node is not None:
        parser.error("--nproc-per-node only applies to torchrun presets")

    override_env = parse_assignments(args.set)
    run_label = slugify(args.run_name or preset.name)
    stamp = timestamp_slug()
    run_dir = RUNS_ROOT / f"{stamp}_{run_label}"
    run_dir.mkdir(parents=True, exist_ok=False)
    run_id = run_label
    train_log_path = run_dir / f"{run_id}.txt"
    launcher_log_path = run_dir / "launcher.log"

    resolved_env = dict(preset.env)
    resolved_env.update(override_env)
    resolved_env["RUN_ID"] = run_id
    if preset.target == "mlx":
        resolved_env["OUT_DIR"] = str(run_dir)
    else:
        resolved_env["LOG_DIR"] = str(run_dir)
        resolved_env["ARTIFACT_DIR"] = str(run_dir)

    nproc_per_node = args.nproc_per_node if args.nproc_per_node is not None else preset.nproc_per_node
    command = build_command(preset, nproc_per_node)

    preflight: dict[str, object] = {}
    if not args.skip_checks:
        preflight["env"] = check_env(preset.target)
        preflight["data"] = check_data(
            resolved_env["DATA_PATH"],
            resolved_env["TOKENIZER_PATH"],
            min_train_shards=preset.min_train_shards,
            seq_len=int(resolved_env.get("TRAIN_SEQ_LEN", "1024")),
        )

    run_spec = {
        "timestamp": iso_now(),
        "preset": preset.name,
        "target": preset.target,
        "description": preset.description,
        "run_name": run_label,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "command": command,
        "resolved_env": resolved_env,
        "nproc_per_node": nproc_per_node,
        "git_commit": git_commit(),
        "git_dirty": git_is_dirty(),
        "preflight": preflight,
    }
    (run_dir / "run_spec.json").write_text(json.dumps(run_spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"run_dir: {run_dir}")
    print(f"preset: {preset.name}")
    print("command: " + " ".join(command))
    if args.dry_run:
        return

    env = os.environ.copy()
    env.update(resolved_env)
    started_at = time.time()
    with launcher_log_path.open("w", encoding="utf-8") as launcher_log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            launcher_log.write(line)
        exit_code = process.wait()
    wall_clock_seconds = round(time.time() - started_at, 3)

    metrics = parse_trainer_log(train_log_path if train_log_path.is_file() else launcher_log_path)
    artifacts = collect_artifacts(run_dir)
    compressed_model_bytes = None
    for name, info in artifacts.items():
        if name.endswith(".ptz"):
            compressed_model_bytes = int(info["bytes"])
            break

    result = {
        **run_spec,
        "status": "completed" if exit_code == 0 else "failed",
        "exit_code": exit_code,
        "wall_clock_seconds": wall_clock_seconds,
        "train_log_path": str(train_log_path) if train_log_path.is_file() else None,
        "launcher_log_path": str(launcher_log_path),
        "artifacts": artifacts,
        "compressed_model_bytes": compressed_model_bytes,
        "entrypoint_code_bytes": entrypoint_code_bytes(preset.entrypoint),
        "submission_budget_estimate_bytes": (
            entrypoint_code_bytes(preset.entrypoint) + compressed_model_bytes
            if compressed_model_bytes is not None
            else None
        ),
        "metrics": metrics,
    }
    result_path = run_dir / "result.json"
    result["result_path"] = str(result_path)
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    append_index(result)

    best_bpb = final_bpb(metrics)
    if best_bpb is None:
        print(f"result: {result['status']} exit_code={exit_code} wall_clock_seconds={wall_clock_seconds}")
    else:
        print(
            f"result: {result['status']} exit_code={exit_code} "
            f"final_val_bpb={best_bpb:.6f} wall_clock_seconds={wall_clock_seconds}"
        )

    if exit_code != 0:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
