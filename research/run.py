#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frontier_cache import (
    CACHE_OVERRIDE_EXPECTATIONS_ENV,
    apply_cache_override_priority,
    cache_config_record,
    cache_override_expectations_json,
    causal_cache_config_from_env,
    format_cache_config,
    validate_cache_override_expectations,
)
from research.frontier_registry import FRONTIER_PRESETS, FRONTIER_SCALES
from research.presets import PRESETS, RUN_SCALES, Preset, RunScale
from research.byte_budget import write_budget_reports
from research.submission_readiness import write_submission_readiness_reports
from research.submission_metrics import (
    canonical_submission_eval,
    canonical_submission_fields,
)
from scripts.check_data import check_data
from scripts.check_env import check_env
from scripts.check_frontier_env import inspect_frontier_env


RESULTS_ROOT = ROOT / "research" / "results"
RUNS_ROOT = RESULTS_ROOT / "runs"
INDEX_JSONL = RESULTS_ROOT / "index.jsonl"
INDEX_CSV = RESULTS_ROOT / "index.csv"
FINAL_BPB_MISSING = 10**9
DYNAMIC_ENV_KEYS = {"RUN_ID", "LOG_DIR", "ARTIFACT_DIR", "OUT_DIR"}
ALL_PRESETS: dict[str, Preset] = {**PRESETS, **FRONTIER_PRESETS}
ALL_SCALES: dict[str, RunScale] = {**RUN_SCALES, **FRONTIER_SCALES}
KNOWN_GPU_PROFILES = {
    "local_mlx",
    "local_cuda",
    "1xh100",
    "8xh100",
    "8xh100_sxm",
}


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
GENERIC_EVAL_LINE_RE = re.compile(
    r"^(?P<label>[A-Za-z0-9_+-]+) val_loss:(?P<val_loss>[-+0-9.eE]+) "
    r"val_bpb:(?P<val_bpb>[-+0-9.eE]+)(?: .*?eval_time:(?P<eval_time_ms>[-+0-9.eE]+)ms)?$"
)
GENERIC_EXACT_LINE_RE = re.compile(
    r"^(?P<label>[A-Za-z0-9_+-]+)_exact val_loss:(?P<val_loss>[-+0-9.eE]+) "
    r"val_bpb:(?P<val_bpb>[-+0-9.eE]+)$"
)
SERIALIZED_MODEL_RE = re.compile(r"^Serialized model int8\+zlib: (?P<bytes>\d+) bytes ")
SERIALIZED_MODEL_MIXED_RE = re.compile(r"^Serialized model mixed\+(?P<compressor>zlib|zstd): (?P<bytes>\d+) bytes$")
SERIALIZED_MODEL_MLX_RE = re.compile(r"^serialized_model_int8_zlib:(?P<bytes>\d+) bytes ")
TOTAL_SUBMISSION_RE = re.compile(r"^Total submission size int8\+zlib: (?P<bytes>\d+) bytes$")
TOTAL_SUBMISSION_GENERIC_RE = re.compile(r"^Total submission size(?: int8\+zlib)?: (?P<bytes>\d+) bytes$")
CODE_SIZE_RE = re.compile(r"^Code size: (?P<bytes>\d+) bytes$")
DIAGNOSTIC_LINE_RE = re.compile(
    r"^DIAGNOSTIC (?P<label>[A-Za-z0-9_+-]+) val_loss:(?P<val_loss>[-+0-9.eE]+) "
    r"val_bpb:(?P<val_bpb>[-+0-9.eE]+) eval_time:(?P<eval_time_ms>[-+0-9.eE]+)ms$"
)
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


def visible_cuda_device_count(raw: str | None) -> int | None:
    if raw is None:
        return None
    raw = raw.strip()
    if not raw:
        return None
    return len([item for item in raw.split(",") if item.strip()])


def build_command(preset: Preset, nproc_per_node: int | None) -> list[str]:
    entrypoint = str(ROOT / preset.entrypoint)
    if preset.launch_mode == "python":
        return [sys.executable, entrypoint]
    if preset.launch_mode == "torchrun":
        launcher = resolved_torchrun_launcher()
        return [*launcher, "--standalone", f"--nproc_per_node={nproc_per_node}", entrypoint]
    raise ValueError(f"Unsupported launch mode {preset.launch_mode!r}")


def resolved_torchrun_launcher() -> list[str]:
    torchrun = shutil.which("torchrun")
    if torchrun is not None:
        return [torchrun]
    return [sys.executable, "-m", "torch.distributed.run"]


def validate_launch_inputs(
    *,
    preset: Preset,
    nproc_per_node: int | None,
    gpu_profile: str | None,
    preflight: dict[str, object],
) -> dict[str, object]:
    validation: dict[str, object] = {
        "gpu_profile": gpu_profile,
        "gpu_profile_known": gpu_profile in KNOWN_GPU_PROFILES if gpu_profile else True,
        "launch_mode": preset.launch_mode,
        "nproc_per_node": nproc_per_node,
    }
    if preset.launch_mode != "torchrun":
        return validation

    launcher = resolved_torchrun_launcher()
    validation["torchrun_launcher"] = launcher

    if nproc_per_node is None or nproc_per_node < 1:
        raise RuntimeError("torchrun presets require nproc_per_node >= 1")

    expected_nproc = {"1xh100": 1, "8xh100": 8, "8xh100_sxm": 8}.get(gpu_profile or "")
    if expected_nproc is not None and nproc_per_node != expected_nproc:
        raise RuntimeError(
            f"gpu_profile={gpu_profile!r} expects nproc_per_node={expected_nproc}, got {nproc_per_node}"
        )

    visible_devices = visible_cuda_device_count(os.environ.get("CUDA_VISIBLE_DEVICES"))
    validation["visible_cuda_device_count"] = visible_devices
    if visible_devices is not None and visible_devices < nproc_per_node:
        raise RuntimeError(
            f"CUDA_VISIBLE_DEVICES exposes only {visible_devices} devices, but nproc_per_node={nproc_per_node}"
        )

    frontier_env = preflight.get("frontier_env") if isinstance(preflight, dict) else None
    detected_devices = None
    if isinstance(frontier_env, dict):
        maybe_count = frontier_env.get("device_count")
        if isinstance(maybe_count, int):
            detected_devices = maybe_count
    validation["detected_cuda_device_count"] = detected_devices
    if preset.target == "cuda" and detected_devices is not None and detected_devices < nproc_per_node:
        raise RuntimeError(
            f"Frontier preflight detected only {detected_devices} CUDA devices, but nproc_per_node={nproc_per_node}"
        )

    return validation


def checkpoint_filename_for(preset: Preset) -> str:
    return "checkpoint_latest.pkl" if preset.target == "mlx" else "checkpoint_latest.pt"


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


def counted_code_paths_for(preset: Preset) -> tuple[str, ...]:
    return preset.counted_code_paths or (preset.entrypoint,)


def counted_code_bytes_for(preset: Preset) -> int:
    return sum(len((ROOT / rel_path).read_bytes()) for rel_path in counted_code_paths_for(preset))


def absolutize_env_paths(env: dict[str, str]) -> dict[str, str]:
    resolved = dict(env)
    for key in ("DATA_PATH", "TOKENIZER_PATH", "RESUME_FROM", "CHECKPOINT_PATH", "SUMMARY_PATH", "OUT_DIR", "LOG_DIR", "ARTIFACT_DIR"):
        value = resolved.get(key)
        if not value:
            continue
        path = Path(value)
        if not path.is_absolute():
            resolved[key] = str((ROOT / path).resolve())
    return resolved


def get_preset(name: str) -> Preset:
    try:
        return ALL_PRESETS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown preset {name!r}. Available presets: {', '.join(sorted(ALL_PRESETS))}") from exc


def get_scale(name: str) -> RunScale:
    try:
        return ALL_SCALES[name]
    except KeyError as exc:
        raise KeyError(f"Unknown scale {name!r}. Available scales: {', '.join(sorted(ALL_SCALES))}") from exc


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
        "named_evals": {},
        "named_evals_exact": {},
        "diagnostics": {},
        "artifact_metrics": {},
    }
    if not log_path.is_file():
        return metrics

    named_evals: dict[str, dict[str, object]] = {}
    named_evals_exact: dict[str, dict[str, object]] = {}
    diagnostics: dict[str, dict[str, object]] = {}
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

        match = GENERIC_EXACT_LINE_RE.match(line)
        if match:
            label = match.group("label")
            payload = {key: to_number(value) for key, value in match.groupdict().items() if key != "label"}
            named_evals_exact[label] = payload
            if label == "final_int8_zlib_roundtrip":
                metrics["final_roundtrip_exact"] = payload
            continue

        match = DIAGNOSTIC_LINE_RE.match(line)
        if match:
            label = match.group("label")
            diagnostics[label] = {key: to_number(value) for key, value in match.groupdict().items() if key != "label"}
            continue

        match = GENERIC_EVAL_LINE_RE.match(line)
        if match:
            label = match.group("label")
            payload = {key: to_number(value) for key, value in match.groupdict().items() if key != "label" and value is not None}
            named_evals[label] = payload
            if label == "final_int8_zlib_roundtrip":
                metrics["final_roundtrip"] = payload
            continue

        match = SERIALIZED_MODEL_RE.match(line)
        if match:
            artifact_metrics["compressed_model_bytes_logged"] = int(match.group("bytes"))
            continue

        match = SERIALIZED_MODEL_MIXED_RE.match(line)
        if match:
            artifact_metrics["compressed_model_bytes_logged"] = int(match.group("bytes"))
            artifact_metrics["compressor"] = match.group("compressor")
            continue

        match = SERIALIZED_MODEL_MLX_RE.match(line)
        if match:
            artifact_metrics["compressed_model_bytes_logged"] = int(match.group("bytes"))
            continue

        match = TOTAL_SUBMISSION_RE.match(line)
        if match:
            artifact_metrics["total_submission_bytes_logged"] = int(match.group("bytes"))
            continue

        match = TOTAL_SUBMISSION_GENERIC_RE.match(line)
        if match:
            artifact_metrics["total_submission_bytes_logged"] = int(match.group("bytes"))
            continue

        match = CODE_SIZE_RE.match(line)
        if match:
            artifact_metrics["code_bytes_logged"] = int(match.group("bytes"))
            continue

    metrics["named_evals"] = named_evals
    metrics["named_evals_exact"] = named_evals_exact
    metrics["diagnostics"] = diagnostics
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


def preferred_eval(metrics: dict[str, object]) -> tuple[str | None, dict[str, object] | None]:
    return canonical_submission_eval(metrics)


def final_bpb(metrics: dict[str, object]) -> float | None:
    _label, candidate = preferred_eval(metrics)
    if isinstance(candidate, dict) and "val_bpb" in candidate:
        return float(candidate["val_bpb"])
    return None


def config_diff(base_env: dict[str, str], resolved_env: dict[str, str]) -> dict[str, dict[str, str | None]]:
    diff: dict[str, dict[str, str | None]] = {}
    keys = sorted(set(base_env) | set(resolved_env))
    for key in keys:
        if key in DYNAMIC_ENV_KEYS:
            continue
        before = base_env.get(key)
        after = resolved_env.get(key)
        if before != after:
            diff[key] = {"from": before, "to": after}
    return diff


def write_legality_note(run_dir: Path, preset: Preset, result: dict[str, object]) -> dict[str, object]:
    status = str(result.get("status") or "incomplete")
    artifact_bytes = result.get("submission_budget_estimate_bytes")
    final_metric_present = final_bpb(result.get("metrics") or {}) is not None
    artifact_budget_ok = artifact_bytes is None or int(artifact_bytes) <= 16_000_000
    if status != "completed":
        legality_status = "incomplete"
    elif not artifact_budget_ok or not final_metric_present:
        legality_status = "invalid"
    else:
        legality_status = "legal"
    note = {
        "preset": preset.name,
        "status": legality_status,
        "artifact_budget_ok": artifact_budget_ok,
        "final_metric_present": final_metric_present,
        "official_submission_metric_label": result.get("official_submission_metric_label") or result.get("final_submission_metric_label"),
        "official_submission_bpb": result.get("final_submission_bpb"),
        "legality_summary": list(preset.legality_summary),
    }
    lines = [
        f"Preset: {preset.name}",
        f"Status: {legality_status}",
        f"Artifact budget ok: {note['artifact_budget_ok']}",
        f"Final metric present: {final_metric_present}",
        f"Official submission metric: {note['official_submission_metric_label']}",
        f"Official submission bpb: {note['official_submission_bpb']}",
        "",
        "Why this run is compliant:",
    ]
    for item in preset.legality_summary:
        lines.append(f"- {item}")
    (run_dir / "legality_note.json").write_text(json.dumps(note, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (run_dir / "legality_note.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return note


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
                "scale": record.get("scale") or "",
                "family": record.get("family"),
                "target": record.get("target"),
                "status": record.get("status"),
                "exit_code": record.get("exit_code"),
                "final_val_bpb": record.get("final_submission_bpb") or final_bpb(metrics) or "",
                "final_val_loss": record.get("final_submission_loss") or (((preferred_eval(metrics)[1]) or {}).get("val_loss") or ""),
                "compressed_model_bytes": record.get("compressed_model_bytes") or artifact_metrics.get("compressed_model_bytes_logged") or "",
                "counted_code_bytes": record.get("counted_code_bytes") or "",
                "submission_budget_estimate_bytes": record.get("submission_budget_estimate_bytes") or artifact_metrics.get("total_submission_bytes_logged") or "",
                "wall_clock_seconds": record.get("wall_clock_seconds"),
                "gpu_profile": record.get("gpu_profile") or "",
                "legality_status": ((record.get("legality_note") or {}).get("status")) or "",
                "git_commit": record.get("git_commit") or "",
                "diff_base": record.get("diff_base") or "",
                "diff_keys": ",".join(sorted((record.get("config_diff_from_base") or {}).keys())),
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
                "scale",
                "family",
                "target",
                "status",
                "exit_code",
                "final_val_bpb",
                "final_val_loss",
                "compressed_model_bytes",
                "counted_code_bytes",
                "submission_budget_estimate_bytes",
                "wall_clock_seconds",
                "gpu_profile",
                "legality_status",
                "git_commit",
                "diff_base",
                "diff_keys",
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
    deduped: dict[str, dict[str, object]] = {}
    ordered: list[dict[str, object]] = []
    for item in existing:
        run_dir = str(item.get("run_dir") or "")
        if run_dir and run_dir in deduped:
            for idx, prior in enumerate(ordered):
                if str(prior.get("run_dir") or "") == run_dir:
                    ordered[idx] = item
                    break
        else:
            ordered.append(item)
        if run_dir:
            deduped[run_dir] = item
    with INDEX_JSONL.open("w", encoding="utf-8") as f:
        for item in ordered:
            f.write(json.dumps(item, sort_keys=True) + "\n")
    regenerate_csv_index(ordered)


def print_presets() -> None:
    for preset_name in sorted(ALL_PRESETS):
        preset = ALL_PRESETS[preset_name]
        print(f"{preset.name}: {preset.description}")
        print(f"  family: {preset.family}")
        if preset.diff_base:
            print(f"  diff_base: {preset.diff_base}")
        if preset.counted_code_paths:
            print("  counted_code_paths: " + ", ".join(preset.counted_code_paths))
        for note in preset.notes:
            print(f"  - {note}")


def print_scales() -> None:
    for scale_name in sorted(ALL_SCALES):
        scale = ALL_SCALES[scale_name]
        print(f"{scale.name}: {scale.description}")
        for note in scale.notes:
            print(f"  - {note}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a named Parameter Golf preset and record structured metadata.")
    parser.add_argument("--preset", help="Preset name to run.")
    parser.add_argument("--scale", help="Optional run-scale overlay such as probe_short or long_local_overnight.")
    parser.add_argument("--run-name", help="Optional human-readable run name. Defaults to the preset name.")
    parser.add_argument("--seed", type=int, help="Set SEED without needing --set SEED=...")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="Override trainer env vars.")
    parser.add_argument("--nproc-per-node", type=int, help="Override torchrun process count for CUDA presets.")
    parser.add_argument("--resume-run-dir", help="Resume from an existing run directory created by this launcher.")
    parser.add_argument("--notes", help="Optional short note saved into run metadata.")
    parser.add_argument("--gpu-profile", help="Optional hardware/profile tag such as local_mlx, local_cuda, 1xh100, 8xh100, or 8xh100_sxm.")
    parser.add_argument("--skip-checks", action="store_true", help="Skip dependency and dataset preflight checks.")
    parser.add_argument("--preflight-only", action="store_true", help="Run dependency, dataset, and distributed-launch validation without starting training.")
    parser.add_argument("--dry-run", action="store_true", help="Resolve the command and write metadata without launching.")
    parser.add_argument("--list", action="store_true", help="List available presets.")
    parser.add_argument("--list-scales", action="store_true", help="List available run scales.")
    args = parser.parse_args()

    if args.list:
        print_presets()
        return
    if args.list_scales:
        print_scales()
        return
    if not args.preset and not args.resume_run_dir:
        parser.error("--preset is required unless --list, --list-scales, or --resume-run-dir is used")

    resume_spec: dict[str, object] | None = None
    resume_run_dir: Path | None = None
    if args.resume_run_dir:
        resume_run_dir = Path(args.resume_run_dir).resolve()
        run_spec_path = resume_run_dir / "run_spec.json"
        if not run_spec_path.is_file():
            parser.error(f"--resume-run-dir requires an existing run_spec.json in {resume_run_dir}")
        resume_spec = json.loads(run_spec_path.read_text(encoding="utf-8"))
        if not args.preset:
            args.preset = str(resume_spec.get("preset"))

    preset = get_preset(args.preset)
    if preset.launch_mode != "torchrun" and args.nproc_per_node is not None:
        parser.error("--nproc-per-node only applies to torchrun presets")
    resume_scale_name = str((resume_spec or {}).get("scale") or "") or None
    scale_name = args.scale or resume_scale_name
    scale: RunScale | None = get_scale(scale_name) if scale_name else None

    override_env = parse_assignments(args.set)
    if resume_spec is not None and resume_run_dir is not None:
        run_dir = resume_run_dir
        run_label = str(resume_spec.get("run_name") or resume_spec.get("run_id") or preset.name)
        run_id = str(resume_spec.get("run_id") or run_label)
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_label = slugify(args.run_name or preset.name)
        stamp = timestamp_slug()
        run_dir = RUNS_ROOT / f"{stamp}_{run_label}"
        run_dir.mkdir(parents=True, exist_ok=False)
        run_id = run_label
    train_log_path = run_dir / f"{run_id}.txt"
    launcher_log_path = run_dir / "launcher.log"

    resolved_env = dict((resume_spec or {}).get("resolved_env") or preset.env)
    if scale is not None:
        resolved_env.update(scale.env)
    resolved_env.update(override_env)
    if args.seed is not None:
        resolved_env["SEED"] = str(args.seed)
    resolved_env["RUN_ID"] = run_id
    if preset.target == "mlx":
        resolved_env["OUT_DIR"] = str(run_dir)
    else:
        resolved_env["LOG_DIR"] = str(run_dir)
        resolved_env["ARTIFACT_DIR"] = str(run_dir)
    resolved_env["SUMMARY_PATH"] = str(run_dir / "run_summary.json")
    checkpoint_path = run_dir / checkpoint_filename_for(preset)
    resolved_env["CHECKPOINT_PATH"] = str(checkpoint_path)
    if resume_spec is not None and "RESUME_FROM" not in override_env and checkpoint_path.is_file():
        resolved_env["RESUME_FROM"] = str(checkpoint_path)
    resolved_env = absolutize_env_paths(resolved_env)
    resolved_env = apply_cache_override_priority(resolved_env, os.environ)
    resolved_cache_config = causal_cache_config_from_env(resolved_env)
    validate_cache_override_expectations(
        resolved_cache_config,
        os.environ,
        context="launcher resolved cache config",
    )
    cache_expectations_json = cache_override_expectations_json(os.environ)
    if cache_expectations_json is not None:
        resolved_env[CACHE_OVERRIDE_EXPECTATIONS_ENV] = cache_expectations_json
    else:
        resolved_env.pop(CACHE_OVERRIDE_EXPECTATIONS_ENV, None)

    nproc_per_node = args.nproc_per_node if args.nproc_per_node is not None else preset.nproc_per_node
    command = build_command(preset, nproc_per_node)
    base_env = dict(get_preset(preset.diff_base).env) if preset.diff_base else {}
    env_diff = config_diff(base_env, resolved_env) if preset.diff_base else {}

    preflight: dict[str, object] = {}
    if not args.skip_checks:
        extra_modules = ["zstandard"] if resolved_env.get("COMPRESSOR") == "zstd" else []
        extra_modules.extend(preset.required_modules)
        preflight["env"] = check_env(preset.target, extra_modules=extra_modules)
        if preset.target == "cuda" and "flash_attn_interface" in preset.required_modules:
            preflight["frontier_env"] = inspect_frontier_env(require_flash_attn=True)
            if not preflight["frontier_env"].get("ok_to_proceed"):
                issues = "; ".join(preflight["frontier_env"].get("issues") or ["frontier env not ready"])
                raise RuntimeError(f"Frontier CUDA preflight failed: {issues}")
        preflight["data"] = check_data(
            resolved_env["DATA_PATH"],
            resolved_env["TOKENIZER_PATH"],
            min_train_shards=preset.min_train_shards,
            seq_len=int(resolved_env.get("TRAIN_SEQ_LEN", "1024")),
        )
    launch_validation = validate_launch_inputs(
        preset=preset,
        nproc_per_node=nproc_per_node,
        gpu_profile=args.gpu_profile,
        preflight=preflight,
    )
    if args.preflight_only:
        print(f"run_dir: {run_dir}")
        print(f"preset: {preset.name}")
        if scale is not None:
            print(f"scale: {scale.name}")
        if any(key.startswith("CAUSAL_CACHE_") for key in resolved_env):
            print("resolved_cache: " + format_cache_config(resolved_cache_config))
        print("launch_validation: " + json.dumps(launch_validation, sort_keys=True))
        print("command: " + " ".join(command))
        if resume_run_dir is None and run_dir.exists() and not any(run_dir.iterdir()):
            run_dir.rmdir()
        return

    run_spec = {
        "timestamp": iso_now(),
        "preset": preset.name,
        "scale": scale.name if scale is not None else None,
        "family": preset.family,
        "diff_base": preset.diff_base,
        "config_diff_from_base": env_diff,
        "target": preset.target,
        "description": preset.description,
        "notes": args.notes,
        "preset_notes": preset.notes,
        "legality_summary": preset.legality_summary,
        "scale_notes": scale.notes if scale is not None else (),
        "resumed_from_run_dir": str(resume_run_dir) if resume_run_dir is not None else None,
        "run_name": run_label,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "gpu_profile": args.gpu_profile,
        "command": command,
        "counted_code_paths": counted_code_paths_for(preset),
        "resolved_env": resolved_env,
        "resolved_cache_config": cache_config_record(resolved_cache_config),
        "nproc_per_node": nproc_per_node,
        "launch_validation": launch_validation,
        "git_commit": git_commit(),
        "git_dirty": git_is_dirty(),
        "preflight": preflight,
    }
    run_spec_path = run_dir / "run_spec.json"
    if resume_run_dir is not None:
        resume_meta_path = run_dir / f"resume_{timestamp_slug()}.json"
        resume_meta_path.write_text(json.dumps(run_spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        run_spec_path.write_text(json.dumps(run_spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"run_dir: {run_dir}")
    print(f"preset: {preset.name}")
    if scale is not None:
        print(f"scale: {scale.name}")
    if any(key.startswith("CAUSAL_CACHE_") for key in resolved_env):
        print("resolved_cache: " + format_cache_config(resolved_cache_config))
    print("launch_validation: " + json.dumps(launch_validation, sort_keys=True))
    print("command: " + " ".join(command))
    if args.dry_run:
        return

    env = os.environ.copy()
    env.update(resolved_env)
    started_at = time.time()
    interrupted = False
    with launcher_log_path.open("a" if resume_run_dir is not None else "w", encoding="utf-8") as launcher_log:
        process = subprocess.Popen(
            command,
            cwd=run_dir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        try:
            for line in process.stdout:
                print(line, end="")
                launcher_log.write(line)
            exit_code = process.wait()
        except KeyboardInterrupt:
            interrupted = True
            launcher_log.write("\nlauncher:event:keyboard_interrupt forwarding_sigint\n")
            process.send_signal(signal.SIGINT)
            exit_code = process.wait()
    wall_clock_seconds = round(time.time() - started_at, 3)

    metrics = parse_trainer_log(train_log_path if train_log_path.is_file() else launcher_log_path)
    submission_fields = canonical_submission_fields(metrics)
    artifacts = collect_artifacts(run_dir)
    compressed_model_bytes = None
    for name, info in artifacts.items():
        if name.endswith(".ptz"):
            compressed_model_bytes = int(info["bytes"])
            break

    budget_report = None
    if exit_code == 0:
        try:
            budget_report = write_budget_reports(run_dir, counted_code_paths_for(preset))
        except Exception as exc:  # noqa: BLE001
            budget_report = {"error": str(exc)}

    result = {
        **run_spec,
        "status": "completed" if exit_code == 0 else ("interrupted" if interrupted else "failed"),
        "exit_code": exit_code,
        "wall_clock_seconds": wall_clock_seconds,
        "train_log_path": str(train_log_path) if train_log_path.is_file() else None,
        "launcher_log_path": str(launcher_log_path),
        "artifacts": artifacts,
        "compressed_model_bytes": compressed_model_bytes,
        "counted_code_bytes": counted_code_bytes_for(preset),
        "submission_budget_estimate_bytes": (
            counted_code_bytes_for(preset) + compressed_model_bytes
            if compressed_model_bytes is not None
            else None
        ),
        "byte_budget": budget_report,
        "metrics": metrics,
        **submission_fields,
    }
    if isinstance(budget_report, dict):
        result["exported_model_bytes"] = budget_report.get("exported_bytes_measured")
        result["artifact_bytes_measured"] = budget_report.get("artifact_bytes_measured")
        result["remaining_headroom_to_16mb"] = budget_report.get("remaining_headroom_to_16MB")
    result["legality_note"] = write_legality_note(run_dir, preset, result)
    result["submission_readiness"] = write_submission_readiness_reports(run_dir, result)
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
