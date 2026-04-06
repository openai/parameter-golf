"""
Run history tracking for parameter-golf experiments.
Stores results in ./run_history.json as a flat JSON array.
"""

import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

HISTORY_FILE = Path("./run_history.json")


def save_run(run_id, params_dict, final_bpb, final_loss, steps_completed,
             duration_seconds, status="completed", artifact_size_bytes=None,
             started_at=None, finished_at=None):
    """Append a completed run to run_history.json."""
    now = datetime.now(timezone.utc).isoformat()
    entry = {
        "run_id": run_id,
        "timestamp": now,
        "started_at": started_at,
        "finished_at": finished_at or now,
        "params": params_dict,
        "final_bpb": final_bpb,
        "final_loss": final_loss,
        "steps_completed": steps_completed,
        "duration_seconds": duration_seconds,
        "artifact_size_bytes": artifact_size_bytes,
        "status": status,
    }
    runs = load_runs()
    runs.append(entry)
    HISTORY_FILE.write_text(json.dumps(runs, indent=2))


def load_runs():
    """Load all runs from run_history.json, sorted by final_bpb (best first)."""
    if not HISTORY_FILE.exists():
        return []
    try:
        runs = json.loads(HISTORY_FILE.read_text())
    except (json.JSONDecodeError, ValueError):
        return []
    # Sort by final_bpb ascending (lower is better), inf/None at the end
    runs.sort(key=lambda r: r.get("final_bpb") if r.get("final_bpb") is not None else float("inf"))
    return runs


MAX_ARTIFACT_BYTES = 16_000_000


def check_artifact_size(log_path):
    """Parse log for artifact size. Returns (size_bytes, is_valid)."""
    path = Path(log_path)
    if not path.exists():
        return None, False
    text = path.read_text()
    m = re.search(r"Total submission size \S+: (\d+) bytes", text)
    if not m:
        return None, False
    size = int(m.group(1))
    return size, size <= MAX_ARTIFACT_BYTES


def parse_log(log_path):
    """Parse a training log file and extract all metrics.

    Returns dict with:
        final_bpb, final_loss, steps, step_avg,
        all_val_points: [{step, val_loss, val_bpb}],
        all_train_points: [{step, train_loss, train_time_ms, step_avg}]
    """
    result = {
        "final_bpb": None,
        "final_loss": None,
        "steps": 0,
        "step_avg": None,
        "artifact_size_bytes": None,
        "started_at": None,
        "finished_at": None,
        "all_val_points": [],
        "all_train_points": [],
    }

    path = Path(log_path)
    if not path.exists():
        return result

    text = path.read_text()

    # Final score — matches any roundtrip format (int8_zlib, int6, int6_lzma, etc.)
    m = re.search(r"final_\S+_roundtrip val_loss:([\d.]+) val_bpb:([\d.]+)", text)
    if m:
        result["final_loss"] = float(m.group(1))
        result["final_bpb"] = float(m.group(2))

    # Validation checkpoints
    for m in re.finditer(
        r"step:(\d+)/\d+ val_loss:([\d.]+) val_bpb:([\d.]+) train_time:(\d+)ms",
        text,
    ):
        result["all_val_points"].append({
            "step": int(m.group(1)),
            "val_loss": float(m.group(2)),
            "val_bpb": float(m.group(3)),
            "train_time_ms": int(m.group(4)),
        })

    # Artifact size — matches any compression format (int8+zlib, int6+lzma, etc.)
    m_art = re.search(r"Total submission size \S+: (\d+) bytes", text)
    if m_art:
        result["artifact_size_bytes"] = int(m_art.group(1))

    # Train loss lines
    for m in re.finditer(
        r"step:(\d+)/\d+ train_loss:([\d.]+) train_time:(\d+)ms step_avg:([\d.]+)ms",
        text,
    ):
        step = int(m.group(1))
        result["all_train_points"].append({
            "step": step,
            "train_loss": float(m.group(2)),
            "train_time_ms": int(m.group(3)),
            "step_avg": float(m.group(4)),
        })
        result["steps"] = max(result["steps"], step)
        result["step_avg"] = float(m.group(4))

    # Extract timestamps: use file mtime for finished_at, estimate started_at from duration
    try:
        mtime = path.stat().st_mtime
        finished = datetime.fromtimestamp(mtime, tz=timezone.utc)
        result["finished_at"] = finished.isoformat()
        # Estimate started_at from first train point's train_time_ms or duration
        if result["all_train_points"]:
            last_train_ms = result["all_train_points"][-1].get("train_time_ms", 0)
            from datetime import timedelta
            result["started_at"] = (finished - timedelta(milliseconds=last_train_ms)).isoformat()
    except Exception:
        pass

    return result


def get_gpu_thermal_status():
    """Returns dict with temp, power, and throttle status."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu,power.draw,clocks_throttle_reasons.active",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        parts = result.stdout.strip().split(", ")
        temp = int(parts[0]) if len(parts) > 0 else None
        power = parts[1].strip() if len(parts) > 1 else None
        throttle_raw = parts[2].strip() if len(parts) > 2 else "Unknown"
        throttled = throttle_raw not in ("0x0000000000000000", "Not Active", "[Not Supported]", "")
        return {"temp_c": temp, "power_w": power, "throttled": throttled}
    except Exception:
        # Fallback: try without throttle query (may not work on DGX Spark UMA)
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=temperature.gpu,power.draw",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            parts = result.stdout.strip().split(", ")
            temp = int(parts[0]) if len(parts) > 0 else None
            power = parts[1].strip() if len(parts) > 1 else None
            return {"temp_c": temp, "power_w": power, "throttled": None}
        except Exception:
            return {"temp_c": None, "power_w": None, "throttled": None}
