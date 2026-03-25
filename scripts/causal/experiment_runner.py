"""Paired Seed Ablation runner (C5).

Runs training scripts under treatment/control configs across multiple seeds,
collects BPB results, and outputs raw_runs.json for statistical analysis.

CLI:
    python scripts/causal/experiment_runner.py \\
        --treatment treatment_config.json \\
        --control control_config.json \\
        --output results/causal/cycle_1/raw_runs.json \\
        --seeds 42,137,256 \\
        --platform mlx|h100
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# ---------------------------------------------------------------------------
# BPB parsing regex: matches val_bpb:<float> anywhere in a line
# ---------------------------------------------------------------------------
_VAL_BPB_RE = re.compile(r"val_bpb[:\s]+(\d+\.\d+)")

# Default timeout buffer added on top of MAX_WALLCLOCK_SECONDS
_TIMEOUT_BUFFER_S = 120
_DEFAULT_WALLCLOCK_S = 600

# Platform script mapping
_PLATFORM_SCRIPTS = {
    "mlx": "train_gpt_mlx.py",
    "h100": "train_gpt.py",
}


# ---------------------------------------------------------------------------
# Config loading and validation
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    """Load a JSON config file and return its contents."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    return json.loads(p.read_text(encoding="utf-8"))


def validate_config(cfg: dict) -> dict:
    """Validate an experiment config dict.

    Required keys:
        - "script": str - path to training script
        - "env_overrides": dict[str, str] - environment variable overrides

    Raises ValueError/KeyError on invalid config.
    """
    if "script" not in cfg:
        raise ValueError("Config missing required 'script' field")
    if not isinstance(cfg["script"], str):
        raise ValueError("'script' must be a string")

    if "env_overrides" not in cfg:
        raise ValueError("Config missing required 'env_overrides' field")
    if not isinstance(cfg["env_overrides"], dict):
        raise ValueError("'env_overrides' must be a dict")

    for key, val in cfg["env_overrides"].items():
        if not isinstance(key, str):
            raise TypeError(f"env_overrides key must be str, got {type(key).__name__}: {key}")
        if not isinstance(val, str):
            raise TypeError(
                f"env_overrides value for '{key}' must be str, got {type(val).__name__}: {val}"
            )

    return cfg


# ---------------------------------------------------------------------------
# BPB parsing
# ---------------------------------------------------------------------------

def parse_last_val_bpb(stdout: str) -> float | None:
    """Parse the LAST occurrence of val_bpb:<float> from stdout.

    Returns the float value, or None if no match found.
    """
    matches = _VAL_BPB_RE.findall(stdout)
    if not matches:
        return None
    return float(matches[-1])


# ---------------------------------------------------------------------------
# Parse JSON-lines metrics from stdout
# ---------------------------------------------------------------------------

def _parse_jsonlines_metrics(stdout: str) -> list[dict]:
    """Extract any JSON-lines metrics from stdout (one JSON object per line)."""
    metrics = []
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                metrics.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return metrics


# ---------------------------------------------------------------------------
# Subprocess execution for a single seed
# ---------------------------------------------------------------------------

def _run_single_seed(
    cfg: dict,
    seed: int,
    timeout: float,
) -> dict:
    """Run a training script for a single seed and return result dict.

    Result keys: seed, val_bpb (float|None), val_loss (float|None),
    wall_time_s, checkpoint_path, train_log_path, stdout, stderr.
    On error: includes "error" key with description.
    """
    env = os.environ.copy()
    env["SEED"] = str(seed)
    for key, val in cfg.get("env_overrides", {}).items():
        env[key] = val

    script = cfg["script"]
    cmd = [sys.executable, script]

    result: dict[str, Any] = {
        "seed": seed,
        "val_bpb": None,
        "val_loss": None,
        "wall_time_s": None,
        "checkpoint_path": None,
        "train_log_path": None,
    }

    start_time = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        wall_time = time.monotonic() - start_time
        result["wall_time_s"] = round(wall_time, 2)

        if proc.returncode != 0:
            result["error"] = (
                f"Process exited with code {proc.returncode}: "
                f"{proc.stderr.strip()[:500] if proc.stderr else 'no stderr'}"
            )
            # Still try to parse any BPB from partial output
            bpb = parse_last_val_bpb(proc.stdout)
            if bpb is not None:
                result["val_bpb"] = bpb
            return result

        # Parse BPB from stdout
        bpb = parse_last_val_bpb(proc.stdout)
        result["val_bpb"] = bpb

        # Parse JSON-lines metrics for extra fields
        metrics = _parse_jsonlines_metrics(proc.stdout)
        if metrics:
            result["metrics"] = metrics

        # Try to extract val_loss from stdout (pattern: val_loss:<float>)
        loss_match = re.findall(r"val_loss[:\s]+(\d+\.\d+)", proc.stdout)
        if loss_match:
            result["val_loss"] = float(loss_match[-1])

        # Try to capture checkpoint path from stdout
        ckpt_match = re.findall(r"checkpoint[:\s]+(\S+\.safetensors)", proc.stdout)
        if ckpt_match:
            result["checkpoint_path"] = ckpt_match[-1]

        # Try to capture log path from stdout
        log_match = re.findall(r"log[:\s]+(\S+\.log)", proc.stdout)
        if log_match:
            result["train_log_path"] = log_match[-1]

    except subprocess.TimeoutExpired:
        wall_time = time.monotonic() - start_time
        result["wall_time_s"] = round(wall_time, 2)
        result["error"] = f"Timed out after {timeout}s"

    return result


# ---------------------------------------------------------------------------
# Run all seeds for a condition
# ---------------------------------------------------------------------------

def run_condition(
    cfg: dict,
    seeds: list[int],
    timeout: float | None = None,
) -> list[dict]:
    """Run a condition (treatment or control) across all seeds.

    Returns list of per-seed result dicts.
    """
    if timeout is None:
        wallclock = int(cfg.get("env_overrides", {}).get(
            "MAX_WALLCLOCK_SECONDS", str(_DEFAULT_WALLCLOCK_S)
        ))
        timeout = wallclock + _TIMEOUT_BUFFER_S

    results = []
    for seed in seeds:
        result = _run_single_seed(cfg, seed, timeout)
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# Condition status classification
# ---------------------------------------------------------------------------

def _classify_condition(results: list[dict]) -> dict:
    """Classify condition results: reduced_power, failed, etc."""
    successful = [r for r in results if "error" not in r and r.get("val_bpb") is not None]
    failed = [r for r in results if "error" in r or r.get("val_bpb") is None]
    total = len(results)
    n_ok = len(successful)
    n_fail = len(failed)

    status: dict[str, Any] = {"n_successful": n_ok, "n_failed": n_fail}

    if n_fail == 0:
        status["status"] = "complete"
    elif n_ok >= 2 and n_fail < total:
        status["status"] = "reduced_power"
        status["reduced_power"] = True
    else:
        status["status"] = "failed"

    return status


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    """CLI entry point for the experiment runner."""
    parser = argparse.ArgumentParser(
        description="Paired Seed Ablation runner (C5)"
    )
    parser.add_argument("--treatment", required=True, help="Path to treatment config JSON")
    parser.add_argument("--control", required=True, help="Path to control config JSON")
    parser.add_argument("--output", required=True, help="Path to output raw_runs.json")
    parser.add_argument("--seeds", default="42,137,256",
                        help="Comma-separated seed list (default: 42,137,256)")
    parser.add_argument("--platform", choices=["mlx", "h100"], default="mlx",
                        help="Platform to run on (default: mlx)")
    args = parser.parse_args(argv)

    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    # Load and validate configs
    treatment_cfg = validate_config(load_config(args.treatment))
    control_cfg = validate_config(load_config(args.control))

    # Run both conditions
    print(f"Running treatment: {treatment_cfg.get('description', 'N/A')}")
    treatment_results = run_condition(treatment_cfg, seeds)

    print(f"Running control: {control_cfg.get('description', 'N/A')}")
    control_results = run_condition(control_cfg, seeds)

    # Classify results
    treatment_status = _classify_condition(treatment_results)
    control_status = _classify_condition(control_results)

    # Build output
    output = {
        "platform": args.platform,
        "seeds": seeds,
        "treatment": {
            "config": treatment_cfg,
            "results": treatment_results,
            "status": treatment_status,
        },
        "control": {
            "config": control_cfg,
            "results": control_results,
            "status": control_status,
        },
    }

    # Write raw_runs.json
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Results written to {output_path}")

    # Append to experiment_log.json
    from scripts.causal.common import log_experiment

    log_dir = output_path.parent
    log_path = log_dir / "experiment_log.json"
    log_entry = {
        "type": "paired_seed_ablation",
        "platform": args.platform,
        "seeds": seeds,
        "treatment_description": treatment_cfg.get("description", ""),
        "control_description": control_cfg.get("description", ""),
        "treatment_status": treatment_status["status"],
        "control_status": control_status["status"],
        "raw_runs_path": str(output_path),
    }
    log_experiment(str(log_path), log_entry)
    print(f"Experiment logged to {log_path}")


if __name__ == "__main__":
    main()
