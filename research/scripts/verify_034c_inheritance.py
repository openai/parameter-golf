#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


IGNORE_KEYS = {
    "artifact_dir",
    "logfile",
    "model_path",
    "quantized_model_path",
    "run_id",
    "distributed",
    "world_size",
    "rank",
    "local_rank",
    "is_main_process",
    "grad_accum_steps",
}


def parse_hparams_from_log(log_path: Path) -> dict[str, str]:
    lines = log_path.read_text().splitlines()
    baseline: dict[str, str] = {}
    inside = False
    for line in lines:
        if line.strip() == "Hyperparameters:":
            inside = True
            continue
        if not inside:
            continue
        if not line.startswith("  "):
            break
        stripped = line.strip()
        if ": " not in stripped:
            continue
        key, value = stripped.split(": ", 1)
        baseline[key] = value
    if not baseline:
        raise SystemExit(f"Failed to parse Hyperparameters block from {log_path}")
    return baseline


def build_current_hparams() -> dict[str, object]:
    sys.path.insert(0, os.getcwd())
    import train_gpt  # pylint: disable=import-error

    h = train_gpt.Hyperparameters()
    return {k: getattr(h, k) for k, _ in vars(type(h)).items() if not k.startswith("_")}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-log", required=True)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--expected-min-lr", required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()

    baseline = parse_hparams_from_log(Path(args.baseline_log))
    current = build_current_hparams()

    out_dir = Path(args.artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(
        json.dumps(current, indent=2, sort_keys=True, default=str) + "\n"
    )

    expected_min_lr = str(float(args.expected_min_lr))
    diffs: dict[str, dict[str, str]] = {}
    for key, baseline_value in baseline.items():
        if key in IGNORE_KEYS:
            continue
        current_value = str(current.get(key))
        if key == "min_lr":
            if current_value != expected_min_lr:
                diffs[key] = {"baseline": baseline_value, "current": current_value}
            continue
        if current_value != baseline_value:
            diffs[key] = {"baseline": baseline_value, "current": current_value}

    (out_dir / "config_diff.json").write_text(
        json.dumps(diffs, indent=2, sort_keys=True) + "\n"
    )
    if diffs:
        raise SystemExit(
            f"Invalid {args.label} config drift: " + json.dumps(diffs, sort_keys=True)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
