"""
Fractal Cadence Auto-Research Sweep
====================================
Runs a grid of fractal experiments on DGX Spark, logs results to CSV.
Each experiment is ~2-3 minutes (300 steps). Full sweep in ~1-2 hours.

Usage:
  source .venv/bin/activate
  python sweep_fractal.py                    # full sweep
  python sweep_fractal.py --quick            # quick sweep (fewer configs)
  python sweep_fractal.py --custom '{"cadence":2,"num_loops":4}'  # single custom run
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

SCRIPT = "train_fractal_cadence.py"
RESULTS_FILE = "sweep_results.csv"
FIELDS = [
    "timestamp", "run_id", "cadence", "cadence_offset", "num_unique_layers",
    "num_loops", "model_dim", "gravity", "val_bpb", "steps", "f_steps",
    "n_steps", "avg_ms", "time_s", "params", "peak_vram", "notes"
]

# ─── SEARCH SPACE ─────────────────────────────────────────────────────────────

FULL_SWEEP = [
    # Cadence variations (core experiment)
    {"cadence": 1, "notes": "always fractal (control)"},
    {"cadence": 0, "notes": "never fractal (control)"},
    {"cadence": 2, "cadence_offset": 0, "notes": "F/N pattern"},
    {"cadence": 3, "cadence_offset": 2, "notes": "N/N/F pattern"},
    {"cadence": 4, "cadence_offset": 0, "notes": "F/N/N/N pattern"},
    {"cadence": 3, "cadence_offset": 0, "notes": "F/N/N pattern"},

    # Loop count variations
    {"cadence": 2, "num_loops": 2, "notes": "2 loops, cadence 2"},
    {"cadence": 2, "num_loops": 4, "notes": "4 loops, cadence 2"},
    {"cadence": 1, "num_loops": 2, "notes": "always fractal, 2 loops"},
    {"cadence": 1, "num_loops": 4, "notes": "always fractal, 4 loops"},

    # Layer count variations (auto-dim adjusts width)
    {"cadence": 2, "num_unique_layers": 2, "notes": "2 layers x3 loops"},
    {"cadence": 2, "num_unique_layers": 4, "notes": "4 layers x3 loops"},
    {"cadence": 2, "num_unique_layers": 5, "notes": "5 layers x3 loops"},

    # Gravity interactions
    {"cadence": 2, "gravity": True, "notes": "cadence 2 + gravity"},
    {"cadence": 1, "gravity": True, "notes": "always fractal + gravity"},

    # Higher cadence (more normalize steps)
    {"cadence": 5, "cadence_offset": 0, "notes": "F/N/N/N/N pattern"},
    {"cadence": 2, "cadence_offset": 1, "notes": "N/F pattern (phase shifted)"},
]

QUICK_SWEEP = [
    {"cadence": 1, "notes": "always fractal (control)"},
    {"cadence": 0, "notes": "never fractal (control)"},
    {"cadence": 2, "cadence_offset": 0, "notes": "F/N pattern"},
    {"cadence": 3, "cadence_offset": 2, "notes": "N/N/F pattern"},
    {"cadence": 2, "num_loops": 2, "notes": "2 loops, cadence 2"},
    {"cadence": 2, "num_loops": 4, "notes": "4 loops, cadence 2"},
]

# ─── DEFAULTS ─────────────────────────────────────────────────────────────────

DEFAULTS = {
    "cadence": 2,
    "cadence_offset": 0,
    "num_unique_layers": 3,
    "num_loops": 3,
    "model_dim": 0,  # 0 = auto-size
    "gravity": False,
    "iterations": 300,
    "eval_tokens": 100000,
    "max_seconds": 300,
    "batch_tokens": 32768,
    "seq_len": 1024,
    "seed": 1337,
}

# ─── RUNNER ───────────────────────────────────────────────────────────────────

def run_experiment(config, run_id):
    """Run one experiment, return parsed results dict."""
    cfg = {**DEFAULTS, **config}

    cmd = [
        sys.executable, SCRIPT,
        "--cadence", str(cfg["cadence"]),
        "--cadence-offset", str(cfg["cadence_offset"]),
        "--num-unique-layers", str(cfg["num_unique_layers"]),
        "--num-loops", str(cfg["num_loops"]),
        "--iterations", str(cfg["iterations"]),
        "--eval-tokens", str(cfg["eval_tokens"]),
        "--max-seconds", str(cfg["max_seconds"]),
        "--batch-tokens", str(cfg["batch_tokens"]),
        "--seq-len", str(cfg["seq_len"]),
        "--seed", str(cfg["seed"]),
        "--run-id", run_id,
    ]
    if cfg.get("model_dim", 0) > 0:
        cmd.extend(["--model-dim", str(cfg["model_dim"])])
    if cfg.get("gravity", False):
        cmd.append("--gravity")

    print(f"\n{'='*60}")
    print(f"RUN: {run_id}")
    print(f"Config: {json.dumps({k:v for k,v in cfg.items() if k in config or k=='cadence'}, indent=2)}")
    print(f"{'='*60}")

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    elapsed = time.time() - t0

    # Parse output
    stdout = result.stdout
    stderr = result.stderr
    if result.returncode != 0:
        print(f"FAILED (exit {result.returncode})")
        print(stderr[-500:] if stderr else "no stderr")
        return None

    # Extract results from output
    parsed = {
        "timestamp": datetime.now().isoformat(),
        "run_id": run_id,
        "cadence": cfg["cadence"],
        "cadence_offset": cfg["cadence_offset"],
        "num_unique_layers": cfg["num_unique_layers"],
        "num_loops": cfg["num_loops"],
        "model_dim": cfg.get("model_dim", 0),
        "gravity": cfg.get("gravity", False),
        "notes": cfg.get("notes", ""),
    }

    for line in stdout.split("\n"):
        if "val_bpb:" in line and "RESULTS" not in line and "val_bpb:enabled" not in line:
            try:
                for part in line.split():
                    if part.startswith("val_bpb:"):
                        parsed["val_bpb"] = float(part.split(":")[1])
            except (ValueError, IndexError):
                pass
        if line.startswith("steps:"):
            try:
                parts = line.split()
                parsed["steps"] = int(parts[0].split(":")[1])
                for p in parts:
                    if p.startswith("(F:"):
                        parsed["f_steps"] = int(p.split(":")[1])
                    if p.startswith("N:"):
                        parsed["n_steps"] = int(p.rstrip(")").split(":")[1])
            except (ValueError, IndexError):
                pass
        if "avg_ms:" in line:
            try:
                for part in line.split():
                    if part.startswith("avg_ms:"):
                        parsed["avg_ms"] = float(part.split(":")[1].rstrip("ms/step"))
            except (ValueError, IndexError):
                pass
        if "time:" in line and "train_time" not in line:
            try:
                for part in line.split():
                    if part.startswith("time:"):
                        parsed["time_s"] = float(part.split(":")[1].rstrip("s"))
            except (ValueError, IndexError):
                pass
        if "params:" in line and "model_params" not in line:
            try:
                for part in line.split():
                    if part.startswith("params:"):
                        parsed["params"] = part.split(":")[1].replace(",", "")
            except (ValueError, IndexError):
                pass
        if "peak_vram:" in line:
            try:
                for part in line.split():
                    if part.startswith("peak_vram:"):
                        parsed["peak_vram"] = part.split(":")[1]
            except (ValueError, IndexError):
                pass

    val_bpb = parsed.get("val_bpb", "?")
    print(f"\n>>> Result: val_bpb={val_bpb} ({elapsed:.0f}s)")
    return parsed

def append_result(result, filepath):
    """Append one result row to CSV."""
    exists = Path(filepath).exists()
    with open(filepath, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if not exists:
            w.writeheader()
        w.writerow(result)

def print_leaderboard(filepath):
    """Print sorted results."""
    if not Path(filepath).exists():
        return
    rows = []
    with open(filepath) as f:
        for row in csv.DictReader(f):
            try:
                row["val_bpb"] = float(row.get("val_bpb", 999))
            except ValueError:
                row["val_bpb"] = 999
            rows.append(row)
    rows.sort(key=lambda r: r["val_bpb"])
    print(f"\n{'='*80}")
    print("LEADERBOARD")
    print(f"{'='*80}")
    print(f"{'#':>3} {'val_bpb':>8} {'cadence':>7} {'layers':>6} {'loops':>5} {'steps':>5} {'notes'}")
    print(f"{'-'*80}")
    for i, r in enumerate(rows[:20]):
        print(f"{i+1:>3} {r['val_bpb']:>8.4f} {r.get('cadence','?'):>7} "
              f"{r.get('num_unique_layers','?'):>6} {r.get('num_loops','?'):>5} "
              f"{r.get('steps','?'):>5} {r.get('notes','')}")

# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="Quick sweep (6 configs)")
    p.add_argument("--custom", type=str, help="Single custom config as JSON")
    p.add_argument("--output", type=str, default=RESULTS_FILE)
    args = p.parse_args()

    if args.custom:
        configs = [json.loads(args.custom)]
    elif args.quick:
        configs = QUICK_SWEEP
    else:
        configs = FULL_SWEEP

    print(f"Fractal Cadence Sweep — {len(configs)} experiments")
    print(f"Results → {args.output}")

    for i, config in enumerate(configs):
        run_id = f"sweep_{i:02d}_{config.get('notes','').replace(' ','_')[:30]}"
        try:
            result = run_experiment(config, run_id)
            if result:
                append_result(result, args.output)
        except subprocess.TimeoutExpired:
            print(f"TIMEOUT: {run_id}")
        except Exception as e:
            print(f"ERROR: {run_id}: {e}")

        print_leaderboard(args.output)

    print(f"\nDone. Full results in {args.output}")

if __name__ == "__main__":
    main()
