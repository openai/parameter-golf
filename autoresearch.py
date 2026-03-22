"""
Fractal Auto-Research: Overnight Autonomous Optimization
=========================================================
Runs continuously on DGX Spark. Phases:
  1. EXPLORE: diverse initial configs to map the landscape
  2. EXPLOIT: hill-climb from best config, perturbing one axis at a time
  3. REFINE: fine-tune the best region with smaller steps
  4. COMBO: test best values from each axis together

Logs everything to autoresearch_results.csv with live leaderboard.
Designed to run unattended for 8+ hours.

Usage:
  source .venv/bin/activate
  nohup python autoresearch.py > autoresearch.log 2>&1 &
  tail -f autoresearch.log
"""

import csv
import json
import math
import os
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

SCRIPT = "train_fractal_cadence.py"
RESULTS_FILE = "autoresearch_results.csv"
FIELDS = [
    "timestamp", "run_id", "phase", "generation", "val_bpb",
    "cadence", "cadence_offset", "num_unique_layers", "num_loops",
    "lr", "grad_clip", "mlp_mult", "model_dim",
    "steps", "f_steps", "n_steps", "avg_ms", "time_s", "params", "notes"
]

# ─── CONFIG SPACE ─────────────────────────────────────────────────────────────

BASE = {
    "cadence": 2,
    "cadence_offset": 0,
    "num_unique_layers": 3,
    "num_loops": 3,
    "lr": 3e-4,
    "grad_clip": 1.0,
    "mlp_mult": 2,
    "model_dim": 0,  # auto-size
}

# Axes to search and their candidate values
AXES = {
    "num_unique_layers": [2, 3, 4, 5, 6],
    "num_loops":         [1, 2, 3, 4],
    "cadence":           [0, 1, 2, 3, 4],
    "lr":                [1e-4, 2e-4, 3e-4, 5e-4, 8e-4, 1e-3],
    "grad_clip":         [0.3, 0.5, 1.0, 1.5, 2.0],
    "mlp_mult":          [2, 3],
}

RUN_DEFAULTS = {
    "iterations": 300,
    "eval_tokens": 100000,
    "max_seconds": 300,
    "batch_tokens": 32768,
    "seq_len": 1024,
    "seed": 1337,
}

# ─── PHASE 1: EXPLORE ────────────────────────────────────────────────────────

def gen_explore_configs():
    """Diverse initial configs to map the landscape."""
    configs = [
        # Controls
        {"cadence": 1, "notes": "ctrl: always fractal 3x3"},
        {"cadence": 0, "notes": "ctrl: never fractal (single pass)"},
        {"cadence": 2, "notes": "ctrl: cadence2 baseline 3x3"},

        # Architecture extremes
        {"num_unique_layers": 2, "num_loops": 4, "cadence": 2, "notes": "deep: 2L x4 loops"},
        {"num_unique_layers": 6, "num_loops": 2, "cadence": 2, "notes": "wide: 6L x2 loops"},
        {"num_unique_layers": 4, "num_loops": 3, "cadence": 2, "notes": "balanced: 4L x3 loops"},
        {"num_unique_layers": 3, "num_loops": 2, "cadence": 2, "notes": "minimal: 3L x2 loops"},
        {"num_unique_layers": 5, "num_loops": 3, "cadence": 1, "notes": "deep always: 5L x3"},

        # LR extremes
        {"lr": 1e-4, "notes": "lr: low"},
        {"lr": 1e-3, "notes": "lr: high"},

        # Grad clip
        {"grad_clip": 0.3, "notes": "clip: tight"},
        {"grad_clip": 2.0, "notes": "clip: loose"},

        # Cadence patterns
        {"cadence": 3, "cadence_offset": 2, "notes": "cadence: N/N/F"},
        {"cadence": 4, "cadence_offset": 0, "notes": "cadence: F/N/N/N"},

        # MLP
        {"mlp_mult": 3, "notes": "mlp: 3x"},
    ]
    return configs


# ─── PHASE 2: EXPLOIT (hill climb) ───────────────────────────────────────────

def gen_exploit_configs(best_config, results):
    """Perturb the best config one axis at a time."""
    configs = []
    for axis, values in AXES.items():
        current = best_config.get(axis, BASE[axis])
        for v in values:
            if v == current:
                continue
            cfg = {**best_config, axis: v}
            # Fix cadence_offset if cadence changed
            if axis == "cadence" and v > 0:
                cfg["cadence_offset"] = 0
            # Skip if cadence=0 with loops>1 (meaningless)
            if cfg.get("cadence", 2) == 0 and cfg.get("num_loops", 3) > 1:
                cfg["num_loops"] = 1
            cfg["notes"] = f"exploit: {axis}={v}"
            # Skip if already tested
            if not already_tested(cfg, results):
                configs.append(cfg)
    return configs


# ─── PHASE 3: REFINE ─────────────────────────────────────────────────────────

def gen_refine_configs(best_config, results):
    """Fine-grained search around the best config."""
    configs = []
    # LR refinement: ±20%, ±40% of best
    best_lr = best_config.get("lr", 3e-4)
    for mult in [0.6, 0.8, 1.2, 1.4, 1.6]:
        lr = round(best_lr * mult, 6)
        cfg = {**best_config, "lr": lr, "notes": f"refine: lr={lr:.1e}"}
        if not already_tested(cfg, results):
            configs.append(cfg)

    # Grad clip refinement
    best_clip = best_config.get("grad_clip", 1.0)
    for mult in [0.7, 0.85, 1.15, 1.3]:
        clip = round(best_clip * mult, 2)
        cfg = {**best_config, "grad_clip": clip, "notes": f"refine: clip={clip}"}
        if not already_tested(cfg, results):
            configs.append(cfg)

    # Try different cadence offsets
    cad = best_config.get("cadence", 2)
    if cad > 1:
        for off in range(cad):
            if off == best_config.get("cadence_offset", 0):
                continue
            cfg = {**best_config, "cadence_offset": off, "notes": f"refine: offset={off}"}
            if not already_tested(cfg, results):
                configs.append(cfg)

    return configs


# ─── PHASE 4: COMBO ──────────────────────────────────────────────────────────

def gen_combo_configs(results):
    """Combine the best value from each axis."""
    if len(results) < 10:
        return []

    # Find best value per axis
    best_per_axis = {}
    for axis in AXES:
        axis_results = {}
        for r in results:
            val = r.get(axis)
            if val is not None:
                bpb = r.get("val_bpb", 999)
                if val not in axis_results or bpb < axis_results[val]:
                    axis_results[val] = bpb
        if axis_results:
            best_val = min(axis_results, key=axis_results.get)
            best_per_axis[axis] = best_val

    configs = []
    # Full combo
    combo = {**BASE, **best_per_axis, "notes": "combo: best-of-each-axis"}
    if not already_tested(combo, results):
        configs.append(combo)

    # Combo with variations
    for axis in ["lr", "grad_clip"]:
        if axis in best_per_axis:
            for mult in [0.8, 1.2]:
                val = best_per_axis[axis] * mult
                if axis == "grad_clip":
                    val = round(val, 2)
                else:
                    val = round(val, 6)
                cfg = {**BASE, **best_per_axis, axis: val,
                       "notes": f"combo+tweak: {axis}={val}"}
                if not already_tested(cfg, results):
                    configs.append(cfg)

    return configs


# ─── RUNNER ───────────────────────────────────────────────────────────────────

def config_key(cfg):
    """Hashable key for dedup."""
    return (
        cfg.get("cadence", 2), cfg.get("cadence_offset", 0),
        cfg.get("num_unique_layers", 3), cfg.get("num_loops", 3),
        round(cfg.get("lr", 3e-4), 6), round(cfg.get("grad_clip", 1.0), 2),
        cfg.get("mlp_mult", 2), cfg.get("model_dim", 0),
    )


def already_tested(cfg, results):
    key = config_key(cfg)
    for r in results:
        rkey = (
            int(r.get("cadence", 2)), int(r.get("cadence_offset", 0)),
            int(r.get("num_unique_layers", 3)), int(r.get("num_loops", 3)),
            round(float(r.get("lr", 3e-4)), 6), round(float(r.get("grad_clip", 1.0)), 2),
            int(r.get("mlp_mult", 2)), int(r.get("model_dim", 0)),
        )
        if rkey == key:
            return True
    return False


def run_one(config, run_id, phase, generation):
    cfg = {**BASE, **RUN_DEFAULTS, **config}
    cmd = [
        sys.executable, SCRIPT,
        "--cadence", str(cfg["cadence"]),
        "--cadence-offset", str(cfg["cadence_offset"]),
        "--num-unique-layers", str(cfg["num_unique_layers"]),
        "--num-loops", str(cfg["num_loops"]),
        "--lr", str(cfg["lr"]),
        "--grad-clip", str(cfg["grad_clip"]),
        "--mlp-mult", str(cfg["mlp_mult"]),
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

    notes = cfg.get("notes", "")
    print(f"\n[{phase} gen:{generation}] {run_id}: {notes}")
    print(f"  layers={cfg['num_unique_layers']} loops={cfg['num_loops']} "
          f"cadence={cfg['cadence']} lr={cfg['lr']:.1e} clip={cfg['grad_clip']}")

    t0 = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        print("  TIMEOUT")
        return None
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  FAILED (exit {result.returncode})")
        stderr = result.stderr
        if stderr:
            print(f"  {stderr[-200:]}")
        return None

    # Parse
    parsed = {
        "timestamp": datetime.now().isoformat(),
        "run_id": run_id, "phase": phase, "generation": generation,
        "cadence": cfg["cadence"], "cadence_offset": cfg["cadence_offset"],
        "num_unique_layers": cfg["num_unique_layers"], "num_loops": cfg["num_loops"],
        "lr": cfg["lr"], "grad_clip": cfg["grad_clip"],
        "mlp_mult": cfg["mlp_mult"], "model_dim": cfg.get("model_dim", 0),
        "notes": notes,
    }
    stdout = result.stdout
    for line in stdout.split("\n"):
        if "val_bpb:" in line and "RESULTS" not in line and "val_bpb:enabled" not in line:
            try:
                for p in line.split():
                    if p.startswith("val_bpb:"):
                        parsed["val_bpb"] = float(p.split(":")[1])
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
                for p in line.split():
                    if p.startswith("avg_ms:"):
                        parsed["avg_ms"] = float(p.split(":")[1].rstrip("ms/step"))
            except (ValueError, IndexError):
                pass
        if "time:" in line and "train_time" not in line:
            try:
                for p in line.split():
                    if p.startswith("time:"):
                        parsed["time_s"] = float(p.split(":")[1].rstrip("s"))
            except (ValueError, IndexError):
                pass
        if "params:" in line and "model_params" not in line:
            try:
                for p in line.split():
                    if p.startswith("params:"):
                        parsed["params"] = p.split(":")[1].replace(",", "")
            except (ValueError, IndexError):
                pass

    bpb = parsed.get("val_bpb", "?")
    print(f"  >>> val_bpb={bpb} ({elapsed:.0f}s)")
    return parsed


def load_results():
    results = []
    if Path(RESULTS_FILE).exists():
        with open(RESULTS_FILE) as f:
            for row in csv.DictReader(f):
                results.append(row)
    return results


def save_result(result):
    exists = Path(RESULTS_FILE).exists()
    with open(RESULTS_FILE, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if not exists:
            w.writeheader()
        w.writerow(result)


def get_best(results):
    valid = [r for r in results if r.get("val_bpb") and float(r.get("val_bpb", 999)) < 100]
    if not valid:
        return BASE, 999.0
    best = min(valid, key=lambda r: float(r["val_bpb"]))
    cfg = {}
    for k in ["cadence", "cadence_offset", "num_unique_layers", "num_loops", "mlp_mult", "model_dim"]:
        if k in best and best[k]:
            cfg[k] = int(best[k])
    for k in ["lr", "grad_clip"]:
        if k in best and best[k]:
            cfg[k] = float(best[k])
    if "notes" in best:
        cfg["notes"] = best["notes"]
    return cfg, float(best["val_bpb"])


def print_leaderboard(results):
    valid = [r for r in results if r.get("val_bpb") and float(r.get("val_bpb", 999)) < 100]
    valid.sort(key=lambda r: float(r["val_bpb"]))
    print(f"\n{'='*90}")
    print(f"LEADERBOARD ({len(valid)} runs)")
    print(f"{'='*90}")
    print(f"{'#':>3} {'bpb':>8} {'phase':>8} {'L':>2}x{'lp':>2} {'cad':>3} {'lr':>9} {'clip':>5} {'notes'}")
    for i, r in enumerate(valid[:15]):
        print(f"{i+1:>3} {float(r['val_bpb']):>8.4f} {r.get('phase','?'):>8} "
              f"{r.get('num_unique_layers','?'):>2}x{r.get('num_loops','?'):>2} "
              f"{r.get('cadence','?'):>3} {float(r.get('lr',0)):>9.1e} "
              f"{float(r.get('grad_clip',0)):>5.1f} {r.get('notes','')[:35]}")
    print()


# ─── MAIN LOOP ────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("FRACTAL AUTO-RESEARCH — Overnight Optimization")
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Results: {RESULTS_FILE}")
    print("=" * 60)

    results = load_results()
    generation = 0
    run_count = len(results)

    # Phase 1: EXPLORE
    if run_count < 15:
        print("\n>>> PHASE 1: EXPLORE")
        explore = gen_explore_configs()
        for cfg in explore:
            if already_tested(cfg, results):
                continue
            run_count += 1
            generation += 1
            rid = f"auto_{run_count:03d}_explore"
            r = run_one(cfg, rid, "explore", generation)
            if r:
                save_result(r)
                results.append(r)
            print_leaderboard(results)

    # Phase 2: EXPLOIT (hill climb from best)
    best_cfg, best_bpb = get_best(results)
    print(f"\n>>> PHASE 2: EXPLOIT (best so far: {best_bpb:.4f})")
    exploit = gen_exploit_configs(best_cfg, results)
    for cfg in exploit:
        run_count += 1
        generation += 1
        rid = f"auto_{run_count:03d}_exploit"
        r = run_one(cfg, rid, "exploit", generation)
        if r:
            save_result(r)
            results.append(r)
        print_leaderboard(results)
        # Update best after each run
        new_best, new_bpb = get_best(results)
        if new_bpb < best_bpb:
            best_cfg, best_bpb = new_best, new_bpb
            print(f"  *** NEW BEST: {best_bpb:.4f} ***")

    # Phase 3: REFINE
    best_cfg, best_bpb = get_best(results)
    print(f"\n>>> PHASE 3: REFINE (best: {best_bpb:.4f})")
    refine = gen_refine_configs(best_cfg, results)
    for cfg in refine:
        run_count += 1
        generation += 1
        rid = f"auto_{run_count:03d}_refine"
        r = run_one(cfg, rid, "refine", generation)
        if r:
            save_result(r)
            results.append(r)
        print_leaderboard(results)
        new_best, new_bpb = get_best(results)
        if new_bpb < best_bpb:
            best_cfg, best_bpb = new_best, new_bpb
            print(f"  *** NEW BEST: {best_bpb:.4f} ***")

    # Phase 4: COMBO
    print(f"\n>>> PHASE 4: COMBO")
    combos = gen_combo_configs(results)
    for cfg in combos:
        run_count += 1
        generation += 1
        rid = f"auto_{run_count:03d}_combo"
        r = run_one(cfg, rid, "combo", generation)
        if r:
            save_result(r)
            results.append(r)
        print_leaderboard(results)

    # Phase 5: REPEAT exploit/refine loop until stopped
    cycle = 0
    while True:
        cycle += 1
        best_cfg, best_bpb = get_best(results)
        print(f"\n>>> CYCLE {cycle}: EXPLOIT+REFINE (best: {best_bpb:.4f})")

        # Random perturbation of best (exploration)
        for _ in range(5):
            cfg = {**best_cfg}
            axis = random.choice(list(AXES.keys()))
            cfg[axis] = random.choice(AXES[axis])
            cfg["notes"] = f"cycle{cycle}: random {axis}={cfg[axis]}"
            if cfg["cadence"] > 0:
                cfg["cadence_offset"] = random.randint(0, max(cfg["cadence"] - 1, 0))
            if not already_tested(cfg, results):
                run_count += 1
                generation += 1
                rid = f"auto_{run_count:03d}_c{cycle}"
                r = run_one(cfg, rid, f"cycle{cycle}", generation)
                if r:
                    save_result(r)
                    results.append(r)

        # Targeted refinement of current best
        best_cfg, best_bpb = get_best(results)
        refine = gen_refine_configs(best_cfg, results)
        for cfg in refine[:5]:  # limit per cycle
            run_count += 1
            generation += 1
            rid = f"auto_{run_count:03d}_c{cycle}r"
            r = run_one(cfg, rid, f"cycle{cycle}", generation)
            if r:
                save_result(r)
                results.append(r)

        print_leaderboard(results)
        new_best, new_bpb = get_best(results)
        if new_bpb < best_bpb:
            print(f"  *** CYCLE {cycle} NEW BEST: {new_bpb:.4f} ***")

        print(f"\nTotal runs: {run_count} | Best: {new_bpb:.4f} | Time: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
