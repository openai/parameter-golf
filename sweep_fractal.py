"""
Fractal/Hybrid Architecture Sweep — 4-hour automated run
=========================================================
Tests systematically: what's the best balance of weight sharing vs flat layers?

Each test: ~300 steps, ~3 min. 4 hours ≈ 80 tests.
Results saved to sweep_fractal_results.csv

Usage:
  source .venv/bin/activate
  nohup python sweep_fractal.py > sweep_fractal.log 2>&1 &
  tail -f sweep_fractal.log
"""

import csv
import os
import subprocess
import sys
import time
from datetime import datetime

RESULTS_FILE = "sweep_fractal_results.csv"
FIELDS = [
    "timestamp", "run_id", "mode", "num_layers", "num_unique_layers", "num_loops",
    "effective_depth", "model_dim", "num_heads", "num_kv_heads", "mlp_mult",
    "lr", "val_bpb", "params", "steps", "avg_ms", "time_s",
    "estimated_h100_steps", "notes"
]

H100_SPEED_FACTOR = 1.5
H100_WALLCLOCK_MS = 600_000

# head_dim must be multiple of 8 for FA3
# 512/16=32ok 512/8=64ok 384/8=48ok 384/12=32ok 448/8=56ok 640/8=80ok

CONFIGS = [
    # === BASELINES ===
    {"mode": "baseline", "num_layers": 11, "model_dim": 512, "num_heads": 8, "num_kv_heads": 4, "mlp_mult": 3, "lr": 3e-4,
     "notes": "SOTA baseline 11L/512d/8H/3xMLP"},
    {"mode": "baseline", "num_layers": 11, "model_dim": 512, "num_heads": 8, "num_kv_heads": 4, "mlp_mult": 4, "lr": 3e-4,
     "notes": "11L + 4xMLP"},
    {"mode": "baseline", "num_layers": 9, "model_dim": 512, "num_heads": 8, "num_kv_heads": 4, "mlp_mult": 3, "lr": 8e-4,
     "notes": "Qwen local winner: 9L high LR"},
    {"mode": "baseline", "num_layers": 9, "model_dim": 512, "num_heads": 8, "num_kv_heads": 4, "mlp_mult": 4, "lr": 5e-4,
     "notes": "9L 4xMLP"},
    {"mode": "baseline", "num_layers": 7, "model_dim": 512, "num_heads": 8, "num_kv_heads": 4, "mlp_mult": 4, "lr": 5e-4,
     "notes": "7L 4xMLP (fast, fewer params)"},
    {"mode": "baseline", "num_layers": 8, "model_dim": 512, "num_heads": 8, "num_kv_heads": 4, "mlp_mult": 4, "lr": 5e-4,
     "notes": "8L 4xMLP"},
    {"mode": "baseline", "num_layers": 13, "model_dim": 384, "num_heads": 8, "num_kv_heads": 4, "mlp_mult": 3, "lr": 3e-4,
     "notes": "13L narrow"},
    {"mode": "baseline", "num_layers": 11, "model_dim": 512, "num_heads": 16, "num_kv_heads": 8, "mlp_mult": 3, "lr": 3e-4,
     "notes": "11L 16H"},

    # === FRACTAL 2-LOOP ===
    {"mode": "fractal", "num_unique_layers": 6, "num_loops": 2, "model_dim": 512, "num_heads": 8, "num_kv_heads": 4, "mlp_mult": 3, "lr": 3e-4,
     "notes": "6x2=12eff SOTA dims"},
    {"mode": "fractal", "num_unique_layers": 6, "num_loops": 2, "model_dim": 512, "num_heads": 8, "num_kv_heads": 4, "mlp_mult": 4, "lr": 3e-4,
     "notes": "6x2=12eff 4xMLP"},
    {"mode": "fractal", "num_unique_layers": 6, "num_loops": 2, "model_dim": 512, "num_heads": 8, "num_kv_heads": 4, "mlp_mult": 4, "lr": 5e-4,
     "notes": "6x2=12eff 4xMLP hi-lr"},
    {"mode": "fractal", "num_unique_layers": 5, "num_loops": 2, "model_dim": 512, "num_heads": 8, "num_kv_heads": 4, "mlp_mult": 3, "lr": 3e-4,
     "notes": "5x2=10eff lighter"},
    {"mode": "fractal", "num_unique_layers": 5, "num_loops": 2, "model_dim": 512, "num_heads": 8, "num_kv_heads": 4, "mlp_mult": 4, "lr": 3e-4,
     "notes": "5x2=10eff 4xMLP"},
    {"mode": "fractal", "num_unique_layers": 7, "num_loops": 2, "model_dim": 512, "num_heads": 8, "num_kv_heads": 4, "mlp_mult": 3, "lr": 3e-4,
     "notes": "7x2=14eff deeper"},
    {"mode": "fractal", "num_unique_layers": 7, "num_loops": 2, "model_dim": 384, "num_heads": 8, "num_kv_heads": 4, "mlp_mult": 4, "lr": 5e-4,
     "notes": "7x2=14eff narrow 4xMLP"},
    {"mode": "fractal", "num_unique_layers": 8, "num_loops": 2, "model_dim": 384, "num_heads": 8, "num_kv_heads": 4, "mlp_mult": 3, "lr": 3e-4,
     "notes": "8x2=16eff narrow"},
    {"mode": "fractal", "num_unique_layers": 8, "num_loops": 2, "model_dim": 448, "num_heads": 8, "num_kv_heads": 4, "mlp_mult": 3, "lr": 3e-4,
     "notes": "8x2=16eff mid-width"},

    # === FRACTAL 3-LOOP ===
    {"mode": "fractal", "num_unique_layers": 4, "num_loops": 3, "model_dim": 512, "num_heads": 8, "num_kv_heads": 4, "mlp_mult": 3, "lr": 3e-4,
     "notes": "4x3=12eff heavy sharing"},
    {"mode": "fractal", "num_unique_layers": 4, "num_loops": 3, "model_dim": 512, "num_heads": 8, "num_kv_heads": 4, "mlp_mult": 4, "lr": 3e-4,
     "notes": "4x3=12eff 4xMLP"},
    {"mode": "fractal", "num_unique_layers": 4, "num_loops": 3, "model_dim": 512, "num_heads": 8, "num_kv_heads": 4, "mlp_mult": 4, "lr": 5e-4,
     "notes": "4x3=12eff 4xMLP hi-lr"},
    {"mode": "fractal", "num_unique_layers": 3, "num_loops": 4, "model_dim": 512, "num_heads": 8, "num_kv_heads": 4, "mlp_mult": 4, "lr": 3e-4,
     "notes": "3x4=12eff extreme sharing"},
    {"mode": "fractal", "num_unique_layers": 5, "num_loops": 3, "model_dim": 384, "num_heads": 8, "num_kv_heads": 4, "mlp_mult": 3, "lr": 3e-4,
     "notes": "5x3=15eff narrow deep"},
    {"mode": "fractal", "num_unique_layers": 5, "num_loops": 3, "model_dim": 448, "num_heads": 8, "num_kv_heads": 4, "mlp_mult": 3, "lr": 3e-4,
     "notes": "5x3=15eff mid-width"},
    {"mode": "fractal", "num_unique_layers": 5, "num_loops": 3, "model_dim": 384, "num_heads": 12, "num_kv_heads": 4, "mlp_mult": 4, "lr": 5e-4,
     "notes": "5x3=15eff narrow 12H 4xMLP"},

    # === GRAVITY/ATTNRES ENHANCEMENTS ===
    {"mode": "fractal", "num_unique_layers": 6, "num_loops": 2, "model_dim": 512, "num_heads": 8, "num_kv_heads": 4, "mlp_mult": 3, "lr": 3e-4,
     "gravity": True, "notes": "6x2=12eff + gravity"},
    {"mode": "fractal", "num_unique_layers": 6, "num_loops": 2, "model_dim": 512, "num_heads": 8, "num_kv_heads": 4, "mlp_mult": 4, "lr": 3e-4,
     "gravity": True, "notes": "6x2=12eff + gravity + 4xMLP"},
    {"mode": "fractal", "num_unique_layers": 4, "num_loops": 3, "model_dim": 512, "num_heads": 8, "num_kv_heads": 4, "mlp_mult": 4, "lr": 3e-4,
     "gravity": True, "notes": "4x3=12eff + gravity + 4xMLP"},
    {"mode": "fractal", "num_unique_layers": 6, "num_loops": 2, "model_dim": 512, "num_heads": 8, "num_kv_heads": 4, "mlp_mult": 3, "lr": 3e-4,
     "gravity": True, "attnres": True, "notes": "6x2=12eff + gravity + attnres"},

    # === LR SWEEP on promising configs ===
    {"mode": "fractal", "num_unique_layers": 6, "num_loops": 2, "model_dim": 512, "num_heads": 8, "num_kv_heads": 4, "mlp_mult": 4, "lr": 1e-4,
     "notes": "6x2 4xMLP lr=1e-4"},
    {"mode": "fractal", "num_unique_layers": 6, "num_loops": 2, "model_dim": 512, "num_heads": 8, "num_kv_heads": 4, "mlp_mult": 4, "lr": 8e-4,
     "notes": "6x2 4xMLP lr=8e-4"},
    {"mode": "fractal", "num_unique_layers": 6, "num_loops": 2, "model_dim": 512, "num_heads": 8, "num_kv_heads": 4, "mlp_mult": 4, "lr": 1.2e-3,
     "notes": "6x2 4xMLP lr=1.2e-3"},

    # === WIDER FRACTALS (spend size savings on more dim) ===
    {"mode": "fractal", "num_unique_layers": 4, "num_loops": 3, "model_dim": 640, "num_heads": 8, "num_kv_heads": 4, "mlp_mult": 3, "lr": 3e-4,
     "notes": "4x3=12eff wide 640d"},
    {"mode": "fractal", "num_unique_layers": 5, "num_loops": 2, "model_dim": 640, "num_heads": 8, "num_kv_heads": 4, "mlp_mult": 3, "lr": 3e-4,
     "notes": "5x2=10eff wide 640d"},
    {"mode": "fractal", "num_unique_layers": 6, "num_loops": 2, "model_dim": 640, "num_heads": 8, "num_kv_heads": 4, "mlp_mult": 3, "lr": 3e-4,
     "notes": "6x2=12eff wide 640d"},
    {"mode": "fractal", "num_unique_layers": 4, "num_loops": 3, "model_dim": 640, "num_heads": 8, "num_kv_heads": 4, "mlp_mult": 4, "lr": 5e-4,
     "notes": "4x3=12eff wide 640d 4xMLP"},
]


def save_result(result):
    exists = os.path.exists(RESULTS_FILE)
    with open(RESULTS_FILE, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if not exists:
            w.writeheader()
        w.writerow({k: result.get(k, "") for k in FIELDS})


def run_one(cfg, run_id):
    mode = cfg.get("mode", "baseline")
    cmd = [sys.executable, "train_local.py", "--mode", mode]

    if mode == "baseline":
        cmd += ["--num-layers", str(cfg.get("num_layers", 9))]
    else:
        cmd += ["--num-unique-layers", str(cfg.get("num_unique_layers", 3))]
        cmd += ["--num-loops", str(cfg.get("num_loops", 3))]
        if cfg.get("gravity"):
            cmd.append("--gravity")
        if cfg.get("attnres"):
            cmd.append("--attnres")

    cmd += [
        "--model-dim", str(cfg["model_dim"]),
        "--num-heads", str(cfg["num_heads"]),
        "--num-kv-heads", str(cfg["num_kv_heads"]),
        "--mlp-mult", str(cfg["mlp_mult"]),
        "--lr", str(cfg["lr"]),
        "--seq-len", "1024",
        "--iterations", "500",
        "--eval-tokens", "100000",
        "--max-seconds", "180",
        "--batch-tokens", "32768",
        "--seed", "1337",
        "--run-id", run_id,
    ]

    t0 = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        print("  TIMEOUT")
        return None
    elapsed = time.time() - t0

    if result.returncode != 0:
        stderr = result.stderr[-300:] if result.stderr else ""
        print(f"  FAILED (exit {result.returncode}): {stderr}")
        return None

    parsed = {"timestamp": datetime.now().isoformat(), "run_id": run_id, "time_s": f"{elapsed:.1f}"}
    parsed["mode"] = mode
    if mode == "baseline":
        parsed["num_layers"] = cfg.get("num_layers", 9)
        parsed["effective_depth"] = cfg.get("num_layers", 9)
    else:
        parsed["num_unique_layers"] = cfg.get("num_unique_layers", 3)
        parsed["num_loops"] = cfg.get("num_loops", 3)
        parsed["effective_depth"] = cfg.get("num_unique_layers", 3) * cfg.get("num_loops", 3)
    parsed["model_dim"] = cfg["model_dim"]
    parsed["num_heads"] = cfg["num_heads"]
    parsed["num_kv_heads"] = cfg["num_kv_heads"]
    parsed["mlp_mult"] = cfg["mlp_mult"]
    parsed["lr"] = cfg["lr"]
    parsed["notes"] = cfg.get("notes", "")

    for line in result.stdout.split("\n"):
        if "val_bpb:" in line and "val_bpb:enabled" not in line:
            try:
                parsed["val_bpb"] = float(line.split("val_bpb:")[1].strip().split()[0])
            except (ValueError, IndexError):
                pass
        if line.strip().startswith("params:"):
            try:
                parsed["params"] = line.split("params:")[1].strip().split()[0].replace(",", "")
            except (ValueError, IndexError):
                pass
        if line.strip().startswith("steps:"):
            try:
                parsed["steps"] = line.split("steps:")[1].strip().split()[0]
            except (ValueError, IndexError):
                pass
        if line.strip().startswith("time:"):
            try:
                ms = float(line.split("time:")[1].strip().split()[0].rstrip("ms"))
                steps = int(parsed.get("steps", 0))
                if steps > 0:
                    parsed["avg_ms"] = f"{ms / steps:.1f}"
                    h100_ms = (ms / steps) / H100_SPEED_FACTOR
                    parsed["estimated_h100_steps"] = int(H100_WALLCLOCK_MS / h100_ms)
            except (ValueError, IndexError):
                pass

    return parsed


def main():
    print(f"Fractal/Hybrid Sweep — {len(CONFIGS)} configs, ~3 min each")
    print(f"Estimated runtime: {len(CONFIGS) * 3.5 / 60:.1f} hours")
    print(f"Results: {RESULTS_FILE}")
    print()

    results = []
    for i, cfg in enumerate(CONFIGS):
        run_id = f"sweep_{i:03d}"
        notes = cfg.get("notes", "")
        mode = cfg.get("mode", "baseline")

        if mode == "baseline":
            depth_str = f"{cfg.get('num_layers', 9)}L"
        else:
            ul = cfg.get("num_unique_layers", 3)
            nl = cfg.get("num_loops", 3)
            depth_str = f"{ul}x{nl}={ul*nl}eff"

        print(f"[{i+1}/{len(CONFIGS)}] {depth_str} {cfg['model_dim']}d/{cfg['num_heads']}H/{cfg['mlp_mult']}xMLP lr={cfg['lr']:.0e} | {notes}")

        r = run_one(cfg, run_id)
        if r:
            save_result(r)
            results.append(r)
            bpb = r.get("val_bpb", "?")
            params = r.get("params", "?")
            h100 = r.get("estimated_h100_steps", "?")
            print(f"  => bpb={bpb} params={params} est_h100_steps={h100}")
        else:
            print("  => FAILED")

        if (i + 1) % 5 == 0 and results:
            valid = [r for r in results if r.get("val_bpb")]
            valid.sort(key=lambda r: float(r["val_bpb"]))
            print(f"\n{'='*80}")
            print(f"LEADERBOARD (top 10 of {len(valid)}) after {i+1} runs")
            print(f"{'='*80}")
            for j, r in enumerate(valid[:10]):
                m = r.get("mode", "?")
                d = r.get("effective_depth", "?")
                dim = r.get("model_dim", "?")
                mlp = r.get("mlp_mult", "?")
                bpb = float(r["val_bpb"])
                h = r.get("estimated_h100_steps", "?")
                n = r.get("notes", "")[:40]
                print(f"  {j+1:>2}. bpb={bpb:>7.4f} | {m:>8} depth={d} dim={dim} mlp={mlp}x h100~{h} | {n}")
            print()

    valid = [r for r in results if r.get("val_bpb")]
    valid.sort(key=lambda r: float(r["val_bpb"]))
    print(f"\n{'='*80}")
    print(f"FINAL LEADERBOARD ({len(valid)} runs)")
    print(f"{'='*80}")
    for j, r in enumerate(valid[:20]):
        m = r.get("mode", "?")
        d = r.get("effective_depth", "?")
        dim = r.get("model_dim", "?")
        mlp = r.get("mlp_mult", "?")
        bpb = float(r["val_bpb"])
        p = r.get("params", "?")
        h = r.get("estimated_h100_steps", "?")
        n = r.get("notes", "")[:50]
        print(f"  {j+1:>2}. bpb={bpb:>7.4f} | {m:>8} depth={d} dim={dim} mlp={mlp}x params={p} h100~{h} | {n}")

    best_flat = [r for r in valid if r.get("mode") == "baseline"]
    best_frac = [r for r in valid if r.get("mode") == "fractal"]
    if best_flat and best_frac:
        print(f"\nBest baseline: {float(best_flat[0]['val_bpb']):.4f} ({best_flat[0].get('notes','')})")
        print(f"Best fractal:  {float(best_frac[0]['val_bpb']):.4f} ({best_frac[0].get('notes','')})")
        gap = float(best_frac[0]["val_bpb"]) - float(best_flat[0]["val_bpb"])
        print(f"Gap: {gap:+.4f} ({'fractal wins' if gap < 0 else 'baseline wins'})")


if __name__ == "__main__":
    main()
