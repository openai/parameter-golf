#!/usr/bin/env python3
"""
Parallel autoresearch sweep — runs 3 experiments simultaneously.
MLX serializes GPU access but CPU/memory overhead is minimal.
On M3 Ultra 96GB, 3 parallel experiments use ~4GB each = ~12GB total.
"""
import sys
import os
import subprocess
import time
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent))

VENV_PYTHON = str(Path(__file__).parent.parent / ".venv/bin/python")
RUNNER = str(Path(__file__).parent / "run_experiment.py")
RESULTS_FILE = Path(__file__).parent / "results.jsonl"

# Skip already-completed experiments
def get_done():
    if not RESULTS_FILE.exists():
        return set()
    done = set()
    for line in RESULTS_FILE.read_text().strip().split("\n"):
        if line:
            done.add(json.loads(line)["name"])
    return done

ITERS = 1000

ALL_EXPERIMENTS = [
    # Already done: baseline_12L, 11L_baseline, 13L, 12L_mlp2x
    ("12L_mlp4x", ["MLP_MULT=4"]),
    ("12L_bigram2048", ["BIGRAM_HASH_SIZE=2048"]),
    ("12L_bigram8192", ["BIGRAM_HASH_SIZE=8192"]),
    ("12L_muon095", ["MUON_MOMENTUM=0.95", "MUON_MOMENTUM_WARMUP_START=0.85"]),
    ("12L_warmdown5000", ["WARMDOWN_ITERS=5000"]),
    ("12L_warmdown2000", ["WARMDOWN_ITERS=2000"]),
    ("12L_qat020", ["QAT_START_FRACTION=0.20"]),
    ("12L_qat010", ["QAT_START_FRACTION=0.10"]),
    ("12L_zloss0", ["Z_LOSS_WEIGHT=0"]),
    ("12L_zloss1e3", ["Z_LOSS_WEIGHT=0.001"]),
    ("12L_noclip", ["GRAD_CLIP_NORM=0"]),
    ("12L_clip05", ["GRAD_CLIP_NORM=0.5"]),
    ("12L_rope32", ["ROPE_DIMS=32"]),
    ("12L_rope0_full", ["ROPE_DIMS=0"]),
    ("12L_nomtp", ["MTP_NUM_HEADS=0"]),
    ("12L_mtp3", ["MTP_NUM_HEADS=3"]),
    ("12L_ema0998", ["EMA_DECAY=0.998"]),
    ("12L_ema0995", ["EMA_DECAY=0.995"]),
]

def run_one(name_and_env):
    name, env_list = name_and_env
    cmd = [VENV_PYTHON, RUNNER, "--name", name, "--iters", str(ITERS)]
    if env_list:
        cmd.extend(["--env"] + env_list)
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200,
                           cwd=str(Path(__file__).parent.parent))
    elapsed = time.time() - t0
    return name, elapsed, result.returncode

def print_leaderboard():
    if not RESULTS_FILE.exists():
        return
    results = []
    for line in RESULTS_FILE.read_text().strip().split("\n"):
        if line:
            results.append(json.loads(line))
    results = [r for r in results if r.get("val_bpb")]
    results.sort(key=lambda r: r.get("val_bpb", 999))
    print(f"\n{'='*65}")
    print(f"{'Name':<30} {'bpb':>8} {'int6':>8} {'time':>8}")
    print(f"{'='*65}")
    for r in results:
        bpb = f"{r.get('val_bpb', 0):.4f}"
        int6 = f"{r.get('int6_bpb', 0):.4f}" if "int6_bpb" in r else "N/A"
        t = f"{r.get('elapsed_s', 0):.0f}s"
        print(f"{r['name']:<30} {bpb:>8} {int6:>8} {t:>8}")

if __name__ == "__main__":
    done = get_done()
    remaining = [(n, e) for n, e in ALL_EXPERIMENTS if n not in done]
    print(f"Done: {len(done)}, Remaining: {len(remaining)}, Parallel: 3")
    print(f"Estimated: {len(remaining) * 28 / 3 / 60:.1f} hours\n")

    with ProcessPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(run_one, exp): exp[0] for exp in remaining}
        for future in as_completed(futures):
            name = futures[future]
            try:
                name, elapsed, code = future.result()
                status = "OK" if code == 0 else f"FAIL({code})"
                print(f"  {name}: {status} ({elapsed:.0f}s)")
            except Exception as e:
                print(f"  {name}: ERROR: {e}")
            print_leaderboard()

    print("\n\nFINAL LEADERBOARD:")
    print_leaderboard()
