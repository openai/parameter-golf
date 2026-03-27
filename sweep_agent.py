#!/usr/bin/env python3
"""
Greedy hyperparameter sweep for Parameter Golf on Mac Mini M1.
Baseline to beat: 2.3113 bpb (200 steps, no innovations)
"""
import subprocess, json, os, time
from pathlib import Path

SEARCH_SPACE = {
    "NUM_LAYERS":       ["8", "9", "10", "11"],
    "MLP_MULT":         ["2", "3", "4"],
    "NUM_HEADS":        ["4", "8", "16"],
    "WEIGHT_DECAY":     ["0.01", "0.04", "0.1"],
}

DEFAULT_CONFIG = {
    "NUM_LAYERS":   "9",
    "MLP_MULT":     "2",
    "NUM_HEADS":    "8",
    "WEIGHT_DECAY": "0.1",
}

SMOKE_ENV = {
    "ITERATIONS":         "200",
    "TRAIN_BATCH_TOKENS": "8192",
    "TRAIN_SEQ_LEN":      "512",
    "VAL_LOSS_EVERY":     "999999",  # skip mid-run val — only care about final
    "WARMUP_STEPS":       "5",
}

def run_smoke(config: dict, run_id: str) -> float:
    env = {**os.environ, **SMOKE_ENV, **config, "RUN_ID": run_id}
    t0 = time.time()
    try:
        result = subprocess.run(
            ["python3", "train_gpt_mlx.py"],
            env=env, capture_output=True, text=True,
            timeout=900, cwd=Path(__file__).parent
        )
    except subprocess.TimeoutExpired:
        print(f"  [{run_id}] TIMEOUT")
        return 9.99

    elapsed = time.time() - t0
    bpb = 9.99

    for line in (result.stdout + result.stderr).splitlines():
        if "final_int8_zlib_roundtrip_exact" in line:
            try:
                bpb = float(line.split("val_bpb:")[-1].strip().split()[0])
            except: pass

    # Fallback: grab last val_bpb if roundtrip line not found
    if bpb == 9.99:
        for line in (result.stdout + result.stderr).splitlines():
            if "val_bpb" in line:
                try:
                    candidate = float(line.split("val_bpb:")[-1].strip().split()[0])
                    if 0.5 < candidate < 5.0:
                        bpb = candidate
                except: pass

    tag = "✅" if bpb < 9.0 else "❌ parse failed"
    print(f"  [{run_id}] bpb={bpb:.4f} | {elapsed:.0f}s {tag}")

    # Save log for debugging
    log_dir = Path.home() / "pg_smoke_logs"
    log_dir.mkdir(exist_ok=True)
    (log_dir / f"{run_id}.log").write_text(result.stdout + result.stderr)

    return bpb

def greedy_sweep():
    results = []
    best_config = DEFAULT_CONFIG.copy()

    print("📊 Running default config as baseline...")
    best_bpb = run_smoke(best_config, "sweep_default")
    results.append({"config": best_config.copy(), "bpb": best_bpb, "id": "default"})
    print(f"Default bpb: {best_bpb:.4f}\n")

    for factor, values in SEARCH_SPACE.items():
        print(f"🔍 Sweeping: {factor} (current: {best_config[factor]})")
        for val in values:
            if val == best_config[factor]:
                continue
            candidate = {**best_config, factor: val}
            run_id = f"{factor}_{val}"
            bpb = run_smoke(candidate, run_id)
            results.append({"config": candidate.copy(), "bpb": bpb, "id": run_id})
            if bpb < best_bpb:
                print(f"  🎯 New best! {factor}={val}: {bpb:.4f} (was {best_bpb:.4f})")
                best_bpb = bpb
                best_config = candidate.copy()

    results.sort(key=lambda x: x["bpb"])
    Path("sweep_results.json").write_text(json.dumps(results, indent=2))

    print(f"\n{'='*50}")
    print(f"🏆 Best bpb:    {best_bpb:.4f}")
    print(f"🏆 Best config: {best_config}")
    print(f"📄 Saved to sweep_results.json")
    print(f"📁 Logs in ~/pg_smoke_logs/")

    print(f"\n# Paste this on Runpod:")
    cfg = " \\\n".join(f"{k}={v}" for k, v in best_config.items())
    print(
        f"OMP_NUM_THREADS=1 \\\nRUN_ID=kl_final_v1 \\\n"
        f"MAX_WALLCLOCK_SECONDS=590 \\\nITERATIONS=500000 \\\n"
        f"TRAIN_BATCH_TOKENS=524288 \\\nTRAIN_SEQ_LEN=1024 \\\n"
        f"{cfg} \\\n"
        f"torchrun --standalone --nproc_per_node=8 train_gpt.py"
    )

if __name__ == "__main__":
    print("🚀 Parameter Golf Sweep — KaiLean")
    print(f"Baseline to beat: 2.3113 bpb")
    print(f"Estimated time: ~2 hours on M1\n")
    greedy_sweep()
