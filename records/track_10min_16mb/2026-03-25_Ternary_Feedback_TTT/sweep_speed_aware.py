#!/usr/bin/env python3
"""Speed-aware hyperparameter sweep for Parameter Golf.

Goal: Find the Pareto-optimal innovation dimensions that maximize BPB
within the strict 10-minute wall-clock constraint. Each configuration is
evaluated for a fixed time budget, and the best BPB wins.

Usage:
    python sweep_speed_aware.py

Results are written to sweep_results.md.
"""
import subprocess
import os
import re
import time

RESULTS_FILE = "sweep_results.md"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Base environment shared by all configs
BASE_ENV = {
    "DATA_PATH": os.environ.get("DATA_PATH", "../../../data/datasets/fineweb10B_sp1024"),
    "TOKENIZER_PATH": os.environ.get("TOKENIZER_PATH", "../../../data/tokenizers/fineweb_1024_bpe.model"),
    "VOCAB_SIZE": "1024",
    "NUM_LAYERS": "4",
    "MODEL_DIM": "256",
    "NUM_HEADS": "4",
    "NUM_KV_HEADS": "2",
    "MLP_MULT": "4",
    "EMBED_DIM": "128",
    "TRAIN_BATCH_TOKENS": "16384",
    "GRAD_ACCUM_STEPS": "2",
    "ITERATIONS": "100000",
    "SEED": "42",
    # Features always on
    "FEEDBACK_ENABLED": "1",
    "BIGRAM_HASH_ENABLED": "1",
    "VRL_ENABLED": "1",
    "CAPSULE_ENABLED": "1",
    "KOOPMAN_ENABLED": "1",
    "XSA_START_LAYER": "2",
    "TURBO_QUANT_KV": "0",
    "NGRAM_CACHE_ENABLED": "1",
    "TTT_ENABLED": "1",
}

# Configurations to sweep: (name, overrides)
# Each one tests a different balance of innovation size vs speed
CONFIGS = [
    ("baseline_plain", {
        "FEEDBACK_ENABLED": "0", "BIGRAM_HASH_ENABLED": "0", "VRL_ENABLED": "0",
        "CAPSULE_ENABLED": "0", "KOOPMAN_ENABLED": "0", "XSA_START_LAYER": "4",
    }),
    ("tiny_innovations", {
        "FEEDBACK_DIM": "16", "FEEDBACK_SKETCH_TOKENS": "1",
        "CAPSULE_NUM": "4", "CAPSULE_DIM": "16",
        "BIGRAM_HASH_DIM": "32", "KOOPMAN_RANK": "2",
        "FEEDBACK_EVERY": "4",
    }),
    ("small_innovations", {
        "FEEDBACK_DIM": "32", "FEEDBACK_SKETCH_TOKENS": "2",
        "CAPSULE_NUM": "8", "CAPSULE_DIM": "32",
        "BIGRAM_HASH_DIM": "64", "KOOPMAN_RANK": "2",
        "FEEDBACK_EVERY": "2",
    }),
    ("medium_innovations", {
        "FEEDBACK_DIM": "64", "FEEDBACK_SKETCH_TOKENS": "4",
        "CAPSULE_NUM": "8", "CAPSULE_DIM": "32",
        "BIGRAM_HASH_DIM": "64", "KOOPMAN_RANK": "4",
        "FEEDBACK_EVERY": "2",
    }),
    ("large_innovations", {
        "FEEDBACK_DIM": "64", "FEEDBACK_SKETCH_TOKENS": "4",
        "CAPSULE_NUM": "16", "CAPSULE_DIM": "64",
        "BIGRAM_HASH_DIM": "128", "KOOPMAN_RANK": "4",
        "FEEDBACK_EVERY": "1",
    }),
    ("turbo_quant_kv", {
        "FEEDBACK_DIM": "32", "FEEDBACK_SKETCH_TOKENS": "2",
        "CAPSULE_NUM": "8", "CAPSULE_DIM": "32",
        "BIGRAM_HASH_DIM": "64", "KOOPMAN_RANK": "2",
        "TURBO_QUANT_KV": "1",
        "FEEDBACK_EVERY": "2",
    }),
]


def merge_env(overrides: dict) -> dict:
    e = os.environ.copy()
    e.update(BASE_ENV)
    e.update(overrides)
    return e


def run_config(name: str, overrides: dict, time_limit_s: int = 600) -> dict:
    """Run one configuration and extract metrics."""
    env = merge_env(overrides)
    env["MAX_WALLCLOCK_SECONDS"] = str(time_limit_s)
    env["RUN_ID"] = f"sweep_{name}"

    cmd = ["bash", "run_mlx_reasoner.sh"]

    print(f"\n{'='*60}")
    print(f"  SWEEP: {name} (time_limit={time_limit_s}s)")
    print(f"{'='*60}")

    t0 = time.time()
    process = subprocess.Popen(
        cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, cwd=SCRIPT_DIR
    )

    final_bpb = "ERROR"
    sliding_bpb = "N/A"
    ttt_bpb = "N/A"
    step_count = 0
    last_step_time = "N/A"

    for line in process.stdout:
        print(line, end='')
        # Capture training step info
        m = re.search(r'step:(\d+)/\d+ loss:[\d.]+ t:\d+ms step:(\d+)ms', line)
        if m:
            step_count = int(m.group(1))
            last_step_time = m.group(2)
        # Capture various eval BPBs
        m = re.search(r'val_bpb:([\d.]+)', line)
        if m:
            final_bpb = m.group(1)
        m = re.search(r'final_sliding val_loss:[\d.]+ val_bpb:([\d.]+)', line)
        if m:
            sliding_bpb = m.group(1)
        m = re.search(r'legal_ttt val_loss:[\d.]+ val_bpb:([\d.]+)', line)
        if m:
            ttt_bpb = m.group(1)

    process.wait()
    elapsed = time.time() - t0

    return {
        "name": name,
        "final_bpb": final_bpb,
        "sliding_bpb": sliding_bpb,
        "ttt_bpb": ttt_bpb,
        "steps": step_count,
        "step_time_ms": last_step_time,
        "elapsed_s": f"{elapsed:.0f}",
    }


def main():
    time_limit = int(os.environ.get("SWEEP_TIME_LIMIT", 600))  # 10 min default
    results = []

    for name, overrides in CONFIGS:
        result = run_config(name, overrides, time_limit)
        results.append(result)

    # Write report
    with open(RESULTS_FILE, "w") as f:
        f.write("# Speed-Aware Sweep Results\\n\\n")
        f.write(f"**Time budget per config:** {time_limit}s\\n\\n")
        f.write("| Config | Steps | Step (ms) | Val BPB | Sliding BPB | TTT BPB |\\n")
        f.write("|--------|-------|-----------|---------|-------------|---------|\\n")
        for r in results:
            f.write(f"| {r['name']} | {r['steps']} | {r['step_time_ms']} "
                    f"| {r['final_bpb']} | {r['sliding_bpb']} | {r['ttt_bpb']} |\\n")

        # Find best
        valid = [(r, float(r['ttt_bpb'])) for r in results
                 if r['ttt_bpb'] not in ("N/A", "ERROR")]
        if valid:
            best = min(valid, key=lambda x: x[1])
            f.write(f"\\n## Winner: `{best[0]['name']}` with TTT BPB = **{best[1]:.4f}**\\n")
            f.write(f"\\nSteps completed: {best[0]['steps']} at ~{best[0]['step_time_ms']}ms/step\\n")

    print(f"\\nSweep complete! Results written to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
