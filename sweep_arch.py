#!/usr/bin/env python3
"""Architecture sweep: find the best (layers, dim, mlp_mult) for a given artifact budget.

Uses fast eval only (no TTT/ngram/kNN). Kitchen sink training techniques always on.
Sizes models to hit the target artifact budget based on actual compression ratios.

Usage:
    python sweep_arch.py                    # sweep with defaults
    python sweep_arch.py --target-mb 15.9   # custom artifact target
    python sweep_arch.py --results-only     # just print results
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from pgolf.config import PROJECT_DIR
from pgolf.runner import run_trial, save_result

SWEEP_RESULTS = os.path.join(PROJECT_DIR, "sweep_results.jsonl")

# Kitchen sink training techniques (always on)
KITCHEN_SINK = {
    "ACTIVATION": "leaky_relu_squared",
    "ENABLE_HYBRIDNORM": 1,
    "ENABLE_SMEARGATE": 1,
    "ENABLE_DIFF_ATTN": 1,
    "ENABLE_WAVELET": 1,
    "ENABLE_VGA": 1,
    "MTP_NUM_HEADS": 1,
    "EMA_DECAY": 0.997,
    "ENABLE_SWA": 1,
    "ENABLE_QAT": 1,
    "ENABLE_OPTROT": 1,
    "ENABLE_GPTQ": 1,
    "ENABLE_PRUNING": 1,
    "PRUNE_FRACTION": 0.02,
}

# Architecture configs to sweep — (label, {overrides})
# Sized to approximately fill 16MB artifact at given quant bits
ARCH_CONFIGS = [
    # int6 configs (proven good quantization quality)
    (
        "9L_512d_mlp2x_int6",
        {
            "NUM_LAYERS": 9,
            "MODEL_DIM": 512,
            "MLP_MULT": 2,
            "NUM_HEADS": 8,
            "NUM_KV_HEADS": 4,
            "QUANT_BITS": 6,
        },
    ),
    (
        "11L_512d_mlp2x_int6",
        {
            "NUM_LAYERS": 11,
            "MODEL_DIM": 512,
            "MLP_MULT": 2,
            "NUM_HEADS": 8,
            "NUM_KV_HEADS": 4,
            "QUANT_BITS": 6,
        },
    ),
    (
        "11L_512d_mlp3x_int6",
        {
            "NUM_LAYERS": 11,
            "MODEL_DIM": 512,
            "MLP_MULT": 3,
            "NUM_HEADS": 8,
            "NUM_KV_HEADS": 4,
            "QUANT_BITS": 6,
        },
    ),
    (
        "13L_512d_mlp2x_int6",
        {
            "NUM_LAYERS": 13,
            "MODEL_DIM": 512,
            "MLP_MULT": 2,
            "NUM_HEADS": 8,
            "NUM_KV_HEADS": 4,
            "QUANT_BITS": 6,
        },
    ),
    (
        "11L_576d_mlp2x_int6",
        {
            "NUM_LAYERS": 11,
            "MODEL_DIM": 576,
            "MLP_MULT": 2,
            "NUM_HEADS": 4,
            "NUM_KV_HEADS": 4,
            "QUANT_BITS": 6,
        },
    ),
    (
        "13L_576d_mlp2x_int6",
        {
            "NUM_LAYERS": 13,
            "MODEL_DIM": 576,
            "MLP_MULT": 2,
            "NUM_HEADS": 4,
            "NUM_KV_HEADS": 4,
            "QUANT_BITS": 6,
        },
    ),
    (
        "14L_512d_mlp3x_int6",
        {
            "NUM_LAYERS": 14,
            "MODEL_DIM": 512,
            "MLP_MULT": 3,
            "NUM_HEADS": 8,
            "NUM_KV_HEADS": 4,
            "QUANT_BITS": 6,
        },
    ),
    (
        "11L_640d_mlp2x_int6",
        {
            "NUM_LAYERS": 11,
            "MODEL_DIM": 640,
            "MLP_MULT": 2,
            "NUM_HEADS": 8,
            "NUM_KV_HEADS": 4,
            "QUANT_BITS": 6,
        },
    ),
    # int5 configs (more params but noisier — may or may not help)
    (
        "11L_576d_mlp3x_int5",
        {
            "NUM_LAYERS": 11,
            "MODEL_DIM": 576,
            "MLP_MULT": 3,
            "NUM_HEADS": 4,
            "NUM_KV_HEADS": 4,
            "QUANT_BITS": 5,
        },
    ),
    (
        "14L_576d_mlp2x_int5",
        {
            "NUM_LAYERS": 14,
            "MODEL_DIM": 576,
            "MLP_MULT": 2,
            "NUM_HEADS": 4,
            "NUM_KV_HEADS": 4,
            "QUANT_BITS": 5,
        },
    ),
    (
        "12L_640d_mlp2x_int5",
        {
            "NUM_LAYERS": 12,
            "MODEL_DIM": 640,
            "MLP_MULT": 2,
            "NUM_HEADS": 8,
            "NUM_KV_HEADS": 4,
            "QUANT_BITS": 5,
        },
    ),
    # Deeper configs
    (
        "16L_448d_mlp3x_int6",
        {
            "NUM_LAYERS": 16,
            "MODEL_DIM": 448,
            "MLP_MULT": 3,
            "NUM_HEADS": 8,
            "NUM_KV_HEADS": 4,
            "QUANT_BITS": 6,
        },
    ),
    (
        "15L_512d_mlp2x_int6",
        {
            "NUM_LAYERS": 15,
            "MODEL_DIM": 512,
            "MLP_MULT": 2,
            "NUM_HEADS": 8,
            "NUM_KV_HEADS": 4,
            "QUANT_BITS": 6,
        },
    ),
]


def load_completed() -> set[str]:
    completed = set()
    if os.path.exists(SWEEP_RESULTS):
        with open(SWEEP_RESULTS) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("status") == "OK":
                        completed.add(r["label"])
                except json.JSONDecodeError:
                    continue
    return completed


def save_sweep_result(result: dict):
    with open(SWEEP_RESULTS, "a") as f:
        f.write(json.dumps(result, default=str) + "\n")


def print_results():
    if not os.path.exists(SWEEP_RESULTS):
        print("No sweep results yet.")
        return

    results = []
    with open(SWEEP_RESULTS) as f:
        for line in f:
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    print(f"\n{'=' * 85}")
    print("ARCHITECTURE SWEEP RESULTS (fast eval, kitchen sink training)")
    print(f"{'=' * 85}")
    print(f"{'Label':<30} {'BPB':>8} {'Size':>10} {'Params':>10} {'Status':>8}")
    print("-" * 85)

    for r in sorted(results, key=lambda x: x.get("val_bpb", 99)):
        bpb = r.get("val_bpb", 99)
        size = r.get("artifact_size", "?")
        label = r.get("label", "?")
        status = r.get("status", "?")
        cfg = r.get("config", {})
        nl = cfg.get("NUM_LAYERS", "?")
        dim = cfg.get("MODEL_DIM", "?")
        mlp = cfg.get("MLP_MULT", "?")
        bits = cfg.get("QUANT_BITS", "?")
        print(
            f"{label:<30} {bpb:>8.4f} {str(size):>10} "
            f"{nl}L/{dim}d/{mlp}x/i{bits} {status:>8}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Architecture sweep for Parameter Golf"
    )
    parser.add_argument("--max-wallclock", type=int, default=300)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--results-only", action="store_true")
    args = parser.parse_args()

    if args.results_only:
        print_results()
        return

    completed = load_completed()
    total = len(ARCH_CONFIGS)

    print(f"\n{'=' * 70}")
    print(f"ARCHITECTURE SWEEP: {total} configs, {len(completed)} already done")
    print(f"Training: {args.iterations} iters, {args.max_wallclock}s wallclock cap")
    print(f"Techniques: kitchen sink (no TTT/ngram/kNN)")
    print(f"Ctrl+C to stop — progress saved, re-run to resume")
    print(f"{'=' * 70}\n")

    done = 0
    for label, arch in ARCH_CONFIGS:
        done += 1
        if label in completed:
            print(f"[{done}/{total}] {label}: SKIP (already completed)")
            continue

        config = {**KITCHEN_SINK, **arch, "RUN_ID": f"sweep_{label}"}

        print(f"[{done}/{total}] {label}")
        print(f"  Arch: {arch}")
        print(f"  Running...", end="", flush=True)

        try:
            result = run_trial(config, args.max_wallclock, args.iterations, label=label)
        except KeyboardInterrupt:
            print(f"\n\nInterrupted at {done}/{total} ({label})")
            sys.exit(0)

        save_sweep_result(result)
        status = result["status"]
        bpb = result["val_bpb"]
        elapsed = result["elapsed"]
        size = result.get("artifact_size", "?")
        print(f" {status} | bpb={bpb:.4f} | size={size} | {elapsed:.0f}s")

        if status != "OK":
            err = result.get("error", "")
            if err:
                print(f"  Error: {err[:200]}")

    print_results()


if __name__ == "__main__":
    main()
