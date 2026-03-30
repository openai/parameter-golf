#!/usr/bin/env python3
"""
Overnight autoresearch sweep for Parameter Golf.
Runs ~12 experiments systematically, each ~30 min on M3 Ultra.
Total: ~6 hours.

Explores: layer count, MLP mult, BigramHash size, muon momentum,
warmdown, QAT threshold, z-loss weight, grad clip, rope dims.
"""
import sys
sys.path.insert(0, ".")
from autoresearch.run_experiment import run_experiment, print_leaderboard

ITERS = 2000  # ~15 min each on M3 Ultra at 42K tok/s

experiments = [
    # Baseline: our current best config
    ("baseline_12L", {}),

    # Layer count ablation
    ("11L_baseline", {"NUM_LAYERS": "11"}),
    ("13L", {"NUM_LAYERS": "13"}),

    # MLP expansion
    ("12L_mlp2x", {"MLP_MULT": "2"}),
    ("12L_mlp4x", {"MLP_MULT": "4"}),

    # BigramHash size
    ("12L_bigram2048", {"BIGRAM_HASH_SIZE": "2048"}),
    ("12L_bigram8192", {"BIGRAM_HASH_SIZE": "8192"}),

    # Muon momentum
    ("12L_muon095", {"MUON_MOMENTUM": "0.95", "MUON_MOMENTUM_WARMUP_START": "0.85"}),

    # Warmdown
    ("12L_warmdown5000", {"WARMDOWN_ITERS": "5000"}),
    ("12L_warmdown2000", {"WARMDOWN_ITERS": "2000"}),

    # QAT threshold
    ("12L_qat020", {"QAT_START_FRACTION": "0.20"}),
    ("12L_qat010", {"QAT_START_FRACTION": "0.10"}),

    # Z-loss weight
    ("12L_zloss0", {"Z_LOSS_WEIGHT": "0"}),
    ("12L_zloss1e3", {"Z_LOSS_WEIGHT": "0.001"}),

    # Grad clip
    ("12L_noclip", {"GRAD_CLIP_NORM": "0"}),
    ("12L_clip05", {"GRAD_CLIP_NORM": "0.5"}),

    # Rope dims
    ("12L_rope32", {"ROPE_DIMS": "32"}),
    ("12L_rope0_full", {"ROPE_DIMS": "0"}),

    # MTP ablation
    ("12L_nomtp", {"MTP_NUM_HEADS": "0"}),
    ("12L_mtp3", {"MTP_NUM_HEADS": "3"}),

    # EMA decay
    ("12L_ema0998", {"EMA_DECAY": "0.998"}),
    ("12L_ema0995", {"EMA_DECAY": "0.995"}),
]

if __name__ == "__main__":
    print(f"Running {len(experiments)} experiments at {ITERS} iters each")
    print(f"Estimated time: {len(experiments) * 15 / 60:.1f} hours")
    print()

    for name, env in experiments:
        try:
            run_experiment(name, ITERS, env if env else None)
        except Exception as e:
            print(f"FAILED: {name}: {e}")
        print()

    print("\n\nFINAL LEADERBOARD:")
    print_leaderboard()
