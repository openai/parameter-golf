#!/usr/bin/env python3
"""
10 Parameter Golf Experiments — Config Definitions

Each experiment is a dict with:
- id: short identifier (E01, E02, etc.)
- name: human-readable name
- description: what this tests
- env_vars: dict of env var overrides
- requires_code_change: bool (false = config-only, true = needs modified train_gpt_mlx.py)

The baseline (M4 local) achieved:
- val_bpb: 3.2178 (post int8 quantization)
- steps: 71
- config: 9L, 512d, 8h/4kv, 2x MLP, WARMDOWN=1200, QK_GAIN=1.5

Target: find configs that improve (lower) the BPB on M4 local testing.
"""

import argparse
from typing import Dict, Optional

EXPERIMENTS = [
    {
        "id": "E01",
        "name": "Minimal Competitive",
        "description": "Combo #4: minimal competitive stack (A1+A2+A6+A11, Q1, O1+O3+O4)",
        "env_vars": {
            "NUM_LAYERS": "11",
            "MLP_MULT": "3",
            "MUON_MOMENTUM": "0.95",  # Default, not 0.99
            "WARMDOWN_ITERS": "3500",
            "QK_GAIN_INIT": "1.5",
        },
        "requires_code_change": False,
    },
    {
        "id": "E02",
        "name": "Warmdown 5000",
        "description": "Combo #6: push warmdown to 5000 (vs default 1200)",
        "env_vars": {
            "WARMDOWN_ITERS": "5000",
        },
        "requires_code_change": False,
    },
    {
        "id": "E03",
        "name": "QK-Gain 4.0",
        "description": "Combo #7: QK-Gain scaling from 1.5 to 4.0",
        "env_vars": {
            "QK_GAIN_INIT": "4.0",
        },
        "requires_code_change": False,
    },
    {
        "id": "E04",
        "name": "EMA-Heavy",
        "description": "Combo #5: EMA=0.998 (vs 0.997), SWA every 30 (vs 50)",
        "env_vars": {
            "NUM_LAYERS": "11",
            "MLP_MULT": "3",
        },
        "requires_code_change": False,
    },
    {
        "id": "E05",
        "name": "Foundation Stack",
        "description": "Combo #2: SOTA minus TTT/GPTQ (11L, 3x MLP, Muon 0.99, WD 3500)",
        "env_vars": {
            "NUM_LAYERS": "11",
            "MLP_MULT": "3",
            "MUON_MOMENTUM": "0.99",
            "WARMDOWN_ITERS": "3500",
        },
        "requires_code_change": False,
    },
    {
        "id": "E06",
        "name": "Focal Loss",
        "description": "Combo #10: Focal loss with gamma=2.0 (down-weight easy tokens)",
        "env_vars": {
            "FOCAL_GAMMA": "2.0",
        },
        "requires_code_change": False,
    },
    {
        "id": "E07",
        "name": "Stride-16 Eval",
        "description": "Combo #9: sliding eval with stride=16 (more precise val eval)",
        "env_vars": {
            "EVAL_SLIDING_STRIDE": "16",
        },
        "requires_code_change": False,  # Will be added if needed
    },
    {
        "id": "E08",
        "name": "QK+Focal+EMA",
        "description": "Combo #20: QK=4.0 + focal loss + EMA=0.998 (combined)",
        "env_vars": {
            "QK_GAIN_INIT": "4.0",
            "FOCAL_GAMMA": "2.0",
            "NUM_LAYERS": "11",
            "MLP_MULT": "3",
        },
        "requires_code_change": False,
    },
    {
        "id": "E09",
        "name": "All April Low-Hanging",
        "description": "Combo #21: QK=4.0 + stride-16 + temperature=0.90",
        "env_vars": {
            "QK_GAIN_INIT": "4.0",
            "EVAL_SLIDING_STRIDE": "16",
            "EVAL_TEMPERATURE": "0.90",
            "NUM_LAYERS": "11",
            "MLP_MULT": "3",
        },
        "requires_code_change": False,
    },
    {
        "id": "E10",
        "name": "Sanath Stack (Partial)",
        "description": "Combo #50: best M4-testable combination (11L, 3x MLP, QK=4.0, focal, warmdown 4000)",
        "env_vars": {
            "NUM_LAYERS": "11",
            "MLP_MULT": "3",
            "QK_GAIN_INIT": "4.0",
            "FOCAL_GAMMA": "2.0",
            "WARMDOWN_ITERS": "4000",
            "MUON_MOMENTUM": "0.99",
        },
        "requires_code_change": False,
    },
]


def get_experiment(exp_id: str) -> Optional[Dict]:
    """Look up experiment by ID."""
    for exp in EXPERIMENTS:
        if exp["id"] == exp_id:
            return exp
    return None


def to_env_string(exp_id: str) -> str:
    """Convert experiment to shell export string for use in run scripts."""
    exp = get_experiment(exp_id)
    if exp is None:
        return ""
    lines = []
    for k, v in exp["env_vars"].items():
        lines.append(f"export {k}={v}")
    return "\n".join(lines)


def list_experiments() -> None:
    """Print all experiments."""
    print("Parameter Golf Experiments:")
    print()
    for exp in EXPERIMENTS:
        print(f"  {exp['id']:3s} | {exp['name']:25s} | {exp['description']}")
        if exp["requires_code_change"]:
            print(f"       |  ** Requires modified train_gpt_mlx.py **")
        print()


def list_experiments_machine() -> None:
    """Print machine-readable experiment rows for shell runners."""
    for exp in EXPERIMENTS:
        env_str = ",".join(f"{k}={v}" for k, v in exp["env_vars"].items())
        print(f"{exp['id']}|{exp['name']}|{exp['description']}|{env_str}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment config utilities")
    parser.add_argument("--machine", action="store_true", help="Print machine-readable rows")
    args = parser.parse_args()
    if args.machine:
        list_experiments_machine()
    else:
        list_experiments()
