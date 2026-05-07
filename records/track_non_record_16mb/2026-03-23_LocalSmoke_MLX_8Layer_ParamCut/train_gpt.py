#!/usr/bin/env python3
"""
Small launcher for this local non-record MLX submission.

This script pins the run configuration used for the 8-layer smoke run and
delegates to the repository-level `train_gpt_mlx.py`.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def main() -> None:
    here = Path(__file__).resolve().parent
    repo_root = here.parents[2]
    target = repo_root / "train_gpt_mlx.py"
    if not target.exists():
        raise FileNotFoundError(f"Could not find {target}")

    env = os.environ.copy()
    env.setdefault("RUN_ID", "mlx_smoke_layers8_local")
    env.setdefault("DATA_PATH", str(repo_root / "data/datasets/fineweb10B_sp1024_smoke"))
    env.setdefault("TOKENIZER_PATH", str(repo_root / "data/tokenizers/fineweb_1024_bpe.model"))
    env.setdefault("ITERATIONS", "60")
    env.setdefault("TRAIN_BATCH_TOKENS", "8192")
    env.setdefault("GRAD_ACCUM_STEPS", "1")
    env.setdefault("VAL_LOSS_EVERY", "0")
    env.setdefault("VAL_BATCH_SIZE", "131072")
    env.setdefault("NUM_LAYERS", "8")

    subprocess.run(["python3", str(target)], check=True, env=env)


if __name__ == "__main__":
    main()
