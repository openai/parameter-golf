#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


RECORD_DIR = Path(__file__).resolve().parent
REPO_ROOT = RECORD_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spectral_flood_walk_v2b import main as spectral_main  # noqa: E402


def ensure_default_arg(flag: str, value: str) -> None:
    if flag not in sys.argv:
        sys.argv.extend([flag, value])


if __name__ == "__main__":
    ensure_default_arg("--auto-log-path", str(RECORD_DIR / "train.log"))
    ensure_default_arg("--output-json", str(RECORD_DIR / "result.json"))
    ensure_default_arg("--model-artifact-path", str(RECORD_DIR / "model_int8.npz"))
    ensure_default_arg("--data-path", str(REPO_ROOT / "data" / "datasets" / "fineweb10B_sp1024"))
    ensure_default_arg("--tokenizer-path", str(REPO_ROOT / "data" / "tokenizers" / "fineweb_1024_bpe.model"))
    spectral_main()
