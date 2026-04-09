#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path


RECORD_DIR = Path(__file__).resolve().parent
REPO_ROOT = RECORD_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def ensure_default_env(name: str, value: str) -> None:
    os.environ.setdefault(name, value)


if __name__ == "__main__":
    ensure_default_env("AUTO_LOG_PATH", str(RECORD_DIR / "train.log"))
    ensure_default_env("OUTPUT_JSON", str(RECORD_DIR / "result.json"))
    ensure_default_env("RAW_MODEL_PATH", str(RECORD_DIR / "final_model.pt"))
    ensure_default_env("MODEL_ARTIFACT_PATH", str(RECORD_DIR / "final_model.int6.ptz"))
    ensure_default_env("RESIDUAL_ARTIFACT_PATH", str(RECORD_DIR / "residual_artifact.npz"))
    ensure_default_env("DATA_PATH", str(REPO_ROOT / "data" / "datasets" / "fineweb10B_sp1024"))
    ensure_default_env("TOKENIZER_PATH", str(REPO_ROOT / "data" / "tokenizers" / "fineweb_1024_bpe.model"))

    from spectral_flood_walk_v2a1_host1233 import main as spectral_main  # noqa: E402

    spectral_main()
