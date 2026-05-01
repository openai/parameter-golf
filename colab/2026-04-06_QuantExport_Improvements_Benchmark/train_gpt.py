from __future__ import annotations

import runpy
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TARGET_SCRIPT = REPO_ROOT / "train_gpt.py"


if __name__ == "__main__":
    runpy.run_path(str(TARGET_SCRIPT), run_name="__main__")
