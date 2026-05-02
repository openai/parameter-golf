import os
import runpy
from pathlib import Path


def main() -> None:
    # Disable looping (depth recurrence) for the base SOTA stack.
    os.environ.setdefault("NUM_LOOPS", "0")

    repo_root = Path(__file__).resolve().parents[3]
    base_trainer = (
        repo_root
        / "records"
        / "track_10min_16mb"
        / "2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT"
        / "train_gpt.py"
    )

    runpy.run_path(str(base_trainer), run_name="__main__")


if __name__ == "__main__":
    main()

