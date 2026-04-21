import runpy
from pathlib import Path


def main() -> None:
    """
    Thin launcher for the Track-P proxy trainer.

    The actual trainer is kept in its original record folder so that this
    submission remains a minimal, reproducible screen of the SP8192 variant.
    """

    repo_root = Path(__file__).resolve().parents[3]
    base_trainer = (
        repo_root
        / "records"
        / "track_10min_16mb"
        / "2026-04-01_Vocab4096_MLPMult4_WD085"
        / "train_gpt.py"
    )

    runpy.run_path(str(base_trainer), run_name="__main__")


if __name__ == "__main__":
    main()

