"""
One-time MLflow setup and convenience helpers.

Usage (from repo root):
    python -m tracking.setup              # creates mlruns/ locally, prints URI
    python -m tracking.setup --ui         # launches mlflow ui on port 5000
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = PROJECT_ROOT / "mlflow.db"
DEFAULT_URI = f"sqlite:///{DEFAULT_DB}"


def ensure_mlflow_installed() -> bool:
    try:
        import mlflow  # noqa: F401
        return True
    except ImportError:
        return False


def setup_local_tracking() -> str:
    uri = DEFAULT_URI
    print(f"MLflow tracking URI: {uri}")
    print(f"MLflow database: {DEFAULT_DB}")
    return uri


def launch_ui(port: int = 5000) -> None:
    uri = setup_local_tracking()
    print(f"Starting MLflow UI on http://127.0.0.1:{port}")
    subprocess.run(
        [sys.executable, "-m", "mlflow", "ui", "--backend-store-uri", uri, "--port", str(port)],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="MLflow setup for Parameter Golf")
    parser.add_argument("--ui", action="store_true", help="Launch MLflow UI after setup")
    parser.add_argument("--port", type=int, default=5000, help="Port for MLflow UI")
    args = parser.parse_args()

    if not ensure_mlflow_installed():
        print("mlflow is not installed. Run: pip install mlflow", file=sys.stderr)
        sys.exit(1)

    if args.ui:
        launch_ui(port=args.port)
    else:
        uri = setup_local_tracking()
        print(f"\nTo start the UI:\n  python -m tracking.setup --ui\n")
        print(f"Set in your script:\n  export MLFLOW_TRACKING_URI={uri}")


if __name__ == "__main__":
    main()
