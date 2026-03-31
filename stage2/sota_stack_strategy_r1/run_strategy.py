#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    here = Path(__file__).resolve()
    orchestrator = here.parents[1] / "h100_matrix_r2" / "orchestrate_stage2.py"
    config = here.with_name("run_configs.json")
    cmd = [sys.executable, str(orchestrator), "--config", str(config), *sys.argv[1:]]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
