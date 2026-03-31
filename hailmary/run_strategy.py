#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    here = Path(__file__).resolve()
    orchestrator = here.with_name("orchestrate_hailmary.py")
    config = here.with_name("run_configs.json")
    cmd = [sys.executable, str(orchestrator), "--config", str(config), *sys.argv[1:]]
    completed = subprocess.run(cmd, check=False)
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()

