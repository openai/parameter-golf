#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    here = Path(__file__).resolve()
    entrypoint = here.with_name("evolve_stage3_2.py")
    cmd = [sys.executable, str(entrypoint), *sys.argv[1:]]
    completed = subprocess.run(cmd, check=False)
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
