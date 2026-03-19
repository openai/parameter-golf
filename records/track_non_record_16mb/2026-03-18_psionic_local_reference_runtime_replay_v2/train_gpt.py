#!/usr/bin/env python3
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
RUNTIME_PAYLOAD = ROOT / "runtime/parameter_golf_submission_runtime"
RUNTIME_MANIFEST = ROOT / "runtime/parameter_golf_submission_runtime.json"

completed = subprocess.run([str(RUNTIME_PAYLOAD), str(RUNTIME_MANIFEST)], cwd=ROOT, check=False)
sys.exit(completed.returncode)
