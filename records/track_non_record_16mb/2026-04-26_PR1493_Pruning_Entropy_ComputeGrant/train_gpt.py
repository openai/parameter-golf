#!/usr/bin/env python3
"""Public entrypoint for the draft PR1493 pruning/entropy checkpoint.

Default behavior:
  python train_gpt.py

runs the local PR1493 training stack in `train_gpt_pr1493.py`. The public
entrypoint disables research-artifact capture by default; helper scripts can be
run explicitly to regenerate pruning and entropy-analysis artifacts.

Helper commands:
  python train_gpt.py cap-masks --artifact-dir artifacts/...
  python train_gpt.py softcap-masks --artifact-dir artifacts/...
  torchrun --standalone --nproc_per_node=8 train_gpt.py eval-pruning --artifact-dir artifacts/...
  python train_gpt.py q-entropy --quantized artifacts/.../final_model.int6.ptz
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


SCRIPT_BY_COMMAND = {
    "cap-masks": "grant_generate_mlp_cap_masks.py",
    "softcap-masks": "grant_generate_mlp_softcap_masks.py",
    "eval-pruning": "grant_eval_mlp_zero_ablation.py",
    "q-entropy": "grant_estimate_q_entropy.py",
}


def _run_script(script_name: str, argv: list[str]) -> None:
    folder = Path(__file__).resolve().parent
    script = folder / script_name
    if not script.is_file():
        raise SystemExit(f"Missing script: {script}")
    sys.argv = [str(script), *argv]
    runpy.run_path(str(script), run_name="__main__")


def main() -> None:
    if len(sys.argv) >= 2 and sys.argv[1] in SCRIPT_BY_COMMAND:
        _run_script(SCRIPT_BY_COMMAND[sys.argv[1]], sys.argv[2:])
        return

    if len(sys.argv) >= 2 and sys.argv[1] in {"helpers", "--helpers"}:
        commands = "\n".join(f"  {name:<14} -> {script}" for name, script in SCRIPT_BY_COMMAND.items())
        print(__doc__)
        print("\nHelper commands:\n" + commands)
        return

    # Normal Parameter Golf entrypoint: run the local PR1493 training stack.
    # Keep research capture off in the public PR-facing path.
    os.environ.setdefault("CAPTURE_RESEARCH_ARTIFACTS", "0")
    os.environ.setdefault("CAPTURE_LATE_SNAPSHOT_FRACS", "")
    os.environ.setdefault("CAPTURE_GPTQ_HESSIANS", "0")
    os.environ.setdefault("CAPTURE_MLP_CHANNEL_STATS", "0")
    os.environ.setdefault("CAPTURE_FINAL_OPTIMIZER", "0")
    _run_script("train_gpt_pr1493.py", sys.argv[1:])


if __name__ == "__main__":
    main()
