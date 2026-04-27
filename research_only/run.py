#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_only.hybrid_registry import RESEARCH_ONLY_PRESETS
from research_only.sequence_mixers import SequenceMixerSpec, build_sequence_mixer


RESULTS_ROOT = ROOT / "research_only" / "results"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or inspect research-only hybrid sequence-mixer presets.")
    parser.add_argument("--preset", help="Research-only preset name.")
    parser.add_argument("--list", action="store_true", help="List research-only presets.")
    parser.add_argument("--dry-run", action="store_true", help="Write run metadata without executing the proxy.")
    args = parser.parse_args()

    if args.list:
        for name in sorted(RESEARCH_ONLY_PRESETS):
            preset = RESEARCH_ONLY_PRESETS[name]
            print(f"{preset.name}: {preset.description}")
        return
    if not args.preset:
        parser.error("--preset is required unless --list is used")

    preset = RESEARCH_ONLY_PRESETS[args.preset]
    run_dir = RESULTS_ROOT / f"{int(time.time())}_{preset.name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    spec = SequenceMixerSpec(
        name=preset.name,
        hidden_dim=int(preset.env["MODEL_DIM"]),
        num_heads=int(preset.env["NUM_HEADS"]),
        state_dim=int(preset.env["STATE_DIM"]),
    )
    build_sequence_mixer(spec, kind=preset.mixer_kind)
    run_spec = {
        "preset": preset.name,
        "description": preset.description,
        "mixer_kind": preset.mixer_kind,
        "env": preset.env,
        "notes": list(preset.notes),
        "research_only": True,
        "submission_safe": False,
        "manual_review_required": True,
    }
    (run_dir / "run_spec.json").write_text(json.dumps(run_spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.dry_run:
        print(f"run_dir: {run_dir}")
        return
    result = {
        **run_spec,
        "status": "proxy_completed",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"run_dir: {run_dir}")


if __name__ == "__main__":
    main()

