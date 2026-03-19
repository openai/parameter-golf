#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from autoresearch_pg.lib.workspace import best_run_for_candidate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check promotion eligibility for a candidate")
    parser.add_argument("--candidate", required=True, help="Candidate id")
    parser.add_argument("--from-tier", required=True, help="Source tier")
    parser.add_argument("--to-tier", required=True, help="Destination tier")
    parser.add_argument(
        "--max-post-quant-val-bpb",
        type=float,
        help="Optional promotion threshold on post-quant val_bpb",
    )
    parser.add_argument(
        "--allow-invalid-bytes",
        action="store_true",
        help="Allow promotion even if the source run was over the artifact cap.",
    )
    parser.add_argument(
        "--launch",
        action="store_true",
        help="Launch the destination tier immediately if eligible.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    best = best_run_for_candidate(args.candidate, tier=args.from_tier, require_valid=False)
    if best is None:
        print(f"no runs found for candidate={args.candidate} tier={args.from_tier}")
        return 1

    objective = best.get("objective", {})
    eligible = True
    reasons: list[str] = []

    if not args.allow_invalid_bytes and not objective.get("valid"):
        eligible = False
        reasons.append("source_run_not_valid")

    if args.max_post_quant_val_bpb is not None:
        observed = objective.get("post_quant_val_bpb")
        if observed is None or observed > args.max_post_quant_val_bpb:
            eligible = False
            reasons.append("post_quant_val_bpb_above_threshold")

    print(
        f"candidate={args.candidate} from={args.from_tier} to={args.to_tier} "
        f"eligible={eligible} post_quant_val_bpb={objective.get('post_quant_val_bpb')} "
        f"bytes_total={objective.get('bytes_total')} reasons={','.join(reasons) or 'ok'}"
    )

    if eligible and args.launch:
        cmd = [
            sys.executable,
            str(SCRIPT_DIR / "run_candidate.py"),
            "--candidate",
            args.candidate,
            "--tier",
            args.to_tier,
        ]
        return subprocess.run(cmd, check=False).returncode

    return 0 if eligible else 1


if __name__ == "__main__":
    raise SystemExit(main())
