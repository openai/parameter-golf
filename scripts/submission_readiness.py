#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.submission_readiness import refresh_submission_artifacts, render_submission_readiness


RUNS_ROOT = ROOT / "research" / "results" / "runs"


def latest_completed_run(*, family: str | None = None) -> Path | None:
    candidates: list[Path] = []
    for run_dir in sorted(RUNS_ROOT.glob("*")):
        result_path = run_dir / "result.json"
        if not result_path.is_file():
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") != "completed":
            continue
        if family is not None and result.get("family") != family:
            continue
        candidates.append(run_dir)
    return candidates[-1] if candidates else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Check or backfill submission-readiness metadata for a completed run.")
    parser.add_argument("--run-dir", help="Path to a completed run directory.")
    parser.add_argument("--latest", action="store_true", help="Use the latest completed run directory.")
    parser.add_argument("--family", help="Optional family filter when using --latest, e.g. frontier.")
    parser.add_argument("--rewrite", action="store_true", help="Rewrite result/summary/legality fields with canonical submission metadata.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    args = parser.parse_args()

    if args.latest:
        run_dir = latest_completed_run(family=args.family)
        if run_dir is None:
            raise SystemExit("No matching completed run directories found.")
    elif args.run_dir:
        run_dir = Path(args.run_dir).resolve()
    else:
        parser.error("Provide --run-dir or --latest")

    refreshed = refresh_submission_artifacts(run_dir, rewrite=args.rewrite)
    report = refreshed["submission_readiness"]
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    print(f"run_dir: {run_dir}")
    print(render_submission_readiness(report), end="")


if __name__ == "__main__":
    main()
