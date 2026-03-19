#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from autoresearch_pg.lib.workspace import iter_run_json_paths, load_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a local autoresearch leaderboard")
    parser.add_argument("--all", action="store_true", help="Include invalid runs")
    parser.add_argument("--limit", type=int, default=20, help="Max rows to render")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rows = []
    for path in iter_run_json_paths():
        payload = load_json(path)
        objective = payload.get("objective", {})
        if not args.all and not objective.get("valid"):
            continue
        rows.append(
            {
                "candidate": payload["candidate_id"],
                "tier": payload["tier"],
                "run_id": payload["run_id"],
                "valid": objective.get("valid"),
                "score": objective.get("proxy_score"),
                "post_quant_val_bpb": objective.get("post_quant_val_bpb"),
                "pre_quant_val_bpb": objective.get("pre_quant_val_bpb"),
                "quant_gap_bpb": objective.get("quant_gap_bpb"),
                "bytes_total": objective.get("bytes_total"),
            }
        )

    rows.sort(key=lambda row: float(row["score"] if row["score"] is not None else 1e9))
    rows = rows[: args.limit]

    print("| candidate | tier | score | post | pre | gap | bytes | valid | run_id |")
    print("|---|---|---:|---:|---:|---:|---:|---|---|")
    for row in rows:
        print(
            f"| {row['candidate']} | {row['tier']} | {row['score']:.6f} | "
            f"{row['post_quant_val_bpb']} | {row['pre_quant_val_bpb']} | "
            f"{row['quant_gap_bpb']} | {row['bytes_total']} | {row['valid']} | {row['run_id']} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
