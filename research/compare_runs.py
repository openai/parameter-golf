#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


INDEX_JSONL = ROOT / "research" / "results" / "index.jsonl"


def load_records() -> list[dict[str, object]]:
    if not INDEX_JSONL.is_file():
        return []
    records: list[dict[str, object]] = []
    with INDEX_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def final_bpb(record: dict[str, object]) -> float:
    metrics = record.get("metrics") or {}
    for key in ("final_roundtrip_exact", "final_roundtrip", "last_val"):
        candidate = metrics.get(key)
        if isinstance(candidate, dict) and candidate.get("val_bpb") is not None:
            return float(candidate["val_bpb"])
    return math.inf


def fmt_float(value: float | None, *, digits: int = 4) -> str:
    if value is None or math.isinf(value):
        return "-"
    return f"{value:.{digits}f}"


def trim(value: str, width: int) -> str:
    if len(value) <= width:
        return value
    return value[: max(width - 1, 0)] + "…"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize structured Parameter Golf runs.")
    parser.add_argument("--preset", help="Filter to a single preset.")
    parser.add_argument("--status", default="completed", help="Filter by run status. Use 'all' to disable.")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    records = load_records()
    if args.preset:
        records = [record for record in records if record.get("preset") == args.preset]
    if args.status != "all":
        records = [record for record in records if record.get("status") == args.status]
    records.sort(key=final_bpb)
    if args.limit > 0:
        records = records[: args.limit]

    if not records:
        print("No matching runs found.")
        return

    header = (
        f"{'run_name':24} {'preset':20} {'status':10} {'val_bpb':10} "
        f"{'artifact_mb':11} {'budget_mb':10} {'wall_s':8} {'commit':10}"
    )
    print(header)
    print("-" * len(header))
    for record in records:
        compressed_bytes = record.get("compressed_model_bytes")
        budget_bytes = record.get("submission_budget_estimate_bytes")
        commit = str(record.get("git_commit") or "")[:10]
        print(
            f"{trim(str(record.get('run_name')), 24):24} "
            f"{trim(str(record.get('preset')), 20):20} "
            f"{trim(str(record.get('status')), 10):10} "
            f"{fmt_float(final_bpb(record), digits=6):10} "
            f"{fmt_float((compressed_bytes or 0) / 1_000_000 if compressed_bytes else None, digits=3):11} "
            f"{fmt_float((budget_bytes or 0) / 1_000_000 if budget_bytes else None, digits=3):10} "
            f"{fmt_float(float(record.get('wall_clock_seconds') or 0.0), digits=1):8} "
            f"{commit:10}"
        )


if __name__ == "__main__":
    main()
