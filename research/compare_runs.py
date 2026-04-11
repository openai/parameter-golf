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
from research.submission_metrics import canonical_submission_eval, metric_payload_by_label


def load_records() -> list[dict[str, object]]:
    if not INDEX_JSONL.is_file():
        return []
    by_run_dir: dict[str, dict[str, object]] = {}
    ordered: list[dict[str, object]] = []
    with INDEX_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            run_dir = str(record.get("run_dir") or "")
            if run_dir and run_dir in by_run_dir:
                for idx, prior in enumerate(ordered):
                    if str(prior.get("run_dir") or "") == run_dir:
                        ordered[idx] = record
                        break
            else:
                ordered.append(record)
            if run_dir:
                by_run_dir[run_dir] = record
    return ordered


def preferred_eval(metrics: dict[str, object]) -> tuple[str | None, dict[str, object] | None]:
    return canonical_submission_eval(metrics)


def final_bpb(record: dict[str, object]) -> float:
    explicit = record.get("final_submission_bpb")
    if explicit is not None:
        return float(explicit)
    if record.get("status") != "completed":
        return math.inf
    _label, candidate = preferred_eval(record.get("metrics") or {})
    if isinstance(candidate, dict) and candidate.get("val_bpb") is not None:
        return float(candidate["val_bpb"])
    return math.inf


def legality_status(record: dict[str, object]) -> str:
    note = record.get("legality_note") or {}
    status = str(note.get("status") or "")
    if status:
        return status
    record_status = str(record.get("status") or "")
    return "legal" if record_status == "completed" else "incomplete"


def train_time_seconds(record: dict[str, object]) -> float | None:
    metrics = record.get("metrics") or {}
    for key in ("last_val", "last_train"):
        candidate = metrics.get(key)
        if isinstance(candidate, dict) and candidate.get("train_time_ms") is not None:
            return float(candidate["train_time_ms"]) / 1000.0
    return None


def eval_time_seconds(record: dict[str, object]) -> float | None:
    explicit_label = record.get("official_submission_metric_label") or record.get("final_submission_metric_label")
    metrics = record.get("metrics") or {}
    if explicit_label:
        candidate = metric_payload_by_label(metrics, str(explicit_label))
        if isinstance(candidate, dict) and candidate.get("eval_time_ms") is not None:
            return float(candidate["eval_time_ms"]) / 1000.0
    if record.get("status") != "completed":
        return None
    _label, candidate = preferred_eval(record.get("metrics") or {})
    if isinstance(candidate, dict) and candidate.get("eval_time_ms") is not None:
        return float(candidate["eval_time_ms"]) / 1000.0
    return None


def budget_value(record: dict[str, object], key: str) -> object:
    budget = record.get("byte_budget") or {}
    if isinstance(budget, dict):
        return budget.get(key)
    return None


def fmt_float(value: float | None, *, digits: int = 4) -> str:
    if value is None or math.isinf(value):
        return "-"
    return f"{value:.{digits}f}"


def fmt_mb(value: object) -> str:
    if value is None:
        return "-"
    return f"{float(value) / 1_000_000:.3f}"


def trim(value: str, width: int) -> str:
    if len(value) <= width:
        return value
    return value[: max(width - 1, 0)] + "…"


def summarized_record(record: dict[str, object]) -> dict[str, object]:
    metrics = record.get("metrics") or {}
    label = record.get("official_submission_metric_label") or record.get("final_submission_metric_label")
    candidate: dict[str, object] | None = metric_payload_by_label(metrics, str(label)) if label else None
    if candidate is None and record.get("status") == "completed":
        label, candidate = preferred_eval(metrics)
    budget = record.get("byte_budget") or {}
    return {
        "run_name": record.get("run_name"),
        "preset": record.get("preset"),
        "scale": record.get("scale"),
        "status": record.get("status"),
        "legality": legality_status(record),
        "final_eval_label": label,
        "val_bpb": record.get("final_submission_bpb") if record.get("final_submission_bpb") is not None else (None if candidate is None else candidate.get("val_bpb")),
        "wall_clock_seconds": record.get("wall_clock_seconds"),
        "train_time_seconds": train_time_seconds(record),
        "eval_time_seconds": eval_time_seconds(record),
        "exported_model_bytes": budget.get("exported_bytes_measured") if isinstance(budget, dict) else None,
        "code_bytes": budget.get("code_bytes_measured") if isinstance(budget, dict) else record.get("counted_code_bytes"),
        "artifact_bytes": budget.get("artifact_bytes_measured") if isinstance(budget, dict) else record.get("submission_budget_estimate_bytes"),
        "remaining_headroom_to_16mb": budget.get("remaining_headroom_to_16MB") if isinstance(budget, dict) else None,
        "gpu_profile": record.get("gpu_profile"),
        "diff_base": record.get("diff_base"),
        "run_dir": record.get("run_dir"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize structured Parameter Golf runs.")
    parser.add_argument("--preset", help="Filter to a single preset.")
    parser.add_argument("--scale", help="Filter to a run scale such as smoke, half_run, or full_run.")
    parser.add_argument("--family", help="Filter to a preset family such as frontier or local.")
    parser.add_argument("--status", default="completed", help="Filter by run status. Use 'all' to disable.")
    parser.add_argument("--legality", help="Filter by legality status: legal, invalid, or incomplete.")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON instead of a table.")
    args = parser.parse_args()

    records = load_records()
    if args.preset:
        records = [record for record in records if record.get("preset") == args.preset]
    if args.scale:
        records = [record for record in records if record.get("scale") == args.scale]
    if args.family:
        records = [record for record in records if record.get("family") == args.family]
    if args.status != "all":
        records = [record for record in records if record.get("status") == args.status]
    if args.legality:
        records = [record for record in records if legality_status(record) == args.legality]
    records.sort(key=final_bpb)
    if args.limit > 0:
        records = records[: args.limit]

    if not records:
        print("No matching runs found.")
        return

    rows = [summarized_record(record) for record in records]
    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
        return

    header = (
        f"{'run_name':24} {'preset':28} {'scale':9} {'legal':10} {'val_bpb':10} "
        f"{'wall_s':8} {'train_s':8} {'eval_s':8} {'export_mb':10} "
        f"{'code_mb':9} {'artifact_mb':11} {'headroom_mb':11}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{trim(str(row.get('run_name')), 24):24} "
            f"{trim(str(row.get('preset')), 28):28} "
            f"{trim(str(row.get('scale') or '-'), 9):9} "
            f"{trim(str(row.get('legality') or '-'), 10):10} "
            f"{fmt_float(None if row.get('val_bpb') is None else float(row['val_bpb']), digits=6):10} "
            f"{fmt_float(None if row.get('wall_clock_seconds') is None else float(row['wall_clock_seconds']), digits=1):8} "
            f"{fmt_float(None if row.get('train_time_seconds') is None else float(row['train_time_seconds']), digits=1):8} "
            f"{fmt_float(None if row.get('eval_time_seconds') is None else float(row['eval_time_seconds']), digits=1):8} "
            f"{fmt_mb(row.get('exported_model_bytes')):10} "
            f"{fmt_mb(row.get('code_bytes')):9} "
            f"{fmt_mb(row.get('artifact_bytes')):11} "
            f"{fmt_mb(row.get('remaining_headroom_to_16mb')):11}"
        )


if __name__ == "__main__":
    main()
