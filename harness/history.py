from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import HISTORY_PATH, STATE_PATH, append_jsonl, ensure_lab_layout, load_jsonl, read_json, write_json


def _compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    compact = dict(metrics)
    train_history = list(compact.pop("train_history", []))
    val_history = list(compact.pop("val_history", []))
    compact["train_history_points"] = len(train_history)
    compact["val_history_points"] = len(val_history)
    compact["train_history_tail"] = train_history[-5:]
    compact["val_history_tail"] = val_history[-5:]
    return compact


def _compact_record(record: dict[str, Any]) -> dict[str, Any]:
    compact = dict(record)
    compact["metrics"] = _compact_metrics(record.get("metrics", {}))
    return compact


def record_is_planner_eligible(record: dict[str, Any]) -> bool:
    result = record.get("result", {})
    metrics = record.get("metrics", {})
    if "planner_eligible" in result:
        return bool(result.get("planner_eligible"))
    if result.get("status") != "completed":
        return False
    if metrics.get("metrics_valid") is False:
        return False
    return bool(
        metrics.get("final_roundtrip_val_bpb") is not None
        or metrics.get("last_pre_quant_val_bpb") is not None
    )


def record_evidence_tier(record: dict[str, Any]) -> str:
    result = record.get("result", {})
    spec = record.get("spec", {})
    metrics = record.get("metrics", {})
    preflight = record.get("preflight", {})
    if result.get("evidence_tier"):
        return str(result["evidence_tier"])
    if metrics.get("metrics_valid") is False:
        return "invalid"
    if spec.get("source") == "bootstrap-record":
        return "record_import"
    comparability = result.get("comparability") or preflight.get("comparability")
    if comparability == "record-candidate":
        return "challenge_record"
    if comparability == "full-comparable":
        return "challenge_candidate"
    if comparability == "subset-only":
        return "subset"
    if str(spec.get("track", "")).startswith("local-"):
        return "smoke"
    return "unknown"


def planner_eligible_rows(history: list[dict[str, Any]], profile_name: str) -> list[dict[str, Any]]:
    return [
        row
        for row in history
        if row.get("spec", {}).get("profile") == profile_name and record_is_planner_eligible(row)
    ]


def load_history() -> list[dict[str, Any]]:
    ensure_lab_layout()
    return load_jsonl(HISTORY_PATH)


def append_record(record: dict[str, Any]) -> None:
    ensure_lab_layout()
    append_jsonl(HISTORY_PATH, _compact_record(record))


def load_state() -> dict[str, Any]:
    ensure_lab_layout()
    return read_json(STATE_PATH, {"next_run_index": 1})


def save_state(state: dict[str, Any]) -> None:
    ensure_lab_layout()
    write_json(STATE_PATH, state)


def next_run_index() -> int:
    state = load_state()
    index = int(state.get("next_run_index", 1))
    state["next_run_index"] = index + 1
    save_state(state)
    return index


def existing_experiment_ids() -> set[str]:
    return {row.get("spec", {}).get("experiment_id") for row in load_history() if row.get("spec", {}).get("experiment_id")}


def find_record(experiment_id: str) -> dict[str, Any] | None:
    for row in reversed(load_history()):
        if row.get("spec", {}).get("experiment_id") == experiment_id:
            return row
    return None
