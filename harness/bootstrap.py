from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .common import REPO_ROOT, utc_now_iso
from .history import append_record, existing_experiment_ids
from .log_parser import parse_log


def _track_from_path(path: Path) -> str:
    parent = path.parent.parent.name
    if parent == "track_10min_16mb":
        return "track_10min_16mb"
    if parent == "track_non_record_16mb":
        return "track_non_record_16mb"
    return parent


def bootstrap_records() -> list[dict[str, Any]]:
    imported: list[dict[str, Any]] = []
    seen = existing_experiment_ids()
    for train_log in sorted(REPO_ROOT.glob("records/*/*/train.log")):
        experiment_id = f"bootstrap::{train_log.parent.parent.name}::{train_log.parent.name}"
        if experiment_id in seen:
            continue
        submission_path = train_log.with_name("submission.json")
        submission = json.loads(submission_path.read_text(encoding="utf-8")) if submission_path.is_file() else {}
        metrics = parse_log(train_log)
        spec = {
            "experiment_id": experiment_id,
            "created_at": utc_now_iso(),
            "source": "bootstrap-record",
            "profile": "record_import",
            "track": _track_from_path(train_log),
            "launcher": "import",
            "script": str(train_log.parent / "train_gpt.py"),
            "supports_autoloop": False,
            "objective": "bootstrap historical context",
            "hypothesis": submission.get("blurb") or "Imported historical record.",
            "rationale": "Existing official/non-record runs are loaded into the harness memory so planning does not start blind.",
            "mutation_name": "bootstrap",
            "parent_experiment_id": None,
            "env": {},
            "tags": ["bootstrap", _track_from_path(train_log)],
        }
        result = {
            "status": "completed",
            "started_at": submission.get("date") or utc_now_iso(),
            "completed_at": submission.get("date") or utc_now_iso(),
            "returncode": 0,
            "duration_sec": 0.0,
            "run_dir": str(train_log.parent),
            "stdout_path": None,
            "log_path": str(train_log),
            "command": "imported from existing record",
            "raw_model_path": None,
            "quant_model_path": None,
        }
        record = {"spec": spec, "result": result, "metrics": metrics, "submission": submission}
        append_record(record)
        imported.append(record)
    return imported

