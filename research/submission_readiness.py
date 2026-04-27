from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from research.submission_metrics import canonical_submission_fields_for_status, metric_payload_by_label


READINESS_JSON = "submission_readiness.json"
READINESS_TEXT = "submission_readiness.txt"
SUMMARY_JSON = "run_summary.json"
RESULT_JSON = "result.json"
LEGALITY_JSON = "legality_note.json"
LEGALITY_TEXT = "legality_note.txt"
BYTE_BUDGET_JSON = "byte_budget.json"


def _load_json(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _dump_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sync_result_submission_fields(result: Mapping[str, object]) -> dict[str, object]:
    updated = dict(result)
    updated.update(
        canonical_submission_fields_for_status(
            updated.get("metrics") or {},
            status=str(updated.get("status") or "completed"),
            track=str(updated.get("track")) if updated.get("track") else None,
        )
    )
    return updated


def sync_summary_submission_fields(
    summary: Mapping[str, object] | None,
    result: Mapping[str, object],
) -> dict[str, object]:
    updated = dict(summary or {})
    training_best_val_bpb = updated.get("training_best_val_bpb", updated.get("best_val_bpb"))
    training_best_val_loss = updated.get("training_best_val_loss", updated.get("best_val_loss"))
    final_label = result.get("official_submission_metric_label") or result.get("final_submission_metric_label")
    final_bpb = result.get("final_submission_bpb")
    final_loss = result.get("final_submission_loss")
    updated["status"] = result.get("status", updated.get("status"))
    updated["training_best_val_bpb"] = training_best_val_bpb
    updated["training_best_val_loss"] = training_best_val_loss
    updated["final_submission_metric_label"] = result.get("final_submission_metric_label")
    updated["official_submission_metric_label"] = final_label
    updated["final_submission_bpb"] = final_bpb
    updated["final_submission_loss"] = final_loss
    updated["best_val_bpb"] = final_bpb if final_bpb is not None else training_best_val_bpb
    updated["best_val_loss"] = final_loss if final_loss is not None else training_best_val_loss
    return updated


def build_legality_note(
    result: Mapping[str, object],
    existing_note: Mapping[str, object] | None = None,
) -> tuple[dict[str, object], str]:
    note = dict(existing_note or {})
    status = str(result.get("status") or "incomplete")
    artifact_bytes = result.get("artifact_bytes_measured")
    if artifact_bytes is None:
        artifact_bytes = result.get("submission_budget_estimate_bytes")
    final_metric_present = result.get("final_submission_bpb") is not None
    artifact_budget_ok = artifact_bytes is None or int(artifact_bytes) <= 16_000_000
    manual_review_required = bool(result.get("requires_manual_rule_review") or result.get("manual_review_required"))
    if status != "completed":
        legality_status = "incomplete"
    elif not artifact_budget_ok or not final_metric_present:
        legality_status = "invalid"
    else:
        legality_status = "legal"
    note.update(
        {
            "preset": result.get("preset"),
            "status": legality_status,
            "artifact_budget_ok": artifact_budget_ok,
            "final_metric_present": final_metric_present,
            "manual_review_required": manual_review_required,
            "official_submission_metric_label": result.get("official_submission_metric_label") or result.get("final_submission_metric_label"),
            "official_submission_bpb": result.get("final_submission_bpb"),
        }
    )
    note.setdefault("legality_summary", [])
    lines = [
        f"Preset: {note.get('preset')}",
        f"Status: {note['status']}",
        f"Artifact budget ok: {note['artifact_budget_ok']}",
        f"Final metric present: {note['final_metric_present']}",
        f"Manual review required: {note['manual_review_required']}",
        f"Official submission metric: {note['official_submission_metric_label']}",
        f"Official submission bpb: {note['official_submission_bpb']}",
        "",
        "Why this run is compliant:",
    ]
    for item in note.get("legality_summary") or []:
        lines.append(f"- {item}")
    return note, "\n".join(lines) + "\n"


def _eval_stage_seconds(metrics: Mapping[str, object] | None) -> dict[str, float]:
    stages: dict[str, float] = {}
    if not isinstance(metrics, Mapping):
        return stages
    diagnostics = metrics.get("diagnostics")
    if isinstance(diagnostics, Mapping):
        for label, payload in diagnostics.items():
            if isinstance(payload, Mapping) and payload.get("eval_time_ms") is not None:
                stages[f"diagnostic_{label}"] = float(payload["eval_time_ms"]) / 1000.0
    named = metrics.get("named_evals")
    if isinstance(named, Mapping):
        for label, payload in named.items():
            if isinstance(payload, Mapping) and payload.get("eval_time_ms") is not None:
                stages[str(label)] = float(payload["eval_time_ms"]) / 1000.0
    return stages


def build_submission_readiness_report(
    result: Mapping[str, object],
    summary: Mapping[str, object] | None,
    legality_note: Mapping[str, object] | None,
    byte_budget: Mapping[str, object] | None,
) -> dict[str, object]:
    official_label = result.get("official_submission_metric_label") or result.get("final_submission_metric_label")
    official_bpb = result.get("final_submission_bpb")
    official_loss = result.get("final_submission_loss")
    metrics = result.get("metrics") if isinstance(result, Mapping) else None
    exported_bytes = None if not isinstance(byte_budget, Mapping) else byte_budget.get("exported_bytes_measured")
    code_bytes = None if not isinstance(byte_budget, Mapping) else byte_budget.get("code_bytes_measured")
    artifact_bytes = None if not isinstance(byte_budget, Mapping) else byte_budget.get("artifact_bytes_measured")
    if artifact_bytes is None:
        artifact_bytes = result.get("submission_budget_estimate_bytes")
    if code_bytes is None:
        code_bytes = result.get("counted_code_bytes")
    train_time_ms = None if not isinstance(summary, Mapping) else summary.get("train_time_ms")
    max_wallclock_seconds = None if not isinstance(summary, Mapping) else summary.get("max_wallclock_seconds")
    train_time_seconds = None if train_time_ms is None else float(train_time_ms) / 1000.0
    eval_stages_seconds = _eval_stage_seconds(metrics if isinstance(metrics, Mapping) else None)
    official_metric_payload = metric_payload_by_label(metrics if isinstance(metrics, Mapping) else None, str(official_label)) if official_label else None
    official_submission_eval_seconds = None
    if isinstance(official_metric_payload, Mapping) and official_metric_payload.get("eval_time_ms") is not None:
        official_submission_eval_seconds = float(official_metric_payload["eval_time_ms"]) / 1000.0
    train_budget_ok = None
    if train_time_ms is not None and max_wallclock_seconds is not None:
        train_budget_ok = float(train_time_ms) <= float(max_wallclock_seconds) * 1000.0 + 1e-6
    official_eval_budget_ok = None
    if official_submission_eval_seconds is not None and max_wallclock_seconds is not None:
        official_eval_budget_ok = official_submission_eval_seconds <= float(max_wallclock_seconds) + 1e-6
    byte_budget_ok = None if artifact_bytes is None else int(artifact_bytes) <= 16_000_000
    end_to_end_budget_ok = None
    if train_time_seconds is not None and official_submission_eval_seconds is not None and max_wallclock_seconds is not None:
        end_to_end_budget_ok = (train_time_seconds + official_submission_eval_seconds) <= float(max_wallclock_seconds) + 1e-6
    manual_review_required = bool(result.get("requires_manual_rule_review") or result.get("manual_review_required"))
    lane = result.get("lane")
    submission_safe = bool(
        result.get("submission_safe")
        and not manual_review_required
        and lane == "stable"
        and (legality_note or {}).get("status") == "legal"
        and byte_budget_ok is not False
        and train_budget_ok is not False
        and official_bpb is not None
    )

    issues: list[str] = []
    if isinstance(summary, Mapping):
        if summary.get("final_submission_metric_label") != result.get("final_submission_metric_label"):
            issues.append("run_summary final_submission_metric_label does not match result.json")
        if summary.get("final_submission_bpb") != result.get("final_submission_bpb"):
            issues.append("run_summary final_submission_bpb does not match result.json")
        if result.get("status") == "completed" and result.get("final_submission_bpb") is not None and summary.get("best_val_bpb") != result.get("final_submission_bpb"):
            issues.append("run_summary best_val_bpb is not aligned to canonical submission metric")
    if isinstance(legality_note, Mapping):
        if legality_note.get("official_submission_metric_label") != official_label:
            issues.append("legality_note official_submission_metric_label does not match result.json")
        if legality_note.get("official_submission_bpb") != official_bpb:
            issues.append("legality_note official_submission_bpb does not match result.json")
        if legality_note.get("manual_review_required") != manual_review_required:
            issues.append("legality_note manual_review_required does not match result.json")

    return {
        "completion_status": result.get("status"),
        "git_commit": result.get("git_commit"),
        "lane": lane,
        "track": result.get("track"),
        "risk": result.get("risk"),
        "final_submission_bpb": result.get("final_submission_bpb"),
        "official_submission_metric_label": official_label,
        "official_submission_bpb": official_bpb,
        "official_submission_loss": official_loss,
        "train_time_seconds": train_time_seconds,
        "official_submission_eval_seconds": official_submission_eval_seconds,
        "export_seconds": result.get("export_seconds"),
        "eval_stages_seconds": eval_stages_seconds,
        "exported_model_bytes": exported_bytes,
        "code_bytes": code_bytes,
        "artifact_bytes": artifact_bytes,
        "total_artifact_bytes": artifact_bytes,
        "legality_status": None if not isinstance(legality_note, Mapping) else legality_note.get("status"),
        "train_budget_ok": train_budget_ok,
        "official_eval_budget_ok": official_eval_budget_ok,
        "artifact_budget_ok": byte_budget_ok,
        "end_to_end_budget_ok": end_to_end_budget_ok,
        "submission_safe": submission_safe,
        "manual_review_required": manual_review_required,
        "wall_clock_constraint_appears_satisfied": train_budget_ok,
        "byte_budget_constraint_appears_satisfied": byte_budget_ok,
        "consistent": not issues,
        "issues": issues,
    }


def render_submission_readiness(report: Mapping[str, object]) -> str:
    lines = [
        f"completion_status: {report.get('completion_status')}",
        f"git_commit: {report.get('git_commit')}",
        f"lane: {report.get('lane')}",
        f"track: {report.get('track')}",
        f"risk: {report.get('risk')}",
        f"official_submission_metric_label: {report.get('official_submission_metric_label')}",
        f"final_submission_bpb: {report.get('final_submission_bpb')}",
        f"official_submission_bpb: {report.get('official_submission_bpb')}",
        f"official_submission_loss: {report.get('official_submission_loss')}",
        f"train_time_seconds: {report.get('train_time_seconds')}",
        f"official_submission_eval_seconds: {report.get('official_submission_eval_seconds')}",
        f"export_seconds: {report.get('export_seconds')}",
        f"exported_model_bytes: {report.get('exported_model_bytes')}",
        f"code_bytes: {report.get('code_bytes')}",
        f"artifact_bytes: {report.get('artifact_bytes')}",
        f"total_artifact_bytes: {report.get('total_artifact_bytes')}",
        f"legality_status: {report.get('legality_status')}",
        f"train_budget_ok: {report.get('train_budget_ok')}",
        f"official_eval_budget_ok: {report.get('official_eval_budget_ok')}",
        f"artifact_budget_ok: {report.get('artifact_budget_ok')}",
        f"end_to_end_budget_ok: {report.get('end_to_end_budget_ok')}",
        f"submission_safe: {report.get('submission_safe')}",
        f"manual_review_required: {report.get('manual_review_required')}",
        f"wall_clock_constraint_appears_satisfied: {report.get('wall_clock_constraint_appears_satisfied')}",
        f"byte_budget_constraint_appears_satisfied: {report.get('byte_budget_constraint_appears_satisfied')}",
        f"consistent: {report.get('consistent')}",
    ]
    eval_stages = report.get("eval_stages_seconds") or {}
    if eval_stages:
        lines.append("eval_stages_seconds:")
        for label, seconds in eval_stages.items():
            lines.append(f"- {label}: {seconds}")
    issues = report.get("issues") or []
    if issues:
        lines.append("issues:")
        for issue in issues:
            lines.append(f"- {issue}")
    return "\n".join(lines) + "\n"


def refresh_submission_artifacts(run_dir: Path, *, rewrite: bool = False) -> dict[str, object]:
    result_path = run_dir / RESULT_JSON
    if not result_path.is_file():
        raise FileNotFoundError(f"Missing result.json in {run_dir}")
    result = sync_result_submission_fields(_load_json(result_path) or {})
    summary_path = run_dir / SUMMARY_JSON
    summary = sync_summary_submission_fields(_load_json(summary_path), result) if summary_path.is_file() else None
    legality_note_path = run_dir / LEGALITY_JSON
    legality_note, legality_text = build_legality_note(result, _load_json(legality_note_path))
    byte_budget = _load_json(run_dir / BYTE_BUDGET_JSON)
    report = build_submission_readiness_report(result, summary, legality_note, byte_budget)
    if rewrite:
        _dump_json(result_path, result)
        if summary is not None:
            _dump_json(summary_path, summary)
        _dump_json(legality_note_path, legality_note)
        (run_dir / LEGALITY_TEXT).write_text(legality_text, encoding="utf-8")
        _dump_json(run_dir / READINESS_JSON, report)
        (run_dir / READINESS_TEXT).write_text(render_submission_readiness(report), encoding="utf-8")
    return {
        "result": result,
        "summary": summary,
        "legality_note": legality_note,
        "byte_budget": byte_budget,
        "submission_readiness": report,
        "legality_text": legality_text,
    }


def write_submission_readiness_reports(run_dir: Path, result: Mapping[str, object]) -> dict[str, object]:
    summary_path = run_dir / SUMMARY_JSON
    summary = sync_summary_submission_fields(_load_json(summary_path), result) if summary_path.is_file() else None
    if summary is not None:
        _dump_json(summary_path, summary)
    legality_note = _load_json(run_dir / LEGALITY_JSON)
    byte_budget = _load_json(run_dir / BYTE_BUDGET_JSON)
    report = build_submission_readiness_report(result, summary, legality_note, byte_budget)
    _dump_json(run_dir / READINESS_JSON, report)
    (run_dir / READINESS_TEXT).write_text(render_submission_readiness(report), encoding="utf-8")
    return report
