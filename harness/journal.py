from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import JOURNAL_PATH, format_env_block


HARNESS_START = "<!-- HARNESS_RUNS_START -->"
HARNESS_END = "<!-- HARNESS_RUNS_END -->"


def ensure_harness_section(path: Path = JOURNAL_PATH) -> None:
    text = path.read_text(encoding="utf-8")
    if HARNESS_START in text and HARNESS_END in text:
        return
    block = (
        "\n## Harness Runs\n\n"
        "Automatically generated entries from the local experiment harness live here.\n"
        f"{HARNESS_START}\n"
        f"{HARNESS_END}\n"
    )
    path.write_text(text.rstrip() + "\n" + block, encoding="utf-8")


def _render_run_entry(record: dict[str, Any]) -> str:
    spec = record["spec"]
    result = record["result"]
    metrics = record["metrics"]
    code_mutation = spec.get("code_mutation") or {}
    lines = [
        f"## {result['completed_at']} - harness::{spec['experiment_id']}",
        "",
        f"- Status: {result['status']}",
        f"- Profile: `{spec['profile']}`",
        f"- Track: `{spec['track']}`",
        f"- Objective: {spec['objective']}",
        f"- Hypothesis: {spec['hypothesis']}",
        f"- Thesis: {spec.get('thesis') or spec['hypothesis']}",
        f"- Hypothesis family: `{spec.get('hypothesis_family', 'unknown')}`",
        f"- Hypothesis id: `{spec.get('hypothesis_id', 'missing')}`",
        f"- Lineage id: `{spec.get('lineage_id', 'missing')}`",
        f"- Parent hypothesis: `{spec.get('parent_hypothesis_id') or 'none'}`",
        f"- Expected upside: {spec.get('expected_upside') or 'unknown'}",
        f"- Risk level: `{spec.get('risk_level', 'unknown')}`",
        f"- Kill criteria: {spec.get('kill_criteria') or 'unknown'}",
        f"- Promotion rule: {spec.get('promotion_rule') or 'unknown'}",
        f"- Rationale: {spec['rationale']}",
        f"- Parent run: `{spec.get('parent_experiment_id') or 'none'}`",
        f"- Mutation tag: `{spec.get('mutation_name', 'manual')}`",
        f"- Mutation kind: `{spec.get('mutation_kind', 'unknown')}`",
        f"- Comparability: `{result.get('comparability', 'unknown')}`",
        f"- Run state: `{result.get('run_state', result.get('status', 'unknown'))}`",
        f"- Evidence tier: `{result.get('evidence_tier', 'unknown')}`",
        f"- Planner eligible: `{result.get('planner_eligible')}`",
        f"- Source script: `{spec['script']}`",
        f"- Materialized script: `{result.get('materialized_script_path') or spec.get('materialized_script') or 'missing'}`",
        f"- Run directory: `{result['run_dir']}`",
        f"- Command: `{result['command']}`",
        "- Important env vars:",
        format_env_block(spec["env"]),
        f"- Log file: `{result.get('log_path') or 'missing'}`",
        f"- Stdout capture: `{result.get('stdout_path') or 'missing'}`",
        f"- Dataset: `{metrics.get('dataset') or 'unknown'}`",
        f"- Train shards: `{metrics.get('train_shards_actual')}` / `{metrics.get('train_shards_expected')}`",
        f"- Best pre-quant `val_bpb`: `{metrics.get('best_val_bpb')}`",
        f"- Last pre-quant `val_bpb`: `{metrics.get('last_pre_quant_val_bpb')}`",
        f"- Final roundtrip `val_bpb`: `{metrics.get('final_roundtrip_val_bpb')}`",
        f"- Quantization gap `val_bpb`: `{metrics.get('quant_gap_bpb')}`",
        f"- Serialized int8+zlib bytes: `{metrics.get('serialized_model_int8_zlib_bytes')}`",
        f"- Total submission bytes: `{metrics.get('total_submission_size_int8_zlib_bytes')}`",
        f"- Raw model bytes: `{metrics.get('raw_model_bytes')}`",
        f"- Code bytes: `{metrics.get('code_bytes')}`",
        f"- Code bytes delta: `{metrics.get('code_bytes_delta')}`",
        f"- Stop step: `{metrics.get('stop_step')}` / `{metrics.get('iterations')}`",
        f"- Stopped on wallclock: `{metrics.get('stopped_early_wallclock')}`",
        f"- Model params: `{metrics.get('model_params')}`",
    ]
    if code_mutation:
        lines.append(f"- Code mutation: `{code_mutation.get('name')}`")
        lines.append(f"- Code mutation family: `{code_mutation.get('family')}`")
        lines.append(f"- Code mutation signature: `{code_mutation.get('signature')}`")
        applied = code_mutation.get("applied") or []
        if applied:
            lines.append(f"- Code mutation changes: `{'; '.join(change['description'] for change in applied)}`")
    preflight = record.get("preflight") or {}
    if preflight:
        lines.append(f"- Preflight launchable: `{preflight.get('can_launch')}`")
        lines.append(f"- Challenge ready: `{preflight.get('challenge_ready')}`")
        lines.append(f"- Ready for execution: `{preflight.get('ready_for_execution')}`")
        if preflight.get("fatal_issues"):
            lines.append(f"- Preflight fatal issues: `{'; '.join(preflight['fatal_issues'])}`")
        if preflight.get("warnings"):
            lines.append(f"- Preflight warnings: `{'; '.join(preflight['warnings'])}`")
    parse_warnings = result.get("parse_warnings") or metrics.get("parse_warnings") or []
    if parse_warnings:
        lines.append(f"- Parse warnings: `{'; '.join(parse_warnings)}`")
    if result.get("failure_reason"):
        lines.append(f"- Failure reason: `{result.get('failure_reason')}`")
    if metrics.get("subset_warning"):
        lines.append("- Notes: local subset warning was present in the trainer log, so this run is not comparable to a full-data baseline.")
    return "\n".join(lines) + "\n"


def append_run_entry(record: dict[str, Any], path: Path = JOURNAL_PATH) -> None:
    ensure_harness_section(path)
    text = path.read_text(encoding="utf-8")
    experiment_id = record["spec"]["experiment_id"]
    if f"harness::{experiment_id}" in text:
        return
    entry = _render_run_entry(record)
    updated = text.replace(HARNESS_END, entry + "\n" + HARNESS_END)
    path.write_text(updated, encoding="utf-8")
