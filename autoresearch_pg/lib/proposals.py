from __future__ import annotations

import difflib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from autoresearch_pg.lib.dedupe import find_duplicate, mutation_fingerprints, rebuild_dedupe_index
from autoresearch_pg.lib.session_registry import update_codex_session
from autoresearch_pg.lib.workspace import (
    best_run_for_candidate,
    bootstrap_candidate,
    candidate_dir,
    config_root,
    dump_json,
    load_json,
    repo_root,
    state_root,
    utc_stamp,
)


def load_codex_config() -> dict[str, Any]:
    return load_json(config_root() / "codex.json", default={})


def codex_reasoning_effort(codex_cfg: dict[str, Any], *, purpose: str = "default") -> str:
    if purpose == "proposal":
        return str(
            codex_cfg.get(
                "proposal_reasoning_effort",
                codex_cfg.get("reasoning_effort", codex_cfg.get("default_reasoning_effort", "high")),
            )
        )
    return str(codex_cfg.get("default_reasoning_effort", codex_cfg.get("reasoning_effort", "high")))


def load_research_starters() -> dict[str, Any]:
    return load_json(config_root() / "research_starters.json", default={"version": 1, "ideas": {}})


def build_proposal_context_pack(
    *,
    family_name: str,
    family_description: str,
    focus_areas: list[str],
    tier: str,
    entrypoint_filename: str,
    allowed_files: list[str],
    parent_candidate_id: str | None,
    inherited_env_overrides: dict[str, Any],
    current_best: dict[str, Any] | None,
    global_best: dict[str, Any] | None,
    best_state: dict[str, Any],
    frontier: dict[str, Any],
    family_stats: dict[str, Any],
    codex_cfg: dict[str, Any],
) -> dict[str, Any]:
    context_cfg = dict(codex_cfg.get("context") or {})
    history_scan_limit = int(context_cfg.get("history_scan_limit", 300))
    family_recent_limit = int(context_cfg.get("family_recent_limit", 6))
    family_failure_limit = int(context_cfg.get("family_failure_limit", 4))
    champion_limit = int(context_cfg.get("champion_limit", 4))
    note_char_limit = int(context_cfg.get("note_char_limit", 280))
    diff_line_limit = int(context_cfg.get("diff_line_limit", 80))

    ledger_rows = _load_ledger_rows(limit=history_scan_limit)
    family_history = [row for row in ledger_rows if row.get("primary_family") == family_name]
    family_tier_history = [row for row in family_history if row.get("tier") == tier]
    valid_family_tier_history = [
        row for row in family_tier_history if bool((row.get("objective") or {}).get("valid"))
    ]

    family_recent = [
        _run_context_line(row)
        for row in family_history[:family_recent_limit]
    ]
    family_failures = [
        _run_context_line(row)
        for row in family_history
        if not bool((row.get("objective") or {}).get("valid"))
    ][:family_failure_limit]
    family_regressions = [
        _run_context_line(row)
        for row in family_tier_history
        if bool((row.get("objective") or {}).get("valid"))
    ]
    family_regressions = family_regressions[:family_recent_limit]

    same_tier_best = (best_state.get("by_tier") or {}).get(tier)
    same_tier_family_best = _best_run_from_history(family_tier_history)
    champion_rows = list(frontier.get("champion_set", []))[:champion_limit]
    parent_context = _parent_context(
        candidate_id=parent_candidate_id,
        entrypoint_filename=entrypoint_filename,
        note_char_limit=note_char_limit,
        diff_line_limit=diff_line_limit,
    )

    stats = ((family_stats.get("families") or {}).get(family_name) or {}).copy()
    observations = _build_observations(
        family_name=family_name,
        tier=tier,
        inherited_env_overrides=inherited_env_overrides,
        family_best=current_best,
        same_tier_best=same_tier_best,
        best_low_quant_gap=frontier.get("best_low_quant_gap"),
        family_stats=stats,
        family_failures=family_failures,
    )
    starter_cfg = dict(codex_cfg.get("research_starters") or {})
    starter_ideas = select_research_starters(
        family_name=family_name,
        registry=load_research_starters(),
        limit=int(starter_cfg.get("context_limit", 5)),
        enabled=bool(starter_cfg.get("enabled", True)),
    )
    proposal = _select_proposal_mode(
        family_name=family_name,
        parent_candidate_id=parent_candidate_id,
        current_best=current_best,
        same_tier_best=same_tier_best,
        same_tier_family_best=same_tier_family_best,
        valid_family_tier_history=valid_family_tier_history,
        codex_cfg=codex_cfg,
    )

    return {
        "objective": {
            "tier": tier,
            "entrypoint_filename": entrypoint_filename,
            "allowed_files": list(allowed_files),
            "byte_cap": 16_000_000,
            "starter_idea_prompt_limit": int(starter_cfg.get("prompt_limit", 3)),
        },
        "family": {
            "name": family_name,
            "description": family_description,
            "focus_areas": list(focus_areas),
            "stats": {
                "scheduled": stats.get("scheduled"),
                "completed": stats.get("completed"),
                "valid": stats.get("valid"),
                "invalid": stats.get("invalid"),
                "wins": stats.get("wins"),
                "best_score": stats.get("best_score"),
                "last_score": stats.get("last_score"),
                "recent_scores": list(stats.get("recent_scores", []))[-5:],
                "duplicates_skipped": stats.get("duplicates_skipped"),
            },
        },
        "base": {
            "parent_candidate_id": parent_candidate_id,
            "inherited_env_overrides": dict(inherited_env_overrides),
        },
        "starter_ideas": starter_ideas,
        "proposal": proposal,
        "parent_context": parent_context,
        "family_frontier": {
            "family_best": _summary_payload(current_best),
            "same_tier_family_best": _run_context_line(same_tier_family_best) if same_tier_family_best else None,
            "recent_family_runs": family_recent,
            "recent_same_tier_valid_runs": family_regressions,
            "recent_family_failures": family_failures,
        },
        "global_frontier": {
            "same_tier_best": _summary_payload(same_tier_best),
            "global_best": _summary_payload(global_best),
            "best_low_quant_gap": _summary_payload(frontier.get("best_low_quant_gap")),
            "best_with_byte_headroom": _summary_payload(frontier.get("best_with_byte_headroom")),
            "champion_set": [_summary_payload(row) for row in champion_rows if row],
        },
        "observations": observations,
    }


def build_proposal_prompt(
    *,
    program_text: str,
    context_pack: dict[str, Any],
) -> str:
    objective = context_pack["objective"]
    family = context_pack["family"]
    base = context_pack["base"]
    starter_ideas = context_pack.get("starter_ideas", [])
    proposal = context_pack["proposal"]
    parent = context_pack["parent_context"]
    family_frontier = context_pack["family_frontier"]
    global_frontier = context_pack["global_frontier"]
    observations = context_pack["observations"]

    focus_lines = "\n".join(f"- {item}" for item in family.get("focus_areas", [])) or "- one clear hypothesis"
    allowed_lines = "\n".join(f"- {name}" for name in objective.get("allowed_files", []))
    observation_lines = "\n".join(f"- {item}" for item in observations) if observations else "- none yet"
    inherited_env = json.dumps(base.get("inherited_env_overrides", {}), sort_keys=True)
    family_stats = family.get("stats", {})
    same_tier_best = global_frontier.get("same_tier_best")
    family_best = family_frontier.get("family_best")
    same_tier_family_best = family_frontier.get("same_tier_family_best")
    best_low_quant_gap = global_frontier.get("best_low_quant_gap")
    best_with_byte_headroom = global_frontier.get("best_with_byte_headroom")
    recent_family_runs = family_frontier.get("recent_same_tier_valid_runs", [])[:4]
    recent_failures = family_frontier.get("recent_family_failures", [])[:2]
    operator_digest = _operator_digest(
        [row for row in family_frontier.get("recent_family_runs", []) if isinstance(row, dict)],
        limit=3,
    )
    parent_hypothesis = parent.get("note_sections", {}).get("Hypothesis", "none")
    parent_knobs = parent.get("note_sections", {}).get("Exact knobs changed", "none")
    parent_symbols = ", ".join(parent.get("changed_symbols", [])[:8]) or "none"
    program_summary = _program_summary(program_text)
    global_note = _global_note(global_frontier.get("global_best"), objective.get("tier"))
    starter_lines = _render_bullet_list(
        [_starter_idea_line(item) for item in starter_ideas[: int(objective.get("starter_idea_prompt_limit", 3))]],
        fallback="- none yet",
    )
    proposal_mode = str(proposal.get("mode", "brief"))
    mode_reason_lines = _render_bullet_list(proposal.get("reasons", []), fallback="- none")
    ambiguity_lines = _render_bullet_list(
        [_compact_run_line(row) for row in proposal.get("ambiguity_competitors", [])],
        fallback="- none",
    )
    deep_extra = ""
    if proposal_mode == "deep":
        champion_lines = _render_bullet_list(
            [_compact_run_line(row) for row in global_frontier.get("champion_set", [])[:4]],
            fallback="- none yet",
        )
        recent_family_all = _render_bullet_list(
            [_compact_run_line(row) for row in family_frontier.get("recent_family_runs", [])[:6]],
            fallback="- none yet",
        )
        diff_excerpt = parent.get("diff_excerpt", "no diff available")
        deep_extra = f"""

Mode rationale:
{mode_reason_lines}

Ambiguous contenders:
{ambiguity_lines}

Champion set:
{champion_lines}

Broader family history:
{recent_family_all}

Parent delta excerpt:
```diff
{diff_excerpt}
```
"""

    return f"""You are preparing one new candidate for the OpenAI Parameter Golf challenge.

Make one coherent proposal in this candidate-local workspace.

Mission:
{program_summary}

Read first:
- `{objective.get("entrypoint_filename")}`
- `notes.md`
- `../../program.md`
- `proposal_artifacts/codex_context_latest.json` if you need deeper history or raw context

Current family:
- family: {family.get("name")}
- proposal_mode: {proposal_mode}
- tier: {objective.get("tier")}
- parent_candidate_id: {base.get("parent_candidate_id")}
- inherited_env_overrides: {inherited_env}
- family_progress: completed={family_stats.get("completed")} valid={family_stats.get("valid")} invalid={family_stats.get("invalid")} wins={family_stats.get("wins")} best={family_stats.get("best_score")} last={family_stats.get("last_score")}

Family focus:
{focus_lines}

Curated starter ideas:
{starter_lines}

Scoreboard:
- same_tier_best: {_compact_run_line(same_tier_best)}
- family_best: {_compact_run_line(family_best)}
- same_tier_family_best: {_compact_run_line(same_tier_family_best)}
- low_quant_gap_anchor: {_compact_run_line(best_low_quant_gap)}
- byte_headroom_anchor: {_compact_run_line(best_with_byte_headroom)}
{global_note}

Recent local evidence:
{_render_bullet_list([_compact_run_line(row) for row in recent_family_runs], fallback="- none yet")}

Recent failure evidence:
{_render_bullet_list([_compact_run_line(row) for row in recent_failures], fallback="- none yet")}

Recent operator pattern:
- {operator_digest}

Observations:
{observation_lines}

Parent context:
- parent_best_run: {_compact_run_line(parent.get("best_run"))}
- parent_hypothesis: {parent_hypothesis}
- parent_knobs: {parent_knobs}
- parent_changed_symbols: {parent_symbols}
- parent_last_proposal: {parent.get("proposal_summary", "none")}
{deep_extra}

Workspace rules:
- Edit only the candidate-local files listed below.
- Keep the change small enough that it is attributable.
- Do not launch training or any long-running experiments.
- Preserve runnability of `{objective.get("entrypoint_filename")}`.
- If you change the hypothesis materially, update `notes.md` sections 1-4.
- Prefer same-tier signals over cross-tier scoreboard noise when they disagree.
- Keep the improvement compatible with the inherited env overrides unless you have a clear reason to change behavior in code.
- Prefer one targeted change that composes with the parent over a broad rewrite.

Allowed files:
{allowed_lines}

Primary task:
- Inspect `{objective.get("entrypoint_filename")}`.
- Implement one promising improvement for the {family.get("name")} family.
- Prefer a small but real code change over commentary.
- Do not touch files outside this candidate workspace.
- Use the context above to avoid duplicating already-tried weak ideas.

Final response:
- Briefly state the hypothesis.
- Mention the exact files you changed.
- Mention expected upside and main risk.
"""


def bootstrap_proposal_candidate(
    *,
    candidate_id: str,
    source_entrypoint: Path,
    parent_candidate_id: str | None,
    family_name: str,
    env_overrides: dict[str, Any],
    note: str,
) -> Path:
    out_dir = bootstrap_candidate(
        candidate_id=candidate_id,
        source_train_gpt=source_entrypoint,
        parent_candidate_id=parent_candidate_id,
        note=note,
    )
    meta_path = out_dir / "meta.json"
    meta = load_json(meta_path, default={})
    meta.update(
        {
            "primary_family": family_name,
            "secondary_tags": ["codex_proposal"],
            "env_overrides": dict(env_overrides),
            "mutation_operator": "codex_proposal",
            "status": "proposal_pending",
            "promotion_history": [],
        }
    )
    dump_json(meta_path, meta)
    return out_dir


def run_codex_proposal(
    *,
    candidate_id: str,
    family_name: str,
    tier: str,
    proposal_mode: str,
    candidate_dir: Path,
    prompt: str,
    context_pack: dict[str, Any],
    codex_cfg: dict[str, Any],
) -> dict[str, Any]:
    proposal_dir = candidate_dir / "proposal_artifacts"
    proposal_dir.mkdir(parents=True, exist_ok=True)
    stamp = utc_stamp()
    prompt_path = proposal_dir / f"codex_prompt_{stamp}.md"
    prompt_latest_path = proposal_dir / "codex_prompt_latest.md"
    context_path = proposal_dir / f"codex_context_{stamp}.json"
    context_latest_path = proposal_dir / "codex_context_latest.json"
    last_message_path = proposal_dir / f"codex_last_message_{stamp}.md"
    last_message_latest_path = proposal_dir / "codex_last_message_latest.md"
    stdout_path = proposal_dir / f"codex_exec_{stamp}.jsonl"
    stdout_latest_path = proposal_dir / "codex_exec_latest.jsonl"
    prompt_path.write_text(prompt, encoding="utf-8")
    prompt_latest_path.write_text(prompt, encoding="utf-8")
    dump_json(context_path, context_pack)
    dump_json(context_latest_path, context_pack)

    cmd = [
        "codex",
        "exec",
        "-C",
        str(candidate_dir),
        "-m",
        str(codex_cfg.get("model", "gpt-5.4")),
        "-c",
        f'reasoning_effort="{codex_reasoning_effort(codex_cfg, purpose="proposal")}"',
        "--output-last-message",
        str(last_message_path),
        "--color",
        "never",
    ]
    sandbox_mode = str(codex_cfg.get("sandbox", "workspace-write"))
    approval_policy = str(codex_cfg.get("approval_policy", "never"))
    if sandbox_mode == "workspace-write" and approval_policy == "never":
        cmd.append("--full-auto")
    else:
        cmd.extend(["-s", sandbox_mode])
    if codex_cfg.get("json_output", True):
        cmd.append("--json")
    if codex_cfg.get("skip_git_repo_check", True):
        cmd.append("--skip-git-repo-check")
    if codex_cfg.get("allow_web_search", False):
        cmd.append("--search")
    cmd.append("-")

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdin is not None
    assert proc.stdout is not None
    proc.stdin.write(prompt)
    proc.stdin.close()

    update_codex_session(
        candidate_id,
        started_at=utc_stamp(),
        status="proposing",
        family=family_name,
        tier=tier,
        proposal_mode=proposal_mode,
        pid=proc.pid,
        candidate_dir=str(candidate_dir),
        prompt_path=str(prompt_path),
        prompt_latest_path=str(prompt_latest_path),
        context_path=str(context_path),
        context_latest_path=str(context_latest_path),
        stdout_path=str(stdout_path),
        stdout_latest_path=str(stdout_latest_path),
        last_message_path=str(last_message_path),
        last_message_latest_path=str(last_message_latest_path),
    )

    tee_stream = bool(codex_cfg.get("tee_stream_to_parent", False))
    with open(stdout_path, "w", encoding="utf-8") as exact_handle, open(
        stdout_latest_path, "w", encoding="utf-8"
    ) as latest_handle:
        for line in proc.stdout:
            exact_handle.write(line)
            exact_handle.flush()
            latest_handle.write(line)
            latest_handle.flush()
            if tee_stream:
                sys.stdout.write(line)
                sys.stdout.flush()
    proc.wait()
    if last_message_path.is_file():
        last_message_latest_path.write_text(last_message_path.read_text(encoding="utf-8"), encoding="utf-8")
    update_codex_session(
        candidate_id,
        status="proposal_done" if proc.returncode == 0 else "proposal_failed",
        proposal_return_code=proc.returncode,
        pid=proc.pid,
    )
    return {
        "command": cmd,
        "return_code": proc.returncode,
        "pid": proc.pid,
        "prompt_path": str(prompt_path),
        "prompt_latest_path": str(prompt_latest_path),
        "context_path": str(context_path),
        "context_latest_path": str(context_latest_path),
        "last_message_path": str(last_message_path),
        "last_message_latest_path": str(last_message_latest_path),
        "stdout_path": str(stdout_path),
        "stdout_latest_path": str(stdout_latest_path),
    }


def finalize_proposal_candidate(
    *,
    candidate_id: str,
    candidate_dir: Path,
    family_name: str,
    parent_candidate_id: str | None,
    entrypoint_filename: str,
    inherited_env_overrides: dict[str, Any],
    context_pack: dict[str, Any],
    codex_cfg: dict[str, Any],
    codex_result: dict[str, Any],
    before_text: str,
) -> dict[str, Any]:
    entrypoint_path = candidate_dir / entrypoint_filename
    after_text = entrypoint_path.read_text(encoding="utf-8")
    changed = after_text != before_text

    prompt_summary = {
        "engine": "codex",
        "model": codex_cfg.get("model"),
        "reasoning_effort": codex_reasoning_effort(codex_cfg, purpose="proposal"),
        "prompt_path": codex_result["prompt_path"],
        "prompt_latest_path": codex_result["prompt_latest_path"],
        "context_path": codex_result["context_path"],
        "context_latest_path": codex_result["context_latest_path"],
        "last_message_path": codex_result["last_message_path"],
        "last_message_latest_path": codex_result["last_message_latest_path"],
        "stdout_path": codex_result["stdout_path"],
        "stdout_latest_path": codex_result["stdout_latest_path"],
        "return_code": codex_result["return_code"],
    }
    mutation_payload = {
        "operator_type": "codex_proposal",
        "proposal_engine": "codex",
        "model": codex_cfg.get("model"),
        "reasoning_effort": codex_reasoning_effort(codex_cfg, purpose="proposal"),
        "proposal_mode": context_pack.get("proposal", {}).get("mode"),
        "proposal_mode_reasons": list(context_pack.get("proposal", {}).get("reasons", [])),
        "parent_candidate_id": parent_candidate_id,
        "artifacts": prompt_summary,
        "changed_entrypoint": changed,
    }
    fingerprints = mutation_fingerprints(
        entrypoint_filename=entrypoint_filename,
        entrypoint_text=after_text,
        env_overrides=inherited_env_overrides,
        primary_family=family_name,
        mutation_operator=str(codex_cfg.get("proposal_operator_id", "codex_proposal")),
        mutation_payload=mutation_payload,
    )

    meta_path = candidate_dir / "meta.json"
    meta = load_json(meta_path, default={})
    meta.update(
        {
            "primary_family": family_name,
            "parent_candidate_id": parent_candidate_id,
            "env_overrides": dict(inherited_env_overrides),
            "mutation_operator": str(codex_cfg.get("proposal_operator_id", "codex_proposal")),
            "mutation_payload": mutation_payload,
            "proposal_engine": "codex",
            "proposal_artifacts": prompt_summary,
            "proposal_mode": context_pack.get("proposal", {}).get("mode"),
            "proposal_mode_reasons": list(context_pack.get("proposal", {}).get("reasons", [])),
            "proposal_changed_entrypoint": changed,
            "entrypoint_hash": fingerprints["entrypoint_hash"],
            "env_overrides_hash": fingerprints["env_overrides_hash"],
            "config_hash": fingerprints["config_hash"],
            "mutation_hash": fingerprints["mutation_hash"],
            "status": "proposed" if changed and codex_result["return_code"] == 0 else "proposal_failed",
        }
    )

    duplicate = None
    if changed and codex_result["return_code"] == 0:
        duplicate = find_duplicate(
            config_hash=fingerprints["config_hash"],
            mutation_hash=fingerprints["mutation_hash"],
            refresh=True,
            exclude_candidate_id=candidate_id,
        )
        if duplicate is not None:
            meta["proposal_duplicate"] = duplicate
            meta["status"] = "proposal_duplicate"

    dump_json(meta_path, meta)
    rebuild_dedupe_index()

    result = {
        "candidate_id": candidate_id,
        "candidate_dir": str(candidate_dir),
        "changed_entrypoint": changed,
        "return_code": codex_result["return_code"],
        "duplicate": duplicate,
        "fingerprints": fingerprints,
        "artifacts": prompt_summary,
        "status": meta["status"],
    }
    return result


def load_program_text() -> str:
    return (repo_root() / "autoresearch_pg" / "program.md").read_text(encoding="utf-8")


def _load_ledger_rows(*, limit: int) -> list[dict[str, Any]]:
    ledger_path = state_root() / "ledger.jsonl"
    if not ledger_path.is_file():
        return []
    lines = ledger_path.read_text(encoding="utf-8").splitlines()
    rows: list[dict[str, Any]] = []
    for line in reversed(lines[-limit:]):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _best_run_from_history(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    valid_rows = [row for row in rows if bool((row.get("objective") or {}).get("valid"))]
    if not valid_rows:
        return None
    return min(
        valid_rows,
        key=lambda row: float((row.get("objective") or {}).get("proxy_score") or 1e9),
    )


def _run_context_line(run_payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if not run_payload:
        return None
    objective = run_payload.get("objective") or {}
    line = {
        "candidate_id": run_payload.get("candidate_id"),
        "tier": run_payload.get("tier"),
        "valid": objective.get("valid"),
        "proxy_score": objective.get("proxy_score"),
        "post_quant_val_bpb": objective.get("post_quant_val_bpb"),
        "pre_quant_val_bpb": objective.get("pre_quant_val_bpb"),
        "quant_gap_bpb": objective.get("quant_gap_bpb"),
        "bytes_total": objective.get("bytes_total"),
        "step_avg_ms": objective.get("step_avg_ms"),
        "mutation_operator": run_payload.get("mutation_operator"),
        "template_id": run_payload.get("template_id"),
        "parent_candidate_id": run_payload.get("parent_candidate_id"),
    }
    return {key: value for key, value in line.items() if value is not None}


def _summary_payload(summary: dict[str, Any] | None) -> dict[str, Any] | None:
    if not summary:
        return None
    payload = {
        "candidate_id": summary.get("candidate_id"),
        "tier": summary.get("tier"),
        "valid": summary.get("valid"),
        "proxy_score": summary.get("proxy_score"),
        "post_quant_val_bpb": summary.get("post_quant_val_bpb"),
        "pre_quant_val_bpb": summary.get("pre_quant_val_bpb"),
        "quant_gap_bpb": summary.get("quant_gap_bpb"),
        "bytes_total": summary.get("bytes_total"),
        "mutation_operator": summary.get("mutation_operator"),
        "template_id": summary.get("template_id"),
        "parent_candidate_id": summary.get("parent_candidate_id"),
    }
    return {key: value for key, value in payload.items() if value is not None}


def _parent_context(
    *,
    candidate_id: str | None,
    entrypoint_filename: str,
    note_char_limit: int,
    diff_line_limit: int,
) -> dict[str, Any]:
    if not candidate_id:
        return {"summary": None, "best_run": None, "note_sections": {}, "proposal_summary": "none", "diff_excerpt": "no parent candidate"}

    meta = load_json(candidate_dir(candidate_id) / "meta.json", default={})
    notes_text = _safe_read(candidate_dir(candidate_id) / "notes.md")
    best_run = best_run_for_candidate(candidate_id, require_valid=False)
    proposal_path = Path(str((meta.get("proposal_artifacts") or {}).get("last_message_path", "")))
    proposal_summary = _compact_text(_safe_read(proposal_path), note_char_limit) if proposal_path.is_file() else "none"

    source_path = Path(str(meta.get("source_train_gpt", "")))
    current_entrypoint = candidate_dir(candidate_id) / entrypoint_filename
    diff_excerpt = _diff_excerpt(source_path, current_entrypoint, limit=diff_line_limit)

    note_sections = {
        key: _compact_text(value, note_char_limit)
        for key, value in _parse_note_sections(notes_text).items()
    }
    summary = {
        "candidate_id": candidate_id,
        "mutation_operator": meta.get("mutation_operator"),
        "template_id": meta.get("template_id"),
        "parent_candidate_id": meta.get("parent_candidate_id"),
        "env_overrides": meta.get("env_overrides"),
    }
    return {
        "summary": {key: value for key, value in summary.items() if value is not None},
        "best_run": _run_context_line(best_run),
        "note_sections": note_sections,
        "proposal_summary": proposal_summary or "none",
        "changed_symbols": _changed_symbol_summary(source_path, current_entrypoint, limit=10),
        "diff_excerpt": diff_excerpt,
    }


def _parse_note_sections(text: str) -> dict[str, str]:
    sections: dict[str, list[str]] = {}
    current: str | None = None
    pattern = re.compile(r"^\d+\.\s+(.+?)\s*$")
    for line in text.splitlines():
        match = pattern.match(line.strip())
        if match:
            current = match.group(1)
            sections.setdefault(current, [])
            continue
        if current is None:
            continue
        sections[current].append(line)
    return {key: "\n".join(value).strip() for key, value in sections.items()}


def _safe_read(path: Path) -> str:
    if not path or not path.is_file():
        return ""
    return path.read_text(encoding="utf-8")


def _diff_excerpt(source_path: Path, current_path: Path, *, limit: int) -> str:
    if not source_path.is_file() or not current_path.is_file():
        return "no diff available"
    source_lines = source_path.read_text(encoding="utf-8").splitlines()
    current_lines = current_path.read_text(encoding="utf-8").splitlines()
    diff_lines = list(
        difflib.unified_diff(
            source_lines,
            current_lines,
            fromfile=str(source_path),
            tofile=str(current_path),
            lineterm="",
        )
    )
    if not diff_lines:
        return "no diff from source"
    if len(diff_lines) > limit:
        omitted = len(diff_lines) - limit
        diff_lines = diff_lines[:limit] + [f"... ({omitted} more diff lines omitted)"]
    return "\n".join(diff_lines)


def _changed_symbol_summary(source_path: Path, current_path: Path, *, limit: int) -> list[str]:
    if not source_path.is_file() or not current_path.is_file():
        return []
    source_lines = source_path.read_text(encoding="utf-8").splitlines()
    current_lines = current_path.read_text(encoding="utf-8").splitlines()
    diff_lines = list(
        difflib.unified_diff(
            source_lines,
            current_lines,
            fromfile=str(source_path),
            tofile=str(current_path),
            lineterm="",
        )
    )
    names: list[str] = []
    seen: set[str] = set()
    patterns = [
        re.compile(r"^[+-]\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\("),
        re.compile(r"^[+-]\s*([A-Za-z_][A-Za-z0-9_]*)\s*[:=]"),
        re.compile(r'^[+-].*os\.environ\.get\("([A-Z0-9_]+)"'),
    ]
    for line in diff_lines:
        for pattern in patterns:
            match = pattern.search(line)
            if not match:
                continue
            name = match.group(1)
            if name in seen:
                continue
            seen.add(name)
            names.append(name)
            if len(names) >= limit:
                return names
    return names


def _compact_text(value: str, limit: int) -> str:
    text = " ".join(value.split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _render_json(value: Any, *, fallback: str) -> str:
    if value in (None, "", [], {}):
        return fallback
    return json.dumps(value, sort_keys=True)


def _render_bullet_list(items: list[dict[str, Any]] | list[str], *, fallback: str) -> str:
    if not items:
        return fallback
    rows: list[str] = []
    for item in items:
        if isinstance(item, str):
            rows.append(f"- {item}")
        else:
            rows.append(f"- {json.dumps(item, sort_keys=True)}")
    return "\n".join(rows)


def select_research_starters(
    *,
    family_name: str,
    registry: dict[str, Any],
    limit: int,
    enabled: bool,
) -> list[dict[str, Any]]:
    if not enabled:
        return []
    ideas = []
    for idea_id, idea in sorted((registry.get("ideas") or {}).items()):
        if not idea.get("enabled", True):
            continue
        primary_family = str(idea.get("primary_family", ""))
        secondary_families = [str(item) for item in idea.get("secondary_families", [])]
        if family_name != primary_family and family_name not in secondary_families:
            continue
        payload = {
            "idea_id": idea_id,
            "primary_family": primary_family,
            "secondary_families": secondary_families,
            "family_match_rank": 0 if family_name == primary_family else 1,
            "priority": int(idea.get("priority", 999)),
            "title": idea.get("title", ""),
            "summary": idea.get("summary", ""),
            "why_it_fits": idea.get("why_it_fits", ""),
            "implementation_sketch": idea.get("implementation_sketch", ""),
            "risk": idea.get("risk", ""),
            "horizon": idea.get("horizon", ""),
            "tags": list(idea.get("tags", [])),
            "source_refs": list(idea.get("source_refs", [])),
        }
        ideas.append(payload)
    ideas.sort(
        key=lambda item: (
            int(item.get("family_match_rank", 1)),
            int(item.get("priority", 999)),
            str(item.get("idea_id")),
        )
    )
    return ideas[:limit]


def _build_observations(
    *,
    family_name: str,
    tier: str,
    inherited_env_overrides: dict[str, Any],
    family_best: dict[str, Any] | None,
    same_tier_best: dict[str, Any] | None,
    best_low_quant_gap: dict[str, Any] | None,
    family_stats: dict[str, Any],
    family_failures: list[dict[str, Any]],
) -> list[str]:
    items: list[str] = []
    if inherited_env_overrides:
        items.append(
            f"Current base already inherits env overrides {json.dumps(inherited_env_overrides, sort_keys=True)}. Build incrementally unless you mean to replace that behavior in code."
        )
    if family_best and family_best.get("quant_gap_bpb") is not None and float(family_best["quant_gap_bpb"]) > 0.004:
        items.append(
            f"{family_name} family best still has a material quant gap ({float(family_best['quant_gap_bpb']):.6f} BPB). Export-aware changes remain valuable."
        )
    if family_best and family_best.get("bytes_total") is not None:
        headroom = 16_000_000 - int(family_best["bytes_total"])
        if headroom < 500_000:
            items.append(
                f"{family_name} family best is close to the byte cap with only {headroom} bytes of headroom. Avoid size growth without a clear score payoff."
            )
    if same_tier_best and same_tier_best.get("candidate_id") and same_tier_best.get("tier") == tier:
        items.append(
            f"Use same-tier results as the main local compass. Current tier best is {same_tier_best.get('candidate_id')} at proxy_score={same_tier_best.get('proxy_score')}."
        )
    if best_low_quant_gap and best_low_quant_gap.get("candidate_id") and best_low_quant_gap.get("primary_family") not in (None, family_name):
        items.append(
            f"Best low-quant-gap candidate currently comes from {best_low_quant_gap.get('primary_family')}. Borrow export-stable habits if relevant."
        )
    invalid = int(family_stats.get("invalid") or 0)
    completed = int(family_stats.get("completed") or 0)
    if completed and invalid / completed >= 0.4:
        items.append(
            f"{family_name} has a high invalid rate ({invalid}/{completed}). Prefer conservative, attributable edits over large speculative patches."
        )
    if family_failures:
        recent_ops = [row.get("mutation_operator") for row in family_failures if row.get("mutation_operator")]
        if recent_ops:
            items.append(
                f"Recent failed family attempts include operators {json.dumps(recent_ops[:3])}. Avoid repeating the same weak pattern without a sharper hypothesis."
            )
    return items


def _starter_idea_line(idea: dict[str, Any]) -> str:
    idea_id = idea.get("idea_id", "idea")
    horizon = idea.get("horizon", "")
    risk = idea.get("risk", "")
    title = idea.get("title", "")
    summary = _compact_text(str(idea.get("summary", "")), 180)
    labels = []
    if horizon:
        labels.append(str(horizon))
    if risk:
        labels.append(f"risk={risk}")
    label_text = f" [{' | '.join(labels)}]" if labels else ""
    return f"`{idea_id}`{label_text}: {title}. {summary}"


def _select_proposal_mode(
    *,
    family_name: str,
    parent_candidate_id: str | None,
    current_best: dict[str, Any] | None,
    same_tier_best: dict[str, Any] | None,
    same_tier_family_best: dict[str, Any] | None,
    valid_family_tier_history: list[dict[str, Any]],
    codex_cfg: dict[str, Any],
) -> dict[str, Any]:
    mode_cfg = dict(codex_cfg.get("modes") or {})
    default_mode = str(mode_cfg.get("default", "brief"))
    deep_families = {str(item) for item in mode_cfg.get("deep_families", ["architecture", "compression"])}
    ambiguity_margin = float(mode_cfg.get("ambiguity_margin_bpb", 0.003))
    ambiguity_top_k = int(mode_cfg.get("ambiguity_top_k", 3))

    reasons: list[str] = []
    if family_name in deep_families:
        reasons.append(f"{family_name} defaults to deep mode")
    if parent_candidate_id is None:
        reasons.append("no parent candidate selected")
    if same_tier_family_best is None:
        reasons.append("no same-tier family best yet")
    elif parent_candidate_id and parent_candidate_id != same_tier_family_best.get("candidate_id"):
        reasons.append("selected parent differs from same-tier family best")

    ranked = sorted(
        valid_family_tier_history,
        key=lambda row: float((row.get("objective") or {}).get("proxy_score") or 1e9),
    )
    compact_ranked = [_run_context_line(row) for row in ranked[:ambiguity_top_k]]
    if len(ranked) >= 2:
        first = float((ranked[0].get("objective") or {}).get("proxy_score") or 1e9)
        second = float((ranked[1].get("objective") or {}).get("proxy_score") or 1e9)
        if abs(second - first) <= ambiguity_margin:
            reasons.append(
                f"top same-tier family contenders are within {ambiguity_margin:.4f} BPB"
            )

    if current_best and same_tier_best:
        family_score = current_best.get("proxy_score")
        tier_score = same_tier_best.get("proxy_score")
        if (
            family_score is not None
            and tier_score is not None
            and current_best.get("candidate_id") != same_tier_best.get("candidate_id")
            and abs(float(family_score) - float(tier_score)) <= ambiguity_margin
        ):
            reasons.append("family best is very close to a different same-tier global anchor")

    mode = "deep" if reasons else default_mode
    return {
        "mode": mode,
        "reasons": reasons,
        "ambiguity_competitors": compact_ranked if mode == "deep" else [],
    }


def _compact_run_line(summary: dict[str, Any] | None) -> str:
    if not summary:
        return "none"
    parts = []
    candidate_id = summary.get("candidate_id")
    if candidate_id:
        parts.append(str(candidate_id))
    tier = summary.get("tier")
    if tier:
        parts.append(f"tier={tier}")
    proxy_score = summary.get("proxy_score")
    if proxy_score is not None:
        parts.append(f"score={float(proxy_score):.6f}")
    quant_gap = summary.get("quant_gap_bpb")
    if quant_gap is not None:
        parts.append(f"gap={float(quant_gap):.6f}")
    bytes_total = summary.get("bytes_total")
    if bytes_total is not None:
        parts.append(f"bytes={int(bytes_total) / 1_000_000:.2f}MB")
    mutation_operator = summary.get("mutation_operator")
    if mutation_operator:
        parts.append(f"op={mutation_operator}")
    template_id = summary.get("template_id")
    if template_id:
        parts.append(f"tpl={template_id}")
    valid = summary.get("valid")
    if valid is not None:
        parts.append(f"valid={str(bool(valid)).lower()}")
    return " ".join(parts) if parts else "none"


def _operator_digest(rows: list[dict[str, Any]], *, limit: int) -> str:
    counts: dict[str, int] = {}
    for row in rows:
        label = str(row.get("template_id") or row.get("mutation_operator") or "unknown")
        counts[label] = counts.get(label, 0) + 1
    if not counts:
        return "none yet"
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]
    return ", ".join(f"{label} x{count}" for label, count in ordered)


def _program_summary(program_text: str) -> str:
    lines = [line.strip() for line in program_text.splitlines()]
    points: list[str] = []
    for line in lines:
        if not line.startswith("- "):
            continue
        content = line[2:].strip()
        if not content:
            continue
        points.append(content)
        if len(points) >= 3:
            break
    if not points:
        return "- minimize post-quant val_bpb\n- keep bytes_total under 16_000_000\n- keep candidates reproducible"
    return "\n".join(f"- {point}" for point in points)


def _global_note(global_best: dict[str, Any] | None, active_tier: str | None) -> str:
    if not global_best:
        return ""
    global_tier = global_best.get("tier")
    if not global_tier or global_tier == active_tier:
        return ""
    return f"- cross_tier_global_best: {_compact_run_line(global_best)} (informational only; do not rank it above same-tier evidence)"
