#!/usr/bin/env python3
from __future__ import annotations

import argparse
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from autoresearch_pg.lib.frontier import (
    bump_duplicate_skip,
    bump_scheduled_count,
    ensure_family_stats_shape,
    ensure_frontier_shape,
    load_family_stats_state,
    load_frontier_state,
    save_family_stats_state,
    save_frontier_state,
    update_family_stats_state,
    update_frontier_state,
)
from autoresearch_pg.lib.frontier_actions import build_recombine_plan, build_replication_plan
from autoresearch_pg.lib.locking import state_lock
from autoresearch_pg.lib.mutations import apply_mutation_plan, build_mutation_plan, load_operator_registry
from autoresearch_pg.lib.proposals import (
    bootstrap_proposal_candidate,
    build_proposal_context_pack,
    build_proposal_prompt,
    codex_reasoning_effort,
    finalize_proposal_candidate,
    load_codex_config,
    load_program_text,
    run_codex_proposal,
)
from autoresearch_pg.lib.session_registry import update_codex_session
from autoresearch_pg.lib.templates import build_template_plan, choose_template, load_template_registry
from autoresearch_pg.lib.workspace import (
    best_run_for_candidate,
    candidate_dir,
    config_root,
    ensure_layout,
    load_json,
    repo_root,
    state_root,
    utc_stamp,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the autoresearch_pg portfolio scheduler")
    parser.add_argument("--cycles", type=int, help="Number of sequential scheduler cycles to run.")
    parser.add_argument("--tier", help="Tier to run. Defaults from scheduler.json for the current platform.")
    parser.add_argument("--family", help="Force one family instead of scheduler sampling.")
    parser.add_argument("--template", help="Force one template id instead of sampling operators/templates.")
    parser.add_argument("--seed", type=int, default=1337, help="RNG seed for action, family, and mutation sampling.")
    parser.add_argument("--dry-run", action="store_true", help="Plan candidates without launching training.")
    parser.add_argument("--no-templates", action="store_true", help="Disable template sampling for mutate actions.")
    parser.add_argument(
        "--source",
        help="Optional root source entrypoint to use when no parent candidate exists. Defaults from the selected tier.",
    )
    parser.add_argument(
        "--action",
        choices=["auto", "mutate", "propose", "replicate", "recombine"],
        default="auto",
        help="Force a portfolio action, or use auto selection.",
    )
    return parser


def load_config(name: str) -> dict[str, Any]:
    return load_json(config_root() / name)


def load_best_state() -> dict[str, Any]:
    return load_json(state_root() / "best.json", default={"global": None, "by_tier": {}})


def default_tier(scheduler_cfg: dict[str, Any]) -> str:
    system_key = f"{platform.system().lower()}_{platform.machine().lower()}"
    defaults = scheduler_cfg.get("tier_defaults", {})
    return defaults.get(system_key, defaults.get("default", "smoke_local"))


def default_source_for_tier(tier: str) -> Path:
    if "mlx" in tier:
        return repo_root() / "train_gpt_mlx.py"
    return repo_root() / "train_gpt.py"


def enabled_family_names(families_cfg: dict[str, Any]) -> list[str]:
    return [
        name
        for name, payload in families_cfg.get("families", {}).items()
        if payload.get("enabled")
    ]


def weighted_choice(items: list[tuple[str, float]], rng: random.Random) -> str:
    total = sum(max(weight, 0.0) for _, weight in items)
    if total <= 0:
        return items[0][0]
    target = rng.uniform(0.0, total)
    running = 0.0
    for key, weight in items:
        running += max(weight, 0.0)
        if running >= target:
            return key
    return items[-1][0]


def choose_family(
    families_cfg: dict[str, Any],
    family_stats: dict[str, Any],
    scheduler_cfg: dict[str, Any],
    rng: random.Random,
    forced_family: str | None,
    *,
    allowed_families: set[str] | None = None,
) -> str:
    families = families_cfg.get("families", {})
    if forced_family:
        if forced_family not in families:
            raise KeyError(f"unknown family {forced_family!r}")
        if not families[forced_family].get("enabled"):
            raise ValueError(f"family {forced_family!r} is disabled in config/families.json")
        if allowed_families is not None and forced_family not in allowed_families:
            raise ValueError(f"family {forced_family!r} is not available for this action")
        return forced_family

    tuning = scheduler_cfg.get("family_selection", {})
    choices: list[tuple[str, float]] = []
    for name in enabled_family_names(families_cfg):
        if allowed_families is not None and name not in allowed_families:
            continue
        family_cfg = families[name]
        stats = family_stats.setdefault("families", {}).setdefault(name, {})
        completed = int(stats.get("completed", 0))
        invalid = int(stats.get("invalid", 0))
        wins = int(stats.get("wins", 0))
        weight = float(family_cfg.get("base_weight", 0.0))
        weight += float(tuning.get("novelty_floor", 0.0))
        weight += float(tuning.get("exploration_bonus", 0.0)) / (1 + completed)
        weight += float(tuning.get("win_bonus", 0.0)) * min(wins, 3)
        weight -= float(tuning.get("failure_penalty", 0.0)) * invalid
        choices.append((name, weight))
    if not choices:
        raise ValueError("no eligible families for the selected action")
    return weighted_choice(choices, rng)


def choose_parent_candidate(
    family_name: str,
    family_cfg: dict[str, Any],
    frontier: dict[str, Any],
    best_state: dict[str, Any],
) -> str | None:
    policy = str(family_cfg.get("parent_policy", "family_best_then_global"))
    family_best = (frontier.get("best_by_family") or {}).get(family_name)
    global_best = frontier.get("global_best") or best_state.get("global")

    if policy.startswith("family_best") and family_best:
        return family_best.get("candidate_id")
    if policy.startswith("global_best") and global_best:
        return global_best.get("candidate_id")
    if family_best:
        return family_best.get("candidate_id")
    if global_best:
        return global_best.get("candidate_id")
    return None


def candidate_source_path(
    parent_candidate_id: str | None,
    tier: str,
    explicit_source: str | None,
) -> tuple[Path, dict[str, Any], str | None]:
    if explicit_source:
        return Path(explicit_source).resolve(), {}, None

    if parent_candidate_id:
        meta = load_json(candidate_dir(parent_candidate_id) / "meta.json", default={})
        entrypoint = meta.get("entrypoint_filename", "train_gpt.py")
        desired_entrypoint = default_source_for_tier(tier).name
        if entrypoint == desired_entrypoint:
            return candidate_dir(parent_candidate_id) / entrypoint, meta, parent_candidate_id
        return default_source_for_tier(tier), {}, None

    return default_source_for_tier(tier), {}, None


def should_try_template(
    *,
    template_registry: dict[str, Any],
    family_name: str,
    scheduler_cfg: dict[str, Any],
    rng: random.Random,
    forced_template_id: str | None,
    disable_templates: bool,
) -> bool:
    if forced_template_id:
        return True
    if disable_templates:
        return False
    template_cfg = scheduler_cfg.get("template_selection", {})
    if not template_cfg.get("enabled", True):
        return False
    family_templates = dict(template_registry.get("templates", {}))
    if not any(
        template.get("family") == family_name and template.get("enabled", True)
        for template in family_templates.values()
    ):
        return False
    return rng.random() < float(template_cfg.get("probability", 0.0))


def choose_action(
    frontier: dict[str, Any],
    scheduler_cfg: dict[str, Any],
    codex_cfg: dict[str, Any],
    rng: random.Random,
    forced_action: str,
) -> str:
    available = ["mutate"]
    if codex_cfg.get("enabled", False):
        available.append("propose")
    if frontier.get("replication_queue"):
        available.append("replicate")
    if has_recombine_opportunity(frontier):
        available.append("recombine")

    if forced_action != "auto":
        if forced_action not in available:
            raise ValueError(f"action {forced_action!r} is unavailable with the current frontier state")
        return forced_action

    action_cfg = scheduler_cfg.get("action_selection", {})
    propose_weight = float(action_cfg.get("propose_probability", 0.0))
    replicate_weight = float(action_cfg.get("replicate_probability", 0.0))
    recombine_weight = float(action_cfg.get("recombine_probability", 0.0))
    mutate_weight = float(
        action_cfg.get(
            "mutate_probability",
            max(0.0, 1.0 - propose_weight - replicate_weight - recombine_weight),
        )
    )

    choices: list[tuple[str, float]] = [("mutate", mutate_weight)]
    if "propose" in available:
        choices.append(("propose", propose_weight))
    if "replicate" in available:
        choices.append(("replicate", replicate_weight))
    if "recombine" in available:
        choices.append(("recombine", recombine_weight))
    return weighted_choice(choices, rng)


def has_recombine_opportunity(frontier: dict[str, Any]) -> bool:
    if not frontier.get("recombine_queue"):
        return False
    try:
        _, _, _, plan, _ = build_recombine_plan(frontier)
    except Exception:
        return False
    return plan.duplicate is None


def available_replication_families(frontier: dict[str, Any]) -> set[str]:
    families: set[str] = set()
    for row in frontier.get("replication_queue", []):
        family = row.get("primary_family")
        if family:
            families.add(str(family))
    return families


def available_recombine_families(frontier: dict[str, Any]) -> set[str]:
    families: set[str] = set()
    for row in frontier.get("recombine_queue", []):
        for family in row.get("families", []):
            if family:
                families.add(str(family))
    return families


def launch_candidate(candidate_id: str, tier: str) -> tuple[int, dict[str, Any] | None]:
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "run_candidate.py"),
        "--candidate",
        candidate_id,
        "--tier",
        tier,
    ]
    proc = subprocess.run(cmd, check=False)
    run_payload = best_run_for_candidate(candidate_id, tier=tier, require_valid=False)
    return proc.returncode, run_payload


def record_run_result(
    *,
    scheduler_cfg: dict[str, Any],
    family_name: str,
    run_payload: dict[str, Any] | None,
) -> tuple[bool, bool]:
    with state_lock():
        frontier = ensure_frontier_shape(load_frontier_state())
        family_stats = ensure_family_stats_shape(load_family_stats_state())
        got_family_win = False
        got_global_win = False
        if run_payload is not None:
            got_family_win, got_global_win = update_frontier_state(
                frontier,
                family_name,
                run_payload,
                recent_limit=int(scheduler_cfg.get("frontier_limits", {}).get("recent_runs", 50)),
                champion_limit=int(scheduler_cfg.get("frontier_limits", {}).get("champion_set", 8)),
            )
            save_frontier_state(frontier)
        update_family_stats_state(family_stats, family_name, run_payload, got_family_win, got_global_win)
        save_family_stats_state(family_stats)
        return got_family_win, got_global_win


def run_proposal_cycle(
    *,
    cycle_index: int,
    tier: str,
    families_cfg: dict[str, Any],
    scheduler_cfg: dict[str, Any],
    codex_cfg: dict[str, Any],
    rng: random.Random,
    forced_family: str | None,
    dry_run: bool,
    explicit_source: str | None,
) -> dict[str, Any]:
    with state_lock():
        frontier = ensure_frontier_shape(load_frontier_state())
        family_stats = ensure_family_stats_shape(load_family_stats_state())
        best_state = load_best_state()

        family_name = choose_family(families_cfg, family_stats, scheduler_cfg, rng, forced_family)
        family_cfg = families_cfg["families"][family_name]
        family_description = str(family_cfg.get("description", ""))
        parent_candidate_id = choose_parent_candidate(family_name, family_cfg, frontier, best_state)
        source_entrypoint, parent_meta, effective_parent_candidate_id = candidate_source_path(
            parent_candidate_id,
            tier,
            explicit_source,
        )
        inherited_env_overrides = dict(parent_meta.get("env_overrides", {}))
        current_best = (frontier.get("best_by_family") or {}).get(family_name)
        global_best = frontier.get("global_best")
        candidate_id = (
            f"{codex_cfg.get('candidate_prefix', 'cod')}_{family_name}_{utc_stamp()}_{rng.randint(1000, 9999)}"
        )
        note = (
            f"Scheduler cycle {cycle_index}: action=propose family={family_name}, "
            f"parent={effective_parent_candidate_id}, source={source_entrypoint.name}"
        )
        candidate_path = bootstrap_proposal_candidate(
            candidate_id=candidate_id,
            source_entrypoint=source_entrypoint,
            parent_candidate_id=effective_parent_candidate_id,
            family_name=family_name,
            env_overrides=inherited_env_overrides,
            note=note,
        )
        program_text = load_program_text()
        context_pack = build_proposal_context_pack(
            family_name=family_name,
            family_description=family_description,
            focus_areas=list((codex_cfg.get("family_focus") or {}).get(family_name, [])),
            tier=tier,
            entrypoint_filename=source_entrypoint.name,
            allowed_files=list(codex_cfg.get("allowed_files", [source_entrypoint.name, "notes.md"])),
            parent_candidate_id=effective_parent_candidate_id,
            inherited_env_overrides=inherited_env_overrides,
            current_best=current_best,
            global_best=global_best,
            best_state=best_state,
            frontier=frontier,
            family_stats=family_stats,
            codex_cfg=codex_cfg,
        )
        prompt = build_proposal_prompt(
            program_text=program_text,
            context_pack=context_pack,
        )
        payload = {
            "cycle": cycle_index,
            "tier": tier,
            "action": "propose",
            "status": "planned" if dry_run else "proposing",
            "candidate_id": candidate_id,
            "candidate_dir": str(candidate_path),
            "family": family_name,
            "operator": str(codex_cfg.get("proposal_operator_id", "codex_proposal")),
            "template_id": None,
            "parent_candidate_id": effective_parent_candidate_id,
            "source_entrypoint": str(source_entrypoint),
            "dry_run": dry_run,
            "model": codex_cfg.get("model"),
            "reasoning_effort": codex_reasoning_effort(codex_cfg, purpose="proposal"),
            "proposal_mode": context_pack.get("proposal", {}).get("mode"),
            "proposal_mode_reasons": list(context_pack.get("proposal", {}).get("reasons", [])),
        }
    print(
        f"[cycle {cycle_index}] action=propose family={family_name} candidate={candidate_id} "
        f"parent={effective_parent_candidate_id} model={codex_cfg.get('model')} "
        f"reasoning={codex_reasoning_effort(codex_cfg, purpose='proposal')}",
        flush=True,
    )
    if dry_run:
        return payload

    before_text = (candidate_path / source_entrypoint.name).read_text(encoding="utf-8")
    codex_result = run_codex_proposal(
        candidate_id=candidate_id,
        family_name=family_name,
        tier=tier,
        proposal_mode=str(context_pack.get("proposal", {}).get("mode", "brief")),
        candidate_dir=candidate_path,
        prompt=prompt,
        context_pack=context_pack,
        codex_cfg=codex_cfg,
    )
    payload["proposal_return_code"] = codex_result["return_code"]
    payload["proposal_artifacts"] = codex_result

    with state_lock():
        finalize = finalize_proposal_candidate(
            candidate_id=candidate_id,
            candidate_dir=candidate_path,
            family_name=family_name,
            parent_candidate_id=effective_parent_candidate_id,
            entrypoint_filename=source_entrypoint.name,
            inherited_env_overrides=inherited_env_overrides,
            context_pack=context_pack,
            codex_cfg=codex_cfg,
            codex_result=codex_result,
            before_text=before_text,
        )
        family_stats = ensure_family_stats_shape(load_family_stats_state())
        if finalize["duplicate"] is not None:
            bump_duplicate_skip(family_stats, family_name)
            save_family_stats_state(family_stats)
        elif finalize["status"] == "proposed":
            bump_scheduled_count(family_stats, family_name)
            save_family_stats_state(family_stats)
    payload.update(
        {
            "status": finalize["status"],
            "changed_entrypoint": finalize["changed_entrypoint"],
            "duplicate": finalize["duplicate"],
            "config_hash": finalize["fingerprints"]["config_hash"],
            "mutation_hash": finalize["fingerprints"]["mutation_hash"],
        }
    )
    update_codex_session(
        candidate_id,
        status=str(finalize["status"]),
        changed_entrypoint=bool(finalize["changed_entrypoint"]),
        duplicate=finalize["duplicate"],
        config_hash=finalize["fingerprints"]["config_hash"],
        mutation_hash=finalize["fingerprints"]["mutation_hash"],
    )
    if finalize["status"] != "proposed":
        return payload

    return_code, run_payload = launch_candidate(candidate_id, tier)
    payload["return_code"] = return_code
    payload["run_found"] = run_payload is not None
    record_run_result(
        scheduler_cfg=scheduler_cfg,
        family_name=family_name,
        run_payload=run_payload,
    )
    update_codex_session(
        candidate_id,
        status="completed" if return_code == 0 else "failed",
        run_return_code=return_code,
        valid=(run_payload or {}).get("objective", {}).get("valid") if run_payload is not None else None,
        run_dir=(run_payload or {}).get("run_dir") if run_payload is not None else None,
        train_log=(run_payload or {}).get("train_log") if run_payload is not None else None,
        stdout_log=(run_payload or {}).get("stdout_log") if run_payload is not None else None,
        post_quant_val_bpb=(run_payload or {}).get("objective", {}).get("post_quant_val_bpb") if run_payload is not None else None,
        bytes_total=(run_payload or {}).get("objective", {}).get("bytes_total") if run_payload is not None else None,
    )
    if run_payload is not None:
        payload["objective"] = run_payload.get("objective", {})
    return payload


def run_mutation_cycle(
    *,
    cycle_index: int,
    tier: str,
    families_cfg: dict[str, Any],
    operator_registry: dict[str, Any],
    template_registry: dict[str, Any],
    family_stats: dict[str, Any],
    frontier: dict[str, Any],
    best_state: dict[str, Any],
    scheduler_cfg: dict[str, Any],
    search_space: dict[str, Any],
    rng: random.Random,
    forced_family: str | None,
    forced_template_id: str | None,
    dry_run: bool,
    disable_templates: bool,
    explicit_source: str | None,
) -> dict[str, Any]:
    with state_lock():
        frontier = ensure_frontier_shape(load_frontier_state())
        family_stats = ensure_family_stats_shape(load_family_stats_state())
        best_state = load_best_state()

        family_name = choose_family(families_cfg, family_stats, scheduler_cfg, rng, forced_family)
        family_cfg = families_cfg["families"][family_name]
        parent_candidate_id = choose_parent_candidate(family_name, family_cfg, frontier, best_state)
        source_entrypoint, parent_meta, effective_parent_candidate_id = candidate_source_path(
            parent_candidate_id,
            tier,
            explicit_source,
        )

        max_attempts = int(scheduler_cfg.get("mutation", {}).get("max_attempts_per_cycle", 8))
        plan = None
        attempt = 0
        while attempt < max_attempts:
            attempt += 1
            use_template = should_try_template(
                template_registry=template_registry,
                family_name=family_name,
                scheduler_cfg=scheduler_cfg,
                rng=rng,
                forced_template_id=forced_template_id,
                disable_templates=disable_templates,
            )
            if use_template:
                template_choice = choose_template(
                    template_registry,
                    family=family_name,
                    rng=rng,
                    forced_template_id=forced_template_id,
                )
                if template_choice is None:
                    if forced_template_id:
                        raise KeyError(f"template {forced_template_id!r} was requested but is unavailable")
                    use_template = False
                else:
                    template_id, template_def = template_choice
                    plan = build_template_plan(
                        template_id=template_id,
                        template_def=template_def,
                        baseline_env=scheduler_cfg.get("baseline_env", {}),
                        source_entrypoint=source_entrypoint,
                        parent_meta=parent_meta,
                        rng=rng,
                        operator_registry=operator_registry,
                        search_space=search_space,
                    )
            if not use_template:
                plan = build_mutation_plan(
                    family_name=family_name,
                    family_cfg=family_cfg,
                    operator_registry=operator_registry,
                    search_space=search_space,
                    baseline_env=scheduler_cfg.get("baseline_env", {}),
                    source_entrypoint=source_entrypoint,
                    parent_meta=parent_meta,
                    rng=rng,
                )
            if plan.duplicate is None:
                break
            print(
                f"[cycle {cycle_index}] action=mutate family={family_name} operator={plan.operator_id} "
                f"matched_candidate={plan.duplicate.get('candidate_id')} attempt={attempt}/{max_attempts}",
                flush=True,
            )
            if not dry_run:
                bump_duplicate_skip(family_stats, family_name)
                save_family_stats_state(family_stats)

        if plan is None:
            raise RuntimeError("mutation planning failed unexpectedly")
        if plan.duplicate is not None and attempt >= max_attempts:
            return {
                "cycle": cycle_index,
                "tier": tier,
                "action": "mutate",
                "family": family_name,
                "status": "skipped_duplicate",
                "duplicate": plan.duplicate,
                "attempts": attempt,
                "source_entrypoint": str(source_entrypoint),
            }

        candidate_id = f"{scheduler_cfg.get('candidate_prefix', 'port')}_{family_name}_{utc_stamp()}_{rng.randint(1000, 9999)}"
        note = (
            f"Scheduler cycle {cycle_index}: action=mutate family={family_name}, operator={plan.operator_id}, "
            f"parent={effective_parent_candidate_id}, env_overrides={plan.env_overrides_merged}"
        )
        payload = {
            "cycle": cycle_index,
            "tier": tier,
            "action": "mutate",
            "status": "planned" if dry_run else "created",
            "candidate_id": candidate_id,
            "candidate_dir": str(candidate_dir(candidate_id)),
            "family": family_name,
            "operator": plan.operator_id,
            "template_id": plan.template_id,
            "overrides": plan.env_overrides_merged,
            "overrides_delta": plan.env_overrides_delta,
            "parent_candidate_id": effective_parent_candidate_id,
            "source_entrypoint": str(source_entrypoint),
            "dry_run": dry_run,
            "attempts": attempt,
            "config_hash": plan.fingerprints["config_hash"],
            "mutation_hash": plan.fingerprints["mutation_hash"],
        }
        print(
            f"[cycle {cycle_index}] action=mutate family={family_name} operator={plan.operator_id} "
            f"template_id={plan.template_id} candidate={candidate_id} parent={effective_parent_candidate_id} "
            f"overrides={plan.env_overrides_merged} attempts={attempt}",
            flush=True,
        )
        if dry_run:
            return payload

        creation = apply_mutation_plan(
            candidate_id=candidate_id,
            plan=plan,
            parent_candidate_id=effective_parent_candidate_id,
            note=note,
        )
        payload["candidate_dir"] = creation["candidate_dir"]

        bump_scheduled_count(family_stats, family_name)
        save_family_stats_state(family_stats)

    return_code, run_payload = launch_candidate(candidate_id, tier)
    payload["return_code"] = return_code
    payload["run_found"] = run_payload is not None
    record_run_result(
        scheduler_cfg=scheduler_cfg,
        family_name=family_name,
        run_payload=run_payload,
    )
    if run_payload is not None:
        payload["objective"] = run_payload.get("objective", {})
    return payload


def run_replication_cycle(
    *,
    cycle_index: int,
    tier: str,
    families_cfg: dict[str, Any],
    family_stats: dict[str, Any],
    frontier: dict[str, Any],
    scheduler_cfg: dict[str, Any],
    rng: random.Random,
    forced_family: str | None,
    dry_run: bool,
) -> dict[str, Any]:
    with state_lock():
        frontier = ensure_frontier_shape(load_frontier_state())
        family_stats = ensure_family_stats_shape(load_family_stats_state())

        allowed = available_replication_families(frontier)
        family_name = choose_family(
            families_cfg,
            family_stats,
            scheduler_cfg,
            rng,
            forced_family,
            allowed_families=allowed if allowed else None,
        )
        source_candidate_id, summary, family_name, plan = build_replication_plan(frontier, family=family_name)
        candidate_id = f"rep_{source_candidate_id}_{utc_stamp()}_{rng.randint(1000, 9999)}"
        payload = {
            "cycle": cycle_index,
            "tier": tier,
            "action": "replicate",
            "status": "planned" if dry_run else "created",
            "candidate_id": candidate_id,
            "candidate_dir": str(candidate_dir(candidate_id)),
            "family": family_name,
            "operator": plan.operator_id,
            "template_id": plan.template_id,
            "parent_candidate_id": source_candidate_id,
            "source_run_id": (summary or {}).get("run_id"),
            "source_tier": (summary or {}).get("tier"),
            "dry_run": dry_run,
            "config_hash": plan.fingerprints["config_hash"],
            "mutation_hash": plan.fingerprints["mutation_hash"],
        }
        print(
            f"[cycle {cycle_index}] action=replicate family={family_name} candidate={candidate_id} "
            f"source={source_candidate_id} source_run={(summary or {}).get('run_id')}",
            flush=True,
        )
        if dry_run:
            return payload

        note = f"Scheduler cycle {cycle_index}: action=replicate source={source_candidate_id}"
        creation = apply_mutation_plan(
            candidate_id=candidate_id,
            plan=plan,
            parent_candidate_id=source_candidate_id,
            note=note,
        )
        payload["candidate_dir"] = creation["candidate_dir"]

        bump_scheduled_count(family_stats, family_name)
        save_family_stats_state(family_stats)

    launch_tier = tier or (summary or {}).get("tier") or default_tier(scheduler_cfg)
    return_code, run_payload = launch_candidate(candidate_id, launch_tier)
    payload["tier"] = launch_tier
    payload["return_code"] = return_code
    payload["run_found"] = run_payload is not None
    record_run_result(
        scheduler_cfg=scheduler_cfg,
        family_name=family_name,
        run_payload=run_payload,
    )
    if run_payload is not None:
        payload["objective"] = run_payload.get("objective", {})
    return payload


def run_recombine_cycle(
    *,
    cycle_index: int,
    tier: str,
    families_cfg: dict[str, Any],
    family_stats: dict[str, Any],
    frontier: dict[str, Any],
    scheduler_cfg: dict[str, Any],
    rng: random.Random,
    forced_family: str | None,
    dry_run: bool,
) -> dict[str, Any]:
    with state_lock():
        frontier = ensure_frontier_shape(load_frontier_state())
        family_stats = ensure_family_stats_shape(load_family_stats_state())

        allowed = available_recombine_families(frontier)
        preferred_family = choose_family(
            families_cfg,
            family_stats,
            scheduler_cfg,
            rng,
            forced_family,
            allowed_families=allowed if allowed else None,
        )
        primary_summary, secondary_summary, family_name, plan, conflicts = build_recombine_plan(
            frontier,
            preferred_family=preferred_family,
        )
        primary_candidate_id = str(primary_summary["candidate_id"])
        secondary_candidate_id = str(secondary_summary["candidate_id"])
        payload = {
            "cycle": cycle_index,
            "tier": tier,
            "action": "recombine",
            "family": family_name,
            "primary_candidate_id": primary_candidate_id,
            "secondary_candidate_id": secondary_candidate_id,
            "operator": plan.operator_id,
            "template_id": None,
            "conflicts": conflicts,
            "duplicate": plan.duplicate,
            "dry_run": dry_run,
            "status": "planned" if dry_run else "created",
            "config_hash": plan.fingerprints["config_hash"],
            "mutation_hash": plan.fingerprints["mutation_hash"],
        }
        if plan.duplicate is not None:
            print(
                f"[cycle {cycle_index}] action=recombine family={family_name} duplicate "
                f"matched_candidate={plan.duplicate.get('candidate_id')} primary={primary_candidate_id} "
                f"secondary={secondary_candidate_id}",
                flush=True,
            )
            if not dry_run:
                bump_duplicate_skip(family_stats, family_name)
                save_family_stats_state(family_stats)
            payload["status"] = "skipped_duplicate"
            return payload

        candidate_id = f"rec_{primary_candidate_id}__{secondary_candidate_id}_{utc_stamp()}_{rng.randint(1000, 9999)}"
        payload["candidate_id"] = candidate_id
        payload["candidate_dir"] = str(candidate_dir(candidate_id))
        print(
            f"[cycle {cycle_index}] action=recombine family={family_name} candidate={candidate_id} "
            f"primary={primary_candidate_id} secondary={secondary_candidate_id} conflicts={conflicts}",
            flush=True,
        )
        if dry_run:
            return payload

        note = (
            f"Scheduler cycle {cycle_index}: action=recombine primary={primary_candidate_id} "
            f"secondary={secondary_candidate_id}"
        )
        creation = apply_mutation_plan(
            candidate_id=candidate_id,
            plan=plan,
            parent_candidate_id=primary_candidate_id,
            note=note,
        )
        payload["candidate_dir"] = creation["candidate_dir"]

        bump_scheduled_count(family_stats, family_name)
        save_family_stats_state(family_stats)

    launch_tier = tier or primary_summary.get("tier") or default_tier(scheduler_cfg)
    return_code, run_payload = launch_candidate(candidate_id, launch_tier)
    payload["tier"] = launch_tier
    payload["return_code"] = return_code
    payload["run_found"] = run_payload is not None
    record_run_result(
        scheduler_cfg=scheduler_cfg,
        family_name=family_name,
        run_payload=run_payload,
    )
    if run_payload is not None:
        payload["objective"] = run_payload.get("objective", {})
    return payload


def run_one_cycle(
    *,
    cycle_index: int,
    tier: str,
    families_cfg: dict[str, Any],
    operator_registry: dict[str, Any],
    template_registry: dict[str, Any],
    family_stats: dict[str, Any],
    frontier: dict[str, Any],
    best_state: dict[str, Any],
    scheduler_cfg: dict[str, Any],
    codex_cfg: dict[str, Any],
    search_space: dict[str, Any],
    rng: random.Random,
    forced_family: str | None,
    forced_template_id: str | None,
    forced_action: str,
    dry_run: bool,
    disable_templates: bool,
    explicit_source: str | None,
) -> dict[str, Any]:
    with state_lock():
        action = choose_action(load_frontier_state(), scheduler_cfg, codex_cfg, rng, forced_action)
    if action == "propose":
        return run_proposal_cycle(
            cycle_index=cycle_index,
            tier=tier,
            families_cfg=families_cfg,
            scheduler_cfg=scheduler_cfg,
            codex_cfg=codex_cfg,
            rng=rng,
            forced_family=forced_family,
            dry_run=dry_run,
            explicit_source=explicit_source,
        )
    if action == "mutate":
        return run_mutation_cycle(
            cycle_index=cycle_index,
            tier=tier,
            families_cfg=families_cfg,
            operator_registry=operator_registry,
            template_registry=template_registry,
            family_stats={},
            frontier={},
            best_state={},
            scheduler_cfg=scheduler_cfg,
            search_space=search_space,
            rng=rng,
            forced_family=forced_family,
            forced_template_id=forced_template_id,
            dry_run=dry_run,
            disable_templates=disable_templates,
            explicit_source=explicit_source,
        )
    if action == "replicate":
        return run_replication_cycle(
            cycle_index=cycle_index,
            tier=tier,
            families_cfg=families_cfg,
            family_stats={},
            frontier={},
            scheduler_cfg=scheduler_cfg,
            rng=rng,
            forced_family=forced_family,
            dry_run=dry_run,
        )
    return run_recombine_cycle(
        cycle_index=cycle_index,
        tier=tier,
        families_cfg=families_cfg,
        family_stats={},
        frontier={},
        scheduler_cfg=scheduler_cfg,
        rng=rng,
        forced_family=forced_family,
        dry_run=dry_run,
    )


def main() -> int:
    args = build_parser().parse_args()
    ensure_layout()

    families_cfg = load_config("families.json")
    operator_registry = load_operator_registry(load_config("mutation_operators.json"))
    template_registry = load_template_registry(seed_if_missing=True)
    scheduler_cfg = load_config("scheduler.json")
    codex_cfg = load_codex_config()
    search_space = load_config("search_space.json")

    forced_family = args.family
    if args.template:
        template_def = dict(template_registry.get("templates", {})).get(args.template)
        if template_def is None:
            raise KeyError(f"unknown template {args.template!r}")
        template_family = str(template_def["family"])
        if forced_family is None:
            forced_family = template_family
        elif forced_family != template_family:
            raise ValueError(
                f"template {args.template!r} belongs to family {template_family!r}, not {forced_family!r}"
            )

    cycles = args.cycles or int(scheduler_cfg.get("default_cycles", 1))
    tier = args.tier or default_tier(scheduler_cfg)
    rng = random.Random(args.seed)

    plans: list[dict[str, Any]] = []
    for cycle_index in range(1, cycles + 1):
        plans.append(
            run_one_cycle(
                cycle_index=cycle_index,
                tier=tier,
                families_cfg=families_cfg,
                operator_registry=operator_registry,
                template_registry=template_registry,
                family_stats={},
                frontier={},
                best_state={},
                scheduler_cfg=scheduler_cfg,
                codex_cfg=codex_cfg,
                search_space=search_space,
                rng=rng,
                forced_family=forced_family,
                forced_template_id=args.template,
                forced_action=args.action,
                dry_run=args.dry_run,
                disable_templates=args.no_templates,
                explicit_source=args.source,
            )
        )

    for plan in plans:
        print(plan)
    print(f"completed_cycles={cycles} tier={tier} dry_run={args.dry_run} action={args.action}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
