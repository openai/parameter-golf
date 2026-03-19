#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from autoresearch_pg.lib.frontier_actions import build_replication_plan
from autoresearch_pg.lib.frontier import (
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
from autoresearch_pg.lib.mutations import apply_mutation_plan
from autoresearch_pg.lib.workspace import best_run_for_candidate, config_root, load_json, slugify, utc_stamp


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replicate an existing candidate exactly and optionally launch a new run")
    parser.add_argument("--candidate", help="Explicit candidate id to replicate.")
    parser.add_argument("--slot", help="Frontier slot to replicate, for example global_best or best_low_quant_gap.")
    parser.add_argument("--family", help="Replicate the current best candidate in a family.")
    parser.add_argument("--new-candidate", help="Optional new candidate id. Defaults to rep_<source>_<stamp>.")
    parser.add_argument("--tier", help="Tier to launch. Defaults to the source candidate's best tier, else scheduler default.")
    parser.add_argument("--launch", action="store_true", help="Launch the replicated candidate immediately.")
    parser.add_argument("--note", help="Optional hypothesis note.")
    return parser


def load_scheduler_config() -> dict[str, Any]:
    return load_json(config_root() / "scheduler.json", default={})


def default_tier(scheduler_cfg: dict[str, Any]) -> str:
    system_key = f"{platform.system().lower()}_{platform.machine().lower()}"
    defaults = scheduler_cfg.get("tier_defaults", {})
    return defaults.get(system_key, defaults.get("default", "smoke_local"))

def main() -> int:
    args = build_parser().parse_args()
    frontier = ensure_frontier_shape(load_frontier_state())
    family_stats = ensure_family_stats_shape(load_family_stats_state())
    scheduler_cfg = load_scheduler_config()

    source_candidate_id, summary, family_name, plan = build_replication_plan(
        frontier,
        candidate_id=args.candidate,
        slot=args.slot,
        family=args.family,
    )

    candidate_id = args.new_candidate or f"rep_{slugify(source_candidate_id)}_{utc_stamp()}"
    note = args.note or (
        f"Replication of candidate {source_candidate_id} "
        f"from {plan.mutation_payload.get('source_tier')}."
    )
    creation = apply_mutation_plan(
        candidate_id=candidate_id,
        plan=plan,
        parent_candidate_id=source_candidate_id,
        note=note,
    )

    payload: dict[str, Any] = {
        "status": "created",
        "candidate_id": candidate_id,
        "source_candidate_id": source_candidate_id,
        "family": family_name,
        "candidate_dir": creation["candidate_dir"],
        "tier": args.tier or (summary or {}).get("tier") or default_tier(scheduler_cfg),
        "launch": args.launch,
    }

    if args.launch:
        tier = payload["tier"]
        bump_scheduled_count(family_stats, family_name)
        save_family_stats_state(family_stats)
        cmd = [
            sys.executable,
            str(SCRIPT_DIR / "run_candidate.py"),
            "--candidate",
            candidate_id,
            "--tier",
            tier,
        ]
        proc = subprocess.run(cmd, check=False)
        payload["return_code"] = proc.returncode
        run_payload = best_run_for_candidate(candidate_id, tier=tier, require_valid=False)
        payload["run_found"] = run_payload is not None
        if run_payload is not None:
            got_family_win, got_global_win = update_frontier_state(
                frontier,
                family_name,
                run_payload,
                recent_limit=int(scheduler_cfg.get("frontier_limits", {}).get("recent_runs", 50)),
                champion_limit=int(scheduler_cfg.get("frontier_limits", {}).get("champion_set", 8)),
            )
            update_family_stats_state(family_stats, family_name, run_payload, got_family_win, got_global_win)
            save_frontier_state(frontier)
            save_family_stats_state(family_stats)
            payload["objective"] = run_payload.get("objective", {})

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
