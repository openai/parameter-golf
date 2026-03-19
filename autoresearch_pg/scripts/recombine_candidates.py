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

from autoresearch_pg.lib.frontier_actions import build_recombine_plan
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
    parser = argparse.ArgumentParser(description="Recombine two existing candidates into one merged candidate")
    parser.add_argument("--primary-candidate", help="Primary candidate id.")
    parser.add_argument("--primary-slot", help="Primary frontier slot.")
    parser.add_argument("--primary-family", help="Primary frontier family.")
    parser.add_argument("--secondary-candidate", help="Secondary candidate id.")
    parser.add_argument("--secondary-slot", help="Secondary frontier slot.")
    parser.add_argument("--secondary-family", help="Secondary frontier family.")
    parser.add_argument("--new-candidate", help="Optional new candidate id.")
    parser.add_argument("--tier", help="Tier to launch. Defaults from the primary source run, else scheduler default.")
    parser.add_argument("--launch", action="store_true", help="Launch the recombined candidate immediately.")
    parser.add_argument(
        "--conflict-policy",
        choices=["primary", "secondary", "error"],
        default="primary",
        help="How to resolve conflicting env override keys.",
    )
    parser.add_argument("--allow-duplicate", action="store_true", help="Create the candidate even if dedupe matches.")
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

    primary_summary, secondary_summary, primary_family, plan, conflicts = build_recombine_plan(
        frontier,
        primary_candidate=args.primary_candidate,
        primary_slot=args.primary_slot,
        primary_family=args.primary_family,
        secondary_candidate=args.secondary_candidate,
        secondary_slot=args.secondary_slot,
        secondary_family=args.secondary_family,
        conflict_policy=args.conflict_policy,
    )
    primary_candidate_id = str(primary_summary["candidate_id"])
    secondary_candidate_id = str(secondary_summary["candidate_id"])
    duplicate = plan.duplicate
    if duplicate is not None and not args.allow_duplicate:
        print(
            json.dumps(
                {"status": "duplicate", "duplicate": duplicate, "fingerprints": plan.fingerprints},
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    candidate_id = args.new_candidate or f"rec_{slugify(primary_candidate_id)}__{slugify(secondary_candidate_id)}_{utc_stamp()}"
    note = args.note or (
        f"Recombine primary={primary_candidate_id} ({primary_family}) "
        f"secondary={secondary_candidate_id} "
        f"conflict_policy={args.conflict_policy}"
    )
    creation = apply_mutation_plan(
        candidate_id=candidate_id,
        plan=plan,
        parent_candidate_id=primary_candidate_id,
        note=note,
    )

    payload: dict[str, Any] = {
        "status": "created",
        "candidate_id": candidate_id,
        "primary_candidate_id": primary_candidate_id,
        "secondary_candidate_id": secondary_candidate_id,
        "family": primary_family,
        "candidate_dir": creation["candidate_dir"],
        "tier": args.tier or primary_summary.get("tier") or default_tier(scheduler_cfg),
        "launch": args.launch,
        "conflicts": conflicts,
    }

    if args.launch:
        tier = payload["tier"]
        bump_scheduled_count(family_stats, primary_family)
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
                primary_family,
                run_payload,
                recent_limit=int(scheduler_cfg.get("frontier_limits", {}).get("recent_runs", 50)),
                champion_limit=int(scheduler_cfg.get("frontier_limits", {}).get("champion_set", 8)),
            )
            update_family_stats_state(family_stats, primary_family, run_payload, got_family_win, got_global_win)
            save_frontier_state(frontier)
            save_family_stats_state(family_stats)
            payload["objective"] = run_payload.get("objective", {})

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
