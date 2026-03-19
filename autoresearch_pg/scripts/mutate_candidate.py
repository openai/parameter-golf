#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import random
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from autoresearch_pg.lib.mutations import apply_mutation_plan, build_mutation_plan, load_operator_registry
from autoresearch_pg.lib.templates import build_template_plan, choose_template, load_template_registry
from autoresearch_pg.lib.workspace import candidate_dir, config_root, ensure_layout, load_json, repo_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a mutated autoresearch_pg candidate with dedupe checks")
    parser.add_argument("--candidate", help="Candidate id. Defaults to a generated mutation id.")
    parser.add_argument("--family", help="Primary family, for example training_recipe or architecture.")
    parser.add_argument("--template", help="Optional template id. If set, family is inferred unless explicitly provided.")
    parser.add_argument("--operator", help="Optional operator id to force instead of sampling from the family.")
    parser.add_argument("--parent", help="Optional parent candidate id.")
    parser.add_argument("--source", help="Optional explicit source entrypoint path.")
    parser.add_argument("--tier", help="Optional tier hint used to choose the default root entrypoint.")
    parser.add_argument("--seed", type=int, default=1337, help="RNG seed for operator sampling.")
    parser.add_argument("--note", help="Optional initial note to place in notes.md.")
    parser.add_argument("--allow-duplicate", action="store_true", help="Create the candidate even if dedupe matches.")
    parser.add_argument("--dry-run", action="store_true", help="Plan the mutation without creating a candidate.")
    return parser


def load_config(name: str) -> dict[str, Any]:
    return load_json(config_root() / name)


def default_tier(scheduler_cfg: dict[str, Any]) -> str:
    system_key = f"{platform.system().lower()}_{platform.machine().lower()}"
    defaults = scheduler_cfg.get("tier_defaults", {})
    return defaults.get(system_key, defaults.get("default", "smoke_local"))


def default_source_for_tier(tier: str) -> Path:
    if "mlx" in tier:
        return repo_root() / "train_gpt_mlx.py"
    return repo_root() / "train_gpt.py"


def resolve_source(args: argparse.Namespace, scheduler_cfg: dict[str, Any]) -> tuple[Path, dict[str, Any], str | None]:
    if args.source:
        return Path(args.source).resolve(), {}, None
    if args.parent:
        parent_meta = load_json(candidate_dir(args.parent) / "meta.json", default={})
        entrypoint_filename = parent_meta.get("entrypoint_filename", "train_gpt.py")
        return candidate_dir(args.parent) / entrypoint_filename, parent_meta, args.parent
    tier = args.tier or default_tier(scheduler_cfg)
    return default_source_for_tier(tier), {}, None


def main() -> int:
    args = build_parser().parse_args()
    ensure_layout()

    families_cfg = load_config("families.json")
    operators_cfg = load_config("mutation_operators.json")
    search_space = load_config("search_space.json")
    scheduler_cfg = load_config("scheduler.json")
    template_registry = load_template_registry(seed_if_missing=True)

    families = families_cfg.get("families", {})
    template_choice = None
    family_name = args.family
    if args.template:
        template_entry = dict(template_registry.get("templates", {})).get(args.template)
        if template_entry is None:
            raise KeyError(f"unknown template {args.template!r}")
        template_family = str(template_entry["family"])
        template_choice = choose_template(
            template_registry,
            family=template_family,
            rng=random.Random(args.seed),
            forced_template_id=args.template,
        )
        family_name = family_name or template_family

    if not family_name:
        raise ValueError("either --family or --template must be provided")
    if family_name not in families:
        raise KeyError(f"unknown family {family_name!r}")
    family_cfg = families[family_name]

    source_entrypoint, parent_meta, effective_parent = resolve_source(args, scheduler_cfg)
    rng = random.Random(args.seed)
    operator_registry = load_operator_registry(operators_cfg)
    if template_choice is not None:
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
    else:
        plan = build_mutation_plan(
            family_name=family_name,
            family_cfg=family_cfg,
            operator_registry=operator_registry,
            search_space=search_space,
            baseline_env=scheduler_cfg.get("baseline_env", {}),
            source_entrypoint=source_entrypoint,
            parent_meta=parent_meta,
            rng=rng,
            forced_operator_id=args.operator,
        )

    payload = {
        "status": "planned",
        "family": family_name,
        "operator_id": plan.operator_id,
        "operator_type": plan.operator_type,
        "template_id": plan.template_id,
        "parent_candidate_id": effective_parent,
        "source_entrypoint": str(source_entrypoint),
        "env_overrides_delta": plan.env_overrides_delta,
        "env_overrides_merged": plan.env_overrides_merged,
        "fingerprints": plan.fingerprints,
        "duplicate": plan.duplicate,
    }
    if args.dry_run:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    if plan.duplicate is not None and not args.allow_duplicate:
        payload["status"] = "duplicate"
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    candidate_id = args.candidate or f"mut_{family_name}_{plan.operator_id}_{args.seed}"
    note = args.note or (
        f"Mutation created by mutate_candidate.py family={family_name} operator={plan.operator_id} "
        f"parent={effective_parent} env_overrides={plan.env_overrides_merged}"
    )
    result = apply_mutation_plan(
        candidate_id=candidate_id,
        plan=plan,
        parent_candidate_id=effective_parent,
        note=note,
    )
    payload.update(result)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
