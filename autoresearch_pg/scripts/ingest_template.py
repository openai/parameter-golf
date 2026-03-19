#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from autoresearch_pg.lib.templates import load_template_registry, save_template_registry, seed_live_templates


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Seed, list, or ingest autoresearch_pg templates")
    parser.add_argument("--seed-defaults", action="store_true", help="Seed the live template registry from config/templates.json.")
    parser.add_argument("--force", action="store_true", help="Force overwrite when seeding defaults.")
    parser.add_argument("--list", action="store_true", help="List the current live templates.")
    parser.add_argument("--from-file", help="Path to a JSON file containing one template or a full templates registry.")
    parser.add_argument("--template-id", help="Template id to create or update.")
    parser.add_argument("--family", help="Primary family for the template.")
    parser.add_argument("--description", help="Short template description.")
    parser.add_argument("--template-type", choices=["env_template", "patch_template", "hybrid_template", "operator_template"])
    parser.add_argument("--operator-id", help="Operator id for operator_template templates.")
    parser.add_argument("--env", action="append", default=[], help="Env override in KEY=VALUE form. Repeatable.")
    parser.add_argument("--replacement", action="append", default=[], help="Patch replacement in FIND=>REPLACE form. Repeatable.")
    parser.add_argument("--reference", action="append", default=[], help="Source reference string. Repeatable.")
    parser.add_argument("--tag", action="append", default=[], help="Tag string. Repeatable.")
    parser.add_argument("--selection-weight", type=float, default=1.0, help="Selection weight used by the scheduler.")
    parser.add_argument("--disable", action="store_true", help="Store the template as disabled.")
    return parser


def parse_scalar(raw: str) -> Any:
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def parse_env_items(items: list[str]) -> dict[str, Any]:
    env: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"invalid --env item {item!r}; expected KEY=VALUE")
        key, value = item.split("=", 1)
        env[key] = parse_scalar(value)
    return env


def parse_replacements(items: list[str]) -> list[dict[str, str]]:
    replacements = []
    for item in items:
        if "=>" not in item:
            raise ValueError(f"invalid --replacement item {item!r}; expected FIND=>REPLACE")
        find, replace = item.split("=>", 1)
        replacements.append({"find": find, "replace": replace})
    return replacements


def load_file_payload(path: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "templates" in payload:
        return payload
    if "template_id" in payload:
        template_id = str(payload["template_id"])
        template = dict(payload)
        template.pop("template_id", None)
        return {"version": 1, "templates": {template_id: template}}
    raise ValueError(f"{path} is neither a registry payload nor a single-template payload")


def upsert_templates(registry: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    templates = dict(registry.get("templates", {}))
    for template_id, template in (incoming.get("templates") or {}).items():
        templates[template_id] = template
    registry["templates"] = templates
    return registry


def build_single_template(args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    if not args.template_id or not args.family or not args.template_type:
        raise ValueError("--template-id, --family, and --template-type are required when not using --from-file")

    mutation: dict[str, Any] = {"type": args.template_type}
    env_overrides = parse_env_items(args.env)
    replacements = parse_replacements(args.replacement)
    if env_overrides:
        mutation["env_overrides"] = env_overrides
    if replacements:
        mutation["replacements"] = replacements
    if args.operator_id:
        mutation["operator_id"] = args.operator_id
    if args.template_type == "operator_template" and not args.operator_id:
        raise ValueError("--operator-id is required for operator_template templates")

    template = {
        "enabled": not args.disable,
        "family": args.family,
        "selection_weight": args.selection_weight,
        "description": args.description or "",
        "source_refs": list(args.reference),
        "tags": list(args.tag),
        "mutation": mutation,
    }
    return args.template_id, template


def main() -> int:
    args = build_parser().parse_args()

    if args.seed_defaults:
        payload = seed_live_templates(force=args.force)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    registry = load_template_registry(seed_if_missing=True)
    if args.list and not args.from_file and not args.template_id:
        print(json.dumps(registry, indent=2, sort_keys=True))
        return 0

    if args.from_file:
        incoming = load_file_payload(args.from_file)
    else:
        template_id, template = build_single_template(args)
        incoming = {"version": 1, "templates": {template_id: template}}

    registry = upsert_templates(registry, incoming)
    save_template_registry(registry)
    print(json.dumps(registry, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
