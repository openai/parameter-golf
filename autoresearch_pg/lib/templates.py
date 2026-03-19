from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from autoresearch_pg.lib.dedupe import find_duplicate, mutation_fingerprints
from autoresearch_pg.lib.mutations import MutationPlan, build_mutation_plan
from autoresearch_pg.lib.workspace import config_root, dump_json, iso_utc, load_json, state_root


def default_templates_path() -> Path:
    return config_root() / "templates.json"


def live_templates_path() -> Path:
    return state_root() / "templates.json"


def seed_live_templates(force: bool = False) -> dict[str, Any]:
    live_path = live_templates_path()
    if live_path.is_file() and not force:
        return load_json(live_path, default={"version": 1, "last_updated": None, "templates": {}})

    seeded = load_json(default_templates_path(), default={"version": 1, "templates": {}})
    seeded["last_updated"] = iso_utc()
    dump_json(live_path, seeded)
    return seeded


def load_template_registry(seed_if_missing: bool = True) -> dict[str, Any]:
    live_path = live_templates_path()
    if live_path.is_file():
        return load_json(live_path, default={"version": 1, "last_updated": None, "templates": {}})
    if seed_if_missing:
        return seed_live_templates(force=False)
    return load_json(default_templates_path(), default={"version": 1, "templates": {}})


def save_template_registry(payload: dict[str, Any]) -> None:
    payload["last_updated"] = iso_utc()
    dump_json(live_templates_path(), payload)


def list_templates(
    registry: dict[str, Any],
    *,
    family: str | None = None,
    enabled_only: bool = True,
) -> list[tuple[str, dict[str, Any]]]:
    templates = []
    for template_id, template in sorted((registry.get("templates") or {}).items()):
        if family is not None and template.get("family") != family:
            continue
        if enabled_only and not template.get("enabled", True):
            continue
        templates.append((template_id, template))
    return templates


def choose_template(
    registry: dict[str, Any],
    *,
    family: str,
    rng: random.Random,
    forced_template_id: str | None = None,
) -> tuple[str, dict[str, Any]] | None:
    templates = dict(registry.get("templates") or {})
    if forced_template_id:
        template = templates.get(forced_template_id)
        if template is None:
            raise KeyError(f"unknown template {forced_template_id!r}")
        if template.get("family") != family:
            raise ValueError(
                f"template {forced_template_id!r} belongs to family {template.get('family')!r}, not {family!r}"
            )
        if not template.get("enabled", True):
            raise ValueError(f"template {forced_template_id!r} is disabled")
        return forced_template_id, template

    options = list_templates(registry, family=family, enabled_only=True)
    if not options:
        return None

    total = sum(max(float(template.get("selection_weight", 1.0)), 0.0) for _, template in options)
    if total <= 0:
        return options[0]
    target = rng.uniform(0.0, total)
    running = 0.0
    for template_id, template in options:
        running += max(float(template.get("selection_weight", 1.0)), 0.0)
        if running >= target:
            return template_id, template
    return options[-1]


def _apply_replacements(source_text: str, replacements: list[dict[str, Any]], template_id: str) -> str:
    patched = source_text
    changed = False
    for item in replacements:
        find = str(item["find"])
        replace = str(item["replace"])
        if find not in patched:
            raise ValueError(f"template {template_id!r} could not find patch target {find!r}")
        patched = patched.replace(find, replace)
        changed = True
    if not changed:
        raise ValueError(f"template {template_id!r} did not modify the source entrypoint")
    return patched


def build_template_plan(
    *,
    template_id: str,
    template_def: dict[str, Any],
    baseline_env: dict[str, Any],
    source_entrypoint: Path,
    parent_meta: dict[str, Any],
    rng: random.Random,
    operator_registry: dict[str, Any],
    search_space: dict[str, Any],
) -> MutationPlan:
    family_name = str(template_def["family"])
    mutation = dict(template_def.get("mutation", {}))
    template_type = str(mutation.get("type", "env_template"))
    tags = list(template_def.get("tags", []))

    compatible_entrypoints = list(mutation.get("compatible_entrypoints", []))
    if compatible_entrypoints and source_entrypoint.name not in compatible_entrypoints:
        raise ValueError(
            f"template {template_id!r} is incompatible with entrypoint {source_entrypoint.name!r}; "
            f"expected one of {compatible_entrypoints}"
        )

    if template_type == "operator_template":
        operator_id = str(mutation["operator_id"])
        family_cfg = {"operator_ids": [operator_id]}
        base_plan = build_mutation_plan(
            family_name=family_name,
            family_cfg=family_cfg,
            operator_registry=operator_registry,
            search_space=search_space,
            baseline_env=baseline_env,
            source_entrypoint=source_entrypoint,
            parent_meta=parent_meta,
            rng=rng,
            forced_operator_id=operator_id,
        )
        source_refs = list(template_def.get("source_refs", []))
        payload = dict(base_plan.mutation_payload)
        payload.update(
            {
                "template_id": template_id,
                "template_type": template_type,
                "base_operator_id": operator_id,
                "source_refs": source_refs,
                "description": template_def.get("description"),
            }
        )
        fingerprints = mutation_fingerprints(
            entrypoint_filename=base_plan.entrypoint_filename,
            entrypoint_text=base_plan.entrypoint_text,
            env_overrides=base_plan.env_overrides_merged,
            primary_family=family_name,
            mutation_operator=template_id,
            mutation_payload=payload,
        )
        duplicate = find_duplicate(
            config_hash=fingerprints["config_hash"],
            mutation_hash=fingerprints["mutation_hash"],
            refresh=True,
        )
        return MutationPlan(
            family_name=family_name,
            operator_id=template_id,
            operator_type="template",
            template_id=template_id,
            source_entrypoint=base_plan.source_entrypoint,
            entrypoint_filename=base_plan.entrypoint_filename,
            entrypoint_text=base_plan.entrypoint_text,
            env_overrides_delta=base_plan.env_overrides_delta,
            env_overrides_merged=base_plan.env_overrides_merged,
            mutation_payload=payload,
            fingerprints=fingerprints,
            duplicate=duplicate,
            secondary_tags=sorted(set(tags + ["template"])),
        )

    source_text = source_entrypoint.read_text(encoding="utf-8")
    entrypoint_text = source_text
    env_delta = dict(mutation.get("env_overrides", {}))
    if template_type in {"patch_template", "hybrid_template"}:
        entrypoint_text = _apply_replacements(source_text, list(mutation.get("replacements", [])), template_id)
    elif template_type != "env_template":
        raise ValueError(f"unsupported template type {template_type!r}")

    merged_env = dict(parent_meta.get("env_overrides", {}))
    merged_env.update(env_delta)
    payload = {
        "operator_type": "template",
        "template_id": template_id,
        "template_type": template_type,
        "env_overrides": env_delta,
        "replacements": list(mutation.get("replacements", [])),
        "source_refs": list(template_def.get("source_refs", [])),
        "description": template_def.get("description"),
    }
    fingerprints = mutation_fingerprints(
        entrypoint_filename=source_entrypoint.name,
        entrypoint_text=entrypoint_text,
        env_overrides=merged_env,
        primary_family=family_name,
        mutation_operator=template_id,
        mutation_payload=payload,
    )
    duplicate = find_duplicate(
        config_hash=fingerprints["config_hash"],
        mutation_hash=fingerprints["mutation_hash"],
        refresh=True,
    )
    return MutationPlan(
        family_name=family_name,
        operator_id=template_id,
        operator_type="template",
        template_id=template_id,
        source_entrypoint=source_entrypoint,
        entrypoint_filename=source_entrypoint.name,
        entrypoint_text=entrypoint_text,
        env_overrides_delta=env_delta,
        env_overrides_merged=merged_env,
        mutation_payload=payload,
        fingerprints=fingerprints,
        duplicate=duplicate,
        secondary_tags=sorted(set(tags + ["template"])),
    )
