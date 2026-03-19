from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from autoresearch_pg.lib.dedupe import find_duplicate, mutation_fingerprints, normalize_env_overrides, rebuild_dedupe_index
from autoresearch_pg.lib.workspace import bootstrap_candidate, candidate_dir, dump_json, load_json


@dataclass
class MutationPlan:
    family_name: str
    operator_id: str
    operator_type: str
    template_id: str | None
    source_entrypoint: Path
    entrypoint_filename: str
    entrypoint_text: str
    env_overrides_delta: dict[str, Any]
    env_overrides_merged: dict[str, Any]
    mutation_payload: dict[str, Any]
    fingerprints: dict[str, Any]
    duplicate: dict[str, Any] | None
    secondary_tags: list[str] = field(default_factory=list)


def load_operator_registry(operators_cfg: dict[str, Any]) -> dict[str, Any]:
    return dict(operators_cfg.get("operators", {}))


def resolve_family_operator_ids(family_cfg: dict[str, Any]) -> list[str]:
    if "operator_ids" in family_cfg:
        return list(family_cfg.get("operator_ids", []))
    # Backwards compatibility with inline Phase 1 config.
    return [str(item["id"]) for item in family_cfg.get("operators", [])]


def resolve_current_env(baseline_env: dict[str, Any], parent_meta: dict[str, Any]) -> dict[str, Any]:
    env = dict(baseline_env)
    env.update(parent_meta.get("env_overrides", {}))
    return env


def _constraint_ok(candidate_env: dict[str, Any], constraint: str) -> bool:
    if constraint == "model_dim_divisible_by_num_heads":
        return int(candidate_env["MODEL_DIM"]) % int(candidate_env["NUM_HEADS"]) == 0
    if constraint == "num_heads_divisible_by_num_kv_heads":
        return int(candidate_env["NUM_HEADS"]) % int(candidate_env["NUM_KV_HEADS"]) == 0
    raise ValueError(f"unsupported mutation constraint {constraint!r}")


def _search_space_values(search_space: dict[str, Any], parameter: str) -> list[Any]:
    values = ((search_space.get("parameters") or {}).get(parameter) or {}).get("values")
    if not values:
        raise KeyError(f"parameter {parameter!r} not found in config/search_space.json")
    return list(values)


def _choose_new_value(values: list[Any], current_value: Any, rng: random.Random) -> Any:
    candidates = [value for value in values if value != current_value]
    if not candidates:
        return current_value
    return rng.choice(candidates)


def _pick_env_choice(
    operator_def: dict[str, Any],
    search_space: dict[str, Any],
    current_env: dict[str, Any],
    rng: random.Random,
) -> dict[str, Any]:
    parameter = str(operator_def["parameter"])
    values = _search_space_values(search_space, parameter)

    if "constraints" in operator_def:
        filtered: list[Any] = []
        for value in values:
            candidate_env = dict(current_env)
            candidate_env[parameter] = value
            if all(_constraint_ok(candidate_env, constraint) for constraint in operator_def["constraints"]):
                filtered.append(value)
        values = filtered

    if not values:
        raise ValueError(f"operator {operator_def.get('id')} has no legal values for parameter {parameter}")

    return {parameter: _choose_new_value(values, current_env.get(parameter), rng)}


def _pick_env_bundle(
    operator_def: dict[str, Any],
    current_env: dict[str, Any],
    rng: random.Random,
) -> dict[str, Any]:
    parameters = list(operator_def.get("parameters", []))
    bundles = list(operator_def.get("bundles", []))
    candidates = []
    for bundle in bundles:
        if any(key not in bundle for key in parameters):
            raise ValueError(f"operator {operator_def.get('id')} bundle is missing one of {parameters}")
        if any(current_env.get(key) != bundle[key] for key in parameters):
            candidates.append(bundle)
    if not candidates:
        raise ValueError(f"operator {operator_def.get('id')} has no alternative bundles")
    return dict(rng.choice(candidates))


def _pick_constrained_bundle(
    operator_def: dict[str, Any],
    search_space: dict[str, Any],
    current_env: dict[str, Any],
    rng: random.Random,
) -> dict[str, Any]:
    parameters = list(operator_def.get("parameters", []))
    if parameters != ["MODEL_DIM", "NUM_HEADS", "NUM_KV_HEADS"]:
        raise ValueError(f"unsupported constrained bundle parameters {parameters}")

    dim_values = _search_space_values(search_space, "MODEL_DIM")
    head_values = _search_space_values(search_space, "NUM_HEADS")
    kv_values = _search_space_values(search_space, "NUM_KV_HEADS")
    current_bundle = tuple(current_env.get(name) for name in parameters)
    candidates: list[dict[str, Any]] = []
    for dim in dim_values:
        for heads in head_values:
            for kv_heads in kv_values:
                candidate_env = dict(current_env)
                candidate_env.update({"MODEL_DIM": dim, "NUM_HEADS": heads, "NUM_KV_HEADS": kv_heads})
                constraints = list(operator_def.get("constraints", []))
                if all(_constraint_ok(candidate_env, constraint) for constraint in constraints):
                    bundle = {
                        "MODEL_DIM": dim,
                        "NUM_HEADS": heads,
                        "NUM_KV_HEADS": kv_heads,
                    }
                    if tuple(bundle[name] for name in parameters) != current_bundle:
                        candidates.append(bundle)
    if not candidates:
        raise ValueError(f"operator {operator_def.get('id')} has no legal constrained bundles")
    return dict(rng.choice(candidates))


def _apply_search_replace_patch(operator_def: dict[str, Any], source_text: str) -> str:
    patched = source_text
    replacements = list(operator_def.get("replacements", []))
    changed = False
    for item in replacements:
        find = str(item["find"])
        replace = str(item["replace"])
        if find not in patched:
            raise ValueError(f"operator {operator_def.get('id')} could not find patch target {find!r}")
        patched = patched.replace(find, replace)
        changed = True
    if not changed:
        raise ValueError(f"operator {operator_def.get('id')} did not modify the source entrypoint")
    return patched


def build_mutation_plan(
    *,
    family_name: str,
    family_cfg: dict[str, Any],
    operator_registry: dict[str, Any],
    search_space: dict[str, Any],
    baseline_env: dict[str, Any],
    source_entrypoint: Path,
    parent_meta: dict[str, Any],
    rng: random.Random,
    forced_operator_id: str | None = None,
) -> MutationPlan:
    operator_ids = resolve_family_operator_ids(family_cfg)
    if not operator_ids:
        raise ValueError(f"family {family_name!r} has no operator ids configured")

    if forced_operator_id:
        if forced_operator_id not in operator_ids:
            raise ValueError(f"operator {forced_operator_id!r} is not registered for family {family_name!r}")
        operator_id = forced_operator_id
    else:
        operator_id = rng.choice(operator_ids)

    operator_def = dict(operator_registry[operator_id])
    operator_type = str(operator_def["type"])
    compatible_entrypoints = list(operator_def.get("compatible_entrypoints", []))
    if compatible_entrypoints and source_entrypoint.name not in compatible_entrypoints:
        raise ValueError(
            f"operator {operator_id!r} is incompatible with entrypoint {source_entrypoint.name!r}; "
            f"expected one of {compatible_entrypoints}"
        )

    current_env = resolve_current_env(baseline_env, parent_meta)
    source_text = source_entrypoint.read_text(encoding="utf-8")
    entrypoint_text = source_text
    env_delta: dict[str, Any] = {}

    if operator_type == "env_choice":
        env_delta = _pick_env_choice(operator_def, search_space, current_env, rng)
    elif operator_type == "env_bundle":
        env_delta = _pick_env_bundle(operator_def, current_env, rng)
    elif operator_type == "constrained_bundle":
        env_delta = _pick_constrained_bundle(operator_def, search_space, current_env, rng)
    elif operator_type == "search_replace_patch":
        entrypoint_text = _apply_search_replace_patch(operator_def, source_text)
    else:
        raise ValueError(f"unsupported operator type {operator_type!r}")

    merged_env = dict(parent_meta.get("env_overrides", {}))
    merged_env.update(env_delta)
    mutation_payload = {
        "operator_type": operator_type,
        "env_overrides": env_delta,
    }
    if operator_type == "search_replace_patch":
        mutation_payload["replacements"] = operator_def.get("replacements", [])

    fingerprints = mutation_fingerprints(
        entrypoint_filename=source_entrypoint.name,
        entrypoint_text=entrypoint_text,
        env_overrides=merged_env,
        primary_family=family_name,
        mutation_operator=operator_id,
        mutation_payload=mutation_payload,
    )
    duplicate = find_duplicate(
        config_hash=fingerprints["config_hash"],
        mutation_hash=fingerprints["mutation_hash"],
        refresh=True,
    )
    return MutationPlan(
        family_name=family_name,
        operator_id=operator_id,
        operator_type=operator_type,
        template_id=None,
        source_entrypoint=source_entrypoint,
        entrypoint_filename=source_entrypoint.name,
        entrypoint_text=entrypoint_text,
        env_overrides_delta=env_delta,
        env_overrides_merged=merged_env,
        mutation_payload=mutation_payload,
        fingerprints=fingerprints,
        duplicate=duplicate,
    )


def apply_mutation_plan(candidate_id: str, plan: MutationPlan, parent_candidate_id: str | None, note: str | None) -> dict[str, Any]:
    out_dir = bootstrap_candidate(
        candidate_id=candidate_id,
        source_train_gpt=plan.source_entrypoint,
        parent_candidate_id=parent_candidate_id,
        note=note,
    )

    entrypoint_path = out_dir / plan.entrypoint_filename
    entrypoint_path.write_text(plan.entrypoint_text, encoding="utf-8")

    meta_path = out_dir / "meta.json"
    notes_path = out_dir / "notes.md"
    meta = load_json(meta_path, default={})
    meta.update(
        {
            "primary_family": plan.family_name,
            "secondary_tags": list(plan.secondary_tags),
            "parent_candidate_id": parent_candidate_id,
            "mutation_operator": plan.operator_id,
            "template_id": plan.template_id,
            "mutation_payload": plan.mutation_payload,
            "env_overrides": plan.env_overrides_merged,
            "status": "planned",
            "promotion_history": [],
            "entrypoint_hash": plan.fingerprints["entrypoint_hash"],
            "env_overrides_hash": plan.fingerprints["env_overrides_hash"],
            "config_hash": plan.fingerprints["config_hash"],
            "mutation_hash": plan.fingerprints["mutation_hash"],
        }
    )
    dump_json(meta_path, meta)

    notes = notes_path.read_text(encoding="utf-8")
    notes += "\n".join(
        [
            "",
            "6. Scheduler Metadata",
            "",
            f"family: {plan.family_name}",
            f"operator: {plan.operator_id}",
            f"template_id: {plan.template_id}",
            f"parent_candidate_id: {parent_candidate_id}",
            f"env_overrides: {normalize_env_overrides(plan.env_overrides_merged)}",
            f"config_hash: {plan.fingerprints['config_hash']}",
            f"mutation_hash: {plan.fingerprints['mutation_hash']}",
            "",
        ]
    )
    notes_path.write_text(notes, encoding="utf-8")

    rebuild_dedupe_index()
    return {
        "status": "created",
        "candidate_id": candidate_id,
        "candidate_dir": str(out_dir),
        "operator_id": plan.operator_id,
        "env_overrides": plan.env_overrides_merged,
        "fingerprints": plan.fingerprints,
    }
