from __future__ import annotations

from typing import Any

from autoresearch_pg.lib.dedupe import find_duplicate, mutation_fingerprints
from autoresearch_pg.lib.frontier import all_frontier_summaries, ensure_frontier_shape, resolve_frontier_entry
from autoresearch_pg.lib.mutations import MutationPlan
from autoresearch_pg.lib.workspace import candidate_dir


def select_replication_source(
    frontier: dict[str, Any],
    *,
    candidate_id: str | None = None,
    slot: str | None = None,
    family: str | None = None,
) -> tuple[str, dict[str, Any] | None]:
    frontier = ensure_frontier_shape(frontier)
    if candidate_id:
        summary = resolve_frontier_entry(frontier, candidate_id=candidate_id)
        return candidate_id, summary

    if slot:
        summary = resolve_frontier_entry(frontier, slot=slot, family=family)
        if summary is None:
            raise ValueError("no matching frontier entry found")
        return str(summary["candidate_id"]), summary

    if family:
        summary = _best_replication_summary_for_family(frontier, family)
        if summary is None:
            raise ValueError(f"no valid frontier entry found for family {family!r}")
        return str(summary["candidate_id"]), summary

    queue = list(frontier.get("replication_queue", []))
    if not queue:
        raise ValueError("replication_queue is empty; provide a candidate, slot, or family")
    source_candidate_id = str(queue[0]["candidate_id"])
    summary = resolve_frontier_entry(frontier, candidate_id=source_candidate_id)
    return source_candidate_id, summary


def build_replication_plan(
    frontier: dict[str, Any],
    *,
    candidate_id: str | None = None,
    slot: str | None = None,
    family: str | None = None,
) -> tuple[str, dict[str, Any] | None, str, MutationPlan]:
    source_candidate_id, summary = select_replication_source(
        frontier,
        candidate_id=candidate_id,
        slot=slot,
        family=family,
    )
    source_meta = _load_candidate_meta(source_candidate_id)
    entrypoint_filename = source_meta.get("entrypoint_filename", "train_gpt.py")
    source_entrypoint = candidate_dir(source_candidate_id) / entrypoint_filename
    entrypoint_text = source_entrypoint.read_text(encoding="utf-8")

    family_name = str(source_meta.get("primary_family") or (summary or {}).get("primary_family") or "training_recipe")
    env_overrides = dict(source_meta.get("env_overrides", {}))
    secondary_tags = sorted(set(list(source_meta.get("secondary_tags", [])) + ["replicate"]))
    mutation_payload = {
        "operator_type": "replicate",
        "replication_of": source_candidate_id,
        "source_run_id": (summary or {}).get("run_id"),
        "source_tier": (summary or {}).get("tier"),
    }
    fingerprints = mutation_fingerprints(
        entrypoint_filename=entrypoint_filename,
        entrypoint_text=entrypoint_text,
        env_overrides=env_overrides,
        primary_family=family_name,
        mutation_operator="replicate",
        mutation_payload=mutation_payload,
    )
    plan = MutationPlan(
        family_name=family_name,
        operator_id="replicate",
        operator_type="replicate",
        template_id=source_meta.get("template_id"),
        source_entrypoint=source_entrypoint,
        entrypoint_filename=entrypoint_filename,
        entrypoint_text=entrypoint_text,
        env_overrides_delta={},
        env_overrides_merged=env_overrides,
        mutation_payload=mutation_payload,
        fingerprints=fingerprints,
        duplicate=None,
        secondary_tags=secondary_tags,
    )
    return source_candidate_id, summary, family_name, plan


def select_recombine_pair(
    frontier: dict[str, Any],
    *,
    primary_candidate: str | None = None,
    primary_slot: str | None = None,
    primary_family: str | None = None,
    secondary_candidate: str | None = None,
    secondary_slot: str | None = None,
    secondary_family: str | None = None,
    preferred_family: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    frontier = ensure_frontier_shape(frontier)
    explicit = any(
        [
            primary_candidate,
            primary_slot,
            primary_family,
            secondary_candidate,
            secondary_slot,
            secondary_family,
        ]
    )
    if explicit:
        primary = _select_summary(frontier, candidate=primary_candidate, slot=primary_slot, family=primary_family)
        secondary = _select_summary(
            frontier,
            candidate=secondary_candidate,
            slot=secondary_slot,
            family=secondary_family,
        )
        return primary, secondary

    queue = list(frontier.get("recombine_queue", []))
    if not queue:
        raise ValueError("recombine_queue is empty; provide explicit selectors")
    if preferred_family:
        for item in queue:
            if preferred_family in list(item.get("families", [])):
                return _resolve_queue_pair(frontier, item)
    return _resolve_queue_pair(frontier, queue[0])


def merge_env(
    primary_env: dict[str, Any],
    secondary_env: dict[str, Any],
    *,
    conflict_policy: str,
) -> tuple[dict[str, Any], dict[str, list[Any]]]:
    merged = dict(primary_env)
    conflicts: dict[str, list[Any]] = {}
    for key, value in secondary_env.items():
        if key not in merged:
            merged[key] = value
            continue
        if merged[key] == value:
            continue
        conflicts[key] = [merged[key], value]
        if conflict_policy == "primary":
            continue
        if conflict_policy == "secondary":
            merged[key] = value
            continue
        raise ValueError(f"conflicting env override for {key}: primary={merged[key]!r} secondary={value!r}")
    return merged, conflicts


def build_recombine_plan(
    frontier: dict[str, Any],
    *,
    primary_candidate: str | None = None,
    primary_slot: str | None = None,
    primary_family: str | None = None,
    secondary_candidate: str | None = None,
    secondary_slot: str | None = None,
    secondary_family: str | None = None,
    preferred_family: str | None = None,
    conflict_policy: str = "primary",
) -> tuple[dict[str, Any], dict[str, Any], str, MutationPlan, dict[str, list[Any]]]:
    explicit = any(
        [
            primary_candidate,
            primary_slot,
            primary_family,
            secondary_candidate,
            secondary_slot,
            secondary_family,
        ]
    )
    if explicit:
        primary_summary, secondary_summary = select_recombine_pair(
            frontier,
            primary_candidate=primary_candidate,
            primary_slot=primary_slot,
            primary_family=primary_family,
            secondary_candidate=secondary_candidate,
            secondary_slot=secondary_slot,
            secondary_family=secondary_family,
            preferred_family=preferred_family,
        )
        return _build_recombine_plan_from_summaries(
            primary_summary,
            secondary_summary,
            conflict_policy=conflict_policy,
        )

    fallback: tuple[dict[str, Any], dict[str, Any], str, MutationPlan, dict[str, list[Any]]] | None = None
    for primary_summary, secondary_summary in _iter_queue_recombine_pairs(frontier, preferred_family=preferred_family):
        built = _build_recombine_plan_from_summaries(
            primary_summary,
            secondary_summary,
            conflict_policy=conflict_policy,
        )
        if built[3].duplicate is None:
            return built
        if fallback is None:
            fallback = built

    if fallback is not None:
        return fallback
    raise ValueError("recombine_queue is empty; provide explicit selectors")


def _load_candidate_meta(candidate_id: str) -> dict[str, Any]:
    meta_path = candidate_dir(candidate_id) / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"candidate metadata not found: {meta_path}")
    import json

    return json.loads(meta_path.read_text(encoding="utf-8"))


def _best_replication_summary_for_family(frontier: dict[str, Any], family: str) -> dict[str, Any] | None:
    summaries = [
        row
        for row in all_frontier_summaries(frontier)
        if row.get("valid") and row.get("primary_family") == family
    ]
    non_replicates = [row for row in summaries if row.get("mutation_operator") != "replicate"]
    candidates = non_replicates or summaries
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda row: float(row.get("proxy_score") if row.get("proxy_score") is not None else 1e9),
    )


def _build_recombine_plan_from_summaries(
    primary_summary: dict[str, Any],
    secondary_summary: dict[str, Any],
    *,
    conflict_policy: str,
) -> tuple[dict[str, Any], dict[str, Any], str, MutationPlan, dict[str, list[Any]]]:
    primary_candidate_id = str(primary_summary["candidate_id"])
    secondary_candidate_id = str(secondary_summary["candidate_id"])
    if primary_candidate_id == secondary_candidate_id:
        raise ValueError("primary and secondary candidates must be different")

    primary_meta = _load_candidate_meta(primary_candidate_id)
    secondary_meta = _load_candidate_meta(secondary_candidate_id)

    primary_entrypoint_filename = primary_meta.get("entrypoint_filename", "train_gpt.py")
    secondary_entrypoint_filename = secondary_meta.get("entrypoint_filename", "train_gpt.py")
    if primary_entrypoint_filename != secondary_entrypoint_filename:
        raise ValueError(
            f"entrypoint mismatch: primary={primary_entrypoint_filename!r} secondary={secondary_entrypoint_filename!r}"
        )

    primary_entrypoint = candidate_dir(primary_candidate_id) / primary_entrypoint_filename
    secondary_entrypoint = candidate_dir(secondary_candidate_id) / secondary_entrypoint_filename
    primary_text = primary_entrypoint.read_text(encoding="utf-8")
    secondary_text = secondary_entrypoint.read_text(encoding="utf-8")
    if primary_text != secondary_text:
        raise ValueError(
            "code recombination across divergent entrypoints is not supported in this first pass; "
            "recombine candidates that share the same entrypoint text."
        )

    merged_env, conflicts = merge_env(
        dict(primary_meta.get("env_overrides", {})),
        dict(secondary_meta.get("env_overrides", {})),
        conflict_policy=conflict_policy,
    )
    primary_family_name = str(primary_meta.get("primary_family") or primary_summary.get("primary_family") or "training_recipe")
    secondary_family_name = str(
        secondary_meta.get("primary_family") or secondary_summary.get("primary_family") or "training_recipe"
    )
    secondary_tags = sorted(
        set(
            list(primary_meta.get("secondary_tags", []))
            + list(secondary_meta.get("secondary_tags", []))
            + ["recombine", f"recombine:{secondary_family_name}"]
        )
    )
    mutation_payload = {
        "operator_type": "recombine",
        "primary_candidate_id": primary_candidate_id,
        "secondary_candidate_id": secondary_candidate_id,
        "primary_template_id": primary_meta.get("template_id"),
        "secondary_template_id": secondary_meta.get("template_id"),
        "primary_operator": primary_meta.get("mutation_operator"),
        "secondary_operator": secondary_meta.get("mutation_operator"),
        "primary_env_overrides": primary_meta.get("env_overrides", {}),
        "secondary_env_overrides": secondary_meta.get("env_overrides", {}),
        "env_overrides": merged_env,
        "conflict_policy": conflict_policy,
        "conflicts": conflicts,
    }
    fingerprints = mutation_fingerprints(
        entrypoint_filename=primary_entrypoint_filename,
        entrypoint_text=primary_text,
        env_overrides=merged_env,
        primary_family=primary_family_name,
        mutation_operator="recombine",
        mutation_payload=mutation_payload,
    )
    duplicate = find_duplicate(
        config_hash=fingerprints["config_hash"],
        mutation_hash=fingerprints["mutation_hash"],
        refresh=True,
    )

    plan = MutationPlan(
        family_name=primary_family_name,
        operator_id="recombine",
        operator_type="recombine",
        template_id=None,
        source_entrypoint=primary_entrypoint,
        entrypoint_filename=primary_entrypoint_filename,
        entrypoint_text=primary_text,
        env_overrides_delta=merged_env,
        env_overrides_merged=merged_env,
        mutation_payload=mutation_payload,
        fingerprints=fingerprints,
        duplicate=duplicate,
        secondary_tags=secondary_tags,
    )
    return primary_summary, secondary_summary, primary_family_name, plan, conflicts


def _select_summary(
    frontier: dict[str, Any],
    *,
    candidate: str | None,
    slot: str | None,
    family: str | None,
) -> dict[str, Any]:
    if candidate:
        summary = resolve_frontier_entry(frontier, candidate_id=candidate)
        if summary is not None:
            return summary
        return {"candidate_id": candidate}
    if slot or family:
        summary = resolve_frontier_entry(frontier, slot=slot, family=family)
        if summary is None:
            raise ValueError("no matching frontier entry found")
        return summary
    raise ValueError("candidate, slot, or family selector is required")


def _resolve_queue_pair(frontier: dict[str, Any], item: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    primary = resolve_frontier_entry(frontier, candidate_id=str(item["primary_candidate_id"]))
    secondary = resolve_frontier_entry(frontier, candidate_id=str(item["secondary_candidate_id"]))
    if primary is None or secondary is None:
        raise ValueError("queued recombination pair no longer resolves in frontier state")
    return primary, secondary


def _iter_queue_recombine_pairs(frontier: dict[str, Any], *, preferred_family: str | None) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    frontier = ensure_frontier_shape(frontier)
    queue = list(frontier.get("recombine_queue", []))
    if preferred_family:
        preferred = [item for item in queue if preferred_family in list(item.get("families", []))]
        other = [item for item in queue if item not in preferred]
        queue = preferred + other
    return [_resolve_queue_pair(frontier, item) for item in queue]
