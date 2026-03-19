from __future__ import annotations

from itertools import combinations
from typing import Any

from autoresearch_pg.lib.workspace import dump_json, iso_utc, load_json, state_root


KNOWN_FAMILIES = [
    "training_recipe",
    "architecture",
    "compression",
    "tokenizer",
    "eval_time",
    "systems",
]


def default_frontier_state() -> dict[str, Any]:
    return {
        "global_best": None,
        "best_by_family": {family: None for family in KNOWN_FAMILIES},
        "best_low_quant_gap": None,
        "best_with_byte_headroom": None,
        "champion_set": [],
        "recent_runs": [],
        "replication_queue": [],
        "recombine_queue": [],
        "last_updated": None,
    }


def default_family_stats_state() -> dict[str, Any]:
    return {
        "families": {
            family: {
                "scheduled": 0,
                "completed": 0,
                "duplicates_skipped": 0,
                "valid": 0,
                "invalid": 0,
                "wins": 0,
                "best_score": None,
                "last_score": None,
                "recent_scores": [],
            }
            for family in KNOWN_FAMILIES
        },
        "last_updated": None,
    }


def ensure_frontier_shape(frontier: dict[str, Any]) -> dict[str, Any]:
    payload = default_frontier_state()
    payload.update(frontier)
    best_by_family = dict(payload.get("best_by_family", {}))
    for family in KNOWN_FAMILIES:
        best_by_family.setdefault(family, None)
    payload["best_by_family"] = best_by_family
    payload.setdefault("champion_set", [])
    payload.setdefault("recent_runs", [])
    payload.setdefault("replication_queue", [])
    payload.setdefault("recombine_queue", [])
    payload.setdefault("last_updated", None)
    frontier.clear()
    frontier.update(payload)
    return frontier


def ensure_family_stats_shape(family_stats: dict[str, Any]) -> dict[str, Any]:
    payload = default_family_stats_state()
    families = dict(payload.get("families", {}))
    for family, stats in (family_stats.get("families") or {}).items():
        merged = dict(families.get(family, {}))
        merged.update(stats)
        merged.setdefault("duplicates_skipped", 0)
        families[family] = merged
    payload["families"] = families
    payload["last_updated"] = family_stats.get("last_updated")
    family_stats.clear()
    family_stats.update(payload)
    return family_stats


def load_frontier_state() -> dict[str, Any]:
    frontier = ensure_frontier_shape(load_json(state_root() / "frontier.json", default=default_frontier_state()))
    refresh_frontier_queues(frontier)
    return frontier


def save_frontier_state(frontier: dict[str, Any]) -> None:
    payload = ensure_frontier_shape(frontier)
    refresh_frontier_queues(payload)
    dump_json(state_root() / "frontier.json", payload)


def load_family_stats_state() -> dict[str, Any]:
    return ensure_family_stats_shape(load_json(state_root() / "family_stats.json", default=default_family_stats_state()))


def save_family_stats_state(family_stats: dict[str, Any]) -> None:
    dump_json(state_root() / "family_stats.json", ensure_family_stats_shape(family_stats))


def summarize_run(run_payload: dict[str, Any]) -> dict[str, Any]:
    objective = run_payload.get("objective", {})
    return {
        "candidate_id": run_payload.get("candidate_id"),
        "primary_family": run_payload.get("primary_family"),
        "tier": run_payload.get("tier"),
        "run_id": run_payload.get("run_id"),
        "run_dir": run_payload.get("run_dir"),
        "valid": objective.get("valid"),
        "proxy_score": objective.get("proxy_score"),
        "post_quant_val_bpb": objective.get("post_quant_val_bpb"),
        "pre_quant_val_bpb": objective.get("pre_quant_val_bpb"),
        "quant_gap_bpb": objective.get("quant_gap_bpb"),
        "bytes_total": objective.get("bytes_total"),
        "template_id": run_payload.get("template_id"),
        "mutation_operator": run_payload.get("mutation_operator"),
        "mutation_payload": run_payload.get("mutation_payload"),
        "parent_candidate_id": run_payload.get("parent_candidate_id"),
    }


def better_run(lhs: dict[str, Any] | None, rhs: dict[str, Any] | None, field: str = "proxy_score") -> bool:
    if rhs is None:
        return False
    if lhs is None:
        return True
    lhs_value = lhs.get(field)
    rhs_value = rhs.get(field)
    if rhs_value is None:
        return False
    if lhs_value is None:
        return True
    return float(rhs_value) < float(lhs_value)


def more_headroom(lhs: dict[str, Any] | None, rhs: dict[str, Any] | None, byte_cap: int = 16_000_000) -> bool:
    if rhs is None or not rhs.get("valid"):
        return False
    rhs_bytes = rhs.get("bytes_total")
    if rhs_bytes is None:
        return False
    rhs_headroom = byte_cap - int(rhs_bytes)
    if lhs is None:
        return True
    lhs_bytes = lhs.get("bytes_total")
    if lhs_bytes is None:
        return True
    lhs_headroom = byte_cap - int(lhs_bytes)
    if rhs_headroom == lhs_headroom:
        return better_run(lhs, rhs)
    return rhs_headroom > lhs_headroom


def all_frontier_summaries(frontier: dict[str, Any]) -> list[dict[str, Any]]:
    frontier = ensure_frontier_shape(frontier)
    seen: set[tuple[str, str]] = set()
    rows: list[dict[str, Any]] = []
    for item in [
        frontier.get("global_best"),
        *(frontier.get("best_by_family") or {}).values(),
        frontier.get("best_low_quant_gap"),
        frontier.get("best_with_byte_headroom"),
        *list(frontier.get("champion_set", [])),
        *list(frontier.get("recent_runs", [])),
    ]:
        if not item:
            continue
        key = (str(item.get("candidate_id")), str(item.get("run_id")))
        if key in seen:
            continue
        seen.add(key)
        rows.append(item)
    return rows


def recompute_champion_set(frontier: dict[str, Any], limit: int) -> None:
    frontier = ensure_frontier_shape(frontier)
    seen: set[tuple[str, str]] = set()
    champions: list[dict[str, Any]] = []
    for item in [
        frontier.get("global_best"),
        *(frontier.get("best_by_family") or {}).values(),
        frontier.get("best_low_quant_gap"),
        frontier.get("best_with_byte_headroom"),
    ]:
        if not item:
            continue
        key = (str(item.get("candidate_id")), str(item.get("run_id")))
        if key in seen:
            continue
        seen.add(key)
        champions.append(item)
    frontier["champion_set"] = champions[:limit]


def refresh_frontier_queues(frontier: dict[str, Any]) -> None:
    frontier = ensure_frontier_shape(frontier)
    summaries = [item for item in all_frontier_summaries(frontier) if item.get("valid")]
    summaries.sort(key=lambda row: float(row.get("proxy_score") if row.get("proxy_score") is not None else 1e9))

    replication_candidates = [
        row
        for row in summaries
        if row.get("primary_family") is not None and row.get("mutation_operator") != "replicate"
    ]
    if not replication_candidates:
        replication_candidates = [row for row in summaries if row.get("primary_family") is not None]

    replication_queue = []
    for row in replication_candidates[:5]:
        replication_queue.append(
            {
                "candidate_id": row.get("candidate_id"),
                "primary_family": row.get("primary_family"),
                "priority": row.get("proxy_score"),
                "reason": "frontier_valid",
                "template_id": row.get("template_id"),
            }
        )

    family_best = []
    for family in KNOWN_FAMILIES:
        family_rows = [
            row
            for row in summaries
            if row.get("primary_family") == family and row.get("primary_family") is not None
        ]
        if not family_rows:
            continue
        non_replicates = [row for row in family_rows if row.get("mutation_operator") != "replicate"]
        family_best.append((non_replicates or family_rows)[0])
    recombine_queue = []
    seen_pairs: set[tuple[str, str]] = set()
    for left, right in combinations(family_best, 2):
        if left.get("primary_family") == right.get("primary_family"):
            continue
        ordered_ids = tuple(sorted([str(left.get("candidate_id")), str(right.get("candidate_id"))]))
        if ordered_ids in seen_pairs:
            continue
        seen_pairs.add(ordered_ids)
        score = float(left.get("proxy_score", 1e9)) + float(right.get("proxy_score", 1e9))
        recombine_queue.append(
            {
                "primary_candidate_id": left.get("candidate_id"),
                "secondary_candidate_id": right.get("candidate_id"),
                "families": [left.get("primary_family"), right.get("primary_family")],
                "priority": score,
                "reason": "cross_family_frontier",
            }
        )
    recombine_queue.sort(key=lambda row: row["priority"])
    frontier["replication_queue"] = replication_queue[:5]
    frontier["recombine_queue"] = recombine_queue[:5]


def update_frontier_state(
    frontier: dict[str, Any],
    family_name: str,
    run_payload: dict[str, Any],
    *,
    recent_limit: int = 50,
    champion_limit: int = 8,
) -> tuple[bool, bool]:
    frontier = ensure_frontier_shape(frontier)
    summary = summarize_run(run_payload)
    got_family_win = False
    got_global_win = False

    if summary.get("valid"):
        if better_run(frontier.get("global_best"), summary):
            frontier["global_best"] = summary
            got_global_win = True
        family_best = (frontier.get("best_by_family") or {}).get(family_name)
        if better_run(family_best, summary):
            frontier.setdefault("best_by_family", {})[family_name] = summary
            got_family_win = True

        current_gap = frontier.get("best_low_quant_gap")
        if summary.get("quant_gap_bpb") is not None:
            current_gap_score = abs(float(current_gap.get("quant_gap_bpb", 1e9))) if current_gap else None
            summary_gap_score = abs(float(summary["quant_gap_bpb"]))
            if current_gap is None or summary_gap_score < float(current_gap_score):
                frontier["best_low_quant_gap"] = summary

        if more_headroom(frontier.get("best_with_byte_headroom"), summary):
            frontier["best_with_byte_headroom"] = summary

    recent_runs = list(frontier.get("recent_runs", []))
    recent_runs.insert(0, summary)
    frontier["recent_runs"] = recent_runs[:recent_limit]
    frontier["last_updated"] = iso_utc()
    recompute_champion_set(frontier, champion_limit)
    refresh_frontier_queues(frontier)
    return got_family_win, got_global_win


def update_family_stats_state(
    family_stats: dict[str, Any],
    family_name: str,
    run_payload: dict[str, Any] | None,
    got_family_win: bool,
    got_global_win: bool,
) -> None:
    family_stats = ensure_family_stats_shape(family_stats)
    stats = family_stats.setdefault("families", {}).setdefault(family_name, default_family_stats_state()["families"][family_name])
    stats["completed"] = int(stats.get("completed", 0)) + 1

    if run_payload is None:
        stats["invalid"] = int(stats.get("invalid", 0)) + 1
        family_stats["last_updated"] = iso_utc()
        return

    objective = run_payload.get("objective", {})
    score = objective.get("proxy_score")
    valid = bool(objective.get("valid"))
    stats["last_score"] = score
    recent_scores = list(stats.get("recent_scores", []))
    recent_scores.append(score)
    stats["recent_scores"] = recent_scores[-10:]

    if valid:
        stats["valid"] = int(stats.get("valid", 0)) + 1
        best_score = stats.get("best_score")
        if best_score is None or (score is not None and float(score) < float(best_score)):
            stats["best_score"] = score
    else:
        stats["invalid"] = int(stats.get("invalid", 0)) + 1

    if got_family_win or got_global_win:
        stats["wins"] = int(stats.get("wins", 0)) + 1
    family_stats["last_updated"] = iso_utc()


def bump_duplicate_skip(family_stats: dict[str, Any], family_name: str) -> None:
    family_stats = ensure_family_stats_shape(family_stats)
    stats = family_stats.setdefault("families", {}).setdefault(family_name, default_family_stats_state()["families"][family_name])
    stats["duplicates_skipped"] = int(stats.get("duplicates_skipped", 0)) + 1
    family_stats["last_updated"] = iso_utc()


def bump_scheduled_count(family_stats: dict[str, Any], family_name: str) -> None:
    family_stats = ensure_family_stats_shape(family_stats)
    stats = family_stats.setdefault("families", {}).setdefault(family_name, default_family_stats_state()["families"][family_name])
    stats["scheduled"] = int(stats.get("scheduled", 0)) + 1
    family_stats["last_updated"] = iso_utc()


def resolve_frontier_entry(
    frontier: dict[str, Any],
    *,
    candidate_id: str | None = None,
    slot: str | None = None,
    family: str | None = None,
) -> dict[str, Any] | None:
    frontier = ensure_frontier_shape(frontier)
    if candidate_id:
        for item in all_frontier_summaries(frontier):
            if item.get("candidate_id") == candidate_id:
                return item
        return None
    if slot:
        if slot == "global_best":
            return frontier.get("global_best")
        if slot == "best_low_quant_gap":
            return frontier.get("best_low_quant_gap")
        if slot == "best_with_byte_headroom":
            return frontier.get("best_with_byte_headroom")
        raise KeyError(f"unknown frontier slot {slot!r}")
    if family:
        return (frontier.get("best_by_family") or {}).get(family)
    return None
