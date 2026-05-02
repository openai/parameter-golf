from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np


W9_RELATIVE_PATH = (
    "records/track_non_record_16mb/"
    "2026-05-01_psyarbor_lab_rebus_calculations_w9/train_gpt.py"
)
DEFAULT_COMMUNITY_FINDER_PATH = Path("C:/Users/jjor3/Dev/Community_Finder")
DEFAULT_REDESIGNED_FISHSTICK_PATH = Path("C:/Users/jjor3/Dev/redesigned-fishstick")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Offline W9 community-selection oracle. It compares W9's self-contained "
            "community assignment against Community_Finder's centered spectral + greedy "
            "k-densest helper path. It does not import service/FastAPI workers."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--w9-path", type=Path, default=repo_root / W9_RELATIVE_PATH)
    parser.add_argument("--input", nargs="*", type=Path, default=[])
    parser.add_argument("--input-root", type=Path, default=repo_root / "runs/w9_training_path")
    parser.add_argument("--pattern", default="*.community_snapshot.json")
    parser.add_argument("--community-finder-path", type=Path, default=DEFAULT_COMMUNITY_FINDER_PATH)
    parser.add_argument("--redesigned-fishstick-path", type=Path, default=DEFAULT_REDESIGNED_FISHSTICK_PATH)
    parser.add_argument("--output-root", type=Path, default=repo_root / "runs/w9_community_oracle")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--greedy-rounds", type=int, default=None)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=json_default), encoding="utf-8")


def json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return str(obj)


def import_w9_module(w9_path: Path):
    if not w9_path.exists():
        raise FileNotFoundError(f"Missing W9 train_gpt.py: {w9_path}")
    module_name = f"w9_community_oracle_{abs(hash(str(w9_path.resolve()))) & 0xFFFFFFFF:x}"
    spec = importlib.util.spec_from_file_location(module_name, w9_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import W9 module from {w9_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def import_community_finder(path: Path) -> tuple[Callable[..., Any], Callable[..., Any], dict[str, Any]]:
    pc_algos_path = path / "pc-algos"
    if not pc_algos_path.exists():
        raise FileNotFoundError(f"Missing Community_Finder pc-algos path: {pc_algos_path}")
    sys.path.insert(0, str(pc_algos_path))
    from pc_algos.greedy import greedy_improve
    from pc_algos.spectral import centered_top_candidates

    meta = {
        "path": str(path),
        "pc_algos_path": str(pc_algos_path),
        "primary_mode": "k_densest",
        "dks_supported": False,
    }
    return centered_top_candidates, greedy_improve, meta


def redesigned_fishstick_meta(path: Path) -> dict[str, Any]:
    tasks_path = path / "planted_clique_service_starter" / "pc_service" / "app" / "tasks.py"
    models_path = path / "planted_clique_service_starter" / "pc_service" / "app" / "models.py"
    dks_rejected = False
    for candidate in (tasks_path, models_path):
        if candidate.exists() and "dks" in candidate.read_text(encoding="utf-8", errors="ignore").lower():
            dks_rejected = True
    return {
        "path": str(path),
        "available": path.exists(),
        "historical_parity_only": True,
        "dks_supported": False,
        "dks_rejected_by_service": dks_rejected,
    }


def discover_snapshot_paths(inputs: list[Path], input_root: Path, pattern: str) -> list[Path]:
    paths: list[Path] = []
    search_roots = inputs or [input_root]
    for raw in search_roots:
        path = raw.resolve()
        if path.is_dir():
            paths.extend(path.rglob(pattern))
            continue
        if not path.exists():
            continue
        if path.name.endswith(".community_snapshot.json"):
            paths.append(path)
            continue
        if path.name in {"matrix_summary.json"} or path.name.endswith(".summary.json"):
            try:
                obj = read_json(path)
            except Exception:
                continue
            records = obj if isinstance(obj, list) else [obj]
            for record in records:
                if not isinstance(record, dict):
                    continue
                snapshot_path = record.get("community_snapshot_path")
                if snapshot_path:
                    candidate = Path(snapshot_path)
                    if candidate.exists():
                        paths.append(candidate.resolve())
    return sorted(set(paths))


def parse_seeds(raw: str) -> list[int]:
    seeds = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            seeds.append(int(part))
    return seeds or [0]


def normalize_affinity(matrix: np.ndarray) -> np.ndarray:
    dense = np.asarray(matrix, dtype=np.float64)
    dense = 0.5 * (dense + dense.T)
    np.fill_diagonal(dense, 0.0)
    max_value = float(np.max(dense)) if dense.size else 0.0
    if max_value > 0.0:
        dense = dense / max_value
    return dense


def offdiag_mean(matrix: np.ndarray) -> float:
    dense = np.asarray(matrix, dtype=np.float64)
    n = int(dense.shape[0])
    if n <= 1:
        return 0.0
    values = dense[~np.eye(n, dtype=bool)]
    return float(values.mean()) if values.size else 0.0


def balanced_partition_sizes(n_items: int, n_groups: int, preferred_size: int) -> list[int]:
    sizes = [int(preferred_size) for _ in range(max(n_groups, 1))]
    while sum(sizes) > n_items:
        for idx in reversed(range(len(sizes))):
            if sizes[idx] > 1 and sum(sizes) > n_items:
                sizes[idx] -= 1
    while sum(sizes) < n_items:
        for idx in range(len(sizes)):
            if sum(sizes) >= n_items:
                break
            sizes[idx] += 1
    return sizes


def assign_leftovers(communities: list[list[int]], remaining: list[int], affinity: np.ndarray, size_plan: list[int]) -> list[list[int]]:
    for node in remaining:
        best_group = None
        best_score = -float("inf")
        for group_idx, group in enumerate(communities):
            if len(group) >= size_plan[group_idx]:
                continue
            score = float(np.mean([affinity[node, peer] for peer in group])) if group else 0.0
            if score > best_score or (score == best_score and (best_group is None or len(group) < len(communities[best_group]))):
                best_score = score
                best_group = group_idx
        if best_group is None:
            best_group = min(range(len(communities)), key=lambda idx: len(communities[idx]))
        communities[best_group].append(int(node))
    return communities


def assignment_from_communities(communities: list[list[int]], n: int) -> np.ndarray:
    assignment = np.full((n,), -1, dtype=np.int64)
    for community_id, members in enumerate(communities):
        for node in members:
            assignment[int(node)] = int(community_id)
    if (assignment < 0).any():
        fallback = len(communities) - 1 if communities else 0
        assignment[assignment < 0] = fallback
    return assignment


def community_finder_partition(
    affinity: np.ndarray,
    *,
    community_count: int,
    community_target_size: int,
    tau: float,
    greedy_rounds: int,
    seed: int,
    centered_top_candidates: Callable[..., Any],
    greedy_improve: Callable[..., Any],
) -> np.ndarray:
    dense = normalize_affinity(affinity)
    n = int(dense.shape[0])
    size_plan = balanced_partition_sizes(n, community_count, community_target_size)
    communities: list[list[int]] = [[] for _ in range(community_count)]
    remaining = list(range(n))
    for community_idx in range(max(community_count - 1, 0)):
        target_size = min(max(1, size_plan[community_idx]), len(remaining))
        if target_size <= 0 or not remaining:
            break
        submatrix = dense[np.ix_(remaining, remaining)]
        if np.allclose(submatrix, 0.0):
            ranked = np.argsort(-submatrix.sum(axis=1), kind="mergesort")
            chosen_local = ranked[:target_size]
        else:
            p = offdiag_mean(submatrix)
            _, _, candidates, _ = centered_top_candidates(submatrix, target_size, p=p, tau=tau, seed=seed + community_idx)
            initial = candidates[:target_size]
            improved = greedy_improve(submatrix, initial, candidates, rounds=greedy_rounds)
            improved = np.asarray(improved, dtype=np.int64)[:target_size]
            if improved.size < target_size:
                pool = [int(idx) for idx in candidates.tolist() if int(idx) not in improved.tolist()]
                improved = np.concatenate([improved, np.array(pool[: target_size - improved.size], dtype=np.int64)])
            chosen_local = improved
        chosen = sorted(remaining[int(idx)] for idx in chosen_local[:target_size])
        communities[community_idx] = [int(node) for node in chosen]
        chosen_set = set(chosen)
        remaining = [node for node in remaining if node not in chosen_set]
    if communities:
        communities[-1].extend(int(node) for node in remaining)
    assign_leftovers(communities, [], dense, size_plan)
    return assignment_from_communities(communities, n)


def communities_from_assignment(assignment: np.ndarray, community_count: int) -> list[list[int]]:
    return [[int(idx) for idx in np.flatnonzero(assignment == cid).tolist()] for cid in range(community_count)]


def community_metrics(affinity: np.ndarray, assignment: np.ndarray, route_ema: np.ndarray | None = None) -> dict[str, Any]:
    dense = normalize_affinity(affinity)
    assignment = np.asarray(assignment, dtype=np.int64)
    n = int(dense.shape[0])
    community_count = int(assignment.max()) + 1 if assignment.size else 0
    communities = communities_from_assignment(assignment, community_count)
    offdiag = dense[~np.eye(n, dtype=bool)] if n > 1 else np.array([], dtype=np.float64)
    baseline = float(offdiag.mean()) if offdiag.size else 0.0
    within_values: list[float] = []
    cross_values: list[float] = []
    within_sum = 0.0
    cross_sum = 0.0
    for left in range(n):
        for right in range(left + 1, n):
            value = float(dense[left, right])
            if int(assignment[left]) == int(assignment[right]):
                within_values.append(value)
                within_sum += value
            else:
                cross_values.append(value)
                cross_sum += value
    within_mean = float(np.mean(within_values)) if within_values else 0.0
    cross_mean = float(np.mean(cross_values)) if cross_values else 0.0
    total_weight = within_sum + cross_sum
    out: dict[str, Any] = {
        "community_members": communities,
        "community_sizes": [len(members) for members in communities],
        "offdiag_affinity_mean": baseline,
        "within_affinity_mean": within_mean,
        "cross_affinity_mean": cross_mean,
        "density_lift": within_mean - baseline,
        "within_affinity_fraction": within_sum / max(total_weight, 1e-12),
        "cross_affinity_fraction": cross_sum / max(total_weight, 1e-12),
        "size_std": float(np.std([len(members) for members in communities])) if communities else 0.0,
    }
    if route_ema is not None and route_ema.size:
        route = np.asarray(route_ema, dtype=np.float64)
        route = route / np.maximum(route.sum(axis=1, keepdims=True), 1e-12)
        own_masses = []
        best_other = []
        for community_id, members in enumerate(communities):
            if community_id >= route.shape[0] or not members:
                continue
            own_masses.append(float(route[community_id, members].sum()))
            other_masses = [
                float(route[community_id, other_members].sum())
                for other_id, other_members in enumerate(communities)
                if other_id != community_id and other_members
            ]
            best_other.append(max(other_masses) if other_masses else 0.0)
        own_arr = np.asarray(own_masses, dtype=np.float64)
        other_arr = np.asarray(best_other, dtype=np.float64)
        out["route"] = {
            "own_mass_mean": float(own_arr.mean()) if own_arr.size else 0.0,
            "cross_community_route_rate": 1.0 - (float(own_arr.mean()) if own_arr.size else 0.0),
            "community_route_separation": float((own_arr - other_arr).mean()) if own_arr.size else 0.0,
        }
    return out


def coassignment_vector(assignment: np.ndarray) -> np.ndarray:
    values = []
    n = int(assignment.size)
    for left in range(n):
        for right in range(left + 1, n):
            values.append(int(assignment[left]) == int(assignment[right]))
    return np.asarray(values, dtype=np.bool_)


def coassignment_agreement(left: np.ndarray, right: np.ndarray) -> float:
    lv = coassignment_vector(left)
    rv = coassignment_vector(right)
    if lv.size == 0 or rv.size == 0:
        return 1.0
    return float((lv == rv).mean())


def stability(assignments: list[np.ndarray]) -> float:
    if len(assignments) < 2:
        return 1.0
    scores = []
    for left_idx in range(len(assignments)):
        for right_idx in range(left_idx + 1, len(assignments)):
            scores.append(coassignment_agreement(assignments[left_idx], assignments[right_idx]))
    return float(np.mean(scores)) if scores else 1.0


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(ys) < 2:
        return None
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if float(x.std()) == 0.0 or float(y.std()) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def analyze_snapshot(
    *,
    snapshot_path: Path,
    snapshot: dict[str, Any],
    w9,
    centered_top_candidates: Callable[..., Any],
    greedy_improve: Callable[..., Any],
    seeds: list[int],
    tau_override: float | None,
    greedy_rounds_override: int | None,
) -> dict[str, Any]:
    affinity = np.asarray(snapshot["assembly_affinity_ema"], dtype=np.float64)
    current_assignment = np.asarray(snapshot["community_assignment"], dtype=np.int64)
    route_ema = np.asarray(snapshot.get("community_route_ema", []), dtype=np.float64)
    cfg = snapshot.get("args", {})
    community_count = int(cfg.get("community_count", int(current_assignment.max()) + 1))
    community_target_size = int(cfg.get("community_target_size", max(1, int(math.ceil(current_assignment.size / max(community_count, 1))))))
    tau = float(tau_override if tau_override is not None else cfg.get("community_tau", 2.0))
    greedy_rounds = int(greedy_rounds_override if greedy_rounds_override is not None else cfg.get("community_greedy_rounds", 10))

    w9_assignments = [
        w9.compute_community_assignment(
            affinity,
            community_count=community_count,
            community_target_size=community_target_size,
            tau=tau,
            greedy_rounds=greedy_rounds,
            seed=seed,
        )
        for seed in seeds
    ]
    cf_assignments = [
        community_finder_partition(
            affinity,
            community_count=community_count,
            community_target_size=community_target_size,
            tau=tau,
            greedy_rounds=greedy_rounds,
            seed=seed,
            centered_top_candidates=centered_top_candidates,
            greedy_improve=greedy_improve,
        )
        for seed in seeds
    ]
    current_metrics = community_metrics(affinity, current_assignment, route_ema)
    w9_metrics = community_metrics(affinity, w9_assignments[0], route_ema)
    cf_metrics = community_metrics(affinity, cf_assignments[0], route_ema)
    return {
        "snapshot_path": str(snapshot_path),
        "run_id": snapshot.get("run_id", snapshot_path.stem),
        "validation": snapshot.get("validation", {}),
        "config": {
            "community_count": community_count,
            "community_target_size": community_target_size,
            "tau": tau,
            "greedy_rounds": greedy_rounds,
            "seeds": seeds,
        },
        "current": {
            "assignment": current_assignment.tolist(),
            "metrics": current_metrics,
        },
        "w9_recomputed": {
            "assignment_seed0": w9_assignments[0].tolist(),
            "metrics_seed0": w9_metrics,
            "stability": stability(w9_assignments),
            "similarity_to_current": coassignment_agreement(current_assignment, w9_assignments[0]),
        },
        "community_finder": {
            "assignment_seed0": cf_assignments[0].tolist(),
            "metrics_seed0": cf_metrics,
            "stability": stability(cf_assignments),
            "similarity_to_current": coassignment_agreement(current_assignment, cf_assignments[0]),
            "similarity_to_w9_seed0": coassignment_agreement(w9_assignments[0], cf_assignments[0]),
        },
    }


def correlation_report(results: list[dict[str, Any]]) -> dict[str, Any]:
    bpbs = [float(item.get("validation", {}).get("current_eval_bpb")) for item in results if item.get("validation", {}).get("current_eval_bpb") is not None]
    if len(bpbs) != len(results):
        return {"available": False, "reason": "not every snapshot had validation.current_eval_bpb"}
    series = {
        "current_within_affinity_fraction": [float(item["current"]["metrics"]["within_affinity_fraction"]) for item in results],
        "current_cross_affinity_fraction": [float(item["current"]["metrics"]["cross_affinity_fraction"]) for item in results],
        "w9_seed0_within_affinity_fraction": [float(item["w9_recomputed"]["metrics_seed0"]["within_affinity_fraction"]) for item in results],
        "community_finder_seed0_within_affinity_fraction": [
            float(item["community_finder"]["metrics_seed0"]["within_affinity_fraction"]) for item in results
        ],
        "community_finder_stability": [float(item["community_finder"]["stability"]) for item in results],
    }
    return {"available": len(results) >= 2, "with_val_bpb": {key: pearson(values, bpbs) for key, values in series.items()}}


def format_optional_float(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(numeric):
        return "n/a"
    return f"{numeric:.{digits}f}"


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# W9 Community Oracle Summary",
        "",
        f"- snapshots: {len(payload['results'])}",
        f"- Community_Finder mode: {payload['community_finder']['primary_mode']}",
        f"- redesigned-fishstick dks supported: {payload['redesigned_fishstick']['dks_supported']}",
        "",
        "| run_id | val_bpb | current_within | w9_within | cf_within | w9_stability | cf_stability | cf_to_w9 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in payload["results"]:
        val_bpb = item.get("validation", {}).get("current_eval_bpb")
        lines.append(
            f"| {item['run_id']} "
            f"| {format_optional_float(val_bpb)} "
            f"| {float(item['current']['metrics']['within_affinity_fraction']):.4f} "
            f"| {float(item['w9_recomputed']['metrics_seed0']['within_affinity_fraction']):.4f} "
            f"| {float(item['community_finder']['metrics_seed0']['within_affinity_fraction']):.4f} "
            f"| {float(item['w9_recomputed']['stability']):.4f} "
            f"| {float(item['community_finder']['stability']):.4f} "
            f"| {float(item['community_finder']['similarity_to_w9_seed0']):.4f} |"
        )
    lines.append("")
    lines.append("Community_Finder is used as an offline k-densest-style oracle only; no service/runtime dependency is introduced.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    cli = parse_args()
    repo_root = cli.repo_root.resolve()
    w9 = import_w9_module(cli.w9_path.resolve())
    centered_top_candidates, greedy_improve, cf_meta = import_community_finder(cli.community_finder_path.resolve())
    fishstick_meta = redesigned_fishstick_meta(cli.redesigned_fishstick_path.resolve())
    seeds = parse_seeds(cli.seeds)
    snapshot_paths = discover_snapshot_paths(cli.input, cli.input_root.resolve(), cli.pattern)
    if not snapshot_paths:
        raise FileNotFoundError(f"No community snapshots found under {cli.input_root} with pattern {cli.pattern}")

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = cli.output_root.resolve() / f"{timestamp}_community_oracle"
    output_dir.mkdir(parents=True, exist_ok=True)
    results = [
        analyze_snapshot(
            snapshot_path=path,
            snapshot=read_json(path),
            w9=w9,
            centered_top_candidates=centered_top_candidates,
            greedy_improve=greedy_improve,
            seeds=seeds,
            tau_override=cli.tau,
            greedy_rounds_override=cli.greedy_rounds,
        )
        for path in snapshot_paths
    ]
    payload = {
        "repo_root": str(repo_root),
        "snapshot_paths": [str(path) for path in snapshot_paths],
        "community_finder": cf_meta,
        "redesigned_fishstick": fishstick_meta,
        "results": results,
        "correlations": correlation_report(results),
    }
    json_path = output_dir / "oracle_summary.json"
    md_path = output_dir / "oracle_summary.md"
    write_json(json_path, payload)
    write_markdown(md_path, payload)
    print(f"ORACLE_SUMMARY_JSON {json_path}")
    print(f"ORACLE_SUMMARY_MD {md_path}")
    for item in results:
        print(
            f"ORACLE {item['run_id']} "
            f"val_bpb={float(item.get('validation', {}).get('current_eval_bpb', float('nan'))):.4f} "
            f"w9_within={float(item['w9_recomputed']['metrics_seed0']['within_affinity_fraction']):.4f} "
            f"cf_within={float(item['community_finder']['metrics_seed0']['within_affinity_fraction']):.4f} "
            f"cf_to_w9={float(item['community_finder']['similarity_to_w9_seed0']):.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
