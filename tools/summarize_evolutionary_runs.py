#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def load_results(paths: list[str]) -> list[tuple[Path, dict[str, Any]]]:
    loaded: list[tuple[Path, dict[str, Any]]] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            for child in sorted(path.rglob("*.json")):
                try:
                    loaded.append((child, json.loads(child.read_text(encoding="utf-8"))))
                except json.JSONDecodeError:
                    continue
            continue
        loaded.append((path, json.loads(path.read_text(encoding="utf-8"))))
    return loaded


def bytes_to_gb(value: float | None) -> float | None:
    if value is None:
        return None
    return float(value / (1024.0**3))


def classify_result(data: dict[str, Any]) -> str:
    if "results" in data and "model_param_mb_estimate" in data:
        return "throughput"
    if "committee_compressibility" in data:
        return "viability"
    if "members" in data and ("crossover" in data or "committee_schedule" in data or "committee_adaptive" in data):
        return "viability"
    if ("evolution" in data and "base" in data) or "recipe_evolution" in data:
        return "evolution"
    return "unknown"


def flatten_throughput(path: Path, data: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for result in data.get("results", []):
        after = result.get("cuda_memory_after") or {}
        rows.append(
            {
                "section": "throughput",
                "run": path.stem,
                "dtype": data.get("config", {}).get("dtype"),
                "spine": data.get("config", {}).get("spine_variant"),
                "population": result.get("population_size"),
                "chunk_size": result.get("population_chunk_size"),
                "num_chunks": result.get("num_chunks"),
                "status": result.get("status"),
                "batch_eval_s": result.get("batch_eval_s"),
                "models_per_second": result.get("models_per_second"),
                "estimated_param_mb": result.get("estimated_population_param_mb"),
                "estimated_chunk_mb": result.get("estimated_chunk_param_mb"),
                "peak_reserved_gb": bytes_to_gb(after.get("max_reserved_bytes")),
            }
        )
    return rows


def flatten_viability(path: Path, data: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    members = data.get("members", [])
    copies = len(members)
    train_seconds_per_member = None
    if members:
        train_seconds_per_member = members[0].get("train", {}).get("train_seconds_target")
    committee_schedule = data.get("committee_schedule", {})
    committee_adaptive = data.get("committee_adaptive", {})
    branch_seconds = None
    stage_copies = committee_schedule.get("stage_copies")
    stage_train_seconds = committee_schedule.get("stage_train_seconds")
    if stage_copies and stage_train_seconds and len(stage_copies) == len(stage_train_seconds):
        branch_seconds = float(
            sum(float(copies_i) * float(seconds_i) for copies_i, seconds_i in zip(stage_copies, stage_train_seconds))
        )
    elif committee_adaptive and data.get("rounds"):
        branch_seconds = float(
            sum(float(row.get("copies", 0)) * float(row.get("train_seconds_target", 0.0)) for row in data.get("rounds", []))
        )
    elif copies and train_seconds_per_member is not None:
        branch_seconds = float(copies) * float(train_seconds_per_member)
    best_member_bpb = min((member.get("eval", {}).get("bpb") for member in members), default=None)
    member_ensemble_results = data.get("member_ensemble_results", [])
    best_member_ensemble = min(member_ensemble_results, key=lambda row: row.get("val_bpb", float("inf")), default=None)
    ensemble_gain_vs_best = None
    if best_member_bpb is not None and best_member_ensemble is not None and best_member_ensemble.get("val_bpb") is not None:
        ensemble_gain_vs_best = float(best_member_bpb - best_member_ensemble["val_bpb"])
    def build_row(*, strategy: str, percentile: str, summary: dict[str, Any] | None) -> dict[str, Any]:
        return {
            "section": "viability",
            "run": path.stem,
            "member_train_mode": data.get("member_train_mode"),
            "copies": copies,
            "train_seconds_per_member": train_seconds_per_member,
            "branch_seconds": branch_seconds,
            "schedule": committee_schedule.get("stage_signature") or committee_adaptive.get("round_signature"),
            "strategy": strategy,
            "percentile": percentile,
            "num_trials": None if summary is None else summary.get("num_trials"),
            "between": None if summary is None else summary.get("fraction_between_parents"),
            "collapse": None if summary is None else summary.get("fraction_collapse"),
            "improves_best": None if summary is None else summary.get("fraction_improves_best_parent"),
            "best_member_bpb": best_member_bpb,
            "ensemble_topk": None if best_member_ensemble is None else best_member_ensemble.get("topk"),
            "ensemble_val_bpb": None if best_member_ensemble is None else best_member_ensemble.get("val_bpb"),
            "ensemble_test_bpb": None if best_member_ensemble is None else best_member_ensemble.get("test_bpb"),
            "ensemble_gain_vs_best": ensemble_gain_vs_best,
        }

    crossover = data.get("crossover", {})
    if crossover:
        for strategy, percentile_map in crossover.items():
            for percentile, payload in percentile_map.items():
                summary = payload.get("summary", {})
                rows.append(build_row(strategy=strategy, percentile=percentile, summary=summary))
    elif data.get("committee_compressibility") is not None:
        committee_compressibility = data.get("committee_compressibility", {})
        schedule = committee_schedule.get("stage_signature")
        for row in data.get("best_by_budget", []):
            rows.append(
                {
                    "section": "viability",
                    "run": path.stem,
                    "member_train_mode": "committee_compressibility",
                    "copies": committee_compressibility.get("member_count"),
                    "train_seconds_per_member": None,
                    "branch_seconds": branch_seconds,
                    "schedule": schedule,
                    "strategy": f"{row.get('strategy')}@{row.get('rank', 'na')}",
                    "percentile": f"{float(row.get('budget_fraction', 0.0)):.2f}",
                    "num_trials": None,
                    "between": None,
                    "collapse": None,
                    "improves_best": None,
                    "best_member_bpb": None,
                    "ensemble_topk": committee_compressibility.get("member_count"),
                    "ensemble_val_bpb": row.get("val_bpb"),
                    "ensemble_test_bpb": row.get("test_bpb"),
                    "ensemble_gain_vs_best": None,
                }
            )
    elif data.get("committee_schedule") is not None:
        rows.append(build_row(strategy="committee_schedule", percentile="na", summary=None))
    elif data.get("committee_adaptive") is not None:
        rows.append(build_row(strategy="committee_adaptive", percentile="na", summary=None))
    return rows


def flatten_evolution(path: Path, data: dict[str, Any]) -> list[dict[str, Any]]:
    if "recipe_evolution" in data:
        recipe = data.get("recipe_evolution", {})
        history = data.get("history", [])
        best_history = min((row.get("best_bpb") for row in history), default=None)
        best = data.get("best") or {}
        confirm_results = data.get("confirm_results", [])
        best_confirm = min(confirm_results, key=lambda row: row.get("val", {}).get("bpb", float("inf")), default=None)
        return [
            {
                "section": "evolution",
                "run": path.stem,
                "strategy": "recipe_evolution",
                "percentile": recipe.get("recipe_profile"),
                "population": recipe.get("population_size"),
                "generations": recipe.get("generations"),
                "base_val_bpb": None,
                "best_history_bpb": best_history,
                "final_val_bpb": None if not best else best.get("val", {}).get("bpb"),
                "final_test_bpb": None if not best else best.get("test", {}).get("bpb"),
                "gain_vs_base": None,
                "ensemble_topk": None,
                "ensemble_val_bpb": None if best_confirm is None else best_confirm.get("val", {}).get("bpb"),
                "ensemble_test_bpb": None if best_confirm is None else best_confirm.get("test", {}).get("bpb"),
                "ensemble_gain_vs_base": None,
                "ensemble_gain_vs_best": None
                if best_confirm is None or not best
                else float(best.get("val", {}).get("bpb") - best_confirm.get("val", {}).get("bpb")),
            }
        ]

    evolution = data.get("evolution", {})
    base = data.get("base", {})
    history = evolution.get("history", [])
    best_history = min((row.get("best_bpb") for row in history), default=None)
    final_val = evolution.get("final_best_val_bpb")
    base_val = base.get("val", {}).get("bpb")
    ensemble_results = evolution.get("ensemble_results", [])
    best_ensemble = min(ensemble_results, key=lambda row: row.get("val_bpb", float("inf")), default=None)
    gain = None
    if final_val is not None and base_val is not None:
        gain = float(base_val - final_val)
    ensemble_gain = None
    ensemble_gain_vs_best = None
    if best_ensemble is not None and base_val is not None and best_ensemble.get("val_bpb") is not None:
        ensemble_gain = float(base_val - best_ensemble["val_bpb"])
    if best_ensemble is not None and final_val is not None and best_ensemble.get("val_bpb") is not None:
        ensemble_gain_vs_best = float(final_val - best_ensemble["val_bpb"])
    return [
        {
            "section": "evolution",
            "run": path.stem,
            "strategy": evolution.get("crossover_strategy"),
            "percentile": evolution.get("crossover_percentile"),
            "population": evolution.get("population_size"),
            "generations": evolution.get("generations"),
            "base_val_bpb": base_val,
            "best_history_bpb": best_history,
            "final_val_bpb": final_val,
            "final_test_bpb": evolution.get("final_best_test_bpb"),
            "gain_vs_base": gain,
            "ensemble_topk": None if best_ensemble is None else best_ensemble.get("topk"),
            "ensemble_val_bpb": None if best_ensemble is None else best_ensemble.get("val_bpb"),
            "ensemble_test_bpb": None if best_ensemble is None else best_ensemble.get("test_bpb"),
            "ensemble_gain_vs_base": ensemble_gain,
            "ensemble_gain_vs_best": ensemble_gain_vs_best,
        }
    ]


def summarize_results(paths: list[str]) -> dict[str, list[dict[str, Any]]]:
    sections: dict[str, list[dict[str, Any]]] = {"throughput": [], "viability": [], "evolution": []}
    for path, data in load_results(paths):
        kind = classify_result(data)
        if kind == "throughput":
            sections["throughput"].extend(flatten_throughput(path, data))
        elif kind == "viability":
            sections["viability"].extend(flatten_viability(path, data))
        elif kind == "evolution":
            sections["evolution"].extend(flatten_evolution(path, data))
    return sections


def format_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def render_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "(no rows)"
    headers = tuple(rows[0].keys())
    widths = {header: len(header) for header in headers}
    for row in rows:
        for header in headers:
            widths[header] = max(widths[header], len(format_value(row.get(header))))
    lines = [
        "  ".join(header.ljust(widths[header]) for header in headers),
        "  ".join("-" * widths[header] for header in headers),
    ]
    for row in rows:
        lines.append("  ".join(format_value(row.get(header)).ljust(widths[header]) for header in headers))
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize evolutionary benchmark JSON outputs")
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--section", choices=("all", "throughput", "viability", "evolution"), default="all")
    parser.add_argument("--format", choices=("table", "json"), default="table")
    return parser


def main(argv: list[str]) -> int:
    parser = build_parser()
    args = parser.parse_args(argv[1:])
    sections = summarize_results(args.paths)
    if args.format == "json":
        if args.section == "all":
            print(json.dumps(sections, indent=2))
        else:
            print(json.dumps(sections[args.section], indent=2))
        return 0

    if args.section == "all":
        printed = False
        for section_name in ("throughput", "viability", "evolution"):
            rows = sections[section_name]
            if not rows:
                continue
            if printed:
                print()
            print(f"[{section_name}]")
            print(render_table(rows))
            printed = True
        if not printed:
            print("(no recognized results)")
        return 0

    print(render_table(sections[args.section]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
