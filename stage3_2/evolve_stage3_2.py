#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from controller_library import ControllerCandidate, candidate_by_key, child_candidate, seed_candidates
from controller_mutations import mutate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evolution helper for stage3_2 controller search.")
    parser.add_argument("--generation", type=int, default=0)
    parser.add_argument("--label", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--population-size", type=int, default=6)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run", action="store_true")
    return parser.parse_args()


def stage_dir() -> Path:
    return Path(__file__).resolve().parent


def base_config_path() -> Path:
    return stage_dir() / "run_configs.json"


def load_base_config() -> dict[str, Any]:
    return json.loads(base_config_path().read_text(encoding="utf-8"))


def evolution_root(label: str) -> Path:
    return stage_dir() / "evolution_runs" / label


def generation_dir(label: str, generation: int) -> Path:
    return evolution_root(label) / f"gen_{generation:03d}"


def generation_config_path(label: str, generation: int) -> Path:
    return generation_dir(label, generation) / "run_configs.json"


def generation_summary_path(label: str, generation: int) -> Path:
    return generation_dir(label, generation) / "generation_summary.json"


def canonical_signature(candidate: ControllerCandidate) -> str:
    spec = candidate.spec.to_dict()
    gates = spec.get("gates", [])
    gates = sorted(gates, key=lambda gate: (gate["feature"], gate["action"], str(gate["value"])))
    signature = {
        "snapshot_mode": spec.get("snapshot", {}).get("mode"),
        "snapshot_score": spec.get("snapshot", {}).get("score"),
        "pulse_mode": spec.get("pulse", {}).get("mode"),
        "gate_actions": [gate["action"] for gate in gates],
        "gate_features": [gate["feature"] for gate in gates],
        "late_actions": sorted(spec.get("phase_defaults", {}).get("late", {}).keys()),
    }
    return json.dumps(signature, sort_keys=True, separators=(",", ":"))


def controls_from_base(base_config: dict[str, Any]) -> list[dict[str, Any]]:
    slots = {slot["slot"]: slot for slot in base_config["slots"]}
    return [deepcopy(slots["R0A"]), deepcopy(slots["R0B"])]


def seeds_for_generation_zero(population_size: int) -> list[ControllerCandidate]:
    seeds = seed_candidates()
    return seeds[:population_size]


def load_selected_parents(label: str, generation: int) -> list[ControllerCandidate]:
    summary_path = generation_summary_path(label, generation - 1)
    if not summary_path.exists():
        raise SystemExit(f"Missing previous generation summary: {summary_path}")
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    return [ControllerCandidate.from_dict(item) for item in data["selected_parents"]]


def build_children(parents: list[ControllerCandidate], population_size: int, seed: int) -> list[ControllerCandidate]:
    rng = random.Random(seed)
    children: list[ControllerCandidate] = []
    seen = {parent.spec.canonical_json() for parent in parents}
    child_index = 0
    while len(children) < population_size:
        parent = parents[len(children) % len(parents)]
        child_spec, lineage = mutate(parent.spec, rng)
        signature = child_spec.canonical_json()
        if signature in seen:
            continue
        seen.add(signature)
        child_key = f"{parent.key}_c{child_index:02d}"
        child_name = f"{parent.name}_c{child_index:02d}"
        mutation_note = lineage["description"]
        children.append(child_candidate(parent, child_key, child_name, child_spec, mutation_note))
        child_index += 1
    return children


def build_generation_candidates(label: str, generation: int, population_size: int, seed: int) -> list[ControllerCandidate]:
    if generation == 0:
        return seeds_for_generation_zero(population_size)
    parents = load_selected_parents(label, generation)
    return build_children(parents, population_size, seed + generation)


def build_generation_config(
    label: str,
    generation: int,
    population_size: int,
    seed: int,
) -> tuple[dict[str, Any], list[ControllerCandidate]]:
    base_config = load_base_config()
    candidates = build_generation_candidates(label, generation, population_size, seed)
    config = deepcopy(base_config)
    config["batch_id"] = f"stage3_2_evo_g{generation:03d}"
    config["description"] = f"Stage 3.2 evolutionary generation {generation}: bounded controller children."
    config["slots"] = controls_from_base(base_config)
    for idx, candidate in enumerate(candidates):
        config["slots"].append(candidate.to_slot(f"C{idx}"))
    config["phases"]["sanity"]["slots"] = ["R0A", "R0B"] + [f"C{idx}" for idx in range(len(candidates))]
    config["phases"]["screen"]["slots"] = ["R0A", "R0B"] + [f"C{idx}" for idx in range(len(candidates))]
    return config, candidates


def run_phase(config_path: Path, label: str, phase: str, gpus: str, dry_run: bool) -> int:
    cmd = [
        sys.executable,
        str(stage_dir() / "orchestrate_stage3_2.py"),
        "--config",
        str(config_path),
        "--phase",
        phase,
        "--label",
        label,
        "--gpus",
        gpus,
    ]
    if dry_run:
        cmd.append("--dry-run")
    completed = subprocess.run(cmd, check=False)
    return completed.returncode


def pick_selected_parents(
    label: str,
    generation: int,
    candidates: list[ControllerCandidate],
    top_k: int,
) -> list[ControllerCandidate]:
    run_root = generation_config_path(label, generation).parent / "runs" / f"g{generation:03d}"
    summary_path = run_root / "screen" / "summary.json"
    if not summary_path.exists():
        raise SystemExit(f"Missing screen summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    slot_to_candidate = {f"C{idx}": candidate for idx, candidate in enumerate(candidates)}
    comparisons = summary.get("comparisons", [])
    ranked_slots = [entry["slot"] for entry in comparisons if entry["slot"] in slot_to_candidate]
    selected: list[ControllerCandidate] = []
    for slot in ranked_slots:
        if len(selected) >= top_k:
            break
        selected.append(slot_to_candidate[slot])
    selected_signatures = {canonical_signature(candidate) for candidate in selected}
    for slot in ranked_slots:
        candidate = slot_to_candidate[slot]
        signature = canonical_signature(candidate)
        if signature not in selected_signatures:
            selected.append(candidate)
            break
    return selected


def write_generation_summary(
    label: str,
    generation: int,
    candidates: list[ControllerCandidate],
    selected_parents: list[ControllerCandidate],
) -> Path:
    out_path = generation_summary_path(label, generation)
    out = {
        "label": label,
        "generation": generation,
        "candidates": [candidate.to_dict() for candidate in candidates],
        "selected_parents": [candidate.to_dict() for candidate in selected_parents],
    }
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return out_path


def main() -> None:
    args = parse_args()
    gen_dir = generation_dir(args.label, args.generation)
    gen_dir.mkdir(parents=True, exist_ok=True)

    config, candidates = build_generation_config(args.label, args.generation, args.population_size, args.seed)
    config_path = generation_config_path(args.label, args.generation)
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    if not args.run:
        print(json.dumps({
            "generation": args.generation,
            "config_path": str(config_path),
            "candidate_keys": [candidate.key for candidate in candidates],
        }, indent=2))
        return

    run_label = f"g{args.generation:03d}"
    rc = run_phase(config_path, run_label, "sanity", args.gpus, args.dry_run)
    if rc != 0:
        raise SystemExit(rc)
    rc = run_phase(config_path, run_label, "screen", args.gpus, args.dry_run)
    if rc != 0:
        raise SystemExit(rc)
    if args.dry_run:
        return
    selected = pick_selected_parents(args.label, args.generation, candidates, args.top_k)
    summary_path = write_generation_summary(args.label, args.generation, candidates, selected)
    print(json.dumps({
        "generation": args.generation,
        "config_path": str(config_path),
        "generation_summary": str(summary_path),
        "selected_keys": [candidate.key for candidate in selected],
    }, indent=2))


if __name__ == "__main__":
    main()
