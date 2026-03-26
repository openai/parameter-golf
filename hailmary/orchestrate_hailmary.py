#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from patches import apply_patches, list_patches


SUPPORTED_RUNNER_MODES = {"train"}
REFERENCE_PACK = "reference"


@dataclass(frozen=True)
class PhaseSpec:
    name: str
    max_wallclock_seconds: int
    nproc_per_slot: int
    slots: list[str] | None = None


@dataclass(frozen=True)
class JobSpec:
    slot_id: str
    pack_name: str
    phase: str
    gpu_spec: str | None
    nproc_per_slot: int
    run_dir: Path
    log_path: Path
    cmd: list[str]
    env: dict[str, str]
    metadata: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hailmary moonshot orchestrator for parameter-golf.")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to a run_configs.json file. Defaults to the file next to this script.",
    )
    parser.add_argument(
        "--phase",
        default="plan",
        choices=["plan", "readiness", "sanity", "screen", "final_single", "champion_8x", "all", "tournament"],
        help="Which phase to execute.",
    )
    parser.add_argument("--label", default=datetime.now().strftime("%Y%m%d_%H%M%S"), help="Run label.")
    parser.add_argument(
        "--pack",
        default=None,
        help="Named pack from run_configs.json. Defaults to config.default_pack.",
    )
    parser.add_argument(
        "--gpus",
        default="0,1,2,3,4,5,6,7",
        help="Comma-separated local GPU ids.",
    )
    parser.add_argument("--slots", nargs="*", help="Override the config-defined slots for the selected phase.")
    parser.add_argument(
        "--final-slots",
        nargs="*",
        help="Explicit final slots to run instead of auto-promoting from screen results.",
    )
    parser.add_argument("--top-k", type=int, default=1, help="How many screen survivors to promote.")
    parser.add_argument("--dry-run", action="store_true", help="Print launch plan without running.")
    parser.add_argument(
        "--run-champion",
        action="store_true",
        help="With --phase all, force an additional champion_8x run after final_single.",
    )
    parser.add_argument(
        "--skip-champion",
        action="store_true",
        help="With --phase all, stop after final_single and do not run champion_8x.",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> dict[str, Any]:
    return json.loads(config_path.read_text(encoding="utf-8"))


def build_slot_map(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {slot["slot"]: slot for slot in config["slots"]}


def parse_gpu_list(raw: str) -> list[str]:
    gpus = [item.strip() for item in raw.split(",") if item.strip()]
    if not gpus:
        raise SystemExit("At least one GPU id is required via --gpus")
    return gpus


def phase_from_config(config: dict[str, Any], phase_name: str) -> PhaseSpec:
    phase = config["phases"][phase_name]
    return PhaseSpec(
        name=phase_name,
        max_wallclock_seconds=int(phase["max_wallclock_seconds"]),
        nproc_per_slot=int(phase["nproc_per_slot"]),
        slots=phase.get("slots"),
    )


def tournament_settings(config: dict[str, Any]) -> dict[str, Any]:
    section = config.get("tournament", {})
    return {
        "primary_packs": list(section.get("primary_packs", ["moonshot_core", "moonshot_second_wave"])),
        "support_packs": list(section.get("support_packs", [])),
        "deferred_rebuild_packs": list(section.get("deferred_rebuild_packs", [])),
        "control_slots": tuple(section.get("control_slots", ["H0", "H1"])),
        "export_slots": tuple(section.get("export_slots", ["G1", "G2"])),
        "final_pack": section.get("final_pack", "tournament_finalists"),
    }


def resolve_pack_name(args: argparse.Namespace, config: dict[str, Any]) -> str:
    pack_name = args.pack or config.get("default_pack", "moonshot_core")
    if pack_name not in config["packs"]:
        raise SystemExit(
            f"Unknown pack {pack_name!r}. Available packs: {', '.join(sorted(config['packs']))}"
        )
    return pack_name


def merge_env_for_slot(
    slot_id: str, slots: dict[str, dict[str, Any]], defaults: dict[str, str]
) -> tuple[dict[str, str], list[str]]:
    slot = slots[slot_id]
    env = dict(defaults)
    lineage: list[str] = []
    parent = slot.get("parent")
    if parent:
        parent_env, parent_lineage = merge_env_for_slot(parent, slots, defaults)
        env.update(parent_env)
        lineage.extend(parent_lineage)
    env.update(slot["env"])
    lineage.append(slot_id)
    return env, lineage


def resolve_paths(script_path: Path) -> tuple[Path, Path]:
    candidate = script_path.resolve().parent
    for _ in range(10):
        if (candidate / "train_gpt.py").exists():
            return candidate.parent, candidate
        candidate = candidate.parent
    raise SystemExit("Cannot find parameter-golf root (directory containing train_gpt.py)")


def phase_slot_defaults(phase: PhaseSpec, config: dict[str, Any], pack_name: str) -> list[str]:
    pack_slots = list(config["packs"][pack_name])
    if phase.slots is None:
        return pack_slots
    phase_slot_set = set(phase.slots)
    intersected = [slot_id for slot_id in pack_slots if slot_id in phase_slot_set]
    return intersected if intersected else pack_slots


def resolve_phase_slots(
    args: argparse.Namespace, config: dict[str, Any], phase: PhaseSpec, pack_name: str
) -> list[str]:
    if args.slots:
        return list(args.slots)
    default_slots = phase_slot_defaults(phase, config, pack_name)
    if not default_slots:
        raise SystemExit(
            f"No slots resolved for phase {phase.name} pack {pack_name}. Use --slots to specify explicitly."
        )
    return default_slots


def resolve_scope_name(args: argparse.Namespace, pack_name: str) -> str:
    return pack_name if not args.slots else "adhoc"


def resolve_final_slots(
    args: argparse.Namespace, run_root: Path, scope_name: str, top_k: int
) -> list[str]:
    if args.final_slots:
        return list(args.final_slots)
    summary_path = run_root / scope_name / "screen" / "summary.json"
    if not summary_path.exists():
        raise SystemExit(
            f"No screen summary found at {summary_path}. Run --phase screen or --phase all first, or pass --final-slots."
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    promotions = summary.get("recommended_promotions", [])
    if not promotions:
        raise SystemExit(f"Screen summary at {summary_path} contains no recommended promotions.")
    return [entry["slot"] for entry in promotions[:top_k]]


def resolve_patch_names(slot: dict[str, Any]) -> list[str]:
    return list(slot.get("patches", []))


def runnable_reason(slot: dict[str, Any]) -> str | None:
    if slot["implementation_state"] != "ready":
        return f"implementation_state={slot['implementation_state']}"
    runner_mode = slot.get("runner_mode", "train")
    if runner_mode not in SUPPORTED_RUNNER_MODES:
        return f"runner_mode={runner_mode} unsupported"
    for patch_name in resolve_patch_names(slot):
        if patch_name not in list_patches():
            return f"patch={patch_name} unavailable"
    return None


def ensure_runnable(slot_ids: list[str], slot_map: dict[str, dict[str, Any]]) -> None:
    blocked = []
    for slot_id in slot_ids:
        reason = runnable_reason(slot_map[slot_id])
        if reason is not None:
            blocked.append(f"{slot_id}({reason})")
    if blocked:
        raise SystemExit(f"Blocked by non-runnable slots: {', '.join(blocked)}")


def scope_root(config_path: Path, label: str, scope_name: str) -> Path:
    return config_path.parent / "runs" / label / scope_name


def materialize_train_script(pgolf_root: Path, run_dir: Path, slot: dict[str, Any]) -> tuple[Path, list[str]]:
    patch_names = resolve_patch_names(slot)
    source_path = (pgolf_root / "train_gpt.py").resolve()
    patched_source = apply_patches(source_path.read_text(encoding="utf-8"), patch_names)
    train_script = run_dir / "train_gpt.py"
    train_script.write_text(patched_source, encoding="utf-8")
    return train_script, patch_names


def build_job(
    script_path: Path,
    config_path: Path,
    config: dict[str, Any],
    slot_id: str,
    phase: PhaseSpec,
    label: str,
    pack_name: str,
    scope_name: str,
    gpu_spec: str | None,
    nproc_per_slot: int | None = None,
) -> JobSpec:
    _, pgolf_root = resolve_paths(script_path)
    slots = build_slot_map(config)
    slot = slots[slot_id]
    merged_env, lineage = merge_env_for_slot(slot_id, slots, config["defaults"])

    env = os.environ.copy()
    env.update(merged_env)
    env["MAX_WALLCLOCK_SECONDS"] = str(phase.max_wallclock_seconds)
    env["RUN_ID"] = f"{config['batch_id']}_{label}_{scope_name}_{phase.name}_{slot_id}"
    env["PGOLF_HAILMARY_BATCH"] = config["batch_id"]
    env["PGOLF_HAILMARY_PACK"] = pack_name
    env["PGOLF_HAILMARY_SCOPE"] = scope_name
    env["PGOLF_HAILMARY_PHASE"] = phase.name
    env["PGOLF_HAILMARY_SLOT"] = slot_id
    env["PGOLF_HAILMARY_RUNNER_MODE"] = slot.get("runner_mode", "train")
    env["PYTHONUNBUFFERED"] = "1"
    if env["DATA_PATH"].startswith("./"):
        env["DATA_PATH"] = str((pgolf_root / "data" / "datasets" / "fineweb10B_sp1024").resolve())
    if env["TOKENIZER_PATH"].startswith("./"):
        env["TOKENIZER_PATH"] = str((pgolf_root / "data" / "tokenizers" / "fineweb_1024_bpe.model").resolve())
    if gpu_spec is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu_spec

    run_dir = scope_root(config_path, label, scope_name) / phase.name / f"{slot_id}_{slot['name']}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.log"

    effective_nproc = nproc_per_slot if nproc_per_slot is not None else phase.nproc_per_slot
    train_script_path, patch_names = materialize_train_script(pgolf_root, run_dir, slot)

    metadata = {
        "batch_id": config["batch_id"],
        "pack": pack_name,
        "scope": scope_name,
        "phase": phase.name,
        "slot": slot_id,
        "name": slot["name"],
        "role": slot["role"],
        "family": slot["family"],
        "runner_mode": slot.get("runner_mode", "train"),
        "implementation_state": slot["implementation_state"],
        "patches": patch_names,
        "label": label,
        "gpu_spec": gpu_spec,
        "nproc_per_slot": effective_nproc,
        "lineage": lineage,
        "compare_to": slot.get("compare_to"),
        "why": slot["why"],
        "validates": slot["validates"],
        "falsifies": slot["falsifies"],
        "notes": slot.get("notes", []),
        "env": {k: env[k] for k in sorted(merged_env)},
    }
    (run_dir / "config.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    python = os.environ.get("PGOLF_PYTHON", "python")
    cmd = [
        python,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={effective_nproc}",
        str(train_script_path),
    ]
    return JobSpec(
        slot_id=slot_id,
        pack_name=pack_name,
        phase=phase.name,
        gpu_spec=gpu_spec,
        nproc_per_slot=effective_nproc,
        run_dir=run_dir,
        log_path=log_path,
        cmd=cmd,
        env=env,
        metadata=metadata,
    )


def parse_float(pattern: str, text: str) -> float | None:
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    if not matches:
        return None
    return float(matches[-1])


def parse_int(pattern: str, text: str) -> int | None:
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    if not matches:
        return None
    return int(matches[-1])


def parse_metrics(log_path: Path) -> dict[str, Any]:
    if not log_path.exists():
        return {
            "returncode": None,
            "steps": None,
            "step_avg_ms": None,
            "pre_quant_bpb": None,
            "post_quant_bpb": None,
            "ttt_lora_bpb": None,
            "sliding_bpb": None,
            "doc_sliding_bpb": None,
            "submission_size_bytes": None,
            "score_bpb": None,
        }
    text = log_path.read_text(encoding="utf-8", errors="replace")
    pre_quant_matches = re.findall(r"step:\d+/\d+ val_loss:[0-9.]+ val_bpb:([0-9.]+)", text)
    pre_quant_bpb = float(pre_quant_matches[-1]) if pre_quant_matches else None
    post_quant_bpb = parse_float(
        r"final_(?:int8_zlib|int6_lzma)_roundtrip_exact val_loss:[0-9.]+ val_bpb:([0-9.]+)", text
    )
    ttt_lora_bpb = parse_float(r"final_int8_ttt_lora val_loss:[0-9.]+ val_bpb:([0-9.]+)", text)
    sliding_bpb = parse_float(
        r"final_(?:int8_zlib|int6_lzma)_roundtrip_sliding(?:_exact)? val_loss:[0-9.]+ val_bpb:([0-9.]+)",
        text,
    )
    doc_sliding_bpb = parse_float(
        r"final_(?:int8_zlib|int6_lzma)_roundtrip_doc_sliding(?:_exact)? val_loss:[0-9.]+ val_bpb:([0-9.]+)",
        text,
    )
    score_bpb = next(
        (
            value
            for value in (doc_sliding_bpb, sliding_bpb, ttt_lora_bpb, post_quant_bpb, pre_quant_bpb)
            if value is not None
        ),
        None,
    )
    return {
        "steps": parse_int(r"step:(\d+)/", text),
        "step_avg_ms": parse_float(r"step_avg:([0-9.]+)ms", text),
        "pre_quant_bpb": pre_quant_bpb,
        "post_quant_bpb": post_quant_bpb,
        "ttt_lora_bpb": ttt_lora_bpb,
        "sliding_bpb": sliding_bpb,
        "doc_sliding_bpb": doc_sliding_bpb,
        "submission_size_bytes": parse_int(r"Total submission size int8\+zlib: (\d+) bytes", text),
        "score_bpb": score_bpb,
    }


def quant_gap(metrics: dict[str, Any]) -> float | None:
    pre = metrics.get("pre_quant_bpb")
    post = metrics.get("post_quant_bpb")
    if pre is None or post is None:
        return None
    return post - pre


def delta(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline is None:
        return None
    return value - baseline


def delta_int(value: int | None, baseline: int | None) -> int | None:
    if value is None or baseline is None:
        return None
    return value - baseline


def benchmark_gap(metrics: dict[str, Any], benchmarks: dict[str, Any], key: str) -> float | None:
    benchmark_value = benchmarks.get(key)
    if benchmark_value is None:
        return None
    return delta(metrics.get("score_bpb"), benchmark_value)


def comparison_entry(
    slot_id: str,
    slot_map: dict[str, dict[str, Any]],
    phase_results: dict[str, dict[str, Any]],
    benchmarks: dict[str, Any],
) -> dict[str, Any] | None:
    slot = slot_map[slot_id]
    compare_to = slot.get("compare_to")
    if not compare_to:
        return None
    control = phase_results.get(compare_to)
    candidate = phase_results.get(slot_id)
    if control is None or candidate is None:
        return None
    candidate_gap = quant_gap(candidate)
    control_gap = quant_gap(control)
    return {
        "slot": slot_id,
        "name": slot["name"],
        "family": slot["family"],
        "compare_to": compare_to,
        "delta_score_bpb": delta(candidate.get("score_bpb"), control.get("score_bpb")),
        "delta_pre_quant_bpb": delta(candidate.get("pre_quant_bpb"), control.get("pre_quant_bpb")),
        "delta_post_quant_bpb": delta(candidate.get("post_quant_bpb"), control.get("post_quant_bpb")),
        "delta_ttt_lora_bpb": delta(candidate.get("ttt_lora_bpb"), control.get("ttt_lora_bpb")),
        "delta_sliding_bpb": delta(candidate.get("sliding_bpb"), control.get("sliding_bpb")),
        "delta_doc_sliding_bpb": delta(candidate.get("doc_sliding_bpb"), control.get("doc_sliding_bpb")),
        "delta_quant_gap": delta(candidate_gap, control_gap),
        "delta_step_avg_ms": delta(candidate.get("step_avg_ms"), control.get("step_avg_ms")),
        "delta_steps": delta_int(candidate.get("steps"), control.get("steps")),
        "delta_submission_size_bytes": delta_int(
            candidate.get("submission_size_bytes"), control.get("submission_size_bytes")
        ),
        "candidate_gap_to_ours": benchmark_gap(candidate, benchmarks, "ours_bpb"),
        "candidate_gap_to_merged_sota": benchmark_gap(candidate, benchmarks, "merged_sota_bpb"),
        "candidate_gap_to_frontier": benchmark_gap(candidate, benchmarks, "frontier_no_ttt_bpb"),
    }


def absolute_result_sort_key(metrics: dict[str, Any]) -> tuple[float, float, float, float, float]:
    return (
        metrics.get("score_bpb") if metrics.get("score_bpb") is not None else float("inf"),
        metrics.get("post_quant_bpb") if metrics.get("post_quant_bpb") is not None else float("inf"),
        metrics.get("pre_quant_bpb") if metrics.get("pre_quant_bpb") is not None else float("inf"),
        metrics.get("step_avg_ms") if metrics.get("step_avg_ms") is not None else float("inf"),
        -(float(metrics.get("steps")) if metrics.get("steps") is not None else -10**9),
    )


def absolute_result_entry(
    slot_id: str,
    slot_map: dict[str, dict[str, Any]],
    metrics: dict[str, Any],
    benchmarks: dict[str, Any],
) -> dict[str, Any]:
    slot = slot_map[slot_id]
    return {
        "slot": slot_id,
        "name": slot["name"],
        "role": slot["role"],
        "family": slot["family"],
        "runner_mode": slot.get("runner_mode", "train"),
        "patches": resolve_patch_names(slot),
        "score_bpb": metrics.get("score_bpb"),
        "pre_quant_bpb": metrics.get("pre_quant_bpb"),
        "post_quant_bpb": metrics.get("post_quant_bpb"),
        "ttt_lora_bpb": metrics.get("ttt_lora_bpb"),
        "sliding_bpb": metrics.get("sliding_bpb"),
        "doc_sliding_bpb": metrics.get("doc_sliding_bpb"),
        "step_avg_ms": metrics.get("step_avg_ms"),
        "steps": metrics.get("steps"),
        "submission_size_bytes": metrics.get("submission_size_bytes"),
        "gap_to_ours": benchmark_gap(metrics, benchmarks, "ours_bpb"),
        "gap_to_merged_sota": benchmark_gap(metrics, benchmarks, "merged_sota_bpb"),
        "gap_to_frontier_no_ttt": benchmark_gap(metrics, benchmarks, "frontier_no_ttt_bpb"),
        "gap_to_frontier_ttt": benchmark_gap(metrics, benchmarks, "frontier_ttt_bpb"),
    }


def best_result_from_phase_results(
    phase_results: dict[str, dict[str, Any]],
    slot_map: dict[str, dict[str, Any]],
    benchmarks: dict[str, Any],
) -> dict[str, Any] | None:
    ranked = sorted(phase_results.items(), key=lambda item: absolute_result_sort_key(item[1]))
    if not ranked:
        return None
    slot_id, metrics = ranked[0]
    return absolute_result_entry(slot_id, slot_map, metrics, benchmarks)


def comparison_sort_key(entry: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        entry["delta_score_bpb"] if entry["delta_score_bpb"] is not None else float("inf"),
        entry["delta_post_quant_bpb"] if entry["delta_post_quant_bpb"] is not None else float("inf"),
        entry["delta_pre_quant_bpb"] if entry["delta_pre_quant_bpb"] is not None else float("inf"),
        entry["delta_step_avg_ms"] if entry["delta_step_avg_ms"] is not None else float("inf"),
    )


def is_promotable(entry: dict[str, Any]) -> bool:
    return any(
        value is not None and value < 0
        for value in (
            entry["delta_score_bpb"],
            entry["delta_post_quant_bpb"],
            entry["delta_pre_quant_bpb"],
        )
    )


def recommend_promotions(
    phase_results: dict[str, dict[str, Any]],
    slot_map: dict[str, dict[str, Any]],
    benchmarks: dict[str, Any],
    top_k: int,
) -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    for slot_id in phase_results:
        slot = slot_map[slot_id]
        if slot["role"] != "candidate":
            continue
        entry = comparison_entry(slot_id, slot_map, phase_results, benchmarks)
        if entry is not None:
            comparisons.append(entry)
    comparisons.sort(key=comparison_sort_key)
    survivors = [entry for entry in comparisons if is_promotable(entry)]
    if not survivors:
        survivors = comparisons[:1]
    return survivors[:top_k]


def pack_slots(config: dict[str, Any], pack_name: str) -> list[str]:
    return list(config["packs"][pack_name])


def pack_readiness(config: dict[str, Any], slot_map: dict[str, dict[str, Any]], pack_name: str) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for slot_id in pack_slots(config, pack_name):
        slot = slot_map[slot_id]
        reason = runnable_reason(slot)
        entries.append(
            {
                "slot": slot_id,
                "name": slot["name"],
                "role": slot["role"],
                "runner_mode": slot.get("runner_mode", "train"),
                "implementation_state": slot["implementation_state"],
                "compare_to": slot.get("compare_to"),
                "parent": slot.get("parent"),
                "runnable": reason is None,
                "blocked_reason": reason,
            }
        )
    runnable = [entry["slot"] for entry in entries if entry["runnable"]]
    blocked = [entry for entry in entries if not entry["runnable"]]
    return {
        "pack": pack_name,
        "total_slots": len(entries),
        "runnable_slots": runnable,
        "blocked_slots": blocked,
        "entries": entries,
    }


def write_phase_summary(
    scope_root_dir: Path,
    phase_name: str,
    pack_name: str,
    phase_results: dict[str, dict[str, Any]],
    slot_map: dict[str, dict[str, Any]],
    benchmarks: dict[str, Any],
    top_k: int,
) -> Path:
    comparisons: list[dict[str, Any]] = []
    for slot_id in phase_results:
        slot = slot_map[slot_id]
        if slot["role"] != "candidate":
            continue
        entry = comparison_entry(slot_id, slot_map, phase_results, benchmarks)
        if entry is not None:
            comparisons.append(entry)
    comparisons.sort(key=comparison_sort_key)

    summary = {
        "pack": pack_name,
        "phase": phase_name,
        "results": phase_results,
        "comparisons": comparisons,
        "best_result": best_result_from_phase_results(phase_results, slot_map, benchmarks),
        "recommended_promotions": recommend_promotions(phase_results, slot_map, benchmarks, top_k),
    }
    summary_path = scope_root_dir / phase_name / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary_path


def print_plan(config: dict[str, Any], slot_map: dict[str, dict[str, Any]], pack_name: str) -> None:
    payload = {
        "batch_id": config["batch_id"],
        "description": config["description"],
        "default_pack": config.get("default_pack"),
        "selected_pack": pack_name,
        "supported_runner_modes": sorted(SUPPORTED_RUNNER_MODES),
        "phases": config["phases"],
        "packs": {
            name: {
                "slots": pack_slots(config, name),
                "readiness": pack_readiness(config, slot_map, name),
            }
            for name in config["packs"]
        },
    }
    print(json.dumps(payload, indent=2))


def partition_gpus(gpus: list[str], slot_ids: list[str], min_nproc_per_slot: int = 1) -> list[tuple[str, str, int]]:
    if not slot_ids:
        raise SystemExit("At least one slot is required to partition GPUs.")
    if len(slot_ids) > len(gpus):
        raise SystemExit(f"Need at least {len(slot_ids)} GPUs for {len(slot_ids)} slots, only {len(gpus)} provided.")
    total = len(gpus)
    slot_count = len(slot_ids)
    base = total // slot_count
    extra = total % slot_count
    if base < min_nproc_per_slot:
        raise SystemExit(
            f"Cannot allocate at least {min_nproc_per_slot} GPU(s) to each of {slot_count} slots with only {total} GPUs."
        )
    allocations: list[tuple[str, str, int]] = []
    offset = 0
    for index, slot_id in enumerate(slot_ids):
        width = base + (1 if index < extra else 0)
        gpu_group = gpus[offset : offset + width]
        allocations.append((slot_id, ",".join(gpu_group), width))
        offset += width
    return allocations


def run_partitioned_phase(
    script_path: Path,
    config_path: Path,
    config: dict[str, Any],
    phase: PhaseSpec,
    label: str,
    pack_name: str,
    scope_name: str,
    slot_ids: list[str],
    gpus: list[str],
    dry_run: bool,
) -> dict[str, dict[str, Any]]:
    allocations = partition_gpus(gpus, slot_ids, min_nproc_per_slot=phase.nproc_per_slot)
    jobs = [
        build_job(
            script_path,
            config_path,
            config,
            slot_id,
            phase,
            label,
            pack_name,
            scope_name,
            gpu_spec,
            nproc_per_slot=nproc_per_slot,
        )
        for slot_id, gpu_spec, nproc_per_slot in allocations
    ]
    if dry_run:
        print(
            json.dumps(
                [job.metadata | {"cmd": job.cmd, "cwd": str(job.run_dir), "log_path": str(job.log_path)} for job in jobs],
                indent=2,
            )
        )
        return {}

    processes: list[tuple[JobSpec, subprocess.Popen[str], Any]] = []
    for job in jobs:
        print(
            f"[launch] pack={job.pack_name} phase={phase.name} gpu={job.gpu_spec} slot={job.slot_id} log={job.log_path}",
            flush=True,
        )
        log_handle = open(job.log_path, "w", encoding="utf-8")
        process = subprocess.Popen(
            job.cmd,
            cwd=job.run_dir,
            env=job.env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        processes.append((job, process, log_handle))

    results: dict[str, dict[str, Any]] = {}
    for job, process, log_handle in processes:
        returncode = process.wait()
        log_handle.close()
        metrics = parse_metrics(job.log_path)
        metrics["returncode"] = returncode
        results[job.slot_id] = metrics
        print(
            f"[done] pack={job.pack_name} phase={phase.name} slot={job.slot_id} rc={returncode} "
            f"score_bpb={metrics.get('score_bpb')} post_quant_bpb={metrics.get('post_quant_bpb')} "
            f"step_avg_ms={metrics.get('step_avg_ms')}",
            flush=True,
        )
    return results


def run_serial_phase(
    script_path: Path,
    config_path: Path,
    config: dict[str, Any],
    phase: PhaseSpec,
    label: str,
    pack_name: str,
    scope_name: str,
    slot_ids: list[str],
    gpu_spec: str,
    dry_run: bool,
) -> dict[str, dict[str, Any]]:
    jobs = [
        build_job(script_path, config_path, config, slot_id, phase, label, pack_name, scope_name, gpu_spec)
        for slot_id in slot_ids
    ]
    if dry_run:
        print(
            json.dumps(
                [job.metadata | {"cmd": job.cmd, "cwd": str(job.run_dir), "log_path": str(job.log_path)} for job in jobs],
                indent=2,
            )
        )
        return {}
    results: dict[str, dict[str, Any]] = {}
    for job in jobs:
        print(
            f"[launch] pack={job.pack_name} phase={phase.name} gpu={job.gpu_spec} slot={job.slot_id} log={job.log_path}",
            flush=True,
        )
        with open(job.log_path, "w", encoding="utf-8") as log_handle:
            completed = subprocess.run(
                job.cmd,
                cwd=job.run_dir,
                env=job.env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        metrics = parse_metrics(job.log_path)
        metrics["returncode"] = completed.returncode
        results[job.slot_id] = metrics
        print(
            f"[done] pack={job.pack_name} phase={phase.name} slot={job.slot_id} rc={completed.returncode} "
            f"score_bpb={metrics.get('score_bpb')} post_quant_bpb={metrics.get('post_quant_bpb')} "
            f"step_avg_ms={metrics.get('step_avg_ms')}",
            flush=True,
        )
    return results


def should_run_champion(args: argparse.Namespace) -> bool:
    if args.skip_champion:
        return False
    if args.run_champion:
        return True
    return args.top_k > 1


def phase_like(base: PhaseSpec, name: str) -> PhaseSpec:
    return PhaseSpec(name=name, max_wallclock_seconds=base.max_wallclock_seconds, nproc_per_slot=base.nproc_per_slot)


def load_summary(summary_path: Path) -> dict[str, Any]:
    return json.loads(summary_path.read_text(encoding="utf-8"))


def pick_pack_winner(summary_path: Path) -> str:
    summary = load_summary(summary_path)
    promotions = summary.get("recommended_promotions", [])
    if not promotions:
        raise SystemExit(f"No recommended promotions found in {summary_path}")
    return promotions[0]["slot"]


def best_finalist_from_summary(summary_path: Path) -> dict[str, Any]:
    summary = load_summary(summary_path)
    comparisons = summary.get("comparisons", [])
    if not comparisons:
        raise SystemExit(f"No comparisons found in finalist summary {summary_path}")
    return comparisons[0]


def best_result_from_summary(summary_path: Path) -> dict[str, Any]:
    summary = load_summary(summary_path)
    best_result = summary.get("best_result")
    if best_result is None:
        raise SystemExit(f"No best_result found in summary {summary_path}")
    return best_result


def assess_result_vs_targets(
    result: dict[str, Any],
    benchmarks: dict[str, Any],
    reference_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    score = result.get("score_bpb")
    targets: list[dict[str, Any]] = []
    for key, label in (
        ("ours_bpb", "ours"),
        ("merged_sota_bpb", "merged_sota"),
        ("frontier_no_ttt_bpb", "frontier_no_ttt"),
        ("frontier_ttt_bpb", "frontier_ttt"),
    ):
        target_value = benchmarks.get(key)
        targets.append(
            {
                "label": label,
                "target_bpb": target_value,
                "gap_bpb": delta(score, target_value),
                "beaten": None if score is None or target_value is None else score < target_value,
            }
        )
    if reference_result is not None:
        reference_score = reference_result.get("score_bpb")
        targets.append(
            {
                "label": "local_reference_best",
                "target_bpb": reference_score,
                "gap_bpb": delta(score, reference_score),
                "beaten": None if score is None or reference_score is None else score < reference_score,
            }
        )
    return {
        "score_bpb": score,
        "distance_to_frontier_no_ttt_bpb": delta(score, benchmarks.get("frontier_no_ttt_bpb")),
        "beats_frontier_no_ttt": None
        if score is None or benchmarks.get("frontier_no_ttt_bpb") is None
        else score < benchmarks["frontier_no_ttt_bpb"],
        "targets": targets,
    }


def merged_env_snapshot(
    slot_id: str, slot_map: dict[str, dict[str, Any]], defaults: dict[str, str]
) -> dict[str, str]:
    merged_env, _ = merge_env_for_slot(slot_id, slot_map, defaults)
    return dict(merged_env)


def merged_patch_names(slot_ids: list[str], slot_map: dict[str, dict[str, Any]]) -> list[str]:
    patch_names: list[str] = []
    for slot_id in slot_ids:
        for patch_name in resolve_patch_names(slot_map[slot_id]):
            if patch_name not in patch_names:
                patch_names.append(patch_name)
    return patch_names


def synthetic_slot_from_source(
    new_slot_id: str,
    source_slot_id: str,
    slot_map: dict[str, dict[str, Any]],
    defaults: dict[str, str],
    *,
    compare_to: str | None,
    name: str | None = None,
    role: str | None = None,
    family: str | None = None,
    notes_suffix: list[str] | None = None,
) -> dict[str, Any]:
    source = slot_map[source_slot_id]
    merged_env = merged_env_snapshot(source_slot_id, slot_map, defaults)
    notes = list(source.get("notes", []))
    notes.append(f"synthetic_from={source_slot_id}")
    if notes_suffix:
        notes.extend(notes_suffix)
    return {
        "slot": new_slot_id,
        "name": name or source["name"],
        "role": role or source["role"],
        "runner_mode": source.get("runner_mode", "train"),
        "implementation_state": "ready",
        "patches": resolve_patch_names(source),
        "parent": None,
        "compare_to": compare_to,
        "family": family or source["family"],
        "why": source["why"],
        "validates": source["validates"],
        "falsifies": source["falsifies"],
        "env": merged_env,
        "notes": notes,
    }


def synthetic_composite_slot(
    new_slot_id: str,
    source_slot_ids: list[str],
    slot_map: dict[str, dict[str, Any]],
    defaults: dict[str, str],
    *,
    base_control_slot: str,
    compare_to: str,
    name: str,
    family: str,
    why: str,
    validates: str,
    falsifies: str,
) -> dict[str, Any]:
    merged_env = merged_env_snapshot(base_control_slot, slot_map, defaults)
    for slot_id in source_slot_ids:
        merged_env.update(merged_env_snapshot(slot_id, slot_map, defaults))
    return {
        "slot": new_slot_id,
        "name": name,
        "role": "candidate",
        "runner_mode": "train",
        "implementation_state": "ready",
        "patches": merged_patch_names(source_slot_ids, slot_map),
        "parent": None,
        "compare_to": compare_to,
        "family": family,
        "why": why,
        "validates": validates,
        "falsifies": falsifies,
        "env": merged_env,
        "notes": [f"synthetic_composite={'+'.join(source_slot_ids)}"],
    }


def config_with_slots_and_pack(
    config: dict[str, Any],
    pack_name: str,
    slots: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    augmented = json.loads(json.dumps(config))
    augmented["packs"][pack_name] = [slot["slot"] for slot in slots]
    existing_ids = {slot["slot"] for slot in slots}
    augmented["slots"] = [slot for slot in augmented["slots"] if slot["slot"] not in existing_ids]
    augmented["slots"].extend(slots)
    return augmented, build_slot_map(augmented)


def build_tournament_finalist_pack(
    config: dict[str, Any],
    slot_map: dict[str, dict[str, Any]],
    winners_by_pack: dict[str, str],
    settings: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    defaults = config["defaults"]
    control_slots = settings["control_slots"]
    export_slots = settings["export_slots"]
    final_pack = settings["final_pack"]
    primary_packs = settings["primary_packs"]

    slots = [
        synthetic_slot_from_source(
            "T0",
            control_slots[0],
            slot_map,
            defaults,
            compare_to=None,
            name="tournament_control_a",
            role="control",
            family="tournament_control",
            notes_suffix=["stage=tournament_finalists"],
        ),
        synthetic_slot_from_source(
            "T1",
            control_slots[1],
            slot_map,
            defaults,
            compare_to="T0",
            name="tournament_control_b",
            role="control",
            family="tournament_control_repeat",
            notes_suffix=["stage=tournament_finalists"],
        ),
    ]
    for index, pack_name in enumerate(primary_packs, start=2):
        slots.append(
            synthetic_slot_from_source(
                f"T{index}",
                winners_by_pack[pack_name],
                slot_map,
                defaults,
                compare_to="T0",
                name=f"winner_{pack_name}",
                family="pack_winner",
                notes_suffix=[f"winner_of={pack_name}"],
            )
        )

    export_index = 2 + len(primary_packs)
    slots.append(
        synthetic_slot_from_source(
            f"T{export_index}",
            export_slots[0],
            slot_map,
            defaults,
            compare_to="T0",
            name="winner_full_gptq",
            family="export_winner",
            notes_suffix=["winner_of=deferred_export_baseline"],
        )
    )
    slots.append(
        synthetic_slot_from_source(
            f"T{export_index + 1}",
            export_slots[1],
            slot_map,
            defaults,
            compare_to=f"T{export_index}",
            name="winner_full_gptq_plus_ema",
            family="export_winner_child",
            notes_suffix=["winner_of=deferred_export_child"],
        )
    )

    mix_sources = [export_slots[0], winners_by_pack[primary_packs[0]]]
    slots.append(
        synthetic_composite_slot(
            f"T{export_index + 2}",
            mix_sources,
            slot_map,
            defaults,
            base_control_slot=control_slots[0],
            compare_to="T0",
            name=f"mix_full_gptq_{primary_packs[0]}",
            family="tournament_mix",
            why="Tests whether the strongest export patch and the highest-priority lead pack reinforce each other.",
            validates="Export alignment and the new lead hailmary lane are complementary.",
            falsifies="One lane dominates enough that the composition is not worth the complexity.",
        )
    )
    return config_with_slots_and_pack(config, final_pack, slots)


def tournament_stage_plan(config: dict[str, Any], slot_map: dict[str, dict[str, Any]], label: str) -> dict[str, Any]:
    settings = tournament_settings(config)
    sanity = phase_from_config(config, "sanity")
    screen = phase_from_config(config, "screen")
    champion = phase_from_config(config, "champion_8x")
    plan: list[dict[str, Any]] = []
    if REFERENCE_PACK in config["packs"]:
        reference_sanity_slots = phase_slot_defaults(sanity, config, REFERENCE_PACK)
        reference_screen_slots = phase_slot_defaults(screen, config, REFERENCE_PACK)
        ensure_runnable(reference_sanity_slots, slot_map)
        ensure_runnable(reference_screen_slots, slot_map)
        plan.append(
            {
                "reference_pack": REFERENCE_PACK,
                "sanity_phase": phase_like(sanity, f"sanity_{REFERENCE_PACK}").name,
                "screen_phase": phase_like(screen, f"screen_{REFERENCE_PACK}").name,
                "sanity_slots": reference_sanity_slots,
                "screen_slots": reference_screen_slots,
                "note": "Reference anchor pack. Not used for finalist promotion.",
            }
        )
    for pack_name in settings["primary_packs"]:
        sanity_slots = phase_slot_defaults(sanity, config, pack_name)
        screen_slots = phase_slot_defaults(screen, config, pack_name)
        ensure_runnable(sanity_slots, slot_map)
        ensure_runnable(screen_slots, slot_map)
        plan.append(
            {
                "pack": pack_name,
                "sanity_phase": phase_like(sanity, f"sanity_{pack_name}").name,
                "screen_phase": phase_like(screen, f"screen_{pack_name}").name,
                "sanity_slots": sanity_slots,
                "screen_slots": screen_slots,
                "promotion_rule": "top candidate relative to matched control",
            }
        )
    for pack_name in settings["support_packs"]:
        sanity_slots = phase_slot_defaults(sanity, config, pack_name)
        screen_slots = phase_slot_defaults(screen, config, pack_name)
        ensure_runnable(sanity_slots, slot_map)
        ensure_runnable(screen_slots, slot_map)
        plan.append(
            {
                "pack": pack_name,
                "sanity_phase": phase_like(sanity, f"sanity_{pack_name}").name,
                "screen_phase": phase_like(screen, f"screen_{pack_name}").name,
                "sanity_slots": sanity_slots,
                "screen_slots": screen_slots,
                "promotion_rule": "support_pack_only_no_direct_finalist_promotion",
            }
        )
    plan.append(
        {
            "finalist_wave": {
                "pack": settings["final_pack"],
                "sanity_phase": "sanity_finalists",
                "long_phase": "finalists_long",
                "shape": ["T0 control", "T1 control_repeat"]
                + [f"T{index} winner_{pack_name}" for index, pack_name in enumerate(settings["primary_packs"], start=2)]
                + [f"T{2 + len(settings['primary_packs'])} winner_full_gptq"]
                + [f"T{3 + len(settings['primary_packs'])} winner_full_gptq_plus_ema"]
                + [f"T{4 + len(settings['primary_packs'])} mix_full_gptq_{settings['primary_packs'][0]}"],
                "promotion_rule": "best finalist after 600s on 1xH100",
            },
            "champion_8x": {
                "phase": champion.name,
                "nproc_per_slot": champion.nproc_per_slot,
                "max_wallclock_seconds": champion.max_wallclock_seconds,
                "note": "Single best finalist gets the full 8xH100 run.",
            },
        }
    )
    return {"mode": "tournament", "label": label, "scope": "tournament", "stages": plan}


def run_pack_screen(
    script_path: Path,
    config_path: Path,
    config: dict[str, Any],
    slot_map: dict[str, dict[str, Any]],
    label: str,
    pack_name: str,
    gpus: list[str],
    top_k: int,
) -> tuple[str, Path]:
    scope_name = "tournament"
    scope_root_dir = scope_root(config_path, label, scope_name)
    sanity = phase_like(phase_from_config(config, "sanity"), f"sanity_{pack_name}")
    screen = phase_like(phase_from_config(config, "screen"), f"screen_{pack_name}")
    sanity_slots = phase_slot_defaults(phase_from_config(config, "sanity"), config, pack_name)
    screen_slots = phase_slot_defaults(phase_from_config(config, "screen"), config, pack_name)
    ensure_runnable(sanity_slots, slot_map)
    ensure_runnable(screen_slots, slot_map)

    print(f"[tournament] pack={pack_name} sanity slots={','.join(sanity_slots)}", flush=True)
    sanity_results = run_partitioned_phase(
        script_path, config_path, config, sanity, label, pack_name, scope_name, sanity_slots, gpus, False
    )
    sanity_summary = write_phase_summary(
        scope_root_dir, sanity.name, pack_name, sanity_results, slot_map, config["benchmarks"], top_k
    )
    print(f"[summary] {sanity_summary}", flush=True)
    if any(result.get("returncode") not in (0, None) for result in sanity_results.values()):
        raise SystemExit(f"Tournament sanity failed for pack {pack_name}; refusing to continue.")

    print(f"[tournament] pack={pack_name} screen slots={','.join(screen_slots)}", flush=True)
    screen_results = run_partitioned_phase(
        script_path, config_path, config, screen, label, pack_name, scope_name, screen_slots, gpus, False
    )
    screen_summary = write_phase_summary(
        scope_root_dir, screen.name, pack_name, screen_results, slot_map, config["benchmarks"], top_k
    )
    print(f"[summary] {screen_summary}", flush=True)
    return pick_pack_winner(screen_summary), screen_summary


def run_tournament(
    script_path: Path,
    config_path: Path,
    config: dict[str, Any],
    slot_map: dict[str, dict[str, Any]],
    args: argparse.Namespace,
    gpus: list[str],
) -> None:
    settings = tournament_settings(config)
    if args.dry_run:
        print(json.dumps(tournament_stage_plan(config, slot_map, args.label), indent=2))
        return

    reference_summary_entry: dict[str, Any] | None = None
    if REFERENCE_PACK in config["packs"]:
        _, reference_screen_summary = run_pack_screen(
            script_path, config_path, config, slot_map, args.label, REFERENCE_PACK, gpus, 1
        )
        reference_summary_entry = {
            "pack": REFERENCE_PACK,
            "summary_path": str(reference_screen_summary),
            "best_result": best_result_from_summary(reference_screen_summary),
        }

    winners_by_pack: dict[str, str] = {}
    primary_results: list[dict[str, Any]] = []
    support_results: list[dict[str, Any]] = []
    for pack_name in settings["primary_packs"]:
        winner_slot, screen_summary = run_pack_screen(
            script_path, config_path, config, slot_map, args.label, pack_name, gpus, args.top_k
        )
        winners_by_pack[pack_name] = winner_slot
        primary_results.append({"pack": pack_name, "winner_slot": winner_slot, "summary_path": str(screen_summary)})
    for pack_name in settings["support_packs"]:
        winner_slot, screen_summary = run_pack_screen(
            script_path, config_path, config, slot_map, args.label, pack_name, gpus, args.top_k
        )
        support_results.append({"pack": pack_name, "winner_slot": winner_slot, "summary_path": str(screen_summary)})

    finalist_config, finalist_slot_map = build_tournament_finalist_pack(config, slot_map, winners_by_pack, settings)
    finalist_scope_name = "tournament"
    finalist_scope_root = scope_root(config_path, args.label, finalist_scope_name)
    finalist_pack = settings["final_pack"]
    finalist_slots = finalist_config["packs"][finalist_pack]
    finalist_sanity = phase_like(phase_from_config(config, "sanity"), "sanity_finalists")
    finalist_long = phase_like(phase_from_config(config, "final_single"), "finalists_long")

    print(f"[tournament] finalists sanity slots={','.join(finalist_slots)}", flush=True)
    finalist_sanity_results = run_partitioned_phase(
        script_path,
        config_path,
        finalist_config,
        finalist_sanity,
        args.label,
        finalist_pack,
        finalist_scope_name,
        finalist_slots,
        gpus,
        False,
    )
    finalist_sanity_summary = write_phase_summary(
        finalist_scope_root,
        finalist_sanity.name,
        finalist_pack,
        finalist_sanity_results,
        finalist_slot_map,
        finalist_config["benchmarks"],
        1,
    )
    print(f"[summary] {finalist_sanity_summary}", flush=True)
    if any(result.get("returncode") not in (0, None) for result in finalist_sanity_results.values()):
        raise SystemExit("Tournament finalist sanity failed; refusing to continue to long finalists.")

    print(f"[tournament] finalists long slots={','.join(finalist_slots)}", flush=True)
    finalist_long_results = run_partitioned_phase(
        script_path,
        config_path,
        finalist_config,
        finalist_long,
        args.label,
        finalist_pack,
        finalist_scope_name,
        finalist_slots,
        gpus,
        False,
    )
    finalist_long_summary = write_phase_summary(
        finalist_scope_root,
        finalist_long.name,
        finalist_pack,
        finalist_long_results,
        finalist_slot_map,
        finalist_config["benchmarks"],
        1,
    )
    print(f"[summary] {finalist_long_summary}", flush=True)
    champion_slot = pick_pack_winner(finalist_long_summary)
    champion_entry = best_finalist_from_summary(finalist_long_summary)
    best_available_result = absolute_result_entry(
        champion_slot,
        finalist_slot_map,
        finalist_long_results[champion_slot],
        config["benchmarks"],
    )

    tournament_summary = {
        "mode": "tournament",
        "label": args.label,
        "reference_pack_result": reference_summary_entry,
        "primary_packs": settings["primary_packs"],
        "support_packs": settings["support_packs"],
        "deferred_rebuild_packs": settings["deferred_rebuild_packs"],
        "primary_results": primary_results,
        "support_results": support_results,
        "winners_by_pack": winners_by_pack,
        "finalist_pack": finalist_pack,
        "finalist_slots": finalist_slots,
        "finalist_summary_path": str(finalist_long_summary),
        "champion_slot": champion_slot,
        "champion_comparison": champion_entry,
        "best_available_result": best_available_result,
        "beat_status": assess_result_vs_targets(
            best_available_result,
            config["benchmarks"],
            None if reference_summary_entry is None else reference_summary_entry["best_result"],
        ),
    }
    tournament_summary_path = finalist_scope_root / "tournament_summary.json"
    tournament_summary_path.parent.mkdir(parents=True, exist_ok=True)
    tournament_summary_path.write_text(json.dumps(tournament_summary, indent=2) + "\n", encoding="utf-8")
    print(f"[summary] {tournament_summary_path}", flush=True)

    if args.skip_champion:
        return

    champion_phase = phase_from_config(config, "champion_8x")
    ensure_runnable([champion_slot], finalist_slot_map)
    print(f"[tournament] champion_8x slot={champion_slot}", flush=True)
    champion_results = run_serial_phase(
        script_path,
        config_path,
        finalist_config,
        champion_phase,
        args.label,
        finalist_pack,
        finalist_scope_name,
        [champion_slot],
        ",".join(gpus[: champion_phase.nproc_per_slot]),
        False,
    )
    champion_summary = write_phase_summary(
        finalist_scope_root,
        champion_phase.name,
        finalist_pack,
        champion_results,
        finalist_slot_map,
        finalist_config["benchmarks"],
        1,
    )
    print(f"[summary] {champion_summary}", flush=True)
    tournament_summary["champion_8x_summary_path"] = str(champion_summary)
    tournament_summary["champion_8x_result"] = best_result_from_summary(champion_summary)
    tournament_summary["beat_status"] = assess_result_vs_targets(
        tournament_summary["champion_8x_result"],
        config["benchmarks"],
        None if reference_summary_entry is None else reference_summary_entry["best_result"],
    )
    tournament_summary_path.write_text(json.dumps(tournament_summary, indent=2) + "\n", encoding="utf-8")
    print(f"[summary] {tournament_summary_path}", flush=True)


def main() -> None:
    args = parse_args()
    script_path = Path(__file__)
    config_path = Path(args.config).resolve() if args.config else script_path.with_name("run_configs.json").resolve()
    config = load_config(config_path)
    slot_map = build_slot_map(config)
    pack_name = resolve_pack_name(args, config)
    scope_name = resolve_scope_name(args, pack_name)
    run_root = config_path.parent / "runs" / args.label
    scope_root_dir = scope_root(config_path, args.label, scope_name)
    gpus = parse_gpu_list(args.gpus)

    if args.phase == "plan":
        print_plan(config, slot_map, pack_name)
        return
    if args.phase == "readiness":
        print(json.dumps(pack_readiness(config, slot_map, pack_name), indent=2))
        return

    if args.phase in {"sanity", "screen"}:
        phase = phase_from_config(config, args.phase)
        slot_ids = resolve_phase_slots(args, config, phase, pack_name)
        ensure_runnable(slot_ids, slot_map)
        results = run_partitioned_phase(
            script_path, config_path, config, phase, args.label, pack_name, scope_name, slot_ids, gpus, args.dry_run
        )
        if not args.dry_run:
            summary_path = write_phase_summary(
                scope_root_dir, phase.name, pack_name, results, slot_map, config["benchmarks"], args.top_k
            )
            print(f"[summary] {summary_path}", flush=True)
        return

    if args.phase in {"final_single", "champion_8x"}:
        phase = phase_from_config(config, args.phase)
        if args.phase == "final_single":
            slot_ids = resolve_final_slots(args, run_root, scope_name, args.top_k)
            ensure_runnable(slot_ids, slot_map)
            results = run_partitioned_phase(
                script_path,
                config_path,
                config,
                phase,
                args.label,
                pack_name,
                scope_name,
                slot_ids,
                gpus,
                args.dry_run,
            )
        else:
            slot_ids = args.final_slots or args.slots
            if not slot_ids:
                slot_ids = resolve_final_slots(args, run_root, scope_name, 1)
            ensure_runnable(list(slot_ids), slot_map)
            results = run_serial_phase(
                script_path,
                config_path,
                config,
                phase,
                args.label,
                pack_name,
                scope_name,
                list(slot_ids),
                ",".join(gpus[: phase.nproc_per_slot]),
                args.dry_run,
            )
        if not args.dry_run:
            summary_path = write_phase_summary(
                scope_root_dir, phase.name, pack_name, results, slot_map, config["benchmarks"], args.top_k
            )
            print(f"[summary] {summary_path}", flush=True)
        return

    if args.phase == "all":
        sanity = phase_from_config(config, "sanity")
        screen = phase_from_config(config, "screen")
        sanity_slots = resolve_phase_slots(args, config, sanity, pack_name)
        screen_slots = resolve_phase_slots(args, config, screen, pack_name)
        ensure_runnable(sanity_slots, slot_map)
        ensure_runnable(screen_slots, slot_map)
        sanity_results = run_partitioned_phase(
            script_path,
            config_path,
            config,
            sanity,
            args.label,
            pack_name,
            scope_name,
            sanity_slots,
            gpus,
            args.dry_run,
        )
        if args.dry_run:
            screen_jobs = []
            for slot_id, gpu_spec, nproc_per_slot in partition_gpus(gpus, screen_slots, min_nproc_per_slot=screen.nproc_per_slot):
                job = build_job(
                    script_path,
                    config_path,
                    config,
                    slot_id,
                    screen,
                    args.label,
                    pack_name,
                    scope_name,
                    gpu_spec,
                    nproc_per_slot=nproc_per_slot,
                )
                screen_jobs.append(job.metadata | {"cmd": job.cmd, "cwd": str(job.run_dir), "log_path": str(job.log_path)})
            print(
                json.dumps(
                    {
                        "pack": pack_name,
                        "next_phase": "screen",
                        "slots": screen_jobs,
                        "auto_promotion": {
                            "final_single_top_k": args.top_k,
                            "champion_8x_after_final_single": should_run_champion(args),
                            "note": "All phases partition visible GPUs across active slots to avoid idle devices.",
                        },
                    },
                    indent=2,
                )
            )
            return
        write_phase_summary(
            scope_root_dir, sanity.name, pack_name, sanity_results, slot_map, config["benchmarks"], args.top_k
        )
        if any(result.get("returncode") not in (0, None) for result in sanity_results.values()):
            raise SystemExit("Sanity phase had failures. Refusing to continue automatically.")
        screen_results = run_partitioned_phase(
            script_path,
            config_path,
            config,
            screen,
            args.label,
            pack_name,
            scope_name,
            screen_slots,
            gpus,
            args.dry_run,
        )
        screen_summary = write_phase_summary(
            scope_root_dir, screen.name, pack_name, screen_results, slot_map, config["benchmarks"], args.top_k
        )
        print(f"[summary] {screen_summary}", flush=True)
        final_slots = resolve_final_slots(args, run_root, scope_name, args.top_k)
        ensure_runnable(final_slots, slot_map)
        final_phase = phase_from_config(config, "final_single")
        final_results = run_partitioned_phase(
            script_path,
            config_path,
            config,
            final_phase,
            args.label,
            pack_name,
            scope_name,
            final_slots,
            gpus,
            args.dry_run,
        )
        final_summary = write_phase_summary(
            scope_root_dir, final_phase.name, pack_name, final_results, slot_map, config["benchmarks"], args.top_k
        )
        print(f"[summary] {final_summary}", flush=True)
        if should_run_champion(args):
            champion_phase = phase_from_config(config, "champion_8x")
            champion_slots = resolve_final_slots(args, run_root, scope_name, 1)
            champion_results = run_serial_phase(
                script_path,
                config_path,
                config,
                champion_phase,
                args.label,
                pack_name,
                scope_name,
                champion_slots,
                ",".join(gpus[: champion_phase.nproc_per_slot]),
                args.dry_run,
            )
            champion_summary = write_phase_summary(
                scope_root_dir,
                champion_phase.name,
                pack_name,
                champion_results,
                slot_map,
                config["benchmarks"],
                1,
            )
            print(f"[summary] {champion_summary}", flush=True)
        return

    if args.phase == "tournament":
        run_tournament(script_path, config_path, config, slot_map, args, gpus)
        return

    raise SystemExit(f"Unsupported phase {args.phase}")


if __name__ == "__main__":
    main()
