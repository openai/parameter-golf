#!/usr/bin/env python3
"""hm2 orchestrator: bootstrap early, hand off late, record the dynamics."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from patches import apply_patches, list_patches


SUPPORTED_RUNNER_MODES = {"train"}


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
    parser = argparse.ArgumentParser(description="hm2 bootstrap-to-handoff tournament orchestrator.")
    parser.add_argument("--config", default=None, help="Path to run_configs.json.")
    parser.add_argument(
        "--phase",
        default="plan",
        choices=["plan", "readiness", "sanity", "screen", "final_single", "champion_8x", "all", "tournament"],
    )
    parser.add_argument("--label", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--pack", default=None, help="Named pack from config. Defaults to config.default_pack.")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--slots", nargs="*")
    parser.add_argument("--final-slots", nargs="*")
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-champion", action="store_true")
    parser.add_argument("--skip-champion", action="store_true")
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


def phase_like(phase: PhaseSpec, name: str) -> PhaseSpec:
    return PhaseSpec(
        name=name,
        max_wallclock_seconds=phase.max_wallclock_seconds,
        nproc_per_slot=phase.nproc_per_slot,
        slots=phase.slots,
    )


def tournament_settings(config: dict[str, Any]) -> dict[str, Any]:
    section = config.get("tournament", {})
    return {
        "primary_packs": list(section.get("primary_packs", ["bootstrap_handoff", "receiver_mix"])),
        "control_slots": tuple(section.get("control_slots", ["B0", "B1"])),
        "final_pack": section.get("final_pack", "tournament_finalists"),
    }


def resolve_pack_name(args: argparse.Namespace, config: dict[str, Any]) -> str:
    pack_name = args.pack or config.get("default_pack", "bootstrap_handoff")
    if pack_name not in config["packs"]:
        raise SystemExit(f"Unknown pack {pack_name!r}. Available packs: {', '.join(sorted(config['packs']))}")
    return pack_name


def phase_slot_defaults(phase: PhaseSpec, config: dict[str, Any], pack_name: str) -> list[str]:
    pack_slots = list(config["packs"][pack_name])
    if phase.slots is None:
        return pack_slots
    allowed = set(phase.slots)
    intersected = [slot_id for slot_id in pack_slots if slot_id in allowed]
    return intersected if intersected else pack_slots


def resolve_phase_slots(args: argparse.Namespace, config: dict[str, Any], phase: PhaseSpec, pack_name: str) -> list[str]:
    if args.slots:
        return list(args.slots)
    slots = phase_slot_defaults(phase, config, pack_name)
    if not slots:
        raise SystemExit(f"No slots resolved for phase={phase.name} pack={pack_name}")
    return slots


def merge_env_for_slot(slot_id: str, slots: dict[str, dict[str, Any]], defaults: dict[str, str]) -> tuple[dict[str, str], list[str]]:
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


def merge_patches_for_slot(slot_id: str, slots: dict[str, dict[str, Any]]) -> list[str]:
    slot = slots[slot_id]
    patches: list[str] = []
    parent = slot.get("parent")
    if parent:
        patches.extend(merge_patches_for_slot(parent, slots))
    patches.extend(slot.get("patches", []))
    deduped: list[str] = []
    seen: set[str] = set()
    for patch_name in patches:
        if patch_name in seen:
            continue
        seen.add(patch_name)
        deduped.append(patch_name)
    return deduped


def runnable_reason(slot_id: str, slot_map: dict[str, dict[str, Any]]) -> str | None:
    slot = slot_map[slot_id]
    if slot["implementation_state"] != "ready":
        return f"implementation_state={slot['implementation_state']}"
    if slot.get("runner_mode", "train") not in SUPPORTED_RUNNER_MODES:
        return f"runner_mode={slot.get('runner_mode')} unsupported"
    for patch_name in merge_patches_for_slot(slot_id, slot_map):
        if patch_name not in list_patches():
            return f"patch={patch_name} unavailable"
    return None


def ensure_runnable(slot_ids: list[str], slot_map: dict[str, dict[str, Any]]) -> None:
    blockers = []
    for slot_id in slot_ids:
        reason = runnable_reason(slot_id, slot_map)
        if reason is not None:
            blockers.append(f"{slot_id}:{reason}")
    if blockers:
        raise SystemExit("Unrunnable slots: " + ", ".join(blockers))


def scope_root(config_path: Path, label: str, scope_name: str) -> Path:
    return config_path.parent / "runs" / label / scope_name


def build_job(
    script_path: Path,
    config_path: Path,
    config: dict[str, Any],
    slot_map: dict[str, dict[str, Any]],
    slot_id: str,
    phase: PhaseSpec,
    label: str,
    pack_name: str,
    scope_name: str,
    gpu_spec: str | None,
    nproc_per_slot: int | None = None,
) -> JobSpec:
    slot = slot_map[slot_id]
    merged_env, lineage = merge_env_for_slot(slot_id, slot_map, config["defaults"])
    patch_names = merge_patches_for_slot(slot_id, slot_map)

    env = os.environ.copy()
    env.update(merged_env)
    env["MAX_WALLCLOCK_SECONDS"] = str(phase.max_wallclock_seconds)
    env["RUN_ID"] = f"{config['batch_id']}_{label}_{scope_name}_{phase.name}_{slot_id}"
    env["PGOLF_STAGE"] = "hm2"
    env["PGOLF_PHASE"] = phase.name
    env["PGOLF_SLOT"] = slot_id
    env["PYTHONUNBUFFERED"] = "1"

    pgolf_root = script_path.resolve().parent.parent
    if env.get("DATA_PATH", "").startswith("./"):
        env["DATA_PATH"] = str((pgolf_root / env["DATA_PATH"][2:]).resolve())
    if env.get("TOKENIZER_PATH", "").startswith("./"):
        env["TOKENIZER_PATH"] = str((pgolf_root / env["TOKENIZER_PATH"][2:]).resolve())
    if gpu_spec is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu_spec

    nproc = nproc_per_slot if nproc_per_slot is not None else phase.nproc_per_slot
    run_dir = scope_root(config_path, label, scope_name) / phase.name / f"{slot_id}_{slot['name']}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.log"

    base_variants = config.get(
        "base_variants",
        {
            "current_local": "base_train_gpt.py",
        },
    )
    base_variant = env.get("HM2_BASE_VARIANT", config.get("default_base_variant", "current_local"))
    if base_variant not in base_variants:
        raise SystemExit(f"Unknown HM2 base variant {base_variant!r}. Available: {', '.join(sorted(base_variants))}")
    base_source = (script_path.parent / base_variants[base_variant]).read_text(encoding="utf-8")
    patched_source = apply_patches(base_source, patch_names) if patch_names else base_source
    patched_script_path = run_dir / "train_gpt.py"
    patched_script_path.write_text(patched_source, encoding="utf-8")

    metadata = {
        "batch_id": config["batch_id"],
        "label": label,
        "scope": scope_name,
        "pack": pack_name,
        "phase": phase.name,
        "slot": slot_id,
        "name": slot["name"],
        "role": slot["role"],
        "family": slot["family"],
        "lane": slot.get("lane", pack_name),
        "why": slot["why"],
        "validates": slot["validates"],
        "falsifies": slot["falsifies"],
        "compare_to": slot.get("compare_to"),
        "lineage": lineage,
        "patches": patch_names,
        "base_variant": base_variant,
        "env": {k: env[k] for k in sorted(merged_env)},
    }
    (run_dir / "config.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    python = sys.executable
    cmd = [python, str(patched_script_path.resolve())] if nproc == 1 else [
        python,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={nproc}",
        str(patched_script_path.resolve()),
    ]
    return JobSpec(
        slot_id=slot_id,
        pack_name=pack_name,
        phase=phase.name,
        gpu_spec=gpu_spec,
        nproc_per_slot=nproc,
        run_dir=run_dir,
        log_path=log_path,
        cmd=cmd,
        env=env,
        metadata=metadata,
    )


def parse_float(pattern: str, text: str) -> float | None:
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    return float(matches[-1]) if matches else None


def parse_int(pattern: str, text: str) -> int | None:
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    return int(matches[-1]) if matches else None


def _bucket_metric(diag: dict[str, Any] | None, bucket: str, key: str) -> float | None:
    if diag is None:
        return None
    return diag.get("buckets", {}).get(bucket, {}).get(key)


def parse_metrics(log_path: Path) -> dict[str, Any]:
    if not log_path.exists():
        return {
            "returncode": None,
            "steps": None,
            "step_avg_ms": None,
            "pre_quant_bpb": None,
            "post_quant_bpb": None,
            "submission_size_bytes": None,
            "phase_diagnostics": None,
        }
    text = log_path.read_text(encoding="utf-8", errors="replace")
    pre_quant_matches = re.findall(r"step:\d+/\d+ val_loss:[0-9.]+ val_bpb:([0-9.]+)", text)
    pre_quant_bpb = float(pre_quant_matches[-1]) if pre_quant_matches else None
    post_quant_bpb = parse_float(r"final_int8_zlib_roundtrip_exact val_loss:[0-9.]+ val_bpb:([0-9.]+)", text)
    diag_path = log_path.parent / "phase_diagnostics.json"
    phase_diag = json.loads(diag_path.read_text(encoding="utf-8")) if diag_path.exists() else None
    return {
        "steps": parse_int(r"step:(\d+)/", text),
        "step_avg_ms": parse_float(r"step_avg:([0-9.]+)ms", text),
        "pre_quant_bpb": pre_quant_bpb,
        "post_quant_bpb": post_quant_bpb,
        "submission_size_bytes": parse_int(r"Total submission size int[68]\+(?:zstd|zlib): (\d+) bytes", text),
        "phase_diagnostics": phase_diag,
        "early_loss_delta": _bucket_metric(phase_diag, "early", "loss_delta"),
        "mid_loss_delta": _bucket_metric(phase_diag, "mid", "loss_delta"),
        "late_loss_delta": _bucket_metric(phase_diag, "late", "loss_delta"),
    }


def fmt_metric(value: Any) -> str:
    if value is None:
        return "?"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def delta(a: float | None, b: float | None) -> float | None:
    return a - b if a is not None and b is not None else None


def delta_int(a: int | None, b: int | None) -> int | None:
    return a - b if a is not None and b is not None else None


def quant_gap(m: dict[str, Any]) -> float | None:
    return delta(m.get("post_quant_bpb"), m.get("pre_quant_bpb"))


def phase_signature(metrics: dict[str, Any]) -> str:
    early = metrics.get("early_loss_delta")
    mid = metrics.get("mid_loss_delta")
    late = metrics.get("late_loss_delta")
    if early is None:
        return "unknown"
    if late is not None and early > 0 and late <= max(0.20 * early, 0.01):
        return "front_loaded"
    if late is not None and late > early:
        return "late_improver"
    if mid is not None and early is not None and late is not None:
        return "balanced"
    return "partial"


def comparison_entry(slot_id: str, slot_map: dict[str, dict[str, Any]], phase_results: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    slot = slot_map[slot_id]
    compare_to = slot.get("compare_to")
    if not compare_to:
        return None
    control = phase_results.get(compare_to)
    candidate = phase_results.get(slot_id)
    if control is None or candidate is None:
        return None
    return {
        "slot": slot_id,
        "name": slot["name"],
        "lane": slot.get("lane", ""),
        "compare_to": compare_to,
        "delta_post_quant_bpb": delta(candidate.get("post_quant_bpb"), control.get("post_quant_bpb")),
        "delta_pre_quant_bpb": delta(candidate.get("pre_quant_bpb"), control.get("pre_quant_bpb")),
        "delta_quant_gap": delta(quant_gap(candidate), quant_gap(control)),
        "delta_step_avg_ms": delta(candidate.get("step_avg_ms"), control.get("step_avg_ms")),
        "delta_steps": delta_int(candidate.get("steps"), control.get("steps")),
        "delta_early_loss_delta": delta(candidate.get("early_loss_delta"), control.get("early_loss_delta")),
        "delta_late_loss_delta": delta(candidate.get("late_loss_delta"), control.get("late_loss_delta")),
        "phase_signature": phase_signature(candidate),
    }


def comparison_sort_key(entry: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        entry["delta_post_quant_bpb"] if entry["delta_post_quant_bpb"] is not None else float("inf"),
        -(entry["delta_late_loss_delta"] if entry["delta_late_loss_delta"] is not None else float("-inf")),
        entry["delta_step_avg_ms"] if entry["delta_step_avg_ms"] is not None else float("inf"),
        -(entry["delta_steps"] if entry["delta_steps"] is not None else -10**9),
    )


def best_result_from_phase_results(phase_results: dict[str, dict[str, Any]], slot_map: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    ranked = [
        (slot_id, metrics)
        for slot_id, metrics in phase_results.items()
        if metrics.get("post_quant_bpb") is not None
    ]
    if not ranked:
        return None
    slot_id, metrics = min(ranked, key=lambda item: item[1]["post_quant_bpb"])
    slot = slot_map[slot_id]
    return {
        "slot": slot_id,
        "name": slot["name"],
        "role": slot["role"],
        "family": slot["family"],
        "post_quant_bpb": metrics.get("post_quant_bpb"),
        "pre_quant_bpb": metrics.get("pre_quant_bpb"),
        "step_avg_ms": metrics.get("step_avg_ms"),
        "steps": metrics.get("steps"),
        "phase_signature": phase_signature(metrics),
    }


def recommend_promotions(phase_results: dict[str, dict[str, Any]], slot_map: dict[str, dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    comparisons = []
    for slot_id, slot in slot_map.items():
        if slot["role"] != "candidate":
            continue
        entry = comparison_entry(slot_id, slot_map, phase_results)
        if entry is not None:
            comparisons.append(entry)
    comparisons.sort(key=comparison_sort_key)
    survivors = [entry for entry in comparisons if entry["delta_post_quant_bpb"] is not None and entry["delta_post_quant_bpb"] < 0]
    return (survivors or comparisons)[:top_k]


def write_phase_summary(
    scope_root_dir: Path,
    phase_name: str,
    pack_name: str,
    phase_results: dict[str, dict[str, Any]],
    slot_map: dict[str, dict[str, Any]],
    top_k: int,
) -> Path:
    comparisons = []
    for slot_id, slot in slot_map.items():
        if slot["role"] != "candidate":
            continue
        entry = comparison_entry(slot_id, slot_map, phase_results)
        if entry is not None:
            comparisons.append(entry)
    comparisons.sort(key=comparison_sort_key)
    summary = {
        "pack": pack_name,
        "phase": phase_name,
        "results": phase_results,
        "comparisons": comparisons,
        "recommended_promotions": recommend_promotions(phase_results, slot_map, top_k),
        "best_result": best_result_from_phase_results(phase_results, slot_map),
    }
    summary_dir = scope_root_dir / phase_name
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"\n{'='*72}", flush=True)
    print(f"PHASE SUMMARY pack={pack_name} phase={phase_name}", flush=True)
    print(f"{'='*72}", flush=True)
    for slot_id, metrics in sorted(phase_results.items()):
        print(
            f"{slot_id:4s} post={fmt_metric(metrics.get('post_quant_bpb')):>10s} "
            f"pre={fmt_metric(metrics.get('pre_quant_bpb')):>10s} "
            f"steps={fmt_metric(metrics.get('steps')):>6s} "
            f"ms={fmt_metric(metrics.get('step_avg_ms')):>8s} "
            f"e/m/l={fmt_metric(metrics.get('early_loss_delta'))}/{fmt_metric(metrics.get('mid_loss_delta'))}/{fmt_metric(metrics.get('late_loss_delta'))} "
            f"sig={phase_signature(metrics)} rc={metrics.get('returncode')}",
            flush=True,
        )
    print(f"{'='*72}\n", flush=True)
    return summary_path


def partition_gpus(gpus: list[str], slot_ids: list[str], min_per_slot: int = 1) -> list[tuple[str, str, int]]:
    n_gpus, n_slots = len(gpus), len(slot_ids)
    if n_slots > n_gpus:
        raise SystemExit(f"Need {n_slots} GPUs for {n_slots} slots, only {n_gpus} available.")
    base, extra = divmod(n_gpus, n_slots)
    if base < min_per_slot:
        raise SystemExit(f"Cannot allocate {min_per_slot} GPU(s)/slot with {n_gpus} GPUs for {n_slots} slots.")
    allocs: list[tuple[str, str, int]] = []
    offset = 0
    for index, slot_id in enumerate(slot_ids):
        width = base + (1 if index < extra else 0)
        allocs.append((slot_id, ",".join(gpus[offset : offset + width]), width))
        offset += width
    return allocs


def run_jobs(jobs: list[JobSpec], dry_run: bool) -> dict[str, dict[str, Any]]:
    if dry_run:
        print(json.dumps([job.metadata | {"cmd": job.cmd, "log_path": str(job.log_path)} for job in jobs], indent=2))
        return {}
    procs: list[tuple[JobSpec, subprocess.Popen[Any], Any]] = []
    for job in jobs:
        print(f"[launch] phase={job.phase} pack={job.pack_name} gpu={job.gpu_spec} slot={job.slot_id} log={job.log_path}", flush=True)
        fh = open(job.log_path, "w", encoding="utf-8")
        proc = subprocess.Popen(job.cmd, cwd=job.run_dir, env=job.env, stdout=fh, stderr=subprocess.STDOUT, text=True)
        procs.append((job, proc, fh))
    results: dict[str, dict[str, Any]] = {}
    for job, proc, fh in procs:
        rc = proc.wait()
        fh.close()
        metrics = parse_metrics(job.log_path)
        metrics["returncode"] = rc
        results[job.slot_id] = metrics
        print(
            f"[done] phase={job.phase} slot={job.slot_id} rc={rc} "
            f"post_quant_bpb={metrics.get('post_quant_bpb')} sig={phase_signature(metrics)}",
            flush=True,
        )
    return results


def run_partitioned_phase(
    script_path: Path,
    config_path: Path,
    config: dict[str, Any],
    slot_map: dict[str, dict[str, Any]],
    phase: PhaseSpec,
    label: str,
    pack_name: str,
    scope_name: str,
    slot_ids: list[str],
    gpus: list[str],
    dry_run: bool,
) -> dict[str, dict[str, Any]]:
    allocs = partition_gpus(gpus, slot_ids, phase.nproc_per_slot)
    jobs = [
        build_job(script_path, config_path, config, slot_map, slot_id, phase, label, pack_name, scope_name, gpu_spec, nproc_per_slot)
        for slot_id, gpu_spec, nproc_per_slot in allocs
    ]
    return run_jobs(jobs, dry_run)


def run_pack_screen(
    script_path: Path,
    config_path: Path,
    config: dict[str, Any],
    slot_map: dict[str, dict[str, Any]],
    label: str,
    pack_name: str,
    gpus: list[str],
    top_k: int,
    dry_run: bool,
) -> tuple[str, Path]:
    scope_name = pack_name
    scope_root_dir = scope_root(config_path, label, scope_name)
    sanity_phase = phase_like(phase_from_config(config, "sanity"), f"sanity_{pack_name}")
    screen_phase = phase_like(phase_from_config(config, "screen"), f"screen_{pack_name}")
    sanity_slots = phase_slot_defaults(phase_from_config(config, "sanity"), config, pack_name)
    screen_slots = phase_slot_defaults(phase_from_config(config, "screen"), config, pack_name)
    ensure_runnable(sanity_slots + screen_slots, slot_map)

    print(f"[tournament] pack={pack_name} sanity slots={','.join(sanity_slots)}", flush=True)
    sanity_results = run_partitioned_phase(
        script_path, config_path, config, slot_map, sanity_phase, label, pack_name, scope_name, sanity_slots, gpus, dry_run
    )
    if dry_run:
        return sanity_slots[0], scope_root_dir / sanity_phase.name / "summary.json"
    sanity_summary = write_phase_summary(scope_root_dir, sanity_phase.name, pack_name, sanity_results, slot_map, top_k)
    print(f"[summary] {sanity_summary}", flush=True)
    if any(result.get("returncode") not in (0, None) for result in sanity_results.values()):
        raise SystemExit(f"Pack {pack_name} sanity failed; refusing to continue to screen.")

    print(f"[tournament] pack={pack_name} screen slots={','.join(screen_slots)}", flush=True)
    screen_results = run_partitioned_phase(
        script_path, config_path, config, slot_map, screen_phase, label, pack_name, scope_name, screen_slots, gpus, dry_run
    )
    if dry_run:
        return screen_slots[0], scope_root_dir / screen_phase.name / "summary.json"
    screen_summary = write_phase_summary(scope_root_dir, screen_phase.name, pack_name, screen_results, slot_map, top_k)
    print(f"[summary] {screen_summary}", flush=True)
    summary = json.loads(screen_summary.read_text(encoding="utf-8"))
    winner_slot = summary["recommended_promotions"][0]["slot"]
    return winner_slot, screen_summary


def best_result_from_summary(summary_path: Path) -> dict[str, Any]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    best_result = summary.get("best_result")
    if best_result is None:
        raise SystemExit(f"No best_result in summary {summary_path}")
    return best_result


def synthetic_slot_from_source(
    new_slot_id: str,
    source_slot_id: str,
    slot_map: dict[str, dict[str, Any]],
    *,
    name: str,
    family: str,
    compare_to: str | None,
    extra_env: dict[str, str] | None = None,
    extra_patches: list[str] | None = None,
    notes_suffix: list[str] | None = None,
) -> dict[str, Any]:
    source = json.loads(json.dumps(slot_map[source_slot_id]))
    notes = list(source.get("notes", []))
    notes.append(f"synthetic_from={source_slot_id}")
    if notes_suffix:
        notes.extend(notes_suffix)
    return {
        "slot": new_slot_id,
        "name": name,
        "role": source["role"],
        "runner_mode": source.get("runner_mode", "train"),
        "implementation_state": "ready",
        "parent": source_slot_id,
        "compare_to": compare_to,
        "family": family,
        "lane": "finalist",
        "why": source["why"],
        "validates": source["validates"],
        "falsifies": source["falsifies"],
        "patches": list(extra_patches or []),
        "env": dict(extra_env or {}),
        "notes": notes,
    }


def build_tournament_finalist_pack(config: dict[str, Any], slot_map: dict[str, dict[str, Any]], winners_by_pack: dict[str, str]) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    settings = tournament_settings(config)
    bootstrap_winner = winners_by_pack["bootstrap_handoff"]
    receiver_winner = winners_by_pack["receiver_mix"]

    finalist_slots = [
        synthetic_slot_from_source("T0", settings["control_slots"][0], slot_map, name="control_a", family="control", compare_to=None),
        synthetic_slot_from_source("T1", settings["control_slots"][1], slot_map, name="control_b", family="control", compare_to="T0"),
        synthetic_slot_from_source("T2", bootstrap_winner, slot_map, name="winner_bootstrap_handoff", family="bootstrap_winner", compare_to="T0"),
        synthetic_slot_from_source("T3", receiver_winner, slot_map, name="winner_receiver_mix", family="receiver_winner", compare_to="T0"),
        synthetic_slot_from_source("T4", "B2", slot_map, name="anchor_static_bigram", family="anchor_static_bigram", compare_to="T0"),
        synthetic_slot_from_source("T5", "R7", slot_map, name="aggressive_receiver_bundle", family="aggressive_receiver_bundle", compare_to="T0"),
        synthetic_slot_from_source(
            "T6",
            bootstrap_winner,
            slot_map,
            name="winner_bootstrap_plus_snapshot",
            family="bootstrap_snapshot_child",
            compare_to="T0",
            extra_patches=["checkpoint_selection"],
            extra_env={
                "SNAPSHOT_SELECT_ENABLE": "1",
                "SNAPSHOT_SELECT_MODE": "deployed_score",
                "SNAPSHOT_SELECT_SCORE": "deployed",
            },
            notes_suffix=["synthetic_child=+checkpoint_selection"],
        ),
        synthetic_slot_from_source(
            "T7",
            bootstrap_winner,
            slot_map,
            name="winner_bootstrap_plus_ttt",
            family="bootstrap_ttt_child",
            compare_to="T0",
            extra_patches=["pre_quant_ttt"],
            extra_env={
                "PRE_QUANT_TTT_ENABLE": "1",
                "PRE_QUANT_TTT_EPOCHS": "3",
                "PRE_QUANT_TTT_FREEZE_BLOCKS": "9",
                "PRE_QUANT_TTT_CHUNKS": "40",
                "PRE_QUANT_TTT_STRIDE": "256",
            },
            notes_suffix=["synthetic_child=+pre_quant_ttt"],
        ),
    ]
    finalist_config = json.loads(json.dumps(config))
    finalist_config["packs"][settings["final_pack"]] = [slot["slot"] for slot in finalist_slots]
    finalist_config["slots"] = finalist_config["slots"] + finalist_slots
    finalist_slot_map = build_slot_map(finalist_config)
    return finalist_config, finalist_slot_map


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
        print(json.dumps(
            {
                "mode": "tournament",
                "primary_packs": settings["primary_packs"],
                "controls": settings["control_slots"],
                "final_pack": settings["final_pack"],
            },
            indent=2,
        ))
        return

    winners_by_pack: dict[str, str] = {}
    pack_results: list[dict[str, Any]] = []
    for pack_name in settings["primary_packs"]:
        winner_slot, screen_summary = run_pack_screen(
            script_path, config_path, config, slot_map, args.label, pack_name, gpus, args.top_k, False
        )
        winners_by_pack[pack_name] = winner_slot
        pack_results.append({"pack": pack_name, "winner_slot": winner_slot, "summary_path": str(screen_summary)})

    finalist_config, finalist_slot_map = build_tournament_finalist_pack(config, slot_map, winners_by_pack)
    finalist_pack = settings["final_pack"]
    finalist_slots = finalist_config["packs"][finalist_pack]
    finalist_scope_name = "tournament"
    finalist_scope_root = scope_root(config_path, args.label, finalist_scope_name)
    finalist_sanity = phase_like(phase_from_config(config, "sanity"), "sanity_finalists")
    finalist_long = phase_like(phase_from_config(config, "final_single"), "finalists_long")

    print(f"[tournament] finalists sanity slots={','.join(finalist_slots)}", flush=True)
    finalist_sanity_results = run_partitioned_phase(
        script_path, config_path, finalist_config, finalist_slot_map, finalist_sanity, args.label, finalist_pack, finalist_scope_name, finalist_slots, gpus, False
    )
    finalist_sanity_summary = write_phase_summary(finalist_scope_root, finalist_sanity.name, finalist_pack, finalist_sanity_results, finalist_slot_map, 1)
    print(f"[summary] {finalist_sanity_summary}", flush=True)
    if any(result.get("returncode") not in (0, None) for result in finalist_sanity_results.values()):
        raise SystemExit("Finalist sanity failed; refusing to continue.")

    print(f"[tournament] finalists long slots={','.join(finalist_slots)}", flush=True)
    finalist_long_results = run_partitioned_phase(
        script_path, config_path, finalist_config, finalist_slot_map, finalist_long, args.label, finalist_pack, finalist_scope_name, finalist_slots, gpus, False
    )
    finalist_long_summary = write_phase_summary(finalist_scope_root, finalist_long.name, finalist_pack, finalist_long_results, finalist_slot_map, 1)
    print(f"[summary] {finalist_long_summary}", flush=True)
    champion_slot = json.loads(finalist_long_summary.read_text(encoding="utf-8"))["recommended_promotions"][0]["slot"]

    tournament_summary = {
        "mode": "tournament",
        "label": args.label,
        "pack_results": pack_results,
        "winners_by_pack": winners_by_pack,
        "finalist_slots": finalist_slots,
        "finalist_summary_path": str(finalist_long_summary),
        "champion_slot": champion_slot,
        "best_available_result": best_result_from_summary(finalist_long_summary),
    }
    tournament_summary_path = finalist_scope_root / "tournament_summary.json"
    tournament_summary_path.write_text(json.dumps(tournament_summary, indent=2) + "\n", encoding="utf-8")
    print(f"[summary] {tournament_summary_path}", flush=True)

    if args.skip_champion:
        return

    champion_phase = phase_from_config(config, "champion_8x")
    print(f"[tournament] champion_8x slot={champion_slot}", flush=True)
    champion_results = run_partitioned_phase(
        script_path,
        config_path,
        finalist_config,
        finalist_slot_map,
        champion_phase,
        args.label,
        finalist_pack,
        finalist_scope_name,
        [champion_slot],
        gpus[: champion_phase.nproc_per_slot],
        False,
    )
    champion_summary = write_phase_summary(finalist_scope_root, champion_phase.name, finalist_pack, champion_results, finalist_slot_map, 1)
    print(f"[summary] {champion_summary}", flush=True)
    tournament_summary["champion_8x_summary_path"] = str(champion_summary)
    tournament_summary["champion_8x_result"] = best_result_from_summary(champion_summary)
    tournament_summary_path.write_text(json.dumps(tournament_summary, indent=2) + "\n", encoding="utf-8")
    print(f"[summary] {tournament_summary_path}", flush=True)


def main() -> None:
    args = parse_args()
    script_path = Path(__file__).resolve()
    config_path = Path(args.config) if args.config else script_path.with_name("run_configs.json")
    config = load_config(config_path)
    slot_map = build_slot_map(config)
    gpus = parse_gpu_list(args.gpus)

    if args.phase == "plan":
        print(json.dumps(
            {
                "batch_id": config["batch_id"],
                "default_pack": config.get("default_pack"),
                "packs": config["packs"],
                "tournament": config.get("tournament", {}),
                "phases": config["phases"],
            },
            indent=2,
        ))
        return

    if args.phase == "readiness":
        pack_name = resolve_pack_name(args, config)
        phase = phase_from_config(config, "screen")
        slot_ids = resolve_phase_slots(args, config, phase, pack_name)
        report = []
        for slot_id in slot_ids:
            slot = slot_map[slot_id]
            report.append(
                {
                    "slot": slot_id,
                    "name": slot["name"],
                    "implementation_state": slot["implementation_state"],
                    "patches": merge_patches_for_slot(slot_id, slot_map),
                    "status": "ready" if runnable_reason(slot_id, slot_map) is None else runnable_reason(slot_id, slot_map),
                }
            )
        print(json.dumps(report, indent=2))
        return

    if args.phase == "tournament":
        run_tournament(script_path, config_path, config, slot_map, args, gpus)
        return

    pack_name = resolve_pack_name(args, config)
    if args.phase in {"sanity", "screen"}:
        phase = phase_from_config(config, args.phase)
        slot_ids = resolve_phase_slots(args, config, phase, pack_name)
        ensure_runnable(slot_ids, slot_map)
        scope_name = pack_name if not args.slots else "adhoc"
        results = run_partitioned_phase(script_path, config_path, config, slot_map, phase, args.label, pack_name, scope_name, slot_ids, gpus, args.dry_run)
        if not args.dry_run:
            summary_path = write_phase_summary(scope_root(config_path, args.label, scope_name), phase.name, pack_name, results, slot_map, args.top_k)
            print(f"[summary] {summary_path}", flush=True)
        return

    if args.phase in {"all", "final_single", "champion_8x"}:
        scope_name = pack_name if not args.slots else "adhoc"
        pack_root = scope_root(config_path, args.label, scope_name)
        if args.phase == "all":
            for pname in ["sanity", "screen"]:
                phase = phase_from_config(config, pname)
                slot_ids = resolve_phase_slots(args, config, phase, pack_name)
                ensure_runnable(slot_ids, slot_map)
                results = run_partitioned_phase(script_path, config_path, config, slot_map, phase, args.label, pack_name, scope_name, slot_ids, gpus, args.dry_run)
                if args.dry_run:
                    return
                summary_path = write_phase_summary(pack_root, phase.name, pack_name, results, slot_map, args.top_k)
                print(f"[summary] {summary_path}", flush=True)
                if pname == "sanity" and any(result.get("returncode") not in (0, None) for result in results.values()):
                    raise SystemExit("Sanity failed; refusing to continue to screen.")

            final_phase = phase_from_config(config, "final_single")
            final_slots = args.final_slots or [entry["slot"] for entry in json.loads((pack_root / "screen" / "summary.json").read_text(encoding="utf-8"))["recommended_promotions"][: args.top_k]]
            ensure_runnable(final_slots, slot_map)
            final_results = run_partitioned_phase(script_path, config_path, config, slot_map, final_phase, args.label, pack_name, scope_name, final_slots, gpus, False)
            final_summary = write_phase_summary(pack_root, final_phase.name, pack_name, final_results, slot_map, 1)
            print(f"[summary] {final_summary}", flush=True)
            if args.skip_champion:
                return
            champion_slot = json.loads(final_summary.read_text(encoding="utf-8"))["recommended_promotions"][0]["slot"]
            champion_phase = phase_from_config(config, "champion_8x")
            champion_results = run_partitioned_phase(
                script_path, config_path, config, slot_map, champion_phase, args.label, pack_name, scope_name, [champion_slot], gpus[: champion_phase.nproc_per_slot], False
            )
            champion_summary = write_phase_summary(pack_root, champion_phase.name, pack_name, champion_results, slot_map, 1)
            print(f"[summary] {champion_summary}", flush=True)
            return

        phase = phase_from_config(config, args.phase)
        slot_ids = args.final_slots or resolve_phase_slots(args, config, phase, pack_name)
        ensure_runnable(slot_ids, slot_map)
        results = run_partitioned_phase(script_path, config_path, config, slot_map, phase, args.label, pack_name, scope_name, slot_ids, gpus, args.dry_run)
        if not args.dry_run:
            summary_path = write_phase_summary(pack_root, phase.name, pack_name, results, slot_map, 1)
            print(f"[summary] {summary_path}", flush=True)
        return

    raise SystemExit(f"Unsupported phase {args.phase}")


if __name__ == "__main__":
    main()
