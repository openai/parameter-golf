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


@dataclass(frozen=True)
class PhaseSpec:
    name: str
    max_wallclock_seconds: int
    nproc_per_slot: int
    slots: list[str] | None = None


@dataclass(frozen=True)
class JobSpec:
    slot_id: str
    phase: str
    gpu_spec: str | None
    nproc_per_slot: int
    run_dir: Path
    log_path: Path
    cmd: list[str]
    env: dict[str, str]
    metadata: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 2 budget-aware H100 orchestrator.")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to a run_configs.json file. Defaults to the file next to this script.",
    )
    parser.add_argument(
        "--phase",
        default="plan",
        choices=["plan", "sanity", "screen", "final_single", "champion_8x", "all"],
        help="Which phase to execute.",
    )
    parser.add_argument("--label", default=datetime.now().strftime("%Y%m%d_%H%M%S"), help="Run label.")
    parser.add_argument(
        "--gpus",
        default="0,1,2,3,4,5,6,7",
        help="Comma-separated local GPU ids. For parallel phases this should list 8 GPUs.",
    )
    parser.add_argument(
        "--slots",
        nargs="*",
        help="Override the config-defined slots for the selected phase.",
    )
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
    # Walk up from script to find the parameter-golf root (contains train_gpt.py)
    candidate = script_path.resolve().parent
    for _ in range(10):
        if (candidate / "train_gpt.py").exists():
            return candidate.parent, candidate
        candidate = candidate.parent
    raise SystemExit("Cannot find parameter-golf root (directory containing train_gpt.py)")


def resolve_phase_slots(args: argparse.Namespace, phase: PhaseSpec) -> list[str]:
    if args.slots:
        return list(args.slots)
    if phase.slots is None:
        raise SystemExit(f"Phase {phase.name} requires --slots or --final-slots")
    return list(phase.slots)


def resolve_final_slots(args: argparse.Namespace, run_root: Path, top_k: int) -> list[str]:
    if args.final_slots:
        return list(args.final_slots)
    summary_path = run_root / "screen" / "summary.json"
    if not summary_path.exists():
        raise SystemExit(
            f"No screen summary found at {summary_path}. Run --phase screen or --phase all first, or pass --final-slots."
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    promotions = summary.get("recommended_promotions", [])
    if not promotions:
        raise SystemExit(f"Screen summary at {summary_path} contains no recommended promotions.")
    return [entry["slot"] for entry in promotions[:top_k]]


def build_job(
    script_path: Path,
    config_path: Path,
    config: dict[str, Any],
    slot_id: str,
    phase: PhaseSpec,
    label: str,
    gpu_spec: str | None,
    nproc_per_slot: int | None = None,
) -> JobSpec:
    repo_root, pgolf_root = resolve_paths(script_path)
    slots = build_slot_map(config)
    slot = slots[slot_id]
    merged_env, lineage = merge_env_for_slot(slot_id, slots, config["defaults"])

    env = os.environ.copy()
    env.update(merged_env)
    env["MAX_WALLCLOCK_SECONDS"] = str(phase.max_wallclock_seconds)
    env["RUN_ID"] = f"{config['batch_id']}_{label}_{phase.name}_{slot_id}"
    env["PGOLF_STAGE2_BATCH"] = config["batch_id"]
    env["PGOLF_STAGE2_PHASE"] = phase.name
    env["PGOLF_STAGE2_SLOT"] = slot_id
    env["PYTHONUNBUFFERED"] = "1"
    if env["DATA_PATH"].startswith("./"):
        env["DATA_PATH"] = str((pgolf_root / "data" / "datasets" / "fineweb10B_sp1024").resolve())
    if env["TOKENIZER_PATH"].startswith("./"):
        env["TOKENIZER_PATH"] = str((pgolf_root / "data" / "tokenizers" / "fineweb_1024_bpe.model").resolve())
    if gpu_spec is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu_spec

    run_dir = config_path.parent / "runs" / label / phase.name / f"{slot_id}_{slot['name']}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.log"

    metadata = {
        "batch_id": config["batch_id"],
        "phase": phase.name,
        "slot": slot_id,
        "name": slot["name"],
        "role": slot["role"],
        "family": slot["family"],
        "label": label,
        "gpu_spec": gpu_spec,
        "nproc_per_slot": nproc_per_slot if nproc_per_slot is not None else phase.nproc_per_slot,
        "lineage": lineage,
        "compare_to": slot.get("compare_to"),
        "why": slot["why"],
        "validates": slot["validates"],
        "falsifies": slot["falsifies"],
        "notes": slot.get("notes", []),
        "env": {k: env[k] for k in sorted(merged_env)},
    }
    (run_dir / "config.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    train_script = str((pgolf_root / "train_gpt.py").resolve())
    python = os.environ.get("PGOLF_PYTHON", "python")
    cmd = [
        python,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={nproc_per_slot if nproc_per_slot is not None else phase.nproc_per_slot}",
        train_script,
    ]
    return JobSpec(
        slot_id=slot_id,
        phase=phase.name,
        gpu_spec=gpu_spec,
        nproc_per_slot=nproc_per_slot if nproc_per_slot is not None else phase.nproc_per_slot,
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
            "submission_size_bytes": None,
        }
    text = log_path.read_text(encoding="utf-8", errors="replace")
    pre_quant_matches = re.findall(r"step:\d+/\d+ val_loss:[0-9.]+ val_bpb:([0-9.]+)", text)
    pre_quant_bpb = float(pre_quant_matches[-1]) if pre_quant_matches else None
    return {
        "steps": parse_int(r"step:(\d+)/", text),
        "step_avg_ms": parse_float(r"step_avg:([0-9.]+)ms", text),
        "pre_quant_bpb": pre_quant_bpb,
        "post_quant_bpb": parse_float(r"final_int8_zlib_roundtrip_exact val_loss:[0-9.]+ val_bpb:([0-9.]+)", text),
        "submission_size_bytes": parse_int(r"Total submission size int8\+zlib: (\d+) bytes", text),
    }


def quant_gap(metrics: dict[str, Any]) -> float | None:
    pre = metrics.get("pre_quant_bpb")
    post = metrics.get("post_quant_bpb")
    if pre is None or post is None:
        return None
    return post - pre


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
        "compare_to": compare_to,
        "delta_post_quant_bpb": delta(candidate.get("post_quant_bpb"), control.get("post_quant_bpb")),
        "delta_pre_quant_bpb": delta(candidate.get("pre_quant_bpb"), control.get("pre_quant_bpb")),
        "delta_quant_gap": delta(candidate_gap, control_gap),
        "delta_step_avg_ms": delta(candidate.get("step_avg_ms"), control.get("step_avg_ms")),
        "delta_steps": delta_int(candidate.get("steps"), control.get("steps")),
        "candidate_post_quant_gap_to_sota": delta(candidate.get("post_quant_bpb"), benchmarks["record_sota_bpb"]),
    }


def delta(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline is None:
        return None
    return value - baseline


def delta_int(value: int | None, baseline: int | None) -> int | None:
    if value is None or baseline is None:
        return None
    return value - baseline


def comparison_sort_key(entry: dict[str, Any]) -> tuple[float, float, float, float]:
    delta_post = entry["delta_post_quant_bpb"]
    delta_gap = entry["delta_quant_gap"]
    delta_step_avg = entry["delta_step_avg_ms"]
    delta_steps = entry["delta_steps"]
    return (
        delta_post if delta_post is not None else float("inf"),
        delta_gap if delta_gap is not None else float("inf"),
        delta_step_avg if delta_step_avg is not None else float("inf"),
        -(delta_steps if delta_steps is not None else -10**9),
    )


def recommend_promotions(
    phase_results: dict[str, dict[str, Any]],
    slot_map: dict[str, dict[str, Any]],
    benchmarks: dict[str, Any],
    top_k: int,
) -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    for slot_id, slot in slot_map.items():
        if slot["role"] != "candidate":
            continue
        entry = comparison_entry(slot_id, slot_map, phase_results, benchmarks)
        if entry is not None:
            comparisons.append(entry)
    comparisons.sort(key=comparison_sort_key)

    survivors = [
        entry
        for entry in comparisons
        if entry["delta_post_quant_bpb"] is not None and entry["delta_post_quant_bpb"] < 0
    ]
    if not survivors:
        survivors = comparisons[:1]
    return survivors[:top_k]


def write_phase_summary(
    run_root: Path,
    phase_name: str,
    phase_results: dict[str, dict[str, Any]],
    slot_map: dict[str, dict[str, Any]],
    benchmarks: dict[str, Any],
    top_k: int,
) -> Path:
    comparisons: list[dict[str, Any]] = []
    for slot_id, slot in slot_map.items():
        if slot["role"] != "candidate":
            continue
        entry = comparison_entry(slot_id, slot_map, phase_results, benchmarks)
        if entry is not None:
            comparisons.append(entry)
    comparisons.sort(key=comparison_sort_key)

    summary = {
        "phase": phase_name,
        "results": phase_results,
        "comparisons": comparisons,
        "recommended_promotions": recommend_promotions(phase_results, slot_map, benchmarks, top_k),
    }
    summary_path = run_root / phase_name / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary_path


def print_plan(config: dict[str, Any]) -> None:
    print(json.dumps(config, indent=2))


def run_parallel_phase(
    script_path: Path,
    config_path: Path,
    config: dict[str, Any],
    phase: PhaseSpec,
    label: str,
    slot_ids: list[str],
    gpus: list[str],
    dry_run: bool,
) -> dict[str, dict[str, Any]]:
    if len(slot_ids) > len(gpus):
        raise SystemExit(f"Phase {phase.name} needs {len(slot_ids)} GPUs but only {len(gpus)} were provided.")

    jobs = [
        build_job(script_path, config_path, config, slot_id, phase, label, gpus[index])
        for index, slot_id in enumerate(slot_ids)
    ]
    if dry_run:
        print(json.dumps([job.metadata | {"cmd": job.cmd, "cwd": str(job.run_dir), "log_path": str(job.log_path)} for job in jobs], indent=2))
        return {}

    processes: list[tuple[JobSpec, subprocess.Popen[str], Any]] = []
    for job in jobs:
        print(f"[launch] phase={phase.name} gpu={job.gpu_spec} slot={job.slot_id} log={job.log_path}", flush=True)
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
            f"[done] phase={phase.name} slot={job.slot_id} rc={returncode} "
            f"post_quant_bpb={metrics.get('post_quant_bpb')} step_avg_ms={metrics.get('step_avg_ms')}",
            flush=True,
        )
    return results


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
        print(f"[launch] phase={phase.name} gpu={job.gpu_spec} slot={job.slot_id} log={job.log_path}", flush=True)
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
            f"[done] phase={phase.name} slot={job.slot_id} rc={returncode} "
            f"post_quant_bpb={metrics.get('post_quant_bpb')} step_avg_ms={metrics.get('step_avg_ms')}",
            flush=True,
        )
    return results


def run_serial_phase(
    script_path: Path,
    config_path: Path,
    config: dict[str, Any],
    phase: PhaseSpec,
    label: str,
    slot_ids: list[str],
    gpu_spec: str,
    dry_run: bool,
) -> dict[str, dict[str, Any]]:
    jobs = [build_job(script_path, config_path, config, slot_id, phase, label, gpu_spec) for slot_id in slot_ids]
    if dry_run:
        print(json.dumps([job.metadata | {"cmd": job.cmd, "cwd": str(job.run_dir), "log_path": str(job.log_path)} for job in jobs], indent=2))
        return {}

    results: dict[str, dict[str, Any]] = {}
    for job in jobs:
        print(f"[launch] phase={phase.name} gpu={job.gpu_spec} slot={job.slot_id} log={job.log_path}", flush=True)
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
            f"[done] phase={phase.name} slot={job.slot_id} rc={completed.returncode} "
            f"post_quant_bpb={metrics.get('post_quant_bpb')} step_avg_ms={metrics.get('step_avg_ms')}",
            flush=True,
        )
    return results


def ensure_ready(slot_ids: list[str], slot_map: dict[str, dict[str, Any]]) -> None:
    blocked = [slot_id for slot_id in slot_ids if slot_map[slot_id]["implementation_state"] != "ready"]
    if blocked:
        raise SystemExit(f"Blocked by non-ready slots: {', '.join(blocked)}")


def should_run_champion(args: argparse.Namespace) -> bool:
    if args.skip_champion:
        return False
    if args.run_champion:
        return True
    return args.top_k > 1


def main() -> None:
    args = parse_args()
    script_path = Path(__file__)
    config_path = Path(args.config).resolve() if args.config else script_path.with_name("run_configs.json").resolve()
    config = load_config(config_path)
    slot_map = build_slot_map(config)
    run_root = config_path.parent / "runs" / args.label
    gpus = parse_gpu_list(args.gpus)

    if args.phase == "plan":
        print_plan(config)
        return

    if args.phase in {"sanity", "screen"}:
        phase = phase_from_config(config, args.phase)
        slot_ids = resolve_phase_slots(args, phase)
        ensure_ready(slot_ids, slot_map)
        results = run_parallel_phase(script_path, config_path, config, phase, args.label, slot_ids, gpus, args.dry_run)
        if not args.dry_run:
            summary_path = write_phase_summary(run_root, phase.name, results, slot_map, config["benchmarks"], args.top_k)
            print(f"[summary] {summary_path}", flush=True)
        return

    if args.phase in {"final_single", "champion_8x"}:
        phase = phase_from_config(config, args.phase)
        if args.phase == "final_single":
            slot_ids = resolve_final_slots(args, run_root, args.top_k)
            ensure_ready(slot_ids, slot_map)
            results = run_partitioned_phase(
                script_path, config_path, config, phase, args.label, slot_ids, gpus, args.dry_run
            )
        else:
            slot_ids = args.final_slots or args.slots
            if not slot_ids:
                slot_ids = resolve_final_slots(args, run_root, 1)
            ensure_ready(slot_ids, slot_map)
            results = run_serial_phase(
                script_path,
                config_path,
                config,
                phase,
                args.label,
                slot_ids,
                ",".join(gpus[: phase.nproc_per_slot]),
                args.dry_run,
            )
        if not args.dry_run:
            summary_path = write_phase_summary(run_root, phase.name, results, slot_map, config["benchmarks"], args.top_k)
            print(f"[summary] {summary_path}", flush=True)
        return

    if args.phase == "all":
        sanity = phase_from_config(config, "sanity")
        screen = phase_from_config(config, "screen")
        ensure_ready(resolve_phase_slots(args, sanity), slot_map)
        ensure_ready(resolve_phase_slots(args, screen), slot_map)
        sanity_results = run_parallel_phase(
            script_path,
            config_path,
            config,
            sanity,
            args.label,
            resolve_phase_slots(args, sanity),
            gpus,
            args.dry_run,
        )
        if args.dry_run:
            screen_slots = resolve_phase_slots(args, screen)
            screen_jobs = [
                build_job(script_path, config_path, config, slot_id, screen, args.label, gpus[index]).metadata
                for index, slot_id in enumerate(screen_slots)
            ]
            print(
                json.dumps(
                    {
                        "next_phase": "screen",
                        "slots": screen_jobs,
                        "auto_promotion": {
                            "final_single_top_k": args.top_k,
                            "champion_8x_after_final_single": should_run_champion(args),
                            "note": "final_single repartitions all visible GPUs across survivors; champion_8x is only auto-run when top_k>1 unless --skip-champion is passed.",
                        },
                    },
                    indent=2,
                )
            )
            return
        write_phase_summary(run_root, sanity.name, sanity_results, slot_map, config["benchmarks"], args.top_k)

        if any(result.get("returncode") not in (0, None) for result in sanity_results.values()):
            raise SystemExit("Sanity phase had failures. Refusing to continue automatically.")

        screen_results = run_parallel_phase(
            script_path,
            config_path,
            config,
            screen,
            args.label,
            resolve_phase_slots(args, screen),
            gpus,
            False,
        )
        screen_summary_path = write_phase_summary(run_root, screen.name, screen_results, slot_map, config["benchmarks"], args.top_k)
        print(f"[summary] {screen_summary_path}", flush=True)

        final_slots = resolve_final_slots(args, run_root, args.top_k)
        print(f"[promote] final_single slots={','.join(final_slots)}", flush=True)
        final_phase = phase_from_config(config, "final_single")
        final_results = run_partitioned_phase(
            script_path, config_path, config, final_phase, args.label, final_slots, gpus, False
        )
        final_summary_path = write_phase_summary(
            run_root, final_phase.name, final_results, slot_map, config["benchmarks"], args.top_k
        )
        print(f"[summary] {final_summary_path}", flush=True)

        if should_run_champion(args):
            champion_phase = phase_from_config(config, "champion_8x")
            champion_slot = final_slots[:1]
            print(f"[promote] champion_8x slot={champion_slot[0]}", flush=True)
            champion_results = run_serial_phase(
                script_path,
                config_path,
                config,
                champion_phase,
                args.label,
                champion_slot,
                ",".join(gpus[: champion_phase.nproc_per_slot]),
                False,
            )
            champion_summary_path = write_phase_summary(
                run_root, champion_phase.name, champion_results, slot_map, config["benchmarks"], 1
            )
            print(f"[summary] {champion_summary_path}", flush=True)
        return

    raise SystemExit(f"Unsupported phase {args.phase}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
