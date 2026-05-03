#!/usr/bin/env python3
"""Stage 3.4 orchestrator: eval-time optimization and late exploitation on Era 6 stack.

Lane B/C bakeoff: export pipeline, eval-time compute (SLOT, n-gram), and late
branching hypotheses.  Data paths derived from config defaults (SP8192).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from patches import apply_patches


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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 3.4 eval-time & late exploitation orchestrator (Era 6).")
    parser.add_argument("--config", default=None, help="Path to run_configs.json.")
    parser.add_argument(
        "--phase",
        default="plan",
        choices=["plan", "lane_bc_bakeoff", "decision", "all"],
    )
    parser.add_argument("--label", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--slots", nargs="*")
    parser.add_argument("--final-slots", nargs="*")
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-champion", action="store_true")
    parser.add_argument("--skip-champion", action="store_true")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

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


def resolve_phase_slots(args: argparse.Namespace, phase: PhaseSpec) -> list[str]:
    if args.slots:
        return list(args.slots)
    if phase.slots is None:
        raise SystemExit(f"Phase {phase.name} requires --slots or --final-slots")
    return list(phase.slots)


def _resolve_checkpoint(run_root: Path, source_slot: str, slot_map: dict[str, dict[str, Any]]) -> Path | None:
    """Find the final_model.pt from a previous phase run of the given slot."""
    for phase_name in ["screen", "sanity"]:
        phase_dir = run_root / phase_name
        if not phase_dir.exists():
            continue
        # Look for run dirs matching the source slot
        for d in sorted(phase_dir.iterdir()):
            if d.is_dir() and d.name.startswith(f"{source_slot}_"):
                ckpt = d / "final_model.pt"
                if ckpt.exists():
                    return ckpt
    return None


def _resolve_checkpoint_for_bakeoff(
    config: dict[str, Any], run_root: Path, slot_map: dict[str, dict[str, Any]],
    phase_key: str = "lane_bc_bakeoff",
) -> Path | None:
    """Resolve checkpoint for a bakeoff phase.

    If checkpoint_source is 'external', requires CHECKPOINT_PATH env var.
    Otherwise falls back to finding a checkpoint from a prior phase run.
    """
    ckpt_source = config["phases"][phase_key].get("checkpoint_source", "external")
    ckpt_env = os.environ.get("CHECKPOINT_PATH")
    if ckpt_env:
        p = Path(ckpt_env)
        if p.exists():
            return p
    if ckpt_source != "external":
        found = _resolve_checkpoint(run_root, ckpt_source, slot_map)
        if found is not None:
            return found
    return None


def resolve_final_slots(args: argparse.Namespace, run_root: Path, top_k: int) -> list[str]:
    if args.final_slots:
        return list(args.final_slots)
    summary_path = run_root / "screen" / "summary.json"
    if not summary_path.exists():
        raise SystemExit(f"No screen summary at {summary_path}. Run --phase screen first, or pass --final-slots.")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    promotions = summary.get("recommended_promotions", [])
    if not promotions:
        raise SystemExit(f"Screen summary at {summary_path} contains no recommended promotions.")
    return [entry["slot"] for entry in promotions[:top_k]]


# ---------------------------------------------------------------------------
# Job building
# ---------------------------------------------------------------------------

def build_job(
    script_path: Path,
    config_path: Path,
    config: dict[str, Any],
    slot_id: str,
    phase: PhaseSpec,
    label: str,
    gpu_spec: str | None,
    nproc_per_slot: int | None = None,
    checkpoint_path: str | None = None,
) -> JobSpec:
    stage_dir = script_path.resolve().parent
    slots = build_slot_map(config)
    slot = slots[slot_id]
    merged_env, lineage = merge_env_for_slot(slot_id, slots, config["defaults"])

    env = os.environ.copy()
    env.update(merged_env)
    if checkpoint_path is not None:
        env["EXPORT_ONLY"] = "1"
        env["CHECKPOINT_PATH"] = checkpoint_path
    env["MAX_WALLCLOCK_SECONDS"] = str(phase.max_wallclock_seconds)
    env["RUN_ID"] = f"{config['batch_id']}_{label}_{phase.name}_{slot_id}"
    env["PGOLF_STAGE"] = "3_4"
    env["PGOLF_PHASE"] = phase.name
    env["PGOLF_SLOT"] = slot_id
    env["PYTHONUNBUFFERED"] = "1"

    # Resolve data paths relative to the parameter-golf root
    pgolf_root = stage_dir.parent
    if env.get("DATA_PATH", "").startswith("./"):
        env["DATA_PATH"] = str((pgolf_root / env["DATA_PATH"].lstrip("./")).resolve())
    if env.get("TOKENIZER_PATH", "").startswith("./"):
        env["TOKENIZER_PATH"] = str((pgolf_root / env["TOKENIZER_PATH"].lstrip("./")).resolve())
    if gpu_spec is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu_spec

    nproc = nproc_per_slot if nproc_per_slot is not None else phase.nproc_per_slot
    run_dir = config_path.parent / "runs" / label / phase.name / f"{slot_id}_{slot['name']}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.log"

    slot_patches = slot.get("patches", [])
    metadata = {
        "batch_id": config["batch_id"],
        "phase": phase.name,
        "slot": slot_id,
        "name": slot["name"],
        "role": slot["role"],
        "family": slot["family"],
        "lane": slot.get("lane", "branching"),
        "label": label,
        "gpu_spec": gpu_spec,
        "nproc_per_slot": nproc,
        "lineage": lineage,
        "compare_to": slot.get("compare_to"),
        "why": slot["why"],
        "validates": slot["validates"],
        "falsifies": slot["falsifies"],
        "patches": slot_patches,
        "notes": slot.get("notes", []),
        "env": {k: env[k] for k in sorted(merged_env)},
    }
    (run_dir / "config.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    # Apply code patches to base_train_gpt.py and write to run_dir
    base_source = (stage_dir / "base_train_gpt.py").read_text(encoding="utf-8")
    if slot_patches:
        patched_source = apply_patches(base_source, slot_patches)
    else:
        patched_source = base_source
    patched_script_path = run_dir / "train_gpt.py"
    patched_script_path.write_text(patched_source, encoding="utf-8")
    train_script = str(patched_script_path.resolve())

    python = os.environ.get("PGOLF_PYTHON", "python")
    cmd = [python, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={nproc}", train_script]

    return JobSpec(
        slot_id=slot_id,
        phase=phase.name,
        gpu_spec=gpu_spec,
        nproc_per_slot=nproc,
        run_dir=run_dir,
        log_path=log_path,
        cmd=cmd,
        env=env,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Metric parsing (matches 1.1233 frontier log format)
# ---------------------------------------------------------------------------

def parse_float(pattern: str, text: str) -> float | None:
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    return float(matches[-1]) if matches else None


def parse_int(pattern: str, text: str) -> int | None:
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    return int(matches[-1]) if matches else None


def parse_metrics(log_path: Path) -> dict[str, Any]:
    if not log_path.exists():
        return {"returncode": None, "steps": None, "step_avg_ms": None,
                "pre_quant_bpb": None, "post_quant_bpb": None, "submission_size_bytes": None}
    text = log_path.read_text(encoding="utf-8", errors="replace")

    # Pre-quant BPB: last val_bpb from training loop
    pre_quant_matches = re.findall(r"step:\d+/\d+ val_loss:[0-9.]+ val_bpb:([0-9.]+)", text)
    pre_quant_bpb = float(pre_quant_matches[-1]) if pre_quant_matches else None

    # Post-quant BPB: the frontier code logs final_int8_zlib_roundtrip_exact for backward compat
    post_quant_bpb = parse_float(r"final_int8_zlib_roundtrip_exact val_loss:[0-9.]+ val_bpb:([0-9.]+)", text)
    # Fallback: try int6 roundtrip
    if post_quant_bpb is None:
        post_quant_bpb = parse_float(r"final_int6_roundtrip_exact val_loss:[0-9.]+ val_bpb:([0-9.]+)", text)

    return {
        "steps": parse_int(r"step:(\d+)/", text),
        "step_avg_ms": parse_float(r"step_avg:([0-9.]+)ms", text),
        "pre_quant_bpb": pre_quant_bpb,
        "post_quant_bpb": post_quant_bpb,
        "submission_size_bytes": parse_int(r"Total submission size int[68]\+(?:zstd|zlib|brotli): (\d+) bytes", text),
    }


def fmt_metric(value: Any) -> str:
    if value is None:
        return "?"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


# ---------------------------------------------------------------------------
# Comparison & promotion
# ---------------------------------------------------------------------------

def delta(a: float | None, b: float | None) -> float | None:
    return a - b if a is not None and b is not None else None


def delta_int(a: int | None, b: int | None) -> int | None:
    return a - b if a is not None and b is not None else None


def quant_gap(m: dict[str, Any]) -> float | None:
    return delta(m.get("post_quant_bpb"), m.get("pre_quant_bpb"))


def comparison_entry(
    slot_id: str, slot_map: dict[str, dict[str, Any]],
    phase_results: dict[str, dict[str, Any]], benchmarks: dict[str, Any],
) -> dict[str, Any] | None:
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
        "lane": slot.get("lane", "branching"),
        "compare_to": compare_to,
        "delta_post_quant_bpb": delta(candidate.get("post_quant_bpb"), control.get("post_quant_bpb")),
        "delta_pre_quant_bpb": delta(candidate.get("pre_quant_bpb"), control.get("pre_quant_bpb")),
        "delta_quant_gap": delta(quant_gap(candidate), quant_gap(control)),
        "delta_step_avg_ms": delta(candidate.get("step_avg_ms"), control.get("step_avg_ms")),
        "delta_steps": delta_int(candidate.get("steps"), control.get("steps")),
        "candidate_post_quant_gap_to_sota": delta(candidate.get("post_quant_bpb"), benchmarks.get("merged_sota_bpb", benchmarks.get("record_sota_bpb"))),
    }


def comparison_sort_key(entry: dict[str, Any]) -> tuple[float, float, float, float]:
    d_post = entry["delta_post_quant_bpb"]
    d_gap = entry["delta_quant_gap"]
    d_step = entry["delta_step_avg_ms"]
    d_steps = entry["delta_steps"]
    return (
        d_post if d_post is not None else float("inf"),
        d_gap if d_gap is not None else float("inf"),
        d_step if d_step is not None else float("inf"),
        -(d_steps if d_steps is not None else -10**9),
    )


def recommend_promotions(
    phase_results: dict[str, dict[str, Any]], slot_map: dict[str, dict[str, Any]],
    benchmarks: dict[str, Any], top_k: int,
) -> list[dict[str, Any]]:
    """Promote branching candidates by deployed score first, then speed/steps tie-breakers."""
    comparisons: list[dict[str, Any]] = []
    for slot_id, slot in slot_map.items():
        if slot["role"] != "candidate":
            continue
        entry = comparison_entry(slot_id, slot_map, phase_results, benchmarks)
        if entry is not None:
            comparisons.append(entry)

    comparisons.sort(key=comparison_sort_key)
    survivors = [e for e in comparisons if e["delta_post_quant_bpb"] is not None and e["delta_post_quant_bpb"] < 0]
    if survivors:
        return survivors[:top_k]
    return comparisons[:top_k]


def write_phase_summary(
    run_root: Path, phase_name: str, phase_results: dict[str, dict[str, Any]],
    slot_map: dict[str, dict[str, Any]], benchmarks: dict[str, Any], top_k: int,
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
    summary_dir = run_root / phase_name
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    # Print human-readable summary
    print(f"\n{'='*60}", flush=True)
    print(f"  PHASE SUMMARY: {phase_name}", flush=True)
    print(f"{'='*60}", flush=True)
    for r_id, r_metrics in sorted(phase_results.items()):
        lane = slot_map.get(r_id, {}).get("lane", "branching")
        print(f"  {r_id:12s} [{lane}]  post_quant={fmt_metric(r_metrics.get('post_quant_bpb')):>10s}  "
              f"pre_quant={fmt_metric(r_metrics.get('pre_quant_bpb')):>10s}  "
              f"steps={fmt_metric(r_metrics.get('steps'))}  "
              f"ms/step={fmt_metric(r_metrics.get('step_avg_ms'))}  "
              f"rc={r_metrics.get('returncode', '?')}", flush=True)
    if comparisons:
        print(f"\n  RANKED CANDIDATES:", flush=True)
        for i, c in enumerate(comparisons):
            dp = c["delta_post_quant_bpb"]
            dp_str = f"{dp:+.4f}" if dp is not None else "?"
            dg = c["delta_quant_gap"]
            dg_str = f"{dg:+.4f}" if dg is not None else "?"
            print(f"  #{i+1} {c['slot']:12s} {c['name']:30s}  delta_post={dp_str}  delta_gap={dg_str}", flush=True)
    promos = summary.get("recommended_promotions", [])
    if promos:
        print(f"\n  PROMOTED:", flush=True)
        for p in promos:
            print(f"    {p['slot']} [{p['lane']}] {p['name']}", flush=True)
    print(f"{'='*60}\n", flush=True)

    return summary_path


# ---------------------------------------------------------------------------
# Execution modes
# ---------------------------------------------------------------------------

def run_parallel_phase(
    script_path: Path, config_path: Path, config: dict[str, Any],
    phase: PhaseSpec, label: str, slot_ids: list[str], gpus: list[str], dry_run: bool,
    checkpoint_path: str | None = None,
) -> dict[str, dict[str, Any]]:
    jobs = [
        build_job(script_path, config_path, config, slot_id, phase, label, gpus[i % len(gpus)],
                  checkpoint_path=checkpoint_path)
        for i, slot_id in enumerate(slot_ids)
    ]
    if dry_run:
        print(json.dumps([j.metadata | {"cmd": j.cmd, "log": str(j.log_path)} for j in jobs], indent=2))
        return {}

    results: dict[str, dict[str, Any]] = {}
    wave_size = len(gpus)
    num_waves = (len(jobs) + wave_size - 1) // wave_size
    for wave_idx in range(num_waves):
        wave = jobs[wave_idx * wave_size : (wave_idx + 1) * wave_size]
        if num_waves > 1:
            print(f"[wave] {wave_idx + 1}/{num_waves} ({len(wave)} slots)", flush=True)

        procs: list[tuple[JobSpec, subprocess.Popen, Any]] = []
        for job in wave:
            print(f"[launch] {phase.name} gpu={job.gpu_spec} slot={job.slot_id} log={job.log_path}", flush=True)
            fh = open(job.log_path, "w", encoding="utf-8")
            p = subprocess.Popen(job.cmd, cwd=job.run_dir, env=job.env,
                                 stdout=fh, stderr=subprocess.STDOUT, text=True)
            procs.append((job, p, fh))

        for job, p, fh in procs:
            rc = p.wait()
            fh.close()
            m = parse_metrics(job.log_path)
            m["returncode"] = rc
            results[job.slot_id] = m
            print(f"[done] {phase.name} slot={job.slot_id} rc={rc} "
                  f"post_quant_bpb={m.get('post_quant_bpb')} step_avg={m.get('step_avg_ms')}", flush=True)
    return results


def partition_gpus(gpus: list[str], slot_ids: list[str], min_per_slot: int = 1) -> list[tuple[str, str, int]]:
    n_gpus, n_slots = len(gpus), len(slot_ids)
    if n_slots > n_gpus:
        raise SystemExit(f"Need {n_slots} GPUs for {n_slots} slots, only {n_gpus} available.")
    base, extra = divmod(n_gpus, n_slots)
    if base < min_per_slot:
        raise SystemExit(f"Cannot allocate {min_per_slot} GPU(s)/slot with {n_gpus} GPUs for {n_slots} slots.")
    allocs, offset = [], 0
    for i, sid in enumerate(slot_ids):
        width = base + (1 if i < extra else 0)
        allocs.append((sid, ",".join(gpus[offset:offset + width]), width))
        offset += width
    return allocs


def run_partitioned_phase(
    script_path: Path, config_path: Path, config: dict[str, Any],
    phase: PhaseSpec, label: str, slot_ids: list[str], gpus: list[str], dry_run: bool,
) -> dict[str, dict[str, Any]]:
    allocs = partition_gpus(gpus, slot_ids, phase.nproc_per_slot)
    jobs = [build_job(script_path, config_path, config, sid, phase, label, gspec, nproc=nproc)
            for sid, gspec, nproc in allocs]
    if dry_run:
        print(json.dumps([j.metadata | {"cmd": j.cmd, "log": str(j.log_path)} for j in jobs], indent=2))
        return {}

    procs: list[tuple[JobSpec, subprocess.Popen, Any]] = []
    for job in jobs:
        print(f"[launch] {phase.name} gpu={job.gpu_spec} slot={job.slot_id} nproc={job.nproc_per_slot}", flush=True)
        fh = open(job.log_path, "w", encoding="utf-8")
        p = subprocess.Popen(job.cmd, cwd=job.run_dir, env=job.env,
                             stdout=fh, stderr=subprocess.STDOUT, text=True)
        procs.append((job, p, fh))

    results: dict[str, dict[str, Any]] = {}
    for job, p, fh in procs:
        rc = p.wait()
        fh.close()
        m = parse_metrics(job.log_path)
        m["returncode"] = rc
        results[job.slot_id] = m
        print(f"[done] {phase.name} slot={job.slot_id} rc={rc} "
              f"post_quant_bpb={m.get('post_quant_bpb')} step_avg={m.get('step_avg_ms')}", flush=True)
    return results


def run_serial_phase(
    script_path: Path, config_path: Path, config: dict[str, Any],
    phase: PhaseSpec, label: str, slot_ids: list[str], gpu_spec: str, dry_run: bool,
) -> dict[str, dict[str, Any]]:
    jobs = [build_job(script_path, config_path, config, sid, phase, label, gpu_spec) for sid in slot_ids]
    if dry_run:
        print(json.dumps([j.metadata | {"cmd": j.cmd, "log": str(j.log_path)} for j in jobs], indent=2))
        return {}

    results: dict[str, dict[str, Any]] = {}
    for job in jobs:
        print(f"[launch] {phase.name} gpu={job.gpu_spec} slot={job.slot_id}", flush=True)
        with open(job.log_path, "w", encoding="utf-8") as fh:
            cp = subprocess.run(job.cmd, cwd=job.run_dir, env=job.env,
                                stdout=fh, stderr=subprocess.STDOUT, text=True, check=False)
        m = parse_metrics(job.log_path)
        m["returncode"] = cp.returncode
        results[job.slot_id] = m
        print(f"[done] {phase.name} slot={job.slot_id} rc={cp.returncode} "
              f"post_quant_bpb={m.get('post_quant_bpb')}", flush=True)
    return results


_HYPOTHESIS_BAR_FIELDS = [
    "mechanism", "math", "placement", "coefficients",
    "failure_modes", "kill_rule", "dominant_metric", "expected_impact",
]


def ensure_ready(slot_ids: list[str], slot_map: dict[str, dict[str, Any]]) -> None:
    blocked = [s for s in slot_ids if slot_map[s]["implementation_state"] != "ready"]
    if blocked:
        raise SystemExit(f"Blocked by non-ready slots: {', '.join(blocked)}")
    # Enforce hypothesis bar: candidate and support slots must have all required fields
    bar_failures: list[str] = []
    for s in slot_ids:
        slot = slot_map[s]
        if slot["role"] in ("candidate", "support"):
            missing = [f for f in _HYPOTHESIS_BAR_FIELDS if f not in slot or not slot[f]]
            if missing:
                bar_failures.append(f"{s}: missing {', '.join(missing)}")
    if bar_failures:
        raise SystemExit(
            "Hypothesis bar enforcement failed. These slots are missing required fields:\n"
            + "\n".join(f"  {f}" for f in bar_failures)
            + "\nSee hypothesis_stage_bar.md for the required template."
        )


def should_run_champion(args: argparse.Namespace) -> bool:
    if args.skip_champion:
        return False
    if args.run_champion:
        return True
    return args.top_k > 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    script_path = Path(__file__)
    config_path = Path(args.config).resolve() if args.config else script_path.with_name("run_configs.json").resolve()
    config = load_config(config_path)
    slot_map = build_slot_map(config)
    run_root = config_path.parent / "runs" / args.label
    gpus = parse_gpu_list(args.gpus)

    if args.phase == "plan":
        print(json.dumps(config, indent=2))
        return

    if args.phase == "lane_bc_bakeoff":
        phase = phase_from_config(config, "lane_bc_bakeoff")
        slot_ids = resolve_phase_slots(args, phase)
        ensure_ready(slot_ids, slot_map)
        ckpt_path = _resolve_checkpoint_for_bakeoff(config, run_root, slot_map)
        if ckpt_path is None and not args.dry_run:
            raise SystemExit(
                "No checkpoint found for lane_bc_bakeoff. "
                "Set CHECKPOINT_PATH env var to point at a trained checkpoint."
            )
        results = run_parallel_phase(
            script_path, config_path, config, phase, args.label, slot_ids, gpus, args.dry_run,
            checkpoint_path=str(ckpt_path) if ckpt_path else None,
        )
        if not args.dry_run:
            sp = write_phase_summary(run_root, phase.name, results, slot_map, config["benchmarks"], args.top_k)
            print(f"[summary] {sp}", flush=True)
        return

    if args.phase == "decision":
        phase = phase_from_config(config, "decision")
        slot_ids = resolve_final_slots(args, run_root, args.top_k)
        ensure_ready(slot_ids, slot_map)
        results = run_partitioned_phase(script_path, config_path, config, phase, args.label, slot_ids, gpus, args.dry_run)
        if not args.dry_run:
            sp = write_phase_summary(run_root, phase.name, results, slot_map, config["benchmarks"], args.top_k)
            print(f"[summary] {sp}", flush=True)
        return

    if args.phase == "all":
        # Full stage order: lane_bc_bakeoff → decision
        lane_bc = phase_from_config(config, "lane_bc_bakeoff")
        lane_bc_slots = resolve_phase_slots(args, lane_bc)
        lane_bc_ready = [s for s in lane_bc_slots if slot_map[s]["implementation_state"] == "ready"]

        if args.dry_run:
            print("[dry-run] Full stage order:", flush=True)
            for pname in ["lane_bc_bakeoff", "decision"]:
                if pname in config["phases"]:
                    pspec = phase_from_config(config, pname)
                    pslots = pspec.slots or ["(from promotion)"]
                    print(f"  {pname}: slots={pslots} wallclock={pspec.max_wallclock_seconds}s", flush=True)
            return

        # 1. Lane B/C bakeoff (export + eval on checkpoint)
        if lane_bc_ready:
            print(f"\n{'='*60}\n  STAGE: lane_bc_bakeoff\n{'='*60}", flush=True)
            ckpt_path = _resolve_checkpoint_for_bakeoff(config, run_root, slot_map)
            if ckpt_path is None:
                raise SystemExit(
                    "No checkpoint found for lane_bc_bakeoff. "
                    "Set CHECKPOINT_PATH env var to point at a trained checkpoint."
                )
            lane_bc_results = run_parallel_phase(
                script_path, config_path, config, lane_bc, args.label, lane_bc_ready, gpus, False,
                checkpoint_path=str(ckpt_path),
            )
            sp = write_phase_summary(run_root, lane_bc.name, lane_bc_results, slot_map, config["benchmarks"], args.top_k)
            print(f"[summary] {sp}", flush=True)

        # 2. Decision (promoted finalists, full-box)
        print(f"\n{'='*60}\n  STAGE: decision\n{'='*60}", flush=True)
        decision_phase = phase_from_config(config, "decision")
        final_slots = resolve_final_slots(args, run_root, args.top_k)
        print(f"[promote] decision slots={','.join(final_slots)}", flush=True)
        ensure_ready(final_slots, slot_map)
        decision_results = run_partitioned_phase(script_path, config_path, config, decision_phase, args.label, final_slots, gpus, False)
        sp = write_phase_summary(run_root, decision_phase.name, decision_results, slot_map, config["benchmarks"], args.top_k)
        print(f"[summary] {sp}", flush=True)
        return

    raise SystemExit(f"Unsupported phase: {args.phase}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
