#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.generate_evolutionary_matrix import EvoMatrixRun, build_matrix, parse_csv, select_runs


@dataclass(frozen=True)
class QueuePlanEntry:
    run: EvoMatrixRun
    output_path: Path
    log_path: Path


@dataclass
class ActiveProcess:
    entry: QueuePlanEntry
    proc: subprocess.Popen[str]
    log_handle: TextIO
    gpu_slot: str | None
    start_time: float


def detect_gpu_slots(spec: str) -> tuple[str, ...]:
    if spec != "auto":
        return tuple(part.strip() for part in spec.split(",") if part.strip())
    env_spec = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env_spec:
        return tuple(part.strip() for part in env_spec.split(",") if part.strip())
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            cwd=REPO_ROOT,
            text=True,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return ()
    return tuple(line.strip() for line in output.splitlines() if line.strip())


def build_queue_plan(
    *,
    runs: list[EvoMatrixRun],
    output_dir: Path,
    log_dir: Path,
) -> list[QueuePlanEntry]:
    return [
        QueuePlanEntry(
            run=run,
            output_path=output_dir / f"{run.name}.json",
            log_path=log_dir / f"{run.name}.log",
        )
        for run in runs
    ]


def launch_run(
    *,
    entry: QueuePlanEntry,
    python_bin: str,
    script_path: str,
    output_dir: str,
    enwik8_path: str,
    cwd: Path,
    gpu_slot: str | None,
) -> ActiveProcess:
    entry.log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = entry.log_path.open("w", encoding="utf-8")
    env = os.environ.copy()
    if gpu_slot is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu_slot
    command = list(entry.run.command_parts(python_bin, script_path, output_dir, enwik8_path))
    proc = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return ActiveProcess(
        entry=entry,
        proc=proc,
        log_handle=log_handle,
        gpu_slot=gpu_slot,
        start_time=time.time(),
    )


def finalize_process(active: ActiveProcess) -> dict[str, object]:
    return_code = active.proc.wait()
    active.log_handle.close()
    return {
        "name": active.entry.run.name,
        "stage": active.entry.run.stage,
        "gpu_slot": active.gpu_slot,
        "return_code": int(return_code),
        "elapsed_s": float(time.time() - active.start_time),
        "output_path": str(active.entry.output_path),
        "log_path": str(active.entry.log_path),
    }


def run_queue(
    *,
    plan: list[QueuePlanEntry],
    python_bin: str,
    script_path: str,
    output_dir: str,
    enwik8_path: str,
    cwd: Path,
    gpu_slots: tuple[str, ...],
    max_workers: int,
    skip_existing: bool,
    fail_fast: bool,
) -> dict[str, object]:
    pending = [entry for entry in plan if not (skip_existing and entry.output_path.exists())]
    skipped = [entry for entry in plan if skip_existing and entry.output_path.exists()]
    completed: list[dict[str, object]] = []
    failed = False
    slot_count = max(1, max_workers)
    active: list[ActiveProcess] = []
    slot_cursor = 0

    while pending or active:
        while pending and len(active) < slot_count and not failed:
            entry = pending.pop(0)
            if gpu_slots:
                occupied_slots = {proc_entry.gpu_slot for proc_entry in active if proc_entry.gpu_slot is not None}
                available_slots = [slot for slot in gpu_slots if slot not in occupied_slots]
                if available_slots:
                    gpu_slot = available_slots[0]
                else:
                    gpu_slot = gpu_slots[slot_cursor % len(gpu_slots)]
                    slot_cursor += 1
            else:
                gpu_slot = None
            active.append(
                launch_run(
                    entry=entry,
                    python_bin=python_bin,
                    script_path=script_path,
                    output_dir=output_dir,
                    enwik8_path=enwik8_path,
                    cwd=cwd,
                    gpu_slot=gpu_slot,
                )
            )
            print(f"[launch] {entry.run.name} stage={entry.run.stage} gpu={gpu_slot or 'cpu'} log={entry.log_path}")

        if not active:
            break

        time.sleep(1.0)
        still_active: list[ActiveProcess] = []
        for proc_entry in active:
            if proc_entry.proc.poll() is None:
                still_active.append(proc_entry)
                continue
            result = finalize_process(proc_entry)
            completed.append(result)
            status = "ok" if result["return_code"] == 0 else "fail"
            print(
                f"[done] {result['name']} status={status} elapsed_s={result['elapsed_s']:.1f} log={result['log_path']}"
            )
            if result["return_code"] != 0 and fail_fast:
                failed = True
        active = still_active

        if failed:
            for proc_entry in active:
                proc_entry.proc.terminate()
            for proc_entry in active:
                result = finalize_process(proc_entry)
                completed.append(result)
            break

    return {
        "requested_runs": int(len(plan)),
        "launched_runs": int(len(completed)),
        "skipped_runs": [entry.run.name for entry in skipped],
        "results": completed,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Queue and execute the staged evolutionary benchmark matrix")
    parser.add_argument("--python-bin", default="python3")
    parser.add_argument("--script-path", default="tools/evolutionary_benchmark.py")
    parser.add_argument("--output-dir", default="runs/evolutionary")
    parser.add_argument("--log-dir", default="runs/evolutionary/logs")
    parser.add_argument("--enwik8-path", default="/workspace/data/enwik8")
    parser.add_argument("--stages", default=None)
    parser.add_argument("--include-tags", default=None)
    parser.add_argument("--exclude-tags", default=None)
    parser.add_argument("--names", default=None)
    parser.add_argument("--gpus", default="auto", help="Comma-separated GPU ids or auto")
    parser.add_argument("--max-workers", type=int, default=0, help="0 means auto from detected GPU slots, otherwise fixed worker count")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summary-json", default=None)
    return parser


def main(argv: list[str]) -> int:
    parser = build_parser()
    args = parser.parse_args(argv[1:])

    runs = select_runs(
        build_matrix(),
        stages=parse_csv(args.stages),
        include_tags=parse_csv(args.include_tags),
        exclude_tags=parse_csv(args.exclude_tags),
        names=parse_csv(args.names),
    )
    output_dir = Path(args.output_dir)
    log_dir = Path(args.log_dir)
    plan = build_queue_plan(runs=runs, output_dir=output_dir, log_dir=log_dir)
    gpu_slots = detect_gpu_slots(args.gpus)
    max_workers = args.max_workers if args.max_workers > 0 else max(len(gpu_slots), 1)

    if args.dry_run:
        preview = {
            "gpu_slots": list(gpu_slots),
            "max_workers": int(max_workers),
            "runs": [
                {
                    "name": entry.run.name,
                    "stage": entry.run.stage,
                    "tags": list(entry.run.tags),
                    "output_path": str(entry.output_path),
                    "log_path": str(entry.log_path),
                    "command": list(entry.run.command_parts(args.python_bin, args.script_path, args.output_dir, args.enwik8_path)),
                }
                for entry in plan
            ],
        }
        print(json.dumps(preview, indent=2))
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    result = run_queue(
        plan=plan,
        python_bin=args.python_bin,
        script_path=args.script_path,
        output_dir=args.output_dir,
        enwik8_path=args.enwik8_path,
        cwd=REPO_ROOT,
        gpu_slots=gpu_slots,
        max_workers=max_workers,
        skip_existing=args.skip_existing,
        fail_fast=args.fail_fast,
    )
    if args.summary_json:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    failed = any(int(entry["return_code"]) != 0 for entry in result["results"])
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
