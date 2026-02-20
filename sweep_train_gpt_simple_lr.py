#!/usr/bin/env python3
"""
Sweep MATRIX_LR and SCALAR_LR for train_gpt_simple.py.

Runs trials sequentially on one host, captures console output, parses the final
validation loss from logs/<RUN_ID>.txt (or console fallback), and writes ranked
summaries to logs/sweeps/train_gpt_simple_matrix_scalar/<timestamp>/.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Trial:
    name: str
    matrix_lr: float
    scalar_lr: float
    env: dict[str, str]


VAL_LOSS_RE = re.compile(r"step:(\d+)/(\d+)\s+val_loss:([0-9]+(?:\.[0-9]+)?)")


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def parse_key_value(items: Iterable[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected KEY=VALUE, got: {item}")
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError(f"Invalid empty key in: {item}")
        parsed[key] = value
    return parsed


def parse_float_list(values_csv: str, name: str) -> list[float]:
    out: list[float] = []
    for raw in values_csv.split(","):
        raw = raw.strip()
        if not raw:
            continue
        try:
            out.append(float(raw))
        except ValueError as exc:
            raise ValueError(f"Invalid float in --{name}: {raw}") from exc
    if not out:
        raise ValueError(f"--{name} must contain at least one float")
    return out


def lr_tag(value: float) -> str:
    return f"{value:.6g}".replace("-", "m").replace(".", "p")


def build_trials(
    matrix_lrs: list[float],
    scalar_lrs: list[float],
    *,
    embed_lr: float,
    head_lr: float,
    beta2: float,
) -> list[Trial]:
    trials: list[Trial] = []
    for matrix_lr, scalar_lr in product(matrix_lrs, scalar_lrs):
        name = f"m{lr_tag(matrix_lr)}_s{lr_tag(scalar_lr)}"
        trials.append(
            Trial(
                name=name,
                matrix_lr=matrix_lr,
                scalar_lr=scalar_lr,
                env={
                    "EMBED_LR": f"{embed_lr:.8g}",
                    "HEAD_LR": f"{head_lr:.8g}",
                    "MATRIX_LR": f"{matrix_lr:.8g}",
                    "SCALAR_LR": f"{scalar_lr:.8g}",
                    "BETA2": f"{beta2:.8g}",
                },
            )
        )
    return trials


def parse_final_val_loss(log_path: Path) -> tuple[int | None, int | None, float | None]:
    if not log_path.exists():
        return None, None, None
    step = None
    total = None
    val_loss = None
    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            match = VAL_LOSS_RE.search(line)
            if match:
                step = int(match.group(1))
                total = int(match.group(2))
                val_loss = float(match.group(3))
    return step, total, val_loss


def parse_final_val_loss_console(console_log: Path) -> tuple[int | None, int | None, float | None]:
    if not console_log.exists():
        return None, None, None
    step = None
    total = None
    val_loss = None
    with console_log.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            match = VAL_LOSS_RE.search(line)
            if match:
                step = int(match.group(1))
                total = int(match.group(2))
                val_loss = float(match.group(3))
    return step, total, val_loss


def ensure_idle(wait: bool, timeout_s: int, poll_s: int) -> None:
    start = time.time()
    while True:
        proc = subprocess.run(
            [
                "bash",
                "-lc",
                r"""pgrep -af "torchrun .*train_gpt(_simple)?\.py|python[0-9.]* .*train_gpt(_simple)?\.py" | grep -v "sweep_train_gpt_simple_lr.py" || true""",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        active = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
        if not active:
            return

        active_msg = " | ".join(active[:3])
        print(f"[sweep] waiting for idle GPUs, active run(s): {active_msg}", flush=True)

        if not wait:
            raise RuntimeError("Detected active train_gpt/train_gpt_simple process")

        elapsed = time.time() - start
        if elapsed > timeout_s:
            raise TimeoutError(
                f"Timed out waiting for idle GPUs after {timeout_s}s; active={active_msg}"
            )
        time.sleep(poll_s)


def run_trial(
    trial: Trial,
    common_env: dict[str, str],
    prefix: str,
    output_dir: Path,
    python_exec: str,
    nproc_per_node: int,
) -> dict[str, object]:
    run_id = f"{prefix}_{trial.name}_{now_utc()}"
    train_log = Path("logs") / f"{run_id}.txt"
    console_log = output_dir / f"{run_id}.console.log"
    started = time.time()

    env = os.environ.copy()
    env.update(common_env)
    env.update(trial.env)
    env["RUN_ID"] = run_id

    cmd = [
        python_exec,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={nproc_per_node}",
        "train_gpt_simple.py",
    ]

    print(
        f"[sweep] trial={trial.name} run_id={run_id} cmd={' '.join(cmd)} env={json.dumps(trial.env, sort_keys=True)}",
        flush=True,
    )
    print(f"[sweep] console_log={console_log} train_log={train_log}", flush=True)

    rc = 1
    with console_log.open("w", encoding="utf-8") as fh:
        fh.write(f"# run_id={run_id}\n")
        fh.write(f"# started_utc={datetime.now(timezone.utc).isoformat()}\n")
        fh.write(f"# cmd={' '.join(cmd)}\n")
        fh.write(f"# trial_env={json.dumps(trial.env, sort_keys=True)}\n")
        fh.flush()

        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            fh.write(line)
        proc.wait()
        rc = proc.returncode

    step, total, val_loss = parse_final_val_loss(train_log)
    if val_loss is None:
        step, total, val_loss = parse_final_val_loss_console(console_log)

    elapsed = time.time() - started
    result = {
        "trial_name": trial.name,
        "matrix_lr": trial.matrix_lr,
        "scalar_lr": trial.scalar_lr,
        "run_id": run_id,
        "return_code": rc,
        "elapsed_seconds": round(elapsed, 2),
        "val_step": step,
        "val_total_steps": total,
        "final_val_loss": val_loss,
        "train_log": str(train_log),
        "console_log": str(console_log),
        "env": trial.env,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    print(
        f"[sweep] done trial={trial.name} rc={rc} val_loss={val_loss} elapsed_s={elapsed:.1f}",
        flush=True,
    )
    return result


def write_outputs(results: list[dict[str, object]], output_dir: Path) -> None:
    jsonl_path = output_dir / "results.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    summary_path = output_dir / "summary.tsv"
    sortable = sorted(
        results,
        key=lambda r: (
            r.get("final_val_loss") is None,
            float(r.get("final_val_loss") or 0.0),
            int(r.get("return_code") or 1),
        ),
    )
    fieldnames = [
        "rank",
        "trial_name",
        "matrix_lr",
        "scalar_lr",
        "run_id",
        "final_val_loss",
        "return_code",
        "elapsed_seconds",
        "val_step",
        "val_total_steps",
        "train_log",
        "console_log",
    ]
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for idx, row in enumerate(sortable, start=1):
            writer.writerow(
                {
                    "rank": idx,
                    "trial_name": row.get("trial_name"),
                    "matrix_lr": row.get("matrix_lr"),
                    "scalar_lr": row.get("scalar_lr"),
                    "run_id": row.get("run_id"),
                    "final_val_loss": row.get("final_val_loss"),
                    "return_code": row.get("return_code"),
                    "elapsed_seconds": row.get("elapsed_seconds"),
                    "val_step": row.get("val_step"),
                    "val_total_steps": row.get("val_total_steps"),
                    "train_log": row.get("train_log"),
                    "console_log": row.get("console_log"),
                }
            )
    print(f"[sweep] wrote {jsonl_path}", flush=True)
    print(f"[sweep] wrote {summary_path}", flush=True)
    if sortable:
        best = sortable[0]
        print(
            "[sweep] best "
            f"trial={best.get('trial_name')} "
            f"matrix_lr={best.get('matrix_lr')} "
            f"scalar_lr={best.get('scalar_lr')} "
            f"run_id={best.get('run_id')} "
            f"val_loss={best.get('final_val_loss')}",
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep MATRIX_LR and SCALAR_LR for train_gpt_simple.py")
    parser.add_argument(
        "--output-root",
        default="logs/sweeps/train_gpt_simple_matrix_scalar",
        help="Directory where sweep artifacts are written",
    )
    parser.add_argument(
        "--prefix",
        default="simple_lr_sweep",
        help="RUN_ID prefix for each trial",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to launch torch.distributed.run",
    )
    parser.add_argument(
        "--nproc-per-node",
        type=int,
        default=int(os.environ.get("NPROC_PER_NODE", "8")),
        help="Number of distributed processes per trial",
    )
    parser.add_argument(
        "--matrix-lrs",
        default=os.environ.get("MATRIX_LR_GRID", "0.0012,0.0015,0.0018,0.0022"),
        help="Comma-separated MATRIX_LR grid values",
    )
    parser.add_argument(
        "--scalar-lrs",
        default=os.environ.get("SCALAR_LR_GRID", "0.0055,0.0060,0.0070,0.0075,0.0080"),
        help="Comma-separated SCALAR_LR grid values",
    )
    parser.add_argument(
        "--embed-lr",
        type=float,
        default=float(os.environ.get("EMBED_LR", "0.60")),
        help="Fixed EMBED_LR applied to all trials",
    )
    parser.add_argument(
        "--head-lr",
        type=float,
        default=float(os.environ.get("HEAD_LR", "0.008")),
        help="Fixed HEAD_LR applied to all trials",
    )
    parser.add_argument(
        "--beta2",
        type=float,
        default=float(os.environ.get("BETA2", "0.95")),
        help="Fixed BETA2 applied to all trials",
    )
    parser.add_argument(
        "--num-scheduled-iterations",
        type=int,
        default=int(os.environ.get("NUM_SCHEDULED_ITERATIONS", "600")),
        help="NUM_SCHEDULED_ITERATIONS for each trial",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=int(os.environ.get("WARMUP_STEPS", "30")),
        help="WARMUP_STEPS for each trial",
    )
    parser.add_argument(
        "--val-loss-every",
        type=int,
        default=int(os.environ.get("VAL_LOSS_EVERY", "100")),
        help="VAL_LOSS_EVERY for each trial",
    )
    parser.add_argument(
        "--val-tokens",
        type=int,
        default=int(os.environ.get("VAL_TOKENS", "1048576")),
        help="VAL_TOKENS for each trial",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra env var override applied to all trials (repeatable)",
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=0,
        help="Run only the first N trials (0 means all)",
    )
    parser.add_argument(
        "--wait-for-idle",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Wait for existing train_gpt/train_gpt_simple jobs to finish before starting",
    )
    parser.add_argument(
        "--idle-timeout-seconds",
        type=int,
        default=3 * 3600,
        help="Max seconds to wait for idle GPUs",
    )
    parser.add_argument(
        "--idle-poll-seconds",
        type=int,
        default=30,
        help="Polling interval while waiting for idle GPUs",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print trial plan and exit",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_stamp = now_utc()
    output_dir = Path(args.output_root) / run_stamp
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix_lrs = parse_float_list(args.matrix_lrs, "matrix-lrs")
    scalar_lrs = parse_float_list(args.scalar_lrs, "scalar-lrs")
    trials = build_trials(
        matrix_lrs,
        scalar_lrs,
        embed_lr=args.embed_lr,
        head_lr=args.head_lr,
        beta2=args.beta2,
    )
    if args.max_trials > 0:
        trials = trials[: args.max_trials]

    common_env = {
        "NUM_SCHEDULED_ITERATIONS": str(args.num_scheduled_iterations),
        "WARMUP_STEPS": str(args.warmup_steps),
        "VAL_LOSS_EVERY": str(args.val_loss_every),
        "VAL_TOKENS": str(args.val_tokens),
        "SAVE_CHECKPOINT": "0",
    }
    common_env.update(parse_key_value(args.set))

    print(f"[sweep] output_dir={output_dir}", flush=True)
    print(f"[sweep] common_env={json.dumps(common_env, sort_keys=True)}", flush=True)
    print(
        f"[sweep] matrix_lrs={','.join(f'{x:g}' for x in matrix_lrs)} scalar_lrs={','.join(f'{x:g}' for x in scalar_lrs)}",
        flush=True,
    )
    print(f"[sweep] trials={len(trials)}", flush=True)

    if args.dry_run:
        for trial in trials:
            print(
                f"[sweep] trial={trial.name} env={json.dumps(trial.env, sort_keys=True)}",
                flush=True,
            )
        return

    ensure_idle(
        wait=args.wait_for_idle,
        timeout_s=args.idle_timeout_seconds,
        poll_s=args.idle_poll_seconds,
    )

    results: list[dict[str, object]] = []
    for trial in trials:
        result = run_trial(
            trial=trial,
            common_env=common_env,
            prefix=args.prefix,
            output_dir=output_dir,
            python_exec=args.python,
            nproc_per_node=args.nproc_per_node,
        )
        results.append(result)
        write_outputs(results, output_dir)

    print("[sweep] finished all trials", flush=True)


if __name__ == "__main__":
    main()
