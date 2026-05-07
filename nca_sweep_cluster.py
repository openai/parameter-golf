#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import fcntl
import os
import re
import subprocess
from pathlib import Path
from typing import Iterable

FINAL_STEP_VAL_RE = re.compile(
    r"step:(?P<step>\d+)/(?P<iters>\d+)\s+val_loss:(?P<val_loss>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
)
FINAL_INT8_VAL_RE = re.compile(
    r"final_int8_zlib_roundtrip_exact\s+val_loss:(?P<val_loss>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
)
FIELDS = [
    "timestamp_utc",
    "nca_mix",
    "nca_steps",
    "max_wallclock_seconds",
    "seed",
    "returncode",
    "status",
    "val_loss",
    "val_loss_source",
    "run_id",
    "log_path",
    "error",
]


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Cluster NCA sweep via torchrun --nproc_per_node=8.")
    p.add_argument("--train-script", default=str(root / "train_gpt.py"))
    p.add_argument("--output-csv", default=str(root / "cluster_nca_results.csv"))
    p.add_argument("--logs-dir", default=str(root / "logs" / "nca_sweep_cluster"))
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--max-wallclock-seconds", type=int, default=120)
    # Default subprocess timeout: 900s (15 min) covers first-run torch.compile warmup
    # (~10 min compilation) + 120s training + eval/export overhead.  Set to 0 to disable.
    p.add_argument("--timeout-sec", type=float, default=900.0, help="Subprocess timeout; <=0 disables.")
    # Shared Inductor / Triton cache dirs — all subprocesses point here so only the
    # first combo pays the torch.compile compilation cost (~10 min on H100).
    p.add_argument(
        "--inductor-cache-dir",
        default=str(root / ".cache" / "inductor"),
        help="Persistent torch.compile / Inductor cache shared across all sweep runs.",
    )
    p.add_argument(
        "--triton-cache-dir",
        default=str(root / ".cache" / "triton"),
        help="Persistent Triton JIT cache shared across all sweep runs.",
    )
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--extra-env", action="append", default=[], metavar="KEY=VALUE")
    return p.parse_args()


def iter_mix_values() -> Iterable[float]:
    for i in range(1, 10):
        yield round(i / 10.0, 1)


def ensure_csv_header_locked(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", newline="", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        f.seek(0, os.SEEK_END)
        if f.tell() == 0:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            f.flush()
            os.fsync(f.fileno())
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def load_completed(path: Path) -> set[tuple[float, int]]:
    done: set[tuple[float, int]] = set()
    if not path.exists():
        return done
    with path.open("r", newline="", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_SH)
        for row in csv.DictReader(f):
            try:
                if row.get("status") == "ok" and row.get("val_loss"):
                    done.add((round(float(row["nca_mix"]), 1), int(row["nca_steps"])))
            except Exception:
                continue
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    return done


def append_row_locked(path: Path, row: dict[str, object]) -> None:
    with path.open("a+", newline="", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        f.seek(0, os.SEEK_END)
        if f.tell() == 0:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writerow(row)
        f.flush()
        os.fsync(f.fileno())
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def apply_extra_env(env: dict[str, str], extra_env_items: list[str]) -> None:
    for item in extra_env_items:
        if "=" not in item:
            raise ValueError(f"Invalid --extra-env value {item!r}; expected KEY=VALUE.")
        k, v = item.split("=", 1)
        k = k.strip()
        if not k:
            raise ValueError(f"Invalid --extra-env value {item!r}; empty key.")
        env[k] = v


def run_one(
    train_script: str,
    logs_dir: Path,
    seed: int,
    timeout_sec: float,
    max_wallclock_seconds: int,
    nca_mix: float,
    nca_steps: int,
    extra_env: list[str],
    inductor_cache_dir: str = "",
    triton_cache_dir: str = "",
) -> dict[str, object]:
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"nca_cluster_mix{nca_mix:.1f}_steps{nca_steps}_{ts}"
    log_path = logs_dir / f"{run_id}.log"
    logs_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "RUN_ID": run_id,
            "SEED": str(seed),
            "NCA_MIX": f"{nca_mix:.1f}",
            "NCA_STEPS": str(nca_steps),
            "WARMUP_STEPS": "0",
            "MAX_WALLCLOCK_SECONDS": str(max_wallclock_seconds),
        }
    )
    # Share compiled kernel cache across all subprocesses so only the first run
    # pays the torch.compile (~10 min) and Triton JIT compilation cost.
    if inductor_cache_dir:
        Path(inductor_cache_dir).mkdir(parents=True, exist_ok=True)
        env["TORCHINDUCTOR_CACHE_DIR"] = inductor_cache_dir
    if triton_cache_dir:
        Path(triton_cache_dir).mkdir(parents=True, exist_ok=True)
        env["TRITON_CACHE_DIR"] = triton_cache_dir
    apply_extra_env(env, extra_env)

    cmd = ["torchrun", "--nproc_per_node=8", train_script]
    rc = -1
    err = ""
    last_val: float | None = None
    last_src = ""
    line_idx = 0

    with log_path.open("w", encoding="utf-8") as lf:
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                line_idx += 1
                lf.write(line)
                m = FINAL_STEP_VAL_RE.search(line)
                if m:
                    last_val = float(m.group("val_loss"))
                    last_src = f"step_val_loss@{line_idx}"
                m2 = FINAL_INT8_VAL_RE.search(line)
                if m2:
                    last_val = float(m2.group("val_loss"))
                    last_src = "final_int8_zlib_roundtrip_exact"
            rc = proc.wait(timeout=timeout_sec if timeout_sec > 0 else None)
        except subprocess.TimeoutExpired:
            err = f"timeout>{timeout_sec}s"
            proc.kill()
            rc = proc.wait()
        except Exception as e:
            err = str(e)

    status = "ok" if rc == 0 and last_val is not None else "failed"
    if not err and last_val is None:
        err = "no_val_loss_found"
    return {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "nca_mix": f"{nca_mix:.1f}",
        "nca_steps": nca_steps,
        "max_wallclock_seconds": max_wallclock_seconds,
        "seed": seed,
        "returncode": rc,
        "status": status,
        "val_loss": f"{last_val:.8f}" if last_val is not None else "",
        "val_loss_source": last_src,
        "run_id": run_id,
        "log_path": str(log_path),
        "error": err,
    }


def main() -> int:
    args = parse_args()
    output_csv = Path(args.output_csv).resolve()
    logs_dir = Path(args.logs_dir).resolve()
    ensure_csv_header_locked(output_csv)
    completed = load_completed(output_csv) if args.resume else set()

    total = 0
    queued = 0
    for mix in iter_mix_values():
        for steps in range(2, 9):
            total += 1
            if (mix, steps) in completed:
                continue
            queued += 1
            if args.dry_run:
                print(f"[plan] mix={mix:.1f} steps={steps}")
                continue
            print(f"[run] mix={mix:.1f} steps={steps}")
            row = run_one(
                train_script=args.train_script,
                logs_dir=logs_dir,
                seed=args.seed,
                timeout_sec=args.timeout_sec,
                max_wallclock_seconds=args.max_wallclock_seconds,
                nca_mix=mix,
                nca_steps=steps,
                extra_env=args.extra_env,
                inductor_cache_dir=args.inductor_cache_dir,
                triton_cache_dir=args.triton_cache_dir,
            )
            append_row_locked(output_csv, row)
            print(
                f"[{row['status']}] mix={row['nca_mix']} steps={row['nca_steps']} "
                f"val_loss={row['val_loss'] or 'NA'} rc={row['returncode']}"
            )
    print(f"[done] total={total} queued={queued} csv={output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
