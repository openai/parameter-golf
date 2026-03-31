#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Any

ROOT = Path(__file__).resolve().parents[2]
COBRA_DIR = Path(__file__).resolve().parent
DEFAULT_CANDIDATES = COBRA_DIR / "candidates.json"
DEFAULT_PROFILE = COBRA_DIR / "profiles" / "cobra_base_quality.env"
DEFAULT_TRAIN_SCRIPT = ROOT / "experiments" / "A_wing" / "green_1" / "train_gpt.py"

RE_BASE = re.compile(r"final_int6_sliding_window_exact val_loss:([0-9.]+) val_bpb:([0-9.]+)")
RE_DIAG = re.compile(r"DIAGNOSTIC post_ema val_loss:([0-9.]+) val_bpb:([0-9.]+)")
RE_STOP = re.compile(r"stopping_early: wallclock_cap train_time:(\d+)ms step:(\d+)/(\d+)")
RE_PEAK = re.compile(r"peak memory allocated: (\d+) MiB")


def parse_env_file(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not path.exists():
        raise FileNotFoundError(path)
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def load_candidates(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError("candidates.json must contain a list")
    return data


def find_candidate(cands: List[Dict[str, Any]], name: str) -> Dict[str, Any]:
    for c in cands:
        if c.get("name") == name:
            return c
    names = ", ".join(x.get("name", "<unnamed>") for x in cands)
    raise KeyError(f"candidate {name} not found. Available: {names}")


def resolved_env_for_candidate(candidate: Dict[str, Any], fallback_profile: Path) -> Dict[str, str]:
    rel_profile = candidate.get("profile")
    profile_path = (COBRA_DIR / rel_profile).resolve() if rel_profile else fallback_profile
    env = parse_env_file(profile_path)
    for k, v in (candidate.get("overrides") or {}).items():
        env[str(k)] = str(v)
    return env


def build_command(
    env_overrides: Dict[str, str],
    seed: int,
    nproc: int,
    train_script: Path,
    log_file: Path,
) -> str:
    env_parts = [f"SEED={seed}"]
    for k in sorted(env_overrides):
        env_parts.append(f"{k}={shlex.quote(env_overrides[k])}")
    env_prefix = " ".join(env_parts)
    cmd = (
        f"cd {shlex.quote(str(ROOT))} && "
        f"{env_prefix} "
        f"torchrun --standalone --nproc_per_node={nproc} "
        f"{shlex.quote(str(train_script))} "
        f"2>&1 | tee {shlex.quote(str(log_file))}"
    )
    return cmd


def parse_log(path: Path) -> Dict[str, Any]:
    text = path.read_text(errors="ignore")
    out: Dict[str, Any] = {"log": str(path), "base_bpb": None, "diag_bpb": None, "step": None, "train_ms": None, "peak_mib": None}

    m = RE_BASE.search(text)
    if m:
        out["base_loss"] = float(m.group(1))
        out["base_bpb"] = float(m.group(2))

    d = RE_DIAG.search(text)
    if d:
        out["diag_loss"] = float(d.group(1))
        out["diag_bpb"] = float(d.group(2))

    s = RE_STOP.search(text)
    if s:
        out["train_ms"] = int(s.group(1))
        out["step"] = int(s.group(2))
        out["iterations"] = int(s.group(3))

    p = RE_PEAK.search(text)
    if p:
        out["peak_mib"] = int(p.group(1))

    return out


def cmd_plan(args: argparse.Namespace) -> int:
    cands = load_candidates(Path(args.candidates))
    print("COBRA plan mode")
    print(f"repo_root      : {ROOT}")
    print(f"train_script   : {args.train_script}")
    print(f"default_profile: {args.profile}")
    print(f"seed           : {args.seed}")
    print(f"nproc          : {args.nproc}")
    print()
    print("Candidates:")
    for c in cands:
        print(f"- {c['name']}: {c.get('description', '')}")

    if args.show_commands:
        print("\nCommand preview:")
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        for c in cands:
            env_map = resolved_env_for_candidate(c, Path(args.profile))
            log_file = ROOT / "logs" / f"cobra_{c['name']}_s{args.seed}_{ts}.log"
            cmd = build_command(env_map, args.seed, args.nproc, Path(args.train_script), log_file)
            print(f"\n[{c['name']}]\n{cmd}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    cands = load_candidates(Path(args.candidates))
    c = find_candidate(cands, args.candidate)
    env_map = resolved_env_for_candidate(c, Path(args.profile))

    if args.max_wallclock is not None:
        env_map["MAX_WALLCLOCK_SECONDS"] = str(args.max_wallclock)

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = ROOT / "logs" / f"cobra_{c['name']}_s{args.seed}_{ts}.log"
    cmd = build_command(env_map, args.seed, args.nproc, Path(args.train_script), log_file)

    print(f"candidate: {c['name']}")
    print(f"log_file : {log_file}")
    print("command  :")
    print(cmd)

    if not args.execute:
        print("\nDry-run only. Add --execute to launch.")
        return 0

    log_file.parent.mkdir(parents=True, exist_ok=True)
    rc = subprocess.call(["/bin/bash", "-lc", cmd])
    print(f"exit_code: {rc}")
    return rc


def cmd_summarize(args: argparse.Namespace) -> int:
    files = [Path(p) for p in sorted(glob.glob(args.glob))]
    if not files:
        print(f"No files matched: {args.glob}")
        return 1

    rows = [parse_log(p) for p in files]
    rows.sort(key=lambda r: (float("inf") if r["base_bpb"] is None else r["base_bpb"], r["log"]))

    print("base_bpb\tdiag_bpb\tstep\ttrain_ms\tpeak_mib\tlog")
    for r in rows:
        def fmt(v: Any) -> str:
            if v is None:
                return "-"
            if isinstance(v, float):
                return f"{v:.8f}"
            return str(v)

        print(
            "\t".join(
                [
                    fmt(r.get("base_bpb")),
                    fmt(r.get("diag_bpb")),
                    fmt(r.get("step")),
                    fmt(r.get("train_ms")),
                    fmt(r.get("peak_mib")),
                    r["log"],
                ]
            )
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="COBRA harness (base-quality plan/run/summarize)")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_plan = sub.add_parser("plan", help="Show candidate plan")
    p_plan.add_argument("--candidates", default=str(DEFAULT_CANDIDATES))
    p_plan.add_argument("--profile", default=str(DEFAULT_PROFILE))
    p_plan.add_argument("--train-script", default=str(DEFAULT_TRAIN_SCRIPT))
    p_plan.add_argument("--seed", type=int, default=1337)
    p_plan.add_argument("--nproc", type=int, default=8)
    p_plan.add_argument("--show-commands", action="store_true")
    p_plan.set_defaults(func=cmd_plan)

    p_run = sub.add_parser("run", help="Run one candidate (dry-run by default)")
    p_run.add_argument("--candidates", default=str(DEFAULT_CANDIDATES))
    p_run.add_argument("--profile", default=str(DEFAULT_PROFILE))
    p_run.add_argument("--train-script", default=str(DEFAULT_TRAIN_SCRIPT))
    p_run.add_argument("--candidate", required=True)
    p_run.add_argument("--seed", type=int, default=1337)
    p_run.add_argument("--nproc", type=int, default=8)
    p_run.add_argument("--max-wallclock", type=float, default=None)
    p_run.add_argument("--execute", action="store_true")
    p_run.set_defaults(func=cmd_run)

    p_sum = sub.add_parser("summarize", help="Summarize Cobra logs")
    p_sum.add_argument("--glob", default=str(ROOT / "logs" / "cobra_*.log"))
    p_sum.set_defaults(func=cmd_summarize)

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
