#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from autoresearch_pg.lib.objective import evaluate_metrics
from autoresearch_pg.lib.locking import state_lock
from autoresearch_pg.lib.session_registry import get_codex_session, update_codex_session
from autoresearch_pg.lib.train_log import parse_train_log
from autoresearch_pg.lib.workspace import (
    append_jsonl,
    candidate_dir,
    config_root,
    dump_json,
    ensure_layout,
    iso_utc,
    relative_to_repo,
    run_dir,
    state_root,
    update_best_state,
    utc_stamp,
    load_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a candidate under a named tier")
    parser.add_argument("--candidate", required=True, help="Candidate id")
    parser.add_argument("--tier", required=True, help="Tier name from config/tiers.json")
    parser.add_argument(
        "--replay-train-log",
        help="Optional path to an existing train.log to replay through the harness without executing torchrun.",
    )
    return parser


def load_tier(name: str) -> dict[str, Any]:
    payload = json.loads((config_root() / "tiers.json").read_text(encoding="utf-8"))
    tiers = payload.get("tiers", {})
    if name not in tiers:
        raise KeyError(f"unknown tier {name!r}")
    return tiers[name]


def expand_value(value: str, mapping: dict[str, str]) -> str:
    out = value
    for key, replacement in mapping.items():
        out = out.replace("${" + key + "}", replacement)
    return out


def load_train_log_path(run_dir_path: Path, run_id: str) -> Path:
    candidate = run_dir_path / "logs" / f"{run_id}.txt"
    if candidate.is_file():
        final_log = run_dir_path / "train.log"
        shutil.copy2(candidate, final_log)
        return final_log
    fallback = run_dir_path / "stdout.log"
    final_log = run_dir_path / "train.log"
    shutil.copy2(fallback, final_log)
    return final_log


def main() -> int:
    args = build_parser().parse_args()
    ensure_layout()

    cand_dir = candidate_dir(args.candidate)
    if not cand_dir.is_dir():
        raise FileNotFoundError(f"candidate directory not found: {cand_dir}")

    tier = load_tier(args.tier)
    run_id = f"{args.candidate}_{args.tier}_{utc_stamp()}"
    run_dir_path = run_dir(args.candidate, args.tier, run_id)
    run_dir_path.mkdir(parents=True, exist_ok=False)

    candidate_meta = load_json(cand_dir / "meta.json", default={})
    entrypoint_filename = candidate_meta.get("entrypoint_filename", "train_gpt.py")
    source_entrypoint = cand_dir / entrypoint_filename
    if not source_entrypoint.is_file():
        raise FileNotFoundError(f"candidate entrypoint not found: {source_entrypoint}")

    local_entrypoint = run_dir_path / entrypoint_filename
    shutil.copy2(source_entrypoint, local_entrypoint)
    if (cand_dir / "meta.json").is_file():
        shutil.copy2(cand_dir / "meta.json", run_dir_path / "candidate_meta.json")
    if (cand_dir / "notes.md").is_file():
        shutil.copy2(cand_dir / "notes.md", run_dir_path / "candidate_notes.md")

    mapping = {
        "REPO_ROOT": str(REPO_ROOT),
        "CANDIDATE_DIR": str(cand_dir.resolve()),
        "RUN_DIR": str(run_dir_path.resolve()),
    }
    env = os.environ.copy()
    env_snapshot_keys = set()
    for key, value in tier.get("env", {}).items():
        env[key] = expand_value(value, mapping)
        env_snapshot_keys.add(key)
    for key, value in candidate_meta.get("env_overrides", {}).items():
        env[key] = str(value)
        env_snapshot_keys.add(key)
    env["RUN_ID"] = run_id
    env_snapshot_keys.add("RUN_ID")
    env.setdefault("PYTHONUNBUFFERED", "1")

    stdout_log_path = run_dir_path / "stdout.log"
    if candidate_meta.get("proposal_engine") == "codex" or get_codex_session(args.candidate) is not None:
        update_codex_session(
            args.candidate,
            status="training",
            tier=args.tier,
            run_id=run_id,
            run_dir=str(run_dir_path),
            train_log=str(run_dir_path / "train.log"),
            stdout_log=str(stdout_log_path),
        )
    if args.replay_train_log:
        replay_src = Path(args.replay_train_log).resolve()
        if not replay_src.is_file():
            raise FileNotFoundError(f"replay train.log not found: {replay_src}")
        cmd = ["replay_train_log", str(replay_src)]
        stdout_log_path.write_text(
            f"replayed existing train.log from {replay_src}\n",
            encoding="utf-8",
        )
        train_log_path = run_dir_path / "train.log"
        shutil.copy2(replay_src, train_log_path)
        return_code = 0
    else:
        launcher = tier.get("launcher", "torchrun")
        if launcher == "python":
            cmd = [sys.executable, str(local_entrypoint)]
        elif launcher == "torchrun":
            torchrun = shutil.which("torchrun")
            if torchrun is None:
                raise FileNotFoundError(
                    "torchrun was not found on PATH. Install the training runtime or use --replay-train-log."
                )
            cmd = [
                torchrun,
                "--standalone",
                f"--nproc_per_node={int(tier['nproc_per_node'])}",
                str(local_entrypoint),
            ]
        else:
            raise ValueError(f"unsupported tier launcher {launcher!r}")

        proc = subprocess.run(
            cmd,
            cwd=run_dir_path,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        stdout_log_path.write_text(proc.stdout, encoding="utf-8")
        train_log_path = load_train_log_path(run_dir_path, run_id)
        return_code = proc.returncode

    parsed = parse_train_log(train_log_path)
    objective = evaluate_metrics(parsed)

    run_payload = {
        "candidate_id": args.candidate,
        "tier": args.tier,
        "run_id": run_id,
        "created_at": iso_utc(),
        "candidate_dir": relative_to_repo(cand_dir),
        "entrypoint": relative_to_repo(source_entrypoint),
        "primary_family": candidate_meta.get("primary_family"),
        "secondary_tags": candidate_meta.get("secondary_tags", []),
        "template_id": candidate_meta.get("template_id"),
        "mutation_operator": candidate_meta.get("mutation_operator"),
        "mutation_payload": candidate_meta.get("mutation_payload"),
        "parent_candidate_id": candidate_meta.get("parent_candidate_id"),
        "run_dir": relative_to_repo(run_dir_path),
        "command": cmd,
        "env": {key: env[key] for key in sorted(env_snapshot_keys) if key in env},
        "return_code": return_code,
        "train_log": relative_to_repo(train_log_path),
        "stdout_log": relative_to_repo(stdout_log_path),
        "metrics": parsed,
        "objective": objective,
    }
    dump_json(run_dir_path / "run.json", run_payload)
    with state_lock():
        append_jsonl(state_root() / "ledger.jsonl", run_payload)
        update_best_state(run_payload)
    if candidate_meta.get("proposal_engine") == "codex" or get_codex_session(args.candidate) is not None:
        update_codex_session(
            args.candidate,
            status="completed" if return_code == 0 else "failed",
            tier=args.tier,
            run_id=run_id,
            run_dir=str(run_dir_path),
            train_log=str(train_log_path),
            stdout_log=str(stdout_log_path),
            run_return_code=return_code,
            valid=objective.get("valid"),
            post_quant_val_bpb=objective.get("post_quant_val_bpb"),
            bytes_total=objective.get("bytes_total"),
        )

    print(
        f"candidate={args.candidate} tier={args.tier} "
        f"valid={objective['valid']} post_quant_val_bpb={objective['post_quant_val_bpb']} "
        f"bytes_total={objective['bytes_total']} proxy_score={objective['proxy_score']:.6f}"
    )
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
