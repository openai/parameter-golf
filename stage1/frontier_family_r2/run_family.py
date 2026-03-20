#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one Stage 1 frontier family config.")
    parser.add_argument("--slot", required=True, help="Portfolio slot id, e.g. P0")
    parser.add_argument("--nproc-per-node", type=int, default=1, help="Processes / GPUs to launch on the local node")
    parser.add_argument("--label", default="manual", help="Run grouping label, e.g. a100 or h100")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved config without launching training")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(__file__).with_name("run_configs.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    slots = {slot["slot"]: slot for slot in config["slots"]}
    if args.slot not in slots:
        raise SystemExit(f"Unknown slot '{args.slot}'. Available: {', '.join(sorted(slots))}")
    if args.nproc_per_node <= 0:
        raise SystemExit("--nproc-per-node must be positive")

    slot = slots[args.slot]
    repo_root = Path(__file__).resolve().parents[4]
    pgolf_root = repo_root / "pgolf" / "parameter-golf"
    runs_root = Path(__file__).resolve().parent / "runs" / args.label
    run_dir = runs_root / f"{slot['slot']}_{slot['name']}"
    run_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update(config["defaults"])
    env.update(slot["env"])
    env["RUN_ID"] = f"{config['batch_id']}_{args.label}_{slot['slot']}_{slot['name']}"
    if env["DATA_PATH"].startswith("./"):
        env["DATA_PATH"] = str((pgolf_root / "data" / "datasets" / "fineweb10B_sp1024").resolve())
    if env["TOKENIZER_PATH"].startswith("./"):
        env["TOKENIZER_PATH"] = str((pgolf_root / "data" / "tokenizers" / "fineweb_1024_bpe.model").resolve())
    env["PGOLF_STAGE1_BATCH"] = config["batch_id"]
    env["PGOLF_STAGE1_SLOT"] = slot["slot"]
    env["PGOLF_STAGE1_FAMILY"] = slot["family"]
    env["PGOLF_STAGE1_LABEL"] = args.label

    metadata = {
        "batch_id": config["batch_id"],
        "slot": slot["slot"],
        "name": slot["name"],
        "family": slot["family"],
        "hypothesis": slot["hypothesis"],
        "role": slot["role"],
        "label": args.label,
        "nproc_per_node": args.nproc_per_node,
        "why": slot["why"],
        "env": {k: env[k] for k in sorted(config["defaults"] | slot["env"])},
    }
    (run_dir / "config.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    python_bin = repo_root / ".venv" / "bin" / "python"
    if not python_bin.exists():
        raise SystemExit(f"Expected repo-local python at {python_bin}")
    train_script = str((pgolf_root / "train_gpt.py").resolve())
    if args.nproc_per_node == 1:
        cmd = [str(python_bin), train_script]
    else:
        cmd = [
            str(python_bin),
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={args.nproc_per_node}",
            train_script,
        ]

    if args.dry_run:
        print(json.dumps({"cwd": str(run_dir), "cmd": cmd, "metadata": metadata}, indent=2))
        return

    print(f"=== Stage1 Frontier Family R2 — {slot['slot']} {slot['name']} ===", flush=True)
    print(f"Run dir: {run_dir}", flush=True)
    print(f"Family: {slot['family']}  label:{args.label}  nproc_per_node:{args.nproc_per_node}", flush=True)
    subprocess.run(cmd, cwd=run_dir, env=env, check=True)


if __name__ == "__main__":
    main()
