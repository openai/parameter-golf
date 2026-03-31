#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one Stage 1 H100 family config.")
    parser.add_argument("--slot", required=True, help="Portfolio slot id, e.g. P0")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved config without launching training")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(__file__).with_name("run_configs.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    slots = {slot["slot"]: slot for slot in config["slots"]}
    if args.slot not in slots:
        raise SystemExit(f"Unknown slot '{args.slot}'. Available: {', '.join(sorted(slots))}")

    slot = slots[args.slot]
    repo_root = Path(__file__).resolve().parents[4]
    pgolf_root = repo_root / "pgolf" / "parameter-golf"
    runs_root = pgolf_root / "stage1" / "h100_family_r1" / "runs"
    run_dir = runs_root / f"{slot['slot']}_{slot['name']}"
    run_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update(config["defaults"])
    env.update(slot["env"])
    env["RUN_ID"] = f"{config['batch_id']}_{slot['slot']}_{slot['name']}"
    env["DATA_PATH"] = str((pgolf_root / "data" / "datasets" / "fineweb10B_sp1024").resolve()) if env["DATA_PATH"].startswith("./") else env["DATA_PATH"]
    env["TOKENIZER_PATH"] = str((pgolf_root / "data" / "tokenizers" / "fineweb_1024_bpe.model").resolve()) if env["TOKENIZER_PATH"].startswith("./") else env["TOKENIZER_PATH"]
    env["PGOLF_STAGE1_BATCH"] = config["batch_id"]
    env["PGOLF_STAGE1_SLOT"] = slot["slot"]
    env["PGOLF_STAGE1_HYPOTHESIS"] = slot["hypothesis"]

    metadata = {
        "batch_id": config["batch_id"],
        "slot": slot["slot"],
        "name": slot["name"],
        "hypothesis": slot["hypothesis"],
        "family": slot["family"],
        "role": slot["role"],
        "why": slot["why"],
        "env": {k: env[k] for k in sorted(config["defaults"] | slot["env"])}
    }
    (run_dir / "config.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    python_bin = repo_root / ".venv" / "bin" / "python"
    if not python_bin.exists():
        raise SystemExit(f"Expected repo-local python at {python_bin}")
    cmd = [str(python_bin), str((pgolf_root / "train_gpt.py").resolve())]
    if args.dry_run:
        print(json.dumps({"cwd": str(run_dir), "cmd": cmd, "metadata": metadata}, indent=2))
        return

    print(f"=== Stage1 H100 Family R1 — {slot['slot']} {slot['name']} ===", flush=True)
    print(f"Run dir: {run_dir}", flush=True)
    print(f"Hypothesis: {slot['hypothesis']} ({slot['family']})", flush=True)
    subprocess.run(cmd, cwd=run_dir, env=env, check=True)


if __name__ == "__main__":
    main()
