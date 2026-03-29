from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_ROOT = ROOT / "experiments"
DEFAULT_SMOKE_ENV = {
    "ITERATIONS": "50",
    "WARMUP_STEPS": "0",
    "TRAIN_BATCH_TOKENS": "8192",
    "VAL_BATCH_SIZE": "524288",
    "VAL_TOKEN_LIMIT": str(1_048_576),
}

FAMILY_TO_DIR = {
    "exp01_mixed_export": EXPERIMENTS_ROOT / "exp01_mixed_export",
    "exp02_factored_embeddings": EXPERIMENTS_ROOT / "exp02_factored_embeddings",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one Parameter Golf candidate family.")
    parser.add_argument("--family", choices=sorted(FAMILY_TO_DIR), required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Additional env var override; may be passed multiple times.",
    )
    return parser.parse_args()


def parse_set_args(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected KEY=VALUE, got: {item}")
        key, value = item.split("=", 1)
        out[key] = value
    return out


def main() -> int:
    args = parse_args()
    exp_dir = FAMILY_TO_DIR[args.family]
    env = os.environ.copy()
    if args.smoke:
        env.update(DEFAULT_SMOKE_ENV)
    env.setdefault("DATA_PATH", str(ROOT / "data" / "datasets" / "fineweb10B_sp1024"))
    env.setdefault("TOKENIZER_PATH", str(ROOT / "data" / "tokenizers" / "fineweb_1024_bpe.model"))
    env["RUN_ID"] = args.run_id or env.get("RUN_ID", f"{args.family}_run")
    env.update(parse_set_args(args.set))

    cmd = ["uv", "run", "python", "train_gpt.py"]
    proc = subprocess.run(
        cmd,
        cwd=exp_dir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    sys.stdout.write(proc.stdout)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
