from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[2]
ABLATION_DIR = ROOT / "ablations"
LOG_DIR = ABLATION_DIR / "logs"
GRAPH_PATH = ABLATION_DIR / "ablations_loss.svg"
SUMMARY_PATH = ABLATION_DIR / "ablations_summary.tsv"
PLOT_SCRIPT = ROOT / "plot_ablation_losses.py"


@dataclass(frozen=True)
class Ablation:
    slug: str
    description: str
    env: dict[str, str]


ABLATIONS = [
    Ablation(
        "baseline",
        "No train adapters and no TTT adapters",
        {"TRAIN_ADAPTER_KIND": "none", "TTT_ADAPTER_KIND": "none"},
    ),
    Ablation(
        "lora_ttt",
        "LoRA TTT control",
        {
            "TRAIN_ADAPTER_KIND": "none",
            "TTT_ADAPTER_KIND": "lora",
            "TTT_ADAPTER_RANK": "8",
            "TTT_ADAPTER_LR": "0.01",
        },
    ),
    Ablation(
        "random_ttt",
        "Random-map TTT only",
        {
            "TRAIN_ADAPTER_KIND": "none",
            "TTT_ADAPTER_KIND": "random_diag",
            "TTT_ADAPTER_RANK": "8",
            "TTT_ADAPTER_LR": "0.01",
            "ADAPTER_RANDOM_DIST": "rademacher",
            "ADAPTER_SEED": "20260331",
        },
    ),
    Ablation(
        "random_train",
        "Random-map train only",
        {
            "TRAIN_ADAPTER_KIND": "random_diag",
            "TRAIN_ADAPTER_RANK": "8",
            "TRAIN_ADAPTER_LR": "0.02",
            "TRAIN_ADAPTER_TARGETS": "q,v,lm_head",
            "TTT_ADAPTER_KIND": "none",
            "ADAPTER_RANDOM_DIST": "rademacher",
            "ADAPTER_SEED": "20260331",
        },
    ),
    Ablation(
        "random_train_ttt",
        "Random-map train + random-map TTT",
        {
            "TRAIN_ADAPTER_KIND": "random_diag",
            "TRAIN_ADAPTER_RANK": "8",
            "TRAIN_ADAPTER_LR": "0.02",
            "TRAIN_ADAPTER_TARGETS": "q,v,lm_head",
            "TTT_ADAPTER_KIND": "random_diag",
            "TTT_ADAPTER_RANK": "8",
            "TTT_ADAPTER_LR": "0.01",
            "ADAPTER_RANDOM_DIST": "rademacher",
            "ADAPTER_SEED": "20260331",
        },
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all random-map ablations and update the loss graph.")
    parser.add_argument("--nproc-per-node", type=int, default=8)
    parser.add_argument("--max-wallclock-seconds", type=float, default=600.0)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--val-loss-every", type=int, default=1000)
    parser.add_argument("--only", type=str, default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    return parser.parse_args()


def selected_ablations(only: str) -> list[Ablation]:
    if not only.strip():
        return ABLATIONS
    wanted = {item.strip() for item in only.split(",") if item.strip()}
    selected = [ablation for ablation in ABLATIONS if ablation.slug in wanted]
    missing = sorted(wanted - {ablation.slug for ablation in selected})
    if missing:
        raise ValueError(f"Unknown ablations in --only: {missing}")
    return selected


def base_env(args: argparse.Namespace) -> dict[str, str]:
    return {
        "DATA_PATH": str(REPO_ROOT / "data" / "datasets" / "fineweb10B_sp1024"),
        "TOKENIZER_PATH": str(REPO_ROOT / "data" / "tokenizers" / "fineweb_1024_bpe.model"),
        "VOCAB_SIZE": "1024",
        "SEED": str(args.seed),
        "VAL_LOSS_EVERY": str(args.val_loss_every),
        "MAX_WALLCLOCK_SECONDS": str(args.max_wallclock_seconds),
        "PYTHONUNBUFFERED": "1",
    }


def validate_paths(env: dict[str, str]) -> None:
    data_path = Path(env["DATA_PATH"])
    tokenizer_path = Path(env["TOKENIZER_PATH"])
    if not data_path.exists():
        raise FileNotFoundError(f"Missing DATA_PATH: {data_path}")
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Missing TOKENIZER_PATH: {tokenizer_path}")


def resolve_torchrun() -> str:
    env_override = os.environ.get("TORCHRUN_BIN")
    if env_override:
        return env_override
    path_hit = shutil.which("torchrun")
    if path_hit is not None:
        return path_hit
    sibling = Path(sys.executable).with_name("torchrun")
    if sibling.exists():
        return str(sibling)
    sibling_exe = Path(sys.executable).with_name("torchrun.exe")
    if sibling_exe.exists():
        return str(sibling_exe)
    raise FileNotFoundError("Could not find `torchrun`; set TORCHRUN_BIN if needed")


def plot_outputs() -> None:
    subprocess.run(
        [
            sys.executable,
            str(PLOT_SCRIPT),
            "--log-dir",
            str(LOG_DIR),
            "--output",
            str(GRAPH_PATH),
            "--summary",
            str(SUMMARY_PATH),
        ],
        cwd=str(ROOT),
        check=True,
    )


def run_one(ablation: Ablation, args: argparse.Namespace, common_env: dict[str, str], torchrun_bin: str) -> int:
    env = os.environ.copy()
    env.update(common_env)
    env.update(ablation.env)
    env["RUN_ID"] = f"abl_{ablation.slug}_8xh100"
    command = [torchrun_bin, "--standalone", f"--nproc_per_node={args.nproc_per_node}", "train_gpt.py"]
    log_path = LOG_DIR / f"{ablation.slug}.log"
    started_at = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{started_at}] starting {ablation.slug}: {ablation.description}", flush=True)
    if args.dry_run:
        print(f"DRY RUN {ablation.slug}: {' '.join(command)}", flush=True)
        return 0

    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"# slug={ablation.slug}\n")
        handle.write(f"# description={ablation.description}\n")
        handle.write(f"# started_at={started_at}\n")
        handle.write(f"# command={' '.join(command)}\n")
        for key in sorted(common_env.keys() | ablation.env.keys() | {'RUN_ID'}):
            handle.write(f"# env.{key}={env[key]}\n")
        handle.write("\n")
        handle.flush()
        try:
            proc = subprocess.run(
                command,
                cwd=str(ROOT),
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                timeout=args.timeout_seconds,
                check=False,
            )
            return_code = proc.returncode
        except subprocess.TimeoutExpired:
            handle.write(f"\nTIMEOUT after {args.timeout_seconds} seconds\n")
            return_code = 124
    finished_at = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{finished_at}] finished {ablation.slug} returncode={return_code} log={log_path}", flush=True)
    plot_outputs()
    return return_code


def main() -> int:
    args = parse_args()
    chosen = selected_ablations(args.only)
    common_env = base_env(args)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    validate_paths(common_env)
    torchrun_bin = resolve_torchrun()

    if args.dry_run:
        print(f"Record dir: {ROOT}")
        print(f"Logs dir:   {LOG_DIR}")
        print(f"Graph path: {GRAPH_PATH}")
        print(f"Torchrun:   {torchrun_bin}")

    exit_code = 0
    for ablation in chosen:
        return_code = run_one(ablation, args, common_env, torchrun_bin)
        if return_code != 0:
            exit_code = return_code
            if args.stop_on_error:
                break

    if not args.dry_run:
        plot_outputs()
        print(f"Summary TSV: {SUMMARY_PATH}")
        print(f"Loss graph:  {GRAPH_PATH}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
