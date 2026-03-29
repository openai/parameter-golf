from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_SMOKE_ENV = {
    "ITERATIONS": "50",
    "WARMUP_STEPS": "0",
    "TRAIN_BATCH_TOKENS": "8192",
    "VAL_BATCH_SIZE": "524288",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optuna runner for local Parameter Golf experiment proxies.")
    parser.add_argument("--experiment", choices=("exp01_mixed_export", "exp02_factored_embeddings"), required=True)
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--study-name", default="parameter-golf-local")
    parser.add_argument("--storage", default=None)
    parser.add_argument("--sampler-seed", type=int, default=1337)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--full-budget", action="store_true")
    return parser.parse_args()


def build_trial_env(experiment: str, trial) -> dict[str, str]:
    env: dict[str, str] = {}
    if experiment == "exp01_mixed_export":
        env["INT4_CLIP_PERCENTILE"] = f"{trial.suggest_float('int4_clip_percentile', 99.90, 99.999, log=False):.6f}"
        env["INT8_KEEP_FLOAT_MAX_NUMEL"] = str(trial.suggest_categorical("int8_keep_float_max_numel", [32768, 65536, 98304]))
        env["WARMDOWN_ITERS"] = str(trial.suggest_categorical("warmdown_iters", [600, 900, 1200, 1800]))
        env["WARMUP_STEPS"] = str(trial.suggest_categorical("warmup_steps", [0, 2, 5]))
    elif experiment == "exp02_factored_embeddings":
        env["FACTORIZED_EMBED_DIM"] = str(trial.suggest_categorical("factorized_embed_dim", [96, 128, 160, 192]))
        env["TIED_EMBED_LR"] = f"{trial.suggest_float('tied_embed_lr', 0.02, 0.08, log=True):.6f}"
        env["MATRIX_LR"] = f"{trial.suggest_float('matrix_lr', 0.02, 0.06, log=True):.6f}"
        env["WARMDOWN_ITERS"] = str(trial.suggest_categorical("warmdown_iters", [600, 900, 1200, 1800]))
        env["WARMUP_STEPS"] = str(trial.suggest_categorical("warmup_steps", [0, 2, 5]))
    else:
        raise ValueError(f"Unsupported experiment: {experiment}")
    return env


def run_trial(experiment: str, env_updates: dict[str, str], full_budget: bool) -> tuple[float, str]:
    exp_dir = ROOT / experiment
    env = os.environ.copy()
    env.setdefault("RUN_ID", f"{experiment}_optuna")
    if not full_budget:
        for key, value in DEFAULT_SMOKE_ENV.items():
            env.setdefault(key, value)
    env.update(env_updates)
    command = ["uv", "run", "python", "train_gpt.py"]
    proc = subprocess.run(
        command,
        cwd=exp_dir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    output = proc.stdout
    if proc.returncode != 0:
        raise RuntimeError(output[-4000:])
    match = re.search(r"final_[^ ]+_roundtrip_exact val_loss:[0-9.]+ val_bpb:([0-9.]+)", output)
    if not match:
        raise RuntimeError(output[-4000:])
    return float(match.group(1)), output


def main() -> None:
    args = parse_args()
    try:
        import optuna
    except ImportError as exc:
        raise SystemExit("optuna is not installed. Use `uv add optuna` or `uv pip install optuna` in this repo.") from exc

    sampler = optuna.samplers.TPESampler(seed=args.sampler_seed)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="minimize",
        load_if_exists=True,
        sampler=sampler,
    )

    def objective(trial):
        env_updates = build_trial_env(args.experiment, trial)
        score, output = run_trial(args.experiment, env_updates, full_budget=args.full_budget)
        trial.set_user_attr("env", env_updates)
        tail = "\n".join(output.strip().splitlines()[-20:])
        trial.set_user_attr("output_tail", tail)
        return score

    study.optimize(objective, n_trials=args.trials, timeout=args.timeout)
    best = study.best_trial
    print(f"best_value={best.value}")
    print(f"best_params={best.params}")
    print(f"best_env={best.user_attrs.get('env')}")


if __name__ == "__main__":
    main()
