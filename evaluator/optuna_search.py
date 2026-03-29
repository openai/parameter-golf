from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optuna search over the structured Parameter Golf evaluator.")
    parser.add_argument("--family", default="exp01_mixed_export", choices=("exp01_mixed_export",))
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--study-name", default="parameter-golf-evaluator")
    parser.add_argument("--storage", default=None)
    parser.add_argument("--sampler-seed", type=int, default=1337)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--full-budget", action="store_true")
    return parser.parse_args()


def build_trial_set_args(family: str, trial) -> list[str]:
    if family != "exp01_mixed_export":
        raise ValueError(f"Unsupported family: {family}")
    return [
        f"LOWBIT_TARGET_BITS={trial.suggest_categorical('lowbit_target_bits', [5, 6, 7])}",
        f"LOWBIT_CLIP_PERCENTILE={trial.suggest_float('lowbit_clip_percentile', 99.95, 99.9999):.6f}",
        f"INT8_KEEP_FLOAT_MAX_NUMEL={trial.suggest_categorical('int8_keep_float_max_numel', [32768, 65536, 98304])}",
        f"WARMDOWN_ITERS={trial.suggest_categorical('warmdown_iters', [600, 900, 1200, 1800])}",
        f"WARMUP_STEPS={trial.suggest_categorical('warmup_steps', [0, 2, 5])}",
    ]


def run_eval(family: str, set_args: list[str], full_budget: bool) -> dict:
    cmd = [
        "uv",
        "run",
        "python",
        "evaluator/evaluate_candidate.py",
        "--family",
        family,
    ]
    if not full_budget:
        cmd.append("--smoke")
    for item in set_args:
        cmd.extend(["--set", item])

    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stdout[-4000:])
    return json.loads(proc.stdout)


def validate_eval_legality(result: dict) -> None:
    constraints = result.get("constraints", {})
    if constraints.get("legal_eval_ok") is not True:
        status = result.get("status", "unknown")
        mode = result.get("metrics", {}).get("evaluation_mode")
        raise RuntimeError(f"Evaluator rejected illegal or unknown eval mode: status={status} evaluation_mode={mode}")


def main() -> None:
    args = parse_args()
    try:
        import optuna
    except ImportError as exc:
        raise SystemExit("optuna is not installed. Use `uv add optuna` in parameter-golf.") from exc

    sampler = optuna.samplers.TPESampler(seed=args.sampler_seed)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="maximize",
        load_if_exists=True,
        sampler=sampler,
    )

    def objective(trial):
        set_args = build_trial_set_args(args.family, trial)
        result = run_eval(args.family, set_args, full_budget=args.full_budget)
        validate_eval_legality(result)
        trial.set_user_attr("set_args", set_args)
        trial.set_user_attr("result", result)
        return float(result["combined_score"])

    study.optimize(objective, n_trials=args.trials, timeout=args.timeout)
    best = study.best_trial
    print(f"best_value={best.value}")
    print(f"best_params={best.params}")
    print(f"best_set_args={best.user_attrs.get('set_args')}")
    print(json.dumps(best.user_attrs.get("result"), indent=2))


if __name__ == "__main__":
    main()
