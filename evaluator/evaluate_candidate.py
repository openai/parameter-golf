from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
import uuid
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
    parser = argparse.ArgumentParser(description="Evaluate one Parameter Golf candidate and emit structured JSON.")
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
    parser.add_argument("--json-indent", type=int, default=2)
    return parser.parse_args()


def parse_set_args(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected KEY=VALUE, got: {item}")
        key, value = item.split("=", 1)
        out[key] = value
    return out


def extract_float(pattern: str, text: str) -> float | None:
    matches = re.findall(pattern, text)
    return float(matches[-1]) if matches else None


def extract_int(pattern: str, text: str) -> int | None:
    matches = re.findall(pattern, text)
    return int(matches[-1]) if matches else None


def extract_roundtrip_prefix(text: str) -> str | None:
    match = re.search(r"(final_[^ ]+_roundtrip)_exact", text)
    return match.group(1) if match else None


def parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def summarize_failed_constraints(constraints: dict[str, bool]) -> list[str]:
    return [name for name, ok in constraints.items() if not ok]


def infer_eval_metadata(env: dict[str, str]) -> dict[str, object]:
    evaluation_mode = env.get("EVALUATION_MODE", "standard_roundtrip")
    uses_ttt = parse_bool(env.get("USES_TTT"), default=False)
    explicit_legal = env.get("LEGAL_EVAL_OK")
    if explicit_legal is not None:
        legal_eval_ok = parse_bool(explicit_legal, default=False)
    else:
        legal_eval_ok = evaluation_mode == "standard_roundtrip" and not uses_ttt
    return {
        "evaluation_mode": evaluation_mode,
        "uses_ttt": uses_ttt,
        "legal_eval_ok": legal_eval_ok,
    }


def build_result(
    stdout: str,
    run_id: str,
    family: str,
    log_path: str,
    returncode: int,
    eval_metadata: dict[str, object],
) -> dict:
    step_prefix = r"step:[0-9]+/[0-9]+"
    pre_export_val_bpb = extract_float(
        rf"{step_prefix} val_loss:[0-9.]+ val_bpb:([0-9.]+)", stdout
    )
    post_export_val_bpb = extract_float(r"final_[^ ]+_roundtrip_exact val_loss:[0-9.]+ val_bpb:([0-9.]+)", stdout)
    artifact_bytes = extract_int(r"Serialized model [^:]+: ([0-9]+) bytes", stdout)
    total_submission_bytes = extract_int(r"Total submission size [^:]*: ([0-9]+) bytes", stdout)
    train_time_ms = extract_int(
        rf"{step_prefix} val_loss:[0-9.]+ val_bpb:[0-9.]+ train_time:([0-9]+)ms", stdout
    )
    eval_time_ms = extract_int(r"final_[^ ]+_roundtrip val_loss:[0-9.]+ val_bpb:[0-9.]+ eval_time:([0-9]+)ms", stdout)
    model_params = extract_int(r"model_params:([0-9]+)", stdout)
    peak_memory_mib = extract_int(r"peak memory allocated: ([0-9]+) MiB", stdout)
    roundtrip_prefix = extract_roundtrip_prefix(stdout)

    completed = returncode == 0 and post_export_val_bpb is not None
    roundtrip_delta = None
    if pre_export_val_bpb is not None and post_export_val_bpb is not None:
        roundtrip_delta = post_export_val_bpb - pre_export_val_bpb

    artifact_ok = bool(total_submission_bytes is not None and total_submission_bytes < 16_000_000)
    train_time_ok = bool(train_time_ms is not None and train_time_ms < 600_000)
    eval_time_ok = bool(eval_time_ms is not None and eval_time_ms < 600_000)
    legal_eval_ok = bool(eval_metadata["legal_eval_ok"])
    constraints = {
        "artifact_ok": artifact_ok,
        "train_time_ok": train_time_ok,
        "eval_time_ok": eval_time_ok,
        "legal_eval_ok": legal_eval_ok,
    }
    failed_constraints = summarize_failed_constraints(constraints)

    if completed and not failed_constraints and post_export_val_bpb is not None:
        artifact_pressure = (artifact_bytes or total_submission_bytes or 16_000_000) / 16_000_000.0
        combined_score = -post_export_val_bpb - 0.5 * (roundtrip_delta or 0.0) - 0.05 * artifact_pressure
        status = "success"
    elif completed:
        combined_score = -9999.0
        status = "constraint_failed"
    else:
        combined_score = -9999.0
        status = "failed"

    return {
        "status": status,
        "combined_score": combined_score,
        "metrics": {
            "pre_export_val_bpb": pre_export_val_bpb,
            "post_export_val_bpb": post_export_val_bpb,
            "roundtrip_delta_bpb": roundtrip_delta,
            "artifact_bytes": artifact_bytes,
            "total_submission_bytes": total_submission_bytes,
            "train_time_ms": train_time_ms,
            "eval_time_ms": eval_time_ms,
            "model_params": model_params,
            "peak_memory_mib": peak_memory_mib,
            "evaluation_mode": str(eval_metadata["evaluation_mode"]),
            "uses_ttt": bool(eval_metadata["uses_ttt"]),
        },
        "constraints": constraints,
        "artifacts": {
            "run_id": run_id,
            "candidate_family": family,
            "log_path": log_path,
            "roundtrip_prefix": roundtrip_prefix,
            "evaluation_mode": str(eval_metadata["evaluation_mode"]),
            "uses_ttt": bool(eval_metadata["uses_ttt"]),
        },
    }


def main() -> int:
    args = parse_args()
    exp_dir = FAMILY_TO_DIR[args.family]
    env = os.environ.copy()
    if args.smoke:
        env.update(DEFAULT_SMOKE_ENV)
    env.setdefault("DATA_PATH", str(ROOT / "data" / "datasets" / "fineweb10B_sp1024"))
    env.setdefault("TOKENIZER_PATH", str(ROOT / "data" / "tokenizers" / "fineweb_1024_bpe.model"))
    env_updates = parse_set_args(args.set)
    env.update(env_updates)
    eval_metadata = infer_eval_metadata(env)

    run_id = args.run_id or env.get("RUN_ID") or f"{args.family}_{uuid.uuid4().hex[:8]}"
    env["RUN_ID"] = run_id

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

    log_path = str(exp_dir / "logs" / f"{run_id}.txt")
    result = build_result(
        proc.stdout,
        run_id=run_id,
        family=args.family,
        log_path=log_path,
        returncode=proc.returncode,
        eval_metadata=eval_metadata,
    )
    result["artifacts"]["env"] = env_updates
    result["artifacts"]["returncode"] = proc.returncode
    result["artifacts"]["failed_constraints"] = summarize_failed_constraints(result["constraints"])
    if proc.returncode != 0:
        result["artifacts"]["failure_tail"] = "\n".join(proc.stdout.strip().splitlines()[-40:])

    json.dump(result, sys.stdout, indent=args.json_indent)
    sys.stdout.write("\n")
    return 0 if proc.returncode == 0 else proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
