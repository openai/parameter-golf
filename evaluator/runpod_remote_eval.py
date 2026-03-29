from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROGRAM_NAME = "train_gpt.py"
RUNTIME_ROOT = ROOT / "runpod_runtime"
COMPILE_RETRY_MARKERS = (
    "FailOnRecompileLimitHit",
    "recompile_limit reached with fullgraph=True",
)
HYPERPARAM_GUARDRAILS = {
    "NUM_LAYERS": ("max", 12),
    "MODEL_DIM": ("max", 576),
    "NUM_HEADS": ("max", 8),
    "NUM_KV_HEADS": ("max", 4),
    "MLP_MULT": ("max", 2),
    "VOCAB_SIZE": ("max", 2048),
    "TRAIN_SEQ_LEN": ("exact", 1024),
}


@dataclass
class CandidateRun:
    returncode: int
    stdout: str
    compile_strategy: str


def _load_evaluate_candidate_module():
    module_path = Path(__file__).with_name("evaluate_candidate.py")
    spec = importlib.util.spec_from_file_location("parameter_golf_evaluate_candidate", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load evaluator helper from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_EVAL = _load_evaluate_candidate_module()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one Parameter Golf candidate on a remote GPU pod.")
    parser.add_argument("candidate_program")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--env-json", default="{}")
    parser.add_argument("--family", default="runpod_remote_candidate")
    parser.add_argument("--json-indent", type=int, default=2)
    return parser.parse_args()


def _default_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(_EVAL.DEFAULT_SMOKE_ENV)
    env["ITERATIONS"] = env.get("ITERATIONS", "30")
    env["TRAIN_BATCH_TOKENS"] = env.get("TRAIN_BATCH_TOKENS", "8192")
    env["VAL_BATCH_SIZE"] = env.get("VAL_BATCH_SIZE", "131072")
    env["VAL_TOKEN_LIMIT"] = env.get("VAL_TOKEN_LIMIT", str(262_144))
    env.setdefault("DATA_PATH", str(ROOT / "data" / "datasets" / "fineweb10B_sp1024"))
    env.setdefault("TOKENIZER_PATH", str(ROOT / "data" / "tokenizers" / "fineweb_1024_bpe.model"))
    env.setdefault("EVALUATION_MODE", "standard_roundtrip")
    env.setdefault("USES_TTT", "0")
    return env


def _prepare_candidate_dir(program_path: Path, run_id: str) -> Path:
    candidate_dir = RUNTIME_ROOT / run_id
    candidate_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(program_path, candidate_dir / PROGRAM_NAME)
    return candidate_dir


def _extract_env_default(source: str, env_name: str) -> int | float | None:
    pattern = re.compile(rf'os\.environ\.get\("{re.escape(env_name)}",\s*([^)]+)\)')
    match = pattern.search(source)
    if not match:
        return None
    raw = match.group(1).strip()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        try:
            value = ast.literal_eval(raw)
        except Exception:
            return None
    if isinstance(value, (int, float)):
        return value
    return None


def _guardrail_violations(source: str) -> list[str]:
    failures: list[str] = []
    for env_name, (mode, bound) in HYPERPARAM_GUARDRAILS.items():
        value = _extract_env_default(source, env_name)
        if value is None:
            continue
        if mode == "max" and value > bound:
            failures.append(f"{env_name}={value} exceeds max {bound}")
        elif mode == "exact" and value != bound:
            failures.append(f"{env_name}={value} must equal {bound}")
    return failures


def _contains_compile_recompile_failure(stdout: str) -> bool:
    return any(marker in stdout for marker in COMPILE_RETRY_MARKERS)


def _make_compile_safe_variant(source: str, strategy: str) -> str:
    if strategy == "fullgraph_false":
        patched = source.replace(
            "torch.compile(base_model, dynamic=False, fullgraph=True)",
            "torch.compile(base_model, dynamic=False, fullgraph=False)",
        )
        patched = patched.replace(
            "torch.compile(eval_model, dynamic=False, fullgraph=True)",
            "torch.compile(eval_model, dynamic=False, fullgraph=False)",
        )
        patched = patched.replace(
            "torch.compile(base_model.forward_logits, dynamic=False, fullgraph=True)",
            "torch.compile(base_model.forward_logits, dynamic=False, fullgraph=False)",
        )
        return patched
    if strategy == "no_compile":
        patched = source.replace(
            "compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)",
            "compiled_model = base_model",
        )
        patched = patched.replace(
            "compiled_model = torch.compile(base_model, dynamic=False, fullgraph=False)",
            "compiled_model = base_model",
        )
        patched = patched.replace(
            "compiled_eval = torch.compile(eval_model, dynamic=False, fullgraph=True)",
            "compiled_eval = eval_model",
        )
        patched = patched.replace(
            "compiled_eval = torch.compile(eval_model, dynamic=False, fullgraph=False)",
            "compiled_eval = eval_model",
        )
        patched = patched.replace(
            "compiled_logits = torch.compile(base_model.forward_logits, dynamic=False, fullgraph=True)",
            "compiled_logits = base_model.forward_logits",
        )
        patched = patched.replace(
            "compiled_logits = torch.compile(base_model.forward_logits, dynamic=False, fullgraph=False)",
            "compiled_logits = base_model.forward_logits",
        )
        return patched
    raise ValueError(f"Unknown compile fallback strategy: {strategy}")


def _run_candidate(candidate_dir: Path, env: dict[str, str], compile_strategy: str) -> CandidateRun:
    proc = subprocess.run(
        [sys.executable, PROGRAM_NAME],
        cwd=candidate_dir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    return CandidateRun(
        returncode=proc.returncode,
        stdout=proc.stdout,
        compile_strategy=compile_strategy,
    )


def _build_guardrail_failure_result(
    *,
    run_id: str,
    family: str,
    source_path: Path,
    candidate_dir: Path,
    eval_metadata: dict[str, object],
    violations: list[str],
) -> dict:
    constraints = {
        "artifact_ok": False,
        "train_time_ok": False,
        "eval_time_ok": False,
        "legal_eval_ok": bool(eval_metadata["legal_eval_ok"]),
        "candidate_guardrails_ok": False,
    }
    return {
        "status": "constraint_failed",
        "combined_score": -9999.0,
        "metrics": {
            "pre_export_val_bpb": None,
            "post_export_val_bpb": None,
            "roundtrip_delta_bpb": None,
            "artifact_bytes": None,
            "total_submission_bytes": None,
            "train_time_ms": None,
            "eval_time_ms": None,
            "model_params": None,
            "peak_memory_mib": None,
            "evaluation_mode": str(eval_metadata["evaluation_mode"]),
            "uses_ttt": bool(eval_metadata["uses_ttt"]),
        },
        "constraints": constraints,
        "artifacts": {
            "run_id": run_id,
            "candidate_family": family,
            "log_path": str(candidate_dir / "train.log"),
            "roundtrip_prefix": None,
            "evaluation_mode": str(eval_metadata["evaluation_mode"]),
            "uses_ttt": bool(eval_metadata["uses_ttt"]),
            "candidate_program_path": str(source_path),
            "candidate_dir": str(candidate_dir),
            "returncode": None,
            "failed_constraints": _EVAL.summarize_failed_constraints(constraints),
            "failure_tail": "Static guardrail rejection: " + "; ".join(violations),
            "compile_strategy": "preflight_reject",
        },
    }


def _flatten_result(result: dict) -> dict:
    metrics = result.get("metrics", {})
    constraints = result.get("constraints", {})
    artifacts = result.get("artifacts", {})
    return {
        "combined_score": result.get("combined_score", -9999.0),
        "status": result.get("status", "failed"),
        "post_export_val_bpb": metrics.get("post_export_val_bpb"),
        "pre_export_val_bpb": metrics.get("pre_export_val_bpb"),
        "roundtrip_delta_bpb": metrics.get("roundtrip_delta_bpb"),
        "artifact_bytes": metrics.get("artifact_bytes"),
        "total_submission_bytes": metrics.get("total_submission_bytes"),
        "train_time_ms": metrics.get("train_time_ms"),
        "eval_time_ms": metrics.get("eval_time_ms"),
        "model_params": metrics.get("model_params"),
        "artifact_ok": constraints.get("artifact_ok"),
        "train_time_ok": constraints.get("train_time_ok"),
        "eval_time_ok": constraints.get("eval_time_ok"),
        "legal_eval_ok": constraints.get("legal_eval_ok"),
        "candidate_guardrails_ok": constraints.get("candidate_guardrails_ok"),
        "evaluation_mode": metrics.get("evaluation_mode"),
        "uses_ttt": metrics.get("uses_ttt"),
        "log_path": artifacts.get("log_path"),
        "roundtrip_prefix": artifacts.get("roundtrip_prefix"),
        "run_id": artifacts.get("run_id"),
        "compile_strategy": artifacts.get("compile_strategy"),
    }


def main() -> int:
    args = parse_args()
    source_path = Path(args.candidate_program).resolve()
    if not source_path.exists():
        raise SystemExit(f"Candidate program not found: {source_path}")

    run_id = args.run_id or f"runpod_pg_{uuid.uuid4().hex[:10]}"
    source_text = source_path.read_text()
    candidate_dir = _prepare_candidate_dir(source_path, run_id)
    env = _default_env()
    env.update({k: str(v) for k, v in json.loads(args.env_json).items()})
    env["RUN_ID"] = run_id
    eval_metadata = _EVAL.infer_eval_metadata(env)

    guardrail_failures = _guardrail_violations(source_text)
    if guardrail_failures:
        result = _build_guardrail_failure_result(
            run_id=run_id,
            family=args.family,
            source_path=source_path,
            candidate_dir=candidate_dir,
            eval_metadata=eval_metadata,
            violations=guardrail_failures,
        )
        log_path = Path(result["artifacts"]["log_path"])
        log_path.write_text(result["artifacts"]["failure_tail"] + "\n")
        flattened = _flatten_result(result)
        flattened["structured_result"] = result
        json.dump(flattened, sys.stdout, indent=args.json_indent)
        sys.stdout.write("\n")
        return 0

    run = _run_candidate(candidate_dir, env, compile_strategy="fullgraph_true")
    if run.returncode != 0 and _contains_compile_recompile_failure(run.stdout):
        fallback_source = _make_compile_safe_variant(source_text, "fullgraph_false")
        (candidate_dir / PROGRAM_NAME).write_text(fallback_source)
        run = _run_candidate(candidate_dir, env, compile_strategy="fullgraph_false")
        if run.returncode != 0 and _contains_compile_recompile_failure(run.stdout):
            fallback_source = _make_compile_safe_variant(fallback_source, "no_compile")
            (candidate_dir / PROGRAM_NAME).write_text(fallback_source)
            run = _run_candidate(candidate_dir, env, compile_strategy="no_compile")

    log_path = candidate_dir / "train.log"
    log_path.write_text(run.stdout)
    result = _EVAL.build_result(
        run.stdout,
        run_id=run_id,
        family=args.family,
        log_path=str(log_path),
        returncode=run.returncode,
        eval_metadata=eval_metadata,
    )
    result["constraints"]["candidate_guardrails_ok"] = True
    result["artifacts"]["env"] = {
        "ITERATIONS": env.get("ITERATIONS"),
        "TRAIN_BATCH_TOKENS": env.get("TRAIN_BATCH_TOKENS"),
        "VAL_TOKEN_LIMIT": env.get("VAL_TOKEN_LIMIT"),
    }
    result["artifacts"]["candidate_program_path"] = str(source_path)
    result["artifacts"]["candidate_dir"] = str(candidate_dir)
    result["artifacts"]["returncode"] = run.returncode
    result["artifacts"]["compile_strategy"] = run.compile_strategy
    result["artifacts"]["failed_constraints"] = _EVAL.summarize_failed_constraints(
        result["constraints"]
    )
    if run.returncode != 0:
        result["artifacts"]["failure_tail"] = "\n".join(run.stdout.strip().splitlines()[-40:])

    flattened = _flatten_result(result)
    flattened["structured_result"] = result
    json.dump(flattened, sys.stdout, indent=args.json_indent)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
