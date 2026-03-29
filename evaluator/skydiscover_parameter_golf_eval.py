from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ROOT = ROOT / "skydiscover_runtime"
PROGRAM_NAME = "train_gpt.py"


def _load_evaluate_candidate_module():
    module_path = Path(__file__).with_name("evaluate_candidate.py")
    spec = importlib.util.spec_from_file_location("parameter_golf_evaluate_candidate", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load evaluator helper from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_EVAL = _load_evaluate_candidate_module()


def _default_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(_EVAL.DEFAULT_SMOKE_ENV)
    env["ITERATIONS"] = "20"
    env["TRAIN_BATCH_TOKENS"] = "8192"
    env["VAL_BATCH_SIZE"] = "131072"
    env["VAL_TOKEN_LIMIT"] = str(262_144)
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
        "evaluation_mode": metrics.get("evaluation_mode"),
        "uses_ttt": metrics.get("uses_ttt"),
        "log_path": artifacts.get("log_path"),
        "roundtrip_prefix": artifacts.get("roundtrip_prefix"),
        "run_id": artifacts.get("run_id"),
    }


def evaluate(program_path: str) -> dict:
    source_path = Path(program_path)
    run_id = f"skyd_pg_{uuid.uuid4().hex[:10]}"
    candidate_dir = _prepare_candidate_dir(source_path, run_id)
    env = _default_env()
    env["RUN_ID"] = run_id

    proc = subprocess.run(
        [sys.executable, PROGRAM_NAME],
        cwd=candidate_dir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )

    log_path = candidate_dir / "train.log"
    log_path.write_text(proc.stdout)

    result = _EVAL.build_result(
        proc.stdout,
        run_id=run_id,
        family="skydiscover_candidate",
        log_path=str(log_path),
        returncode=proc.returncode,
        eval_metadata=_EVAL.infer_eval_metadata(env),
    )
    result["artifacts"]["env"] = {
        "ITERATIONS": env.get("ITERATIONS"),
        "TRAIN_BATCH_TOKENS": env.get("TRAIN_BATCH_TOKENS"),
        "VAL_TOKEN_LIMIT": env.get("VAL_TOKEN_LIMIT"),
    }
    result["artifacts"]["candidate_program_path"] = str(source_path)
    result["artifacts"]["candidate_dir"] = str(candidate_dir)
    result["artifacts"]["returncode"] = proc.returncode
    result["artifacts"]["failed_constraints"] = _EVAL.summarize_failed_constraints(
        result["constraints"]
    )
    if proc.returncode != 0:
        result["artifacts"]["failure_tail"] = "\n".join(proc.stdout.strip().splitlines()[-40:])

    flattened = _flatten_result(result)
    flattened["structured_result"] = result
    return flattened


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        raise SystemExit("usage: python skydiscover_parameter_golf_eval.py <program_path>")
    print(json.dumps(evaluate(sys.argv[1]), indent=2))
