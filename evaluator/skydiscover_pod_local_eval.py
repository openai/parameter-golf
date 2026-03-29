from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_module(name: str, filename: str):
    module_path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load {filename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_REMOTE = _load_module("parameter_golf_runpod_remote_eval", "runpod_remote_eval.py")


def evaluate(program_path: str) -> dict:
    source_path = Path(program_path).resolve()
    run_id = f"podlocal_{source_path.stem}"
    source_text = source_path.read_text()
    candidate_dir = _REMOTE._prepare_candidate_dir(source_path, run_id)
    env = _REMOTE._default_env()
    env.update(
        {
            "ITERATIONS": "20",
            "TRAIN_BATCH_TOKENS": "8192",
            "VAL_BATCH_SIZE": "131072",
            "VAL_TOKEN_LIMIT": str(262_144),
            "RUN_ID": run_id,
        }
    )
    eval_metadata = _REMOTE._EVAL.infer_eval_metadata(env)

    guardrail_failures = _REMOTE._guardrail_violations(source_text)
    if guardrail_failures:
        result = _REMOTE._build_guardrail_failure_result(
            run_id=run_id,
            family="skydiscover_pod_local_candidate",
            source_path=source_path,
            candidate_dir=candidate_dir,
            eval_metadata=eval_metadata,
            violations=guardrail_failures,
        )
        Path(result["artifacts"]["log_path"]).write_text(result["artifacts"]["failure_tail"] + "\n")
        flattened = _REMOTE._flatten_result(result)
        flattened["structured_result"] = result
        return flattened

    run = _REMOTE._run_candidate(candidate_dir, env, compile_strategy="fullgraph_true")
    if run.returncode != 0 and _REMOTE._contains_compile_recompile_failure(run.stdout):
        fallback_source = _REMOTE._make_compile_safe_variant(source_text, "fullgraph_false")
        (candidate_dir / _REMOTE.PROGRAM_NAME).write_text(fallback_source)
        run = _REMOTE._run_candidate(candidate_dir, env, compile_strategy="fullgraph_false")
        if run.returncode != 0 and _REMOTE._contains_compile_recompile_failure(run.stdout):
            fallback_source = _REMOTE._make_compile_safe_variant(fallback_source, "no_compile")
            (candidate_dir / _REMOTE.PROGRAM_NAME).write_text(fallback_source)
            run = _REMOTE._run_candidate(candidate_dir, env, compile_strategy="no_compile")

    log_path = candidate_dir / "train.log"
    log_path.write_text(run.stdout)
    result = _REMOTE._EVAL.build_result(
        run.stdout,
        run_id=run_id,
        family="skydiscover_pod_local_candidate",
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
    result["artifacts"]["failed_constraints"] = _REMOTE._EVAL.summarize_failed_constraints(
        result["constraints"]
    )
    if run.returncode != 0:
        result["artifacts"]["failure_tail"] = "\n".join(run.stdout.strip().splitlines()[-40:])

    flattened = _REMOTE._flatten_result(result)
    flattened["structured_result"] = result
    return flattened


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        raise SystemExit("usage: python skydiscover_pod_local_eval.py <program_path>")
    print(json.dumps(evaluate(sys.argv[1]), indent=2))
