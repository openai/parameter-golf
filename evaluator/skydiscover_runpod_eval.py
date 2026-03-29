from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_runpod_client_module():
    module_path = Path(__file__).with_name("runpod_client_eval.py")
    spec = importlib.util.spec_from_file_location("parameter_golf_runpod_client_eval", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load RunPod client from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_RUNPOD = _load_runpod_client_module()


def evaluate(program_path: str) -> dict:
    env_json = json.dumps(
        {
            "ITERATIONS": "20",
            "TRAIN_BATCH_TOKENS": "8192",
            "VAL_BATCH_SIZE": "131072",
            "VAL_TOKEN_LIMIT": str(262_144),
        }
    )
    return _RUNPOD.evaluate_remote(
        program_path,
        env_json=env_json,
        family="skydiscover_runpod_candidate",
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        raise SystemExit("usage: python skydiscover_runpod_eval.py <program_path>")
    print(json.dumps(evaluate(sys.argv[1]), indent=2))
