from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from .code_mutations import preview_code_mutation
from .common import REPO_ROOT, preferred_python, utc_now_iso


def _python_has_module(python_exe: str, name: str) -> bool:
    code = (
        "import importlib.util, sys; "
        f"sys.exit(0 if importlib.util.find_spec({name!r}) is not None else 1)"
    )
    completed = subprocess.run([python_exe, "-c", code], check=False)
    return completed.returncode == 0


def _torch_cuda_device_count(python_exe: str) -> int | None:
    if not _python_has_module(python_exe, "torch"):
        return None
    try:
        code = "import torch; print(torch.cuda.device_count())"
        completed = subprocess.run([python_exe, "-c", code], check=False, capture_output=True, text=True)
        if completed.returncode != 0:
            return None
        return int((completed.stdout or "0").strip())
    except Exception:
        return None


def _python_can_run_module(python_exe: str, name: str) -> bool:
    completed = subprocess.run([python_exe, "-m", name, "--help"], check=False, capture_output=True, text=True)
    return completed.returncode == 0


def _load_manifest() -> dict[str, Any] | None:
    path = REPO_ROOT / "data" / "manifest.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _dataset_expectations(data_path: Path) -> dict[str, Any]:
    manifest = _load_manifest()
    if manifest is None:
        return {"manifest_found": False, "dataset_name": data_path.name}
    for dataset in manifest.get("datasets", []):
        if dataset.get("name") == data_path.name or Path(dataset.get("path", "")).name == data_path.name:
            stats = dataset.get("stats") or {}
            return {
                "manifest_found": True,
                "dataset_name": dataset.get("name") or data_path.name,
                "expected_train_shards": stats.get("files_train"),
                "expected_val_shards": stats.get("files_val"),
                "tokenizer_name": dataset.get("tokenizer_name"),
                "tokenizer_kind": dataset.get("tokenizer_kind"),
            }
    return {"manifest_found": True, "dataset_name": data_path.name}


def run_preflight(spec: dict[str, Any]) -> dict[str, Any]:
    data_path = Path(spec["env"].get("DATA_PATH", ""))
    tokenizer_path = Path(spec["env"].get("TOKENIZER_PATH", ""))
    script_path = Path(spec["script"])
    python_exe = preferred_python()
    result: dict[str, Any] = {
        "checked_at": utc_now_iso(),
        "profile": spec["profile"],
        "track": spec["track"],
        "script_path": str(script_path),
        "data_path": str(data_path),
        "tokenizer_path": str(tokenizer_path),
        "python_path": python_exe,
        "fatal_issues": [],
        "warnings": [],
        "required_modules": list(spec.get("required_modules", [])),
        "available_modules": {},
        "launcher_available": True,
        "require_challenge_ready": bool(spec.get("require_challenge_ready", False)),
        "required_gpus": spec.get("required_gpus"),
        "available_gpus": _torch_cuda_device_count(python_exe),
        "code_mutation_valid": True,
        "code_mutation_signature": None,
        "comparability": "unknown",
        "challenge_ready": False,
        "launch_policy_ok": False,
        "ready_for_execution": False,
        "can_launch": False,
    }

    if not script_path.is_file():
        result["fatal_issues"].append(f"trainer script missing: {script_path}")
    if not data_path.is_dir():
        result["fatal_issues"].append(f"data path missing: {data_path}")
    if not tokenizer_path.is_file():
        result["fatal_issues"].append(f"tokenizer path missing: {tokenizer_path}")

    if script_path.is_file() and spec.get("code_mutation"):
        try:
            preview = preview_code_mutation(script_path, spec.get("code_mutation"))
            result["code_mutation_signature"] = preview["code_mutation"]["signature"]
            result["code_mutation_source_hash"] = preview["source_hash"]
        except Exception as exc:
            result["code_mutation_valid"] = False
            result["fatal_issues"].append(f"code mutation invalid: {exc}")

    if spec["launcher"] == "torchrun" and not _python_can_run_module(python_exe, "torch.distributed.run"):
        result["fatal_issues"].append(f"preferred python cannot launch torch.distributed.run: {python_exe}")
        result["launcher_available"] = False

    for module_name in spec.get("required_modules", []):
        ok = _python_has_module(python_exe, module_name)
        result["available_modules"][module_name] = ok
        if not ok:
            result["fatal_issues"].append(f"required module missing: {module_name}")

    if spec.get("required_gpus") is not None:
        gpu_count = result["available_gpus"]
        if gpu_count is None:
            result["fatal_issues"].append("unable to determine available CUDA GPU count")
        elif int(gpu_count) < int(spec["required_gpus"]):
            result["fatal_issues"].append(
                f"requires {spec['required_gpus']} GPUs but only {gpu_count} detected"
            )

    expectations = _dataset_expectations(data_path)
    result.update(expectations)
    actual_train = len(list(data_path.glob("fineweb_train_*.bin"))) if data_path.is_dir() else 0
    actual_val = len(list(data_path.glob("fineweb_val_*.bin"))) if data_path.is_dir() else 0
    result["actual_train_shards"] = actual_train
    result["actual_val_shards"] = actual_val

    expected_train = expectations.get("expected_train_shards")
    expected_val = expectations.get("expected_val_shards")
    if expected_train is not None and actual_train < int(expected_train):
        result["warnings"].append(f"dataset subset detected: train shards {actual_train}/{expected_train}")
    if expected_val is not None and actual_val < int(expected_val):
        result["warnings"].append(f"validation subset detected: val shards {actual_val}/{expected_val}")

    comparability = "full-comparable"
    if spec["track"].startswith("local-"):
        comparability = "smoke-only"
    if expected_train is not None and actual_train < int(expected_train):
        comparability = "subset-only"
    if expected_val is not None and actual_val < int(expected_val):
        comparability = "subset-only"
    if spec["track"] == "record-10min-16mb" and comparability == "full-comparable":
        comparability = "record-candidate"

    result["comparability"] = comparability
    result["challenge_ready"] = comparability in {"full-comparable", "record-candidate"}
    result["launch_policy_ok"] = (not result["require_challenge_ready"]) or result["challenge_ready"]
    result["can_launch"] = not result["fatal_issues"]
    result["ready_for_execution"] = result["can_launch"] and result["launch_policy_ok"]
    return result
