#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def expand_tokens(value: Any, mapping: dict[str, str]) -> Any:
    if isinstance(value, str):
        out = value
        for token, replacement in mapping.items():
            out = out.replace(token, replacement)
        return out
    if isinstance(value, list):
        return [expand_tokens(v, mapping) for v in value]
    if isinstance(value, dict):
        return {k: expand_tokens(v, mapping) for k, v in value.items()}
    return value


def detect_repo_root(config_path: Path) -> Path:
    env_repo_root = os.environ.get("REPO_ROOT")
    if env_repo_root:
        return Path(env_repo_root).expanduser().resolve()

    config_dir = config_path.resolve().parent
    try:
        proc = subprocess.run(
            ["git", "-C", str(config_dir), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        )
        candidate = Path(proc.stdout.strip()).resolve()
        if candidate.exists():
            return candidate
    except Exception:
        pass

    for candidate in (config_dir, *config_dir.parents):
        if (candidate / "analysis").exists() and (candidate / "records").exists():
            return candidate

    parents = config_path.resolve().parents
    return parents[3] if len(parents) > 3 else config_dir


def load_config(config_path: Path) -> dict[str, Any]:
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    record_dir = Path(os.environ.get("RECORD_DIR", str(config_path.resolve().parent))).expanduser().resolve()
    repo_root = detect_repo_root(config_path)
    mapping = {
        "${REPO_ROOT}": str(repo_root),
        "${RECORD_DIR}": str(record_dir),
    }
    return expand_tokens(raw, mapping)


def emit_and_exit(payload: dict[str, Any], exit_code: int) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))
    raise SystemExit(exit_code)


def run() -> None:
    parser = argparse.ArgumentParser(description="Runpod preflight checks for V5.9 launch packet")
    parser.add_argument("--config", default=str(Path(__file__).resolve().parent / "launch_config.json"))
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Collect diagnostics without enforcing hard-failure exit for hardware capacity checks",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)

    checks: list[dict[str, Any]] = []

    def add_check(
        check_id: str,
        ok: bool,
        message: str,
        *,
        hard: bool,
        details: dict[str, Any] | None = None,
        dry_run_downgrade: bool = False,
    ) -> None:
        details = details or {}
        is_hard = bool(hard)
        status = "pass" if ok else "fail"
        if (not ok) and dry_run_downgrade and args.dry_run:
            is_hard = False
            status = "warn"
        checks.append(
            {
                "id": check_id,
                "status": status,
                "ok": bool(ok),
                "hard": is_hard,
                "message": message,
                "details": details,
            }
        )

    hw = cfg.get("hardware_expectations", {})
    paths = cfg.get("paths", {})

    py_major = int(hw.get("python_min_major", 3))
    py_minor = int(hw.get("python_min_minor", 10))
    version_ok = (sys.version_info.major, sys.version_info.minor) >= (py_major, py_minor)
    add_check(
        "python.version",
        version_ok,
        f"Python >= {py_major}.{py_minor} required",
        hard=True,
        details={
            "python_executable": sys.executable,
            "python_version": sys.version.split()[0],
            "required_min": f"{py_major}.{py_minor}",
        },
    )

    module_results: dict[str, dict[str, Any]] = {}
    for mod_name in ("torch", "sentencepiece"):
        try:
            mod = importlib.import_module(mod_name)
            module_results[mod_name] = {
                "available": True,
                "version": getattr(mod, "__version__", "unknown"),
            }
            add_check(
                f"module.{mod_name}",
                True,
                f"Module '{mod_name}' importable",
                hard=True,
                details=module_results[mod_name],
            )
        except Exception as exc:
            module_results[mod_name] = {
                "available": False,
                "error": repr(exc),
            }
            add_check(
                f"module.{mod_name}",
                False,
                f"Module '{mod_name}' import failed",
                hard=True,
                details=module_results[mod_name],
            )

    cuda_details: dict[str, Any] = {
        "cuda_available": False,
        "device_count": 0,
        "expected_cuda_devices": int(hw.get("expected_cuda_devices", 8)),
        "minimum_cuda_devices": int(hw.get("minimum_cuda_devices", 1)),
    }
    if module_results.get("torch", {}).get("available"):
        torch = importlib.import_module("torch")
        cuda_details["cuda_available"] = bool(torch.cuda.is_available())
        cuda_details["device_count"] = int(torch.cuda.device_count()) if cuda_details["cuda_available"] else 0
        if cuda_details["cuda_available"] and cuda_details["device_count"] > 0:
            cuda_details["device_names"] = [torch.cuda.get_device_name(i) for i in range(cuda_details["device_count"])]

    cuda_ok = bool(cuda_details["cuda_available"]) and int(cuda_details["device_count"]) >= int(cuda_details["minimum_cuda_devices"])
    add_check(
        "cuda.runtime",
        cuda_ok,
        "CUDA must be available with enough visible devices",
        hard=True,
        details=cuda_details,
        dry_run_downgrade=True,
    )

    output_root = Path(paths["output_root"]).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(output_root)
    free_gb = usage.free / (1024**3)
    min_free_gb = float(hw.get("minimum_free_disk_gb", 40))
    disk_ok = free_gb >= min_free_gb
    add_check(
        "disk.free_space",
        disk_ok,
        f"At least {min_free_gb:.1f} GB free disk required",
        hard=True,
        details={
            "path": str(output_root),
            "free_gb": round(free_gb, 3),
            "minimum_free_gb": min_free_gb,
        },
        dry_run_downgrade=True,
    )

    required_paths = {
        "dataset_train": Path(paths["data_path"]),
        "dataset_localbench": Path(paths["localbench_data"]),
        "tokenizer": Path(paths["tokenizer_path"]),
        "train_script": Path(paths["train_script"]),
        "optimizer_script": Path(paths["optimizer_script"]),
        "export_wrapper_script": Path(paths["export_wrapper_script"]),
        "serializer_autochooser_script": Path(paths["serializer_autochooser_script"]),
        "base_policy_json": Path(paths["base_policy_json"]),
        "promotion_ranking_csv": Path(paths["promotion_ranking_csv"]),
    }
    for label, path in required_paths.items():
        exists = path.exists()
        kind = "dir" if path.is_dir() else "file"
        add_check(
            f"path.{label}",
            exists,
            f"Required {kind} exists",
            hard=True,
            details={"path": str(path), "exists": exists},
        )

    writable_targets = [
        output_root,
        output_root / "T0",
        output_root / "T1",
        output_root / "D",
        Path(paths["analysis_dir"]).resolve(),
    ]
    for target in writable_targets:
        ok = True
        error = ""
        try:
            target.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(prefix="preflight_", dir=target, delete=True) as tmp:
                tmp.write(b"ok")
                tmp.flush()
        except Exception as exc:
            ok = False
            error = repr(exc)
        add_check(
            f"write.{target.name}",
            ok,
            "Output location must be writable",
            hard=True,
            details={"path": str(target), "error": error},
            dry_run_downgrade=True,
        )

    hard_failures = [c for c in checks if c["status"] == "fail" and c["hard"]]
    warnings = [c for c in checks if c["status"] == "warn"]
    payload = {
        "tool": "runpod_preflight_v5_9",
        "timestamp_utc": datetime.now(tz=timezone.utc).isoformat(),
        "dry_run": bool(args.dry_run),
        "config_path": str(config_path),
        "resolved_paths": paths,
        "summary": {
            "checks_total": len(checks),
            "hard_failures": len(hard_failures),
            "warnings": len(warnings),
            "ok": len(hard_failures) == 0,
        },
        "checks": checks,
    }

    emit_and_exit(payload, 0 if len(hard_failures) == 0 else 1)


if __name__ == "__main__":
    run()
