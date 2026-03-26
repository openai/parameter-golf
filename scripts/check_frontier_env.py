#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _import_optional(module_name: str):
    try:
        return importlib.import_module(module_name), None
    except Exception as exc:  # noqa: BLE001
        return None, exc


def _parse_nvcc_version() -> tuple[str | None, str | None]:
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        return None, None
    result = subprocess.run([nvcc, "--version"], capture_output=True, text=True, check=False)
    output = (result.stdout or "") + (result.stderr or "")
    match = re.search(r"release\s+(\d+\.\d+)", output)
    return nvcc, match.group(1) if match else None


def _cloud_image_likely() -> bool:
    if any(key.startswith("RUNPOD_") for key in os.environ):
        return True
    return Path("/workspace").exists()


def inspect_frontier_env(*, require_flash_attn: bool = True) -> dict[str, object]:
    torch_mod, torch_exc = _import_optional("torch")
    if torch_mod is None:
        return {
            "status": "stop",
            "ok_to_proceed": False,
            "reason": f"torch import failed: {torch_exc}",
            "next_steps": [
                "Use a CUDA / PyTorch image or install a self-managed stack with requirements-local.txt.",
            ],
        }

    torch = torch_mod
    cuda_available = bool(torch.cuda.is_available())
    device_count = int(torch.cuda.device_count()) if cuda_available else 0
    cuda_bf16_supported = bool(torch.cuda.is_bf16_supported()) if cuda_available and hasattr(torch.cuda, "is_bf16_supported") else None
    devices = []
    if cuda_available:
        for idx in range(device_count):
            props = torch.cuda.get_device_properties(idx)
            devices.append(
                {
                    "index": idx,
                    "name": props.name,
                    "total_memory_gb": round(props.total_memory / (1024**3), 2),
                }
            )

    flash_attn_mod, flash_attn_exc = _import_optional("flash_attn_interface")
    flash_attn_path = None if flash_attn_mod is None else getattr(flash_attn_mod, "__file__", "unknown")
    if flash_attn_mod is None:
        flash_attn_ok = False
        flash_attn_error = f"{type(flash_attn_exc).__name__}: {flash_attn_exc}"
        flash_attn_source = None
    else:
        summary_fn = getattr(flash_attn_mod, "flash_attention_import_summary", None)
        summary = summary_fn() if callable(summary_fn) else {}
        flash_attn_ok = bool(summary.get("flash_attn_available"))
        flash_attn_error = summary.get("flash_attn_import_error")
        flash_attn_source = summary.get("flash_attn_source")

    flash_attn_pkg_mod, _flash_attn_pkg_exc = _import_optional("flash_attn")
    flash_attn_package_version = None if flash_attn_pkg_mod is None else getattr(flash_attn_pkg_mod, "__version__", "unknown")
    flash_attn_package_path = None if flash_attn_pkg_mod is None else getattr(flash_attn_pkg_mod, "__file__", "unknown")

    nvcc_path, nvcc_version = _parse_nvcc_version()
    torch_cuda_version = str(torch.version.cuda or "")
    mismatch_risk = bool(
        nvcc_version
        and torch_cuda_version
        and nvcc_version.split(".")[:2] != torch_cuda_version.split(".")[:2]
    )

    issues: list[str] = []
    next_steps: list[str] = []
    if not cuda_available:
        issues.append("torch.cuda.is_available() is False")
        next_steps.append("Use a CUDA-capable image and verify the NVIDIA runtime before training.")
    if not torch_cuda_version:
        issues.append("torch.version.cuda is empty")
        next_steps.append("Use a CUDA-enabled torch build for frontier presets.")
    if nvcc_path is None:
        issues.append("nvcc was not found in PATH")
        next_steps.append("Install FlashAttention only in environments that include a matching CUDA toolchain.")
    if mismatch_risk:
        issues.append(
            f"torch CUDA version ({torch_cuda_version}) does not match nvcc release ({nvcc_version}); "
            "FlashAttention build/import risk is high"
        )
        next_steps.append("Do not upgrade torch on top of the image. Recreate the pod with a matching PyTorch image.")
    if require_flash_attn and not flash_attn_ok:
        issues.append(f"flash_attn_interface import failed: {flash_attn_error}")
        next_steps.append("Run ./scripts/install_cloud.sh or install flash-attn against the existing torch stack.")

    ok_to_proceed = not issues
    status = "ok" if ok_to_proceed else ("warn" if (not require_flash_attn and len(issues) == 1 and not flash_attn_ok) else "stop")
    if status == "warn":
        issues = [f"flash_attn_interface import failed: {flash_attn_error}"]
        next_steps = ["Install flash-attn against the existing torch stack, then rerun this check without --allow-missing-flash-attn."]

    return {
        "status": status,
        "ok_to_proceed": ok_to_proceed,
        "cloud_image_likely": _cloud_image_likely(),
        "in_virtualenv": sys.prefix != getattr(sys, "base_prefix", sys.prefix),
        "python_executable": sys.executable,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch_cuda_version,
        "torch_path": getattr(torch, "__file__", "unknown"),
        "cuda_available": cuda_available,
        "cuda_bf16_supported": cuda_bf16_supported,
        "device_count": device_count,
        "devices": devices,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvcc_path": nvcc_path,
        "nvcc_version": nvcc_version,
        "flash_attn_interface_ok": flash_attn_ok,
        "flash_attn_interface_error": flash_attn_error,
        "flash_attn_interface_path": flash_attn_path,
        "flash_attn_source": flash_attn_source,
        "flash_attn_package_version": flash_attn_package_version,
        "flash_attn_package_path": flash_attn_package_path,
        "mismatch_risk": mismatch_risk,
        "issues": issues,
        "next_steps": next_steps,
    }


def _print_summary(summary: dict[str, object]) -> None:
    status = str(summary["status"]).upper()
    verdict = "OK to proceed" if summary.get("ok_to_proceed") else "STOP: likely environment mismatch"
    if summary.get("status") == "warn":
        verdict = "NOT READY YET: install FlashAttention, then re-check"
    print(f"{status}: {verdict}")
    print(f"cloud_image_likely: {summary.get('cloud_image_likely')}")
    print(f"in_virtualenv: {summary.get('in_virtualenv')}")
    print(f"python_executable: {summary.get('python_executable')}")
    print(f"torch_version: {summary.get('torch_version')}")
    print(f"torch_cuda_version: {summary.get('torch_cuda_version')}")
    print(f"torch_path: {summary.get('torch_path')}")
    print(f"cuda_available: {summary.get('cuda_available')}")
    print(f"cuda_bf16_supported: {summary.get('cuda_bf16_supported')}")
    print(f"device_count: {summary.get('device_count')}")
    for device in summary.get("devices") or []:
        print(f"  gpu[{device['index']}]: {device['name']} ({device['total_memory_gb']} GiB)")
    print(f"nvcc_path: {summary.get('nvcc_path')}")
    print(f"nvcc_version: {summary.get('nvcc_version')}")
    print(f"flash_attn_interface_ok: {summary.get('flash_attn_interface_ok')}")
    print(f"flash_attn_interface_path: {summary.get('flash_attn_interface_path')}")
    print(f"flash_attn_source: {summary.get('flash_attn_source')}")
    print(f"flash_attn_package_version: {summary.get('flash_attn_package_version')}")
    print(f"flash_attn_package_path: {summary.get('flash_attn_package_path')}")
    if summary.get("flash_attn_interface_error"):
        print(f"flash_attn_interface_error: {summary.get('flash_attn_interface_error')}")
    print(f"mismatch_risk: {summary.get('mismatch_risk')}")
    issues = summary.get("issues") or []
    if issues:
        print("issues:")
        for issue in issues:
            print(f"  - {issue}")
    next_steps = summary.get("next_steps") or []
    if next_steps:
        print("next_steps:")
        for item in next_steps:
            print(f"  - {item}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Frontier-specific CUDA / FlashAttention readiness check.")
    parser.add_argument(
        "--allow-missing-flash-attn",
        action="store_true",
        help="Do not fail only because flash_attn_interface is missing; useful before running the installer.",
    )
    parser.add_argument("--json", action="store_true", help="Print the summary as JSON.")
    args = parser.parse_args()

    summary = inspect_frontier_env(require_flash_attn=not args.allow_missing_flash_attn)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        _print_summary(summary)
    if not summary.get("ok_to_proceed") and summary.get("status") != "warn":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
