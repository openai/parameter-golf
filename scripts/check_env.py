#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import platform
import shutil
import sys


def _module_version(module) -> str:
    return str(getattr(module, "__version__", "unknown"))


def check_env(target: str) -> dict[str, object]:
    if sys.version_info < (3, 9):
        raise RuntimeError(f"Python 3.9+ is required, found {platform.python_version()}")

    summary: dict[str, object] = {
        "target": target,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": {},
    }
    packages: dict[str, str] = {}

    required_modules = ["numpy", "sentencepiece", "huggingface_hub", "datasets", "tqdm"]
    if target == "mlx":
        required_modules.append("mlx")
    elif target == "cuda":
        required_modules.append("torch")
    else:
        raise ValueError(f"Unsupported target {target!r}; expected 'mlx' or 'cuda'")

    for module_name in required_modules:
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            raise RuntimeError(f"Missing Python package: {module_name}") from exc
        packages[module_name] = _module_version(module)

    summary["packages"] = packages

    if target == "mlx":
        if platform.system() != "Darwin" or platform.machine() not in {"arm64", "aarch64"}:
            raise RuntimeError("The MLX preset expects Apple Silicon (Darwin arm64)")
        import mlx.core as mx

        summary["accelerator"] = {
            "backend": "mlx",
            "mlx_version": mx.__version__,
        }
    else:
        if shutil.which("torchrun") is None:
            raise RuntimeError("torchrun was not found in PATH")
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda.is_available() is False")
        device_count = torch.cuda.device_count()
        devices = []
        for idx in range(device_count):
            props = torch.cuda.get_device_properties(idx)
            devices.append(
                {
                    "index": idx,
                    "name": props.name,
                    "total_memory_gb": round(props.total_memory / (1024**3), 2),
                }
            )
        summary["accelerator"] = {
            "backend": "cuda",
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "device_count": device_count,
            "devices": devices,
        }

    return summary


def _print_summary(summary: dict[str, object]) -> None:
    print(f"target: {summary['target']}")
    print(f"python: {summary['python']}")
    print(f"platform: {summary['platform']}")
    print(f"machine: {summary['machine']}")
    print("packages:")
    for name, version in sorted((summary.get("packages") or {}).items()):
        print(f"  {name}: {version}")
    accelerator = summary.get("accelerator") or {}
    if accelerator:
        print(f"accelerator_backend: {accelerator.get('backend')}")
        if accelerator.get("backend") == "cuda":
            print(f"cuda_version: {accelerator.get('cuda_version')}")
            print(f"device_count: {accelerator.get('device_count')}")
            for device in accelerator.get("devices", []):
                print(
                    f"  gpu[{device['index']}]: {device['name']} "
                    f"({device['total_memory_gb']} GiB)"
                )
        else:
            print(f"mlx_version: {accelerator.get('mlx_version')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fail-fast dependency and hardware checks for Parameter Golf presets.")
    parser.add_argument("--target", choices=("mlx", "cuda"), required=True)
    parser.add_argument("--json", action="store_true", help="Print the summary as JSON.")
    args = parser.parse_args()

    try:
        summary = check_env(args.target)
    except Exception as exc:
        print(f"check_env failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        _print_summary(summary)


if __name__ == "__main__":
    main()
