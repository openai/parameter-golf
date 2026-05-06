#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import importlib
import json
import os
import sys
from pathlib import Path


REQUIRED_PACKAGES = ("torch", "sentencepiece", "zstandard")


def check_package(name: str) -> tuple[bool, str | None]:
    try:
        module = importlib.import_module(name)
    except Exception as exc:  # pragma: no cover - surfaced directly to operator
        return False, str(exc)
    version = getattr(module, "__version__", None)
    return True, version


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate reproduction environment for parameter-golf.")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--min-gpus", type=int, default=8)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()

    package_results: dict[str, dict[str, object]] = {}
    failures: list[str] = []
    torch_module = None

    for package_name in REQUIRED_PACKAGES:
        ok, detail = check_package(package_name)
        package_results[package_name] = {"ok": ok, "detail": detail}
        if not ok:
            failures.append(f"missing package: {package_name} ({detail})")
        elif package_name == "torch":
            torch_module = importlib.import_module(package_name)

    data_path = Path(args.data_path)
    tokenizer_path = Path(args.tokenizer_path)
    train_shards = sorted(glob.glob(str(data_path / "fineweb_train_*.bin")))
    val_shards = sorted(glob.glob(str(data_path / "fineweb_val_*.bin")))

    data_ok = data_path.is_dir() and bool(train_shards) and bool(val_shards)
    tokenizer_ok = tokenizer_path.is_file()
    if not data_ok:
        failures.append(
            f"dataset missing or incomplete at {data_path} "
            f"(train_shards={len(train_shards)} val_shards={len(val_shards)})"
        )
    if not tokenizer_ok:
        failures.append(f"tokenizer missing at {tokenizer_path}")

    cuda_available = False
    gpu_count = 0
    gpu_names: list[str] = []
    torch_version = None
    if torch_module is not None:
        torch_version = getattr(torch_module, "__version__", None)
        cuda_available = bool(torch_module.cuda.is_available())
        gpu_count = int(torch_module.cuda.device_count())
        for index in range(gpu_count):
            gpu_names.append(str(torch_module.cuda.get_device_name(index)))
        if gpu_count < args.min_gpus:
            failures.append(f"requires at least {args.min_gpus} CUDA GPUs, found {gpu_count}")

    result = {
        "cwd": os.getcwd(),
        "python": sys.executable,
        "torch_version": torch_version,
        "cuda_available": cuda_available,
        "gpu_count": gpu_count,
        "gpu_names": gpu_names,
        "packages": package_results,
        "data_path": str(data_path),
        "train_shards": len(train_shards),
        "val_shards": len(val_shards),
        "tokenizer_path": str(tokenizer_path),
        "ok": not failures,
        "failures": failures,
    }

    if args.as_json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"python={result['python']}")
        print(f"torch_version={torch_version}")
        print(f"cuda_available={cuda_available}")
        print(f"gpu_count={gpu_count}")
        if gpu_names:
            print(f"gpu_names={', '.join(gpu_names)}")
        print(f"data_path={data_path} train_shards={len(train_shards)} val_shards={len(val_shards)}")
        print(f"tokenizer_path={tokenizer_path} exists={tokenizer_ok}")
        for package_name in REQUIRED_PACKAGES:
            package_result = package_results[package_name]
            status = "ok" if package_result["ok"] else "missing"
            detail = package_result["detail"]
            print(f"package:{package_name} status={status} detail={detail}")
        if failures:
            print("failures:")
            for failure in failures:
                print(f"  - {failure}")
        else:
            print("preflight:ok")

    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
