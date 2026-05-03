#!/usr/bin/env python3
"""Build and verify Parameter Golf submission byte-budget metadata.

The official budget is decimal bytes:

    counted code bytes + compressed model bytes < 16,000,000

This script is intentionally conservative. It only counts local files supplied
by the caller, never assumes custom Rust binaries are free, and fails closed
when any requested path is missing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


LIMIT_BYTES = 16_000_000


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _iter_counted_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"counted code path does not exist: {path}")
        if path.is_file():
            files.append(path)
            continue
        for child in sorted(path.rglob("*")):
            if child.is_file():
                files.append(child)
    return files


def _file_bytes(paths: list[Path]) -> int:
    return sum(path.stat().st_size for path in paths)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-artifact", required=True, type=Path)
    parser.add_argument("--code", required=True, nargs="+", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=LIMIT_BYTES)
    args = parser.parse_args()

    if not args.model_artifact.is_file():
        raise FileNotFoundError(f"model artifact does not exist: {args.model_artifact}")

    counted_files = _iter_counted_files(args.code)
    model_bytes = args.model_artifact.stat().st_size
    code_bytes = _file_bytes(counted_files)
    total_bytes = model_bytes + code_bytes
    under_budget = total_bytes < args.limit

    result = {
        "event": "submission_byte_budget",
        "limit": args.limit,
        "model_artifact": str(args.model_artifact),
        "model_bytes": model_bytes,
        "model_sha256": _sha256_file(args.model_artifact),
        "code_paths": [str(path) for path in args.code],
        "code_file_count": len(counted_files),
        "code_bytes": code_bytes,
        "code_sha256": hashlib.sha256(
            b"".join(_sha256_file(path).encode("ascii") for path in counted_files)
        ).hexdigest(),
        "total_bytes": total_bytes,
        "under_budget": under_budget,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True), flush=True)
    if not under_budget:
        raise SystemExit(
            f"submission exceeds byte budget: {total_bytes} >= {args.limit}"
        )


if __name__ == "__main__":
    main()
