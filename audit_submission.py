from __future__ import annotations

import json
import os
import sys
import zlib
from pathlib import Path

REQUIRED_PATHS = [
    Path("records/non-record/multi-cube-face-letter-assignment/submission.template.json"),
    Path("records/non-record/multi-cube-face-letter-assignment/README.submission.md"),
]

OPTIONAL_ARTIFACT_CANDIDATES = [
    Path("final_model.int8.ptz"),
    Path("artifact.ptz"),
    Path("model.ptz"),
    Path("submission_model.ptz"),
]


def file_size(path: Path) -> int:
    return path.stat().st_size


def maybe_decompress_ptz(path: Path) -> tuple[bool, int | None, str | None]:
    try:
        raw = path.read_bytes()
        dec = zlib.decompress(raw)
        return True, len(dec), None
    except Exception as exc:  # noqa: BLE001
        return False, None, str(exc)


def main() -> int:
    root = Path.cwd()
    print(f"[audit] repo root: {root}")

    missing = [str(p) for p in REQUIRED_PATHS if not p.exists()]
    if missing:
        print("[audit] missing required files:")
        for p in missing:
            print(f"  - {p}")
    else:
        print("[audit] required submission files are present")

    found_artifacts = [p for p in OPTIONAL_ARTIFACT_CANDIDATES if p.exists()]
    if not found_artifacts:
        print("[audit] no known artifact candidate found in repo root")
    else:
        for artifact in found_artifacts:
            size = file_size(artifact)
            print(f"[audit] artifact: {artifact} -> {size} bytes")
            if size >= 16_000_000:
                print("  [FAIL] exceeds 16,000,000 byte limit")
            else:
                print("  [OK] under 16,000,000 byte limit")
            ok, dec_size, err = maybe_decompress_ptz(artifact)
            if ok:
                print(f"  [info] zlib-decompressed size: {dec_size} bytes")
            else:
                print(f"  [info] not zlib-decompressed: {err}")

    submission_path = Path("records/non-record/multi-cube-face-letter-assignment/submission.template.json")
    if submission_path.exists():
        try:
            data = json.loads(submission_path.read_text())
            print("[audit] submission JSON parses correctly")
            placeholders = []
            def walk(obj, prefix=""):
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        walk(v, f"{prefix}.{k}" if prefix else k)
                elif isinstance(obj, list):
                    for i, v in enumerate(obj):
                        walk(v, f"{prefix}[{i}]")
                elif isinstance(obj, str) and ("TODO" in obj or "UNVERIFIED" in obj):
                    placeholders.append((prefix, obj))
            walk(data)
            if placeholders:
                print("[audit] placeholders that must be resolved before claiming final submission:")
                for k, v in placeholders:
                    print(f"  - {k}: {v}")
            else:
                print("[audit] no TODO/UNVERIFIED strings found")
        except Exception as exc:  # noqa: BLE001
            print(f"[audit] submission JSON parse failed: {exc}")
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
