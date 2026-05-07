#!/usr/bin/env python3
"""
Pre-submission validation for Parameter Golf records.

Checks folder structure, required files, submission.json schema,
train_gpt.py syntax, and artifact size before opening a PR.

Usage:
    python3 validate_submission.py records/track_10min_16mb/2026-03-24_MySubmission/
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

SIZE_CAP = 16_000_000  # 16 MB decimal, per challenge rules

VALID_TRACKS = (
    "records/track_10min_16mb",
    "records/track_non_record_16mb",
)

# Fields that must be present and non-null in submission.json.
# Each entry maps a canonical name to (accepted_aliases, expected_type).
REQUIRED_FIELDS = {
    "val_bpb": (["val_bpb", "mean_val_bpb"], (int, float)),
    "name": (["name", "run_name"], str),
    "date": (["date"], str),
}

# Fields that should be present but won't fail validation.
RECOMMENDED_FIELDS = {
    "author": str,
    "github_id": str,
    "blurb": str,
}


def check_folder_location(folder: Path, repo_root: Path) -> tuple[bool, str]:
    """Verify the folder is under a valid records/ track."""
    try:
        rel = folder.resolve().relative_to(repo_root.resolve())
    except ValueError:
        return False, f"Folder is not inside the repository: {folder}"
    rel_str = str(rel).replace(os.sep, "/")
    for track in VALID_TRACKS:
        if rel_str.startswith(track + "/"):
            return True, f"Track: {rel_str.split('/')[1]}"
    return False, (
        f"Folder must be under {' or '.join(VALID_TRACKS)}, "
        f"got: {rel_str}"
    )


def check_required_files(folder: Path) -> tuple[bool, list[str]]:
    """Check that README.md, submission.json, and a train_gpt*.py exist."""
    missing = []
    if not (folder / "README.md").is_file():
        missing.append("README.md")
    if not (folder / "submission.json").is_file():
        missing.append("submission.json")
    # Accept train_gpt.py or variants like train_gpt_v5.py.
    train_scripts = list(folder.glob("train_gpt*.py"))
    if not train_scripts:
        missing.append("train_gpt.py (or train_gpt_*.py)")
    return len(missing) == 0, missing


def check_train_log(folder: Path) -> tuple[bool, str]:
    """Check for at least one train log file (.log, .tsv, or .txt)."""
    logs = list(folder.glob("*.log")) + list(folder.glob("*.tsv")) + list(folder.glob("*.txt"))
    if logs:
        names = ", ".join(p.name for p in logs[:5])
        extra = f" (+{len(logs) - 5} more)" if len(logs) > 5 else ""
        return True, f"Found: {names}{extra}"
    return False, "No .log, .tsv, or .txt file found (train log required)"


def check_submission_json(folder: Path) -> tuple[bool, list[str]]:
    """Validate submission.json schema."""
    path = folder / "submission.json"
    errors: list[str] = []
    warnings: list[str] = []

    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        return False, [f"Invalid JSON: {e}"]

    if not isinstance(data, dict):
        return False, ["submission.json must be a JSON object, not a list or scalar"]

    # Required fields (with alias support).
    for canonical, (aliases, expected_type) in REQUIRED_FIELDS.items():
        found_key = None
        for alias in aliases:
            if alias in data:
                found_key = alias
                break
        if found_key is None:
            errors.append(f"Missing required field: {canonical} (also accepts: {', '.join(aliases[1:])})" if len(aliases) > 1 else f"Missing required field: {canonical}")
        elif data[found_key] is None:
            if isinstance(expected_type, tuple):
                want = "/".join(t.__name__ for t in expected_type)
            else:
                want = expected_type.__name__
            errors.append(f"Field '{found_key}' is null (must be {want})")
        elif not isinstance(data[found_key], expected_type):
            got = type(data[found_key]).__name__
            if isinstance(expected_type, tuple):
                want = "/".join(t.__name__ for t in expected_type)
            else:
                want = expected_type.__name__
            errors.append(f"Field '{found_key}' should be {want}, got {got}")

    # Recommended fields (warnings only).
    for field, expected_type in RECOMMENDED_FIELDS.items():
        if field not in data:
            warnings.append(f"Recommended field missing: {field}")
        elif data[field] is not None and not isinstance(data[field], expected_type):
            warnings.append(f"Field '{field}' should be {expected_type.__name__}")

    return len(errors) == 0, errors + [f"(warn) {w}" for w in warnings]


def find_train_script(folder: Path) -> Path | None:
    """Find the train_gpt*.py script in the submission folder."""
    exact = folder / "train_gpt.py"
    if exact.is_file():
        return exact
    variants = list(folder.glob("train_gpt*.py"))
    return variants[0] if variants else None


def check_syntax(folder: Path) -> tuple[bool, str]:
    """Verify the training script has valid Python syntax."""
    path = find_train_script(folder)
    if path is None:
        return False, "No train_gpt*.py found"
    try:
        compile(path.read_text(), str(path), "exec")
        return True, f"OK ({path.name})"
    except SyntaxError as e:
        return False, f"Syntax error in {path.name} at line {e.lineno}: {e.msg}"


def check_size(folder: Path) -> tuple[bool, str]:
    """Report total folder size and warn if close to / over the 16 MB cap."""
    total = sum(f.stat().st_size for f in folder.rglob("*") if f.is_file())
    mb = total / 1_000_000
    if total > SIZE_CAP:
        return False, f"Total size {mb:.2f} MB exceeds {SIZE_CAP / 1_000_000:.0f} MB cap"
    if total > SIZE_CAP * 0.95:
        return True, f"Total size {mb:.2f} MB (close to {SIZE_CAP / 1_000_000:.0f} MB cap)"
    return True, f"Total size {mb:.2f} MB"


def find_repo_root(folder: Path) -> Path | None:
    """Walk up from folder to find the repository root (contains README.md + records/)."""
    current = folder.resolve()
    for _ in range(10):
        if (current / "README.md").is_file() and (current / "records").is_dir():
            return current
        parent = current.parent
        if parent == current:
            break
        current = parent
    return None


def main() -> int:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <submission_folder>")
        print(f"Example: {sys.argv[0]} records/track_10min_16mb/2026-03-24_MyRun/")
        return 1

    folder = Path(sys.argv[1])
    if not folder.is_dir():
        print(f"ERROR: Not a directory: {folder}")
        return 1

    repo_root = find_repo_root(folder)
    if repo_root is None:
        print("WARNING: Could not locate repository root. Skipping folder location check.")

    passed = 0
    failed = 0
    warned = 0

    checks: list[tuple[str, bool, str | list[str]]] = []

    # 1. Folder location.
    if repo_root:
        ok, msg = check_folder_location(folder, repo_root)
        checks.append(("Folder location", ok, msg))
    else:
        checks.append(("Folder location", True, "Skipped (repo root not found)"))

    # 2. Required files.
    ok, missing = check_required_files(folder)
    if ok:
        checks.append(("Required files", True, "README.md, submission.json, train_gpt.py"))
    else:
        checks.append(("Required files", False, f"Missing: {', '.join(missing)}"))

    # 3. Train log.
    ok, msg = check_train_log(folder)
    checks.append(("Train log", ok, msg))

    # 4. submission.json schema (only if file exists).
    if (folder / "submission.json").is_file():
        ok, messages = check_submission_json(folder)
        detail = "; ".join(messages) if messages else "OK"
        checks.append(("submission.json", ok, detail))
    else:
        checks.append(("submission.json", False, "File missing"))

    # 5. Training script syntax (only if a train_gpt*.py exists).
    if find_train_script(folder):
        ok, msg = check_syntax(folder)
        checks.append(("Training script syntax", ok, msg))
    else:
        checks.append(("Training script syntax", False, "No train_gpt*.py found"))

    # 6. Size check.
    ok, msg = check_size(folder)
    checks.append(("Size check", ok, msg))

    # Print results.
    print(f"\nValidating: {folder}\n")
    for name, ok, detail in checks:
        if ok:
            warn_parts = []
            detail_str = detail if isinstance(detail, str) else "; ".join(detail)
            if "(warn)" in detail_str:
                status = "WARN"
                warned += 1
            else:
                status = "PASS"
                passed += 1
        else:
            status = "FAIL"
            failed += 1
            detail_str = detail if isinstance(detail, str) else "; ".join(detail)
        print(f"  [{status}] {name}: {detail_str}")

    print()
    if failed > 0:
        print(f"Result: FAIL ({failed} failed, {passed} passed, {warned} warnings)")
        return 1
    elif warned > 0:
        print(f"Result: PASS with warnings ({passed} passed, {warned} warnings)")
        return 0
    else:
        print(f"Result: PASS ({passed} passed)")
        return 0


if __name__ == "__main__":
    sys.exit(main())
