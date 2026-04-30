"""
watchdog.py — Parameter Golf / GhostWrench Safety Monitor
SturdivantAI Lab · April 2026

Three independent monitoring functions:
  1. Golf Barrier     — artifact size enforcement during training
  2. GhostWrench Safety — terminal command whitelist enforcement
  3. Logic Drift      — hero.md manifest comparison

All three share a structured audit log (watchdog_audit.jsonl).
Run as a background process alongside any Parameter Golf sprint or GhostWrench session.

Usage:
  python watchdog.py --mode golf          # Golf Barrier only
  python watchdog.py --mode ghostwrench   # GhostWrench Safety only
  python watchdog.py --mode drift         # Logic Drift check only
  python watchdog.py --mode all           # All three monitors
  python watchdog.py --regenerate-manifest  # Rebuild hero.md hashes (requires signoff)

Official scoring formula:
  Total = Bytes(train_gpt.py) + Bytes(zlib.compress(weights, level=9))
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import io
import json
import os
import re
import sys
import threading
import time
import zlib
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent
HERO_MD_PATH = PROJECT_ROOT / "hero.md"
AUDIT_LOG_PATH = PROJECT_ROOT / "watchdog_audit.jsonl"
SPRINT_LOG_PATH = PROJECT_ROOT / "golf_sprint_log.md"
TRAIN_SCRIPT_PATH = PROJECT_ROOT / "train_gpt.py"
REQUIREMENTS_PATH = PROJECT_ROOT / "requirements.txt"

# Golf Barrier thresholds — total artifact (compressed weights + code bytes)
THRESHOLD_INFO: int = 14_000_000
THRESHOLD_HALT: int = 15_500_000
THRESHOLD_KILL: int = 15_900_000
HARD_BUDGET: int = 16_000_000

# Blacklist categories for Logic Drift
BLACKLIST_DATA_SNEAKING: set[str] = {
    "transformers",
    "datasets",
    "huggingface_hub",
}
BLACKLIST_EXTERNAL_CALLS: set[str] = {
    "requests",
    "urllib3",
    "httpx",
    "selenium",
    "aiohttp",
    "urllib",
}
BLACKLIST_PURE_TORCH_VIOLATIONS: set[str] = {
    "numpy",
    "np",  # common alias
    "scipy",
    "sklearn",
    "pandas",
}
BLACKLIST_VISUALISATION: set[str] = {
    "matplotlib",
    "seaborn",
    "plotly",
    "tensorboard",
    "wandb",
    "tqdm",
}

WHITELIST_LIBRARIES: set[str] = {
    "torch", "torchaudio", "torchvision",
    "flash_attn", "bitsandbytes", "apex",
    "auto_gptq", "autoawq", "optimum",
    "triton", "cupy", "pynvml",
    "sentencepiece", "tokenizers",
    # standard library — always permitted
    "zlib", "io", "os", "hashlib", "re", "json",
    "threading", "time", "datetime", "pathlib", "sys",
    "argparse", "typing", "collections", "math",
    "struct", "itertools", "functools", "abc",
    "contextlib", "dataclasses", "enum",
}

# GhostWrench OpenShell whitelist patterns (regex)
GHOSTWRENCH_WHITELIST_PATTERNS: list[str] = [
    r"^python\s",
    r"^python3\s",
    r"^pip\s",
    r"^conda\s",
    r"^git\s(status|log|diff|add|commit|push|pull|clone|checkout|branch)",
    r"^ls\s",
    r"^cat\s",
    r"^echo\s",
    r"^mkdir\s",
    r"^cp\s",
    r"^mv\s",
    r"^rm\s(?!-rf\s/)",  # allow rm but block rm -rf /
    r"^aws\s(s3|sagemaker|iam|ec2)\s",
    r"^nvidia-smi",
    r"^nvcc\s",
    r"^torch\s",
    r"^pytest\s",
]

# Commands that always BLOCK regardless of whitelist
GHOSTWRENCH_BLOCK_PATTERNS: list[str] = [
    r"rm\s+-rf\s+/",          # wipe root
    r"chmod\s+777",            # world-writable
    r"curl\s.*\|\s*(bash|sh)", # curl pipe to shell
    r"wget\s.*\|\s*(bash|sh)", # wget pipe to shell
    r">\s*/etc/",              # write to /etc
    r"sudo\s+su",              # privilege escalation
    r"dd\s+if=",               # disk operations
    r"mkfs\.",                 # format disk
    r":()\{.*\};:",            # fork bomb
]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _log_audit(entry: dict[str, Any]) -> None:
    """Append a structured entry to the audit log."""
    entry["timestamp"] = datetime.datetime.utcnow().isoformat() + "Z"
    with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")


def _md5(path: Path) -> str:
    """Compute MD5 hash of a file."""
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _format_mb(n: int) -> str:
    return f"{n / 1_000_000:.3f} MB"


# ---------------------------------------------------------------------------
# 1. Golf Barrier
# ---------------------------------------------------------------------------

def measure_artifact(model_state_dict: dict, code_path: Path = TRAIN_SCRIPT_PATH) -> int:
    """
    Official artifact measurement.
    Total = Bytes(train_gpt.py) + Bytes(zlib.compress(weights, level=9))
    """
    buffer = io.BytesIO()
    import torch
    torch.save(model_state_dict, buffer)
    raw: bytes = buffer.getvalue()
    compressed: bytes = zlib.compress(raw, level=9)
    code_bytes: int = os.path.getsize(code_path)
    return len(compressed) + code_bytes


def golf_barrier_check(
    model_state_dict: dict,
    epoch: int,
    session_id: str = "unknown",
) -> str:
    """
    Run Golf Barrier check after each epoch.
    Returns verdict: 'ok' | 'info' | 'halt' | 'kill'
    """
    total = measure_artifact(model_state_dict)
    verdict = "ok"

    if total >= THRESHOLD_KILL:
        verdict = "kill"
        msg = (
            f"[GOLF BARRIER — KILL] Epoch {epoch} | "
            f"Artifact: {_format_mb(total)} >= {_format_mb(THRESHOLD_KILL)} kill threshold. "
            f"TRAINING TERMINATED. Surface offending layer parameter count before restart."
        )
        print(msg, file=sys.stderr)
    elif total >= THRESHOLD_HALT:
        verdict = "halt"
        msg = (
            f"[GOLF BARRIER — HALT] Epoch {epoch} | "
            f"Artifact: {_format_mb(total)} >= {_format_mb(THRESHOLD_HALT)} halt threshold. "
            f"PAUSE — surface layer-by-layer contribution before continuing."
        )
        print(msg, file=sys.stderr)
    elif total >= THRESHOLD_INFO:
        verdict = "info"
        msg = (
            f"[GOLF BARRIER — INFO] Epoch {epoch} | "
            f"Artifact: {_format_mb(total)} — approaching budget. Review architecture."
        )
        print(msg)
    else:
        msg = (
            f"[GOLF BARRIER — OK] Epoch {epoch} | "
            f"Artifact: {_format_mb(total)} / {_format_mb(HARD_BUDGET)} budget."
        )
        print(msg)

    # GPU memory check via pynvml (if available)
    gpu_util_pct: float | None = None
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu_util_pct = mem_info.used / mem_info.total * 100
        if gpu_util_pct > 95.0:
            print(
                f"[GOLF BARRIER — GPU WARNING] VRAM at {gpu_util_pct:.1f}% — "
                f"next epoch risks OOM kill.",
                file=sys.stderr,
            )
    except Exception:
        pass  # pynvml not available on this machine (Lenovo NPU, not NVIDIA)

    _log_audit({
        "monitor": "golf_barrier",
        "epoch": epoch,
        "artifact_bytes": total,
        "verdict": verdict,
        "gpu_vram_pct": gpu_util_pct,
        "session_id": session_id,
    })

    if verdict == "kill":
        raise RuntimeError(
            f"Golf Barrier KILL: artifact {_format_mb(total)} exceeds "
            f"{_format_mb(THRESHOLD_KILL)} kill threshold."
        )

    return verdict


# ---------------------------------------------------------------------------
# 2. GhostWrench Safety
# ---------------------------------------------------------------------------

def _classify_command(command: str) -> str:
    """
    Evaluate a shell command against the OpenShell whitelist.
    Returns: 'PASS' | 'REVIEW' | 'BLOCK'
    """
    command = command.strip()

    # Check hard-block patterns first
    for pattern in GHOSTWRENCH_BLOCK_PATTERNS:
        if re.search(pattern, command, re.IGNORECASE):
            return "BLOCK"

    # Check whitelist patterns
    for pattern in GHOSTWRENCH_WHITELIST_PATTERNS:
        if re.match(pattern, command, re.IGNORECASE):
            return "PASS"

    # Default to REVIEW for unrecognised commands
    return "REVIEW"


def ghostwrench_safety_check(
    command: str,
    session_id: str = "unknown",
    opus47_classifier_pre_check: str = "not_run",
    operator_override: bool = False,
) -> str:
    """
    Intercept and evaluate a GhostWrench shell command.
    Returns verdict: 'PASS' | 'REVIEW' | 'BLOCK'
    Writes structured entry to audit log.
    """
    verdict = _classify_command(command)

    if verdict == "BLOCK" and not operator_override:
        print(
            f"[GHOSTWRENCH SAFETY — BLOCK] Command blocked: `{command}`\n"
            f"  Reason: matches hard-block pattern. Do not retry — reframe the task.",
            file=sys.stderr,
        )
    elif verdict == "REVIEW":
        print(
            f"[GHOSTWRENCH SAFETY — REVIEW] Command requires review: `{command}`\n"
            f"  Not on whitelist — surface to operator before execution.",
            file=sys.stderr,
        )
    else:
        print(f"[GHOSTWRENCH SAFETY — PASS] `{command}`")

    _log_audit({
        "monitor": "ghostwrench_safety",
        "raw_command": command,
        "whitelist_verdict": verdict,
        "opus47_classifier_pre_check": opus47_classifier_pre_check,
        "operator_override": operator_override,
        "session_id": session_id,
    })

    return verdict


# ---------------------------------------------------------------------------
# 3. Logic Drift
# ---------------------------------------------------------------------------

def _parse_imports(py_path: Path) -> set[str]:
    """Extract top-level library names from import statements in a Python file."""
    libs: set[str] = set()
    try:
        content = py_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return libs

    for line in content.splitlines():
        line = line.strip()
        # import X or import X as Y
        m = re.match(r"^import\s+([\w]+)", line)
        if m:
            libs.add(m.group(1))
        # from X import ...
        m = re.match(r"^from\s+([\w]+)", line)
        if m:
            libs.add(m.group(1))
    return libs


def _parse_requirements(req_path: Path) -> set[str]:
    """Extract library names from requirements.txt."""
    libs: set[str] = set()
    try:
        content = req_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return libs
    for line in content.splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            # Strip version specifiers
            name = re.split(r"[>=<!;\[]", line)[0].strip().replace("-", "_").lower()
            if name:
                libs.add(name)
    return libs


def _load_hero_hashes() -> dict[str, str]:
    """Parse MD5 hashes from hero.md Section 5."""
    hashes: dict[str, str] = {}
    if not HERO_MD_PATH.exists():
        return hashes
    content = HERO_MD_PATH.read_text(encoding="utf-8")
    # Match table rows: | filename | hash | ...
    for m in re.finditer(r"\|\s*([\w./\-]+\.\w+)\s*\|\s*([a-f0-9]{32})\s*\|", content):
        hashes[m.group(1).strip()] = m.group(2).strip()
    return hashes


def logic_drift_check(session_id: str = "unknown") -> list[dict[str, str]]:
    """
    Compare current project state against hero.md manifest.
    Returns list of violations (empty = clean).
    """
    violations: list[dict[str, str]] = []

    # --- Library check ---
    all_imports: set[str] = set()
    for py_file in PROJECT_ROOT.glob("*.py"):
        all_imports |= _parse_imports(py_file)
    all_imports |= _parse_requirements(REQUIREMENTS_PATH)

    blacklists = {
        "Data Sneaking": BLACKLIST_DATA_SNEAKING,
        "External Calls": BLACKLIST_EXTERNAL_CALLS,
        "Pure Torch Violations": BLACKLIST_PURE_TORCH_VIOLATIONS,
        "Visualisation/Logging": BLACKLIST_VISUALISATION,
    }

    for category, blocked in blacklists.items():
        found = all_imports & blocked
        for lib in found:
            msg = (
                f"[LOGIC DRIFT — ARCHITECTURAL VIOLATION] "
                f"Blocked library `{lib}` (category: {category}) "
                f"found in imports or requirements.txt."
            )
            print(msg, file=sys.stderr)
            violations.append({"type": "blocked_library", "library": lib, "category": category})

    # --- File hash check ---
    hero_hashes = _load_hero_hashes()
    for filename, expected_hash in hero_hashes.items():
        if expected_hash.upper() == "PENDING":
            continue
        fpath = PROJECT_ROOT / filename
        if not fpath.exists():
            violations.append({"type": "missing_file", "file": filename})
            print(
                f"[LOGIC DRIFT — MISSING FILE] `{filename}` in manifest but not found on disk.",
                file=sys.stderr,
            )
            continue
        actual_hash = _md5(fpath)
        if actual_hash != expected_hash:
            msg = (
                f"[LOGIC DRIFT — HASH MISMATCH] `{filename}` "
                f"expected {expected_hash}, got {actual_hash}. "
                f"File was modified outside the approved workflow."
            )
            print(msg, file=sys.stderr)
            violations.append({
                "type": "hash_mismatch",
                "file": filename,
                "expected": expected_hash,
                "actual": actual_hash,
            })

    if not violations:
        print("[LOGIC DRIFT — CLEAN] Project state matches hero.md manifest.")

    _log_audit({
        "monitor": "logic_drift",
        "violations_count": len(violations),
        "violations": violations,
        "session_id": session_id,
    })

    return violations


# ---------------------------------------------------------------------------
# Manifest Regeneration
# ---------------------------------------------------------------------------

def regenerate_manifest() -> None:
    """
    Rebuild hero.md Section 5 hashes from current source files.
    Requires architect signoff — will prompt before writing.

    IMPORTANT: Never use this as a shortcut around a Logic Drift flag.
    Investigate the drift first.
    """
    print("\n[MANIFEST REGENERATION]")
    print("WARNING: This updates hero.md with current file hashes.")
    print("Only run this after a legitimate architectural change approved by Kiro.\n")

    signoff = input("Architect signoff reason (or CTRL+C to abort): ").strip()
    if not signoff:
        print("Aborted — no signoff provided.")
        return

    files_to_hash = [
        PROJECT_ROOT / "train_gpt.py",
        PROJECT_ROOT / "watchdog.py",
        PROJECT_ROOT / "requirements.txt",
        PROJECT_ROOT / ".kiro" / "skills" / "parameter-golf.md",
    ]

    print("\nComputing hashes...")
    new_hashes: list[tuple[str, str]] = []
    for fpath in files_to_hash:
        if fpath.exists():
            rel = str(fpath.relative_to(PROJECT_ROOT)).replace("\\", "/")
            h = _md5(fpath)
            new_hashes.append((rel, h))
            print(f"  {rel}: {h}")
        else:
            print(f"  {fpath.name}: NOT FOUND — skipping")

    # Read current hero.md and replace Section 5 table
    content = HERO_MD_PATH.read_text(encoding="utf-8")
    now = datetime.datetime.utcnow().isoformat() + "Z"

    # Update regeneration metadata
    content = re.sub(
        r"regenerated_at:.*",
        f"regenerated_at: {now}",
        content,
    )
    content = re.sub(
        r"architect_signoff:.*",
        f"architect_signoff: {signoff}",
        content,
    )

    # Rebuild Section 5 table
    table_header = (
        "| File                   | MD5 Hash                         | Last verified |\n"
        "|------------------------|----------------------------------|---------------|\n"
    )
    rows = "".join(
        f"| {rel:<22} | {h}                 | {now[:10]} |\n"
        for rel, h in new_hashes
    )
    new_section5 = f"## Section 5 — Source File Hashes\n\n{table_header}{rows}"

    content = re.sub(
        r"## Section 5 — Source File Hashes.*",
        new_section5,
        content,
        flags=re.DOTALL,
    )

    HERO_MD_PATH.write_text(content, encoding="utf-8")
    print(f"\nManifest updated: {HERO_MD_PATH}")
    print(f"Signoff recorded: {signoff}")

    _log_audit({
        "monitor": "manifest_regeneration",
        "files_hashed": [rel for rel, _ in new_hashes],
        "architect_signoff": signoff,
        "session_id": "manual",
    })


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="watchdog.py — Parameter Golf / GhostWrench Safety Monitor"
    )
    parser.add_argument(
        "--mode",
        choices=["golf", "ghostwrench", "drift", "all"],
        default="drift",
        help="Which monitor to run (default: drift)",
    )
    parser.add_argument(
        "--regenerate-manifest",
        action="store_true",
        help="Rebuild hero.md hashes from current source files (requires architect signoff)",
    )
    parser.add_argument(
        "--command",
        type=str,
        default=None,
        help="Shell command to evaluate (ghostwrench mode)",
    )
    parser.add_argument(
        "--session-id",
        type=str,
        default="manual",
        help="Session identifier for audit log",
    )
    args = parser.parse_args()

    if args.regenerate_manifest:
        regenerate_manifest()
        return

    if args.mode in ("drift", "all"):
        print("\n--- Logic Drift Check ---")
        violations = logic_drift_check(session_id=args.session_id)
        if violations:
            print(f"\n{len(violations)} violation(s) found. Resolve before sprint.")

    if args.mode in ("ghostwrench", "all") and args.command:
        print("\n--- GhostWrench Safety Check ---")
        ghostwrench_safety_check(args.command, session_id=args.session_id)

    if args.mode == "golf":
        print(
            "Golf Barrier runs as a callback during training — import and call "
            "golf_barrier_check(model.state_dict(), epoch=N) after each epoch."
        )


if __name__ == "__main__":
    main()
