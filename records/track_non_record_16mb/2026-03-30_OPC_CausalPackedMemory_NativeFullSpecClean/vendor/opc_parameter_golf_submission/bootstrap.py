from __future__ import annotations

from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parent
VENDOR_ROOT = PROJECT_ROOT / "vendor"


def _resolve_repo(vendor_name: str, workspace_name: str) -> Path:
    vendored = VENDOR_ROOT / vendor_name
    if vendored.exists():
        return vendored
    return WORKSPACE_ROOT / workspace_name


OPC_ROOT = _resolve_repo("open_predictive_coder", "open-predictive-coder")
LEDGER_ROOT = _resolve_repo("conker_ledger", "conker-ledger")
CONKER_ROOT = WORKSPACE_ROOT / "conker"
DETECT_ROOT = WORKSPACE_ROOT / "conker-detect"


def add_local_sources() -> None:
    for root in (OPC_ROOT, LEDGER_ROOT):
        src = root / "src"
        if src.exists() and str(src) not in sys.path:
            sys.path.insert(0, str(src))


def git_head(root: Path) -> str | None:
    if not (root / ".git").exists():
        return None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    value = result.stdout.strip()
    return value or None


def git_snapshot_ref(root: Path) -> str | None:
    head = git_head(root)
    if head is None:
        return None
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return head
    return f"{head}-dirty" if result.stdout.strip() else head


__all__ = [
    "PROJECT_ROOT",
    "WORKSPACE_ROOT",
    "VENDOR_ROOT",
    "OPC_ROOT",
    "LEDGER_ROOT",
    "CONKER_ROOT",
    "DETECT_ROOT",
    "add_local_sources",
    "git_head",
    "git_snapshot_ref",
]
