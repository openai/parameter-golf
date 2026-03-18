from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
LAB_DIR = REPO_ROOT / "lab"
RUNS_DIR = LAB_DIR / "runs"
HISTORY_PATH = LAB_DIR / "experiments.jsonl"
STATE_PATH = LAB_DIR / "state.json"
JOURNAL_PATH = REPO_ROOT / "docs" / "EXPERIMENT_JOURNAL.md"
VENV_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def timestamp_slug() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_lab_layout() -> None:
    LAB_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)


def read_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def format_env_block(env: dict[str, str]) -> str:
    if not env:
        return "```bash\n# no overrides\n```"
    lines = [f"{key}={value}" for key, value in sorted(env.items())]
    return "```bash\n" + "\n".join(lines) + "\n```"


def canonical_env(env: dict[str, str]) -> str:
    return json.dumps({k: str(v) for k, v in sorted(env.items())}, sort_keys=True, separators=(",", ":"))


def canonical_code_mutation(code_mutation: dict[str, Any] | None) -> str:
    if not code_mutation:
        return "null"
    normalized = {
        "name": code_mutation.get("name"),
        "family": code_mutation.get("family"),
        "params": code_mutation.get("params", {}),
        "script_basename": code_mutation.get("script_basename")
        or Path(str(code_mutation.get("script", ""))).name
        or None,
        "source_hash": code_mutation.get("source_hash"),
    }
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"))


def preferred_python() -> str:
    if VENV_PYTHON.is_file():
        return str(VENV_PYTHON)
    return "python3"
