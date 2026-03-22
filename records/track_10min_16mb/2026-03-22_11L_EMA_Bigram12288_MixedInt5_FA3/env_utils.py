from __future__ import annotations

import os
from pathlib import Path

PATH_LIKE_KEYS = frozenset({"DATA_PATH", "TOKENIZER_PATH", "SAMPLE_CHECKPOINT"})


def resolve_path_value(script_dir: Path, raw_value: str) -> str:
    path = Path(raw_value.strip())
    if path.is_absolute():
        return str(path)

    candidates = [
        (script_dir / path).resolve(),
        (script_dir.parent / path).resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[0])


def load_env_file(script_dir: Path, filename: str = ".env") -> None:
    env_path = Path(filename)
    if not env_path.is_absolute():
        env_path = script_dir / env_path
    if not env_path.is_file():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or key in os.environ:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if key in PATH_LIKE_KEYS and value:
            value = resolve_path_value(script_dir, value)
        os.environ[key] = value
