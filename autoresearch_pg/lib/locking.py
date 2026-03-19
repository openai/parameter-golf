from __future__ import annotations

import fcntl
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from autoresearch_pg.lib.workspace import ensure_layout, state_root


def state_lock_path() -> Path:
    ensure_layout()
    return state_root() / ".state.lock"


@contextmanager
def state_lock() -> Iterator[None]:
    lock_path = state_lock_path()
    with open(lock_path, "a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
