from __future__ import annotations

import os
from typing import Any

from autoresearch_pg.lib.locking import state_lock
from autoresearch_pg.lib.workspace import dump_json, iso_utc, load_json, state_root


def codex_sessions_path() -> str:
    return str(state_root() / "codex_sessions.json")


def default_codex_sessions_state() -> dict[str, Any]:
    return {
        "version": 1,
        "last_updated": None,
        "sessions": {},
    }


def ensure_codex_sessions_shape(payload: dict[str, Any]) -> dict[str, Any]:
    base = default_codex_sessions_state()
    base.update(payload)
    base["sessions"] = dict(base.get("sessions", {}))
    return base


def load_codex_sessions_state() -> dict[str, Any]:
    return ensure_codex_sessions_shape(
        load_json(state_root() / "codex_sessions.json", default=default_codex_sessions_state())
    )


def save_codex_sessions_state(payload: dict[str, Any]) -> None:
    payload = ensure_codex_sessions_shape(payload)
    payload["last_updated"] = iso_utc()
    dump_json(state_root() / "codex_sessions.json", payload)


def update_codex_session(candidate_id: str, **fields: Any) -> dict[str, Any]:
    with state_lock():
        payload = load_codex_sessions_state()
        sessions = payload.setdefault("sessions", {})
        row = dict(sessions.get(candidate_id, {}))
        row.setdefault("candidate_id", candidate_id)
        row.update(fields)
        row["updated_at"] = iso_utc()
        sessions[candidate_id] = row
        save_codex_sessions_state(payload)
        return row


def get_codex_session(candidate_id: str) -> dict[str, Any] | None:
    payload = load_codex_sessions_state()
    return (payload.get("sessions") or {}).get(candidate_id)


def list_codex_sessions(*, active_only: bool = False) -> list[dict[str, Any]]:
    payload = load_codex_sessions_state()
    rows = list((payload.get("sessions") or {}).values())
    if active_only:
        rows = [row for row in rows if str(row.get("status")) in {"proposing", "proposal_done", "training"}]
    rows.sort(key=lambda row: str(row.get("started_at") or row.get("updated_at") or ""), reverse=True)
    return rows


def pid_is_alive(pid: int | None) -> bool:
    if pid is None or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True
