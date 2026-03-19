from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from autoresearch_pg.lib.workspace import candidate_dir, candidates_root, dump_json, iso_utc, load_json, state_root


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_env_overrides(env_overrides: dict[str, Any] | None) -> dict[str, str]:
    return {
        str(key): str(value)
        for key, value in sorted((env_overrides or {}).items(), key=lambda item: item[0])
    }


def mutation_fingerprints(
    *,
    entrypoint_filename: str,
    entrypoint_text: str,
    env_overrides: dict[str, Any] | None,
    primary_family: str | None,
    mutation_operator: str | None,
    mutation_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    normalized_env = normalize_env_overrides(env_overrides)
    entrypoint_hash = sha256_text(entrypoint_text)
    env_overrides_hash = sha256_text(_canonical_json(normalized_env))
    config_hash = sha256_text(
        _canonical_json(
            {
                "entrypoint_filename": entrypoint_filename,
                "entrypoint_hash": entrypoint_hash,
                "env_overrides": normalized_env,
            }
        )
    )
    mutation_hash = sha256_text(
        _canonical_json(
            {
                "primary_family": primary_family,
                "mutation_operator": mutation_operator,
                "mutation_payload": mutation_payload or {},
                "entrypoint_filename": entrypoint_filename,
                "entrypoint_hash": entrypoint_hash,
            }
        )
    )
    return {
        "entrypoint_hash": entrypoint_hash,
        "env_overrides_hash": env_overrides_hash,
        "config_hash": config_hash,
        "mutation_hash": mutation_hash,
    }


def candidate_record(candidate_id: str) -> dict[str, Any] | None:
    cand_dir = candidate_dir(candidate_id)
    meta_path = cand_dir / "meta.json"
    if not meta_path.is_file():
        return None

    meta = load_json(meta_path, default={})
    entrypoint_filename = meta.get("entrypoint_filename", "train_gpt.py")
    entrypoint_path = cand_dir / entrypoint_filename
    if not entrypoint_path.is_file():
        return None

    fingerprints = mutation_fingerprints(
        entrypoint_filename=entrypoint_filename,
        entrypoint_text=entrypoint_path.read_text(encoding="utf-8"),
        env_overrides=meta.get("env_overrides", {}),
        primary_family=meta.get("primary_family"),
        mutation_operator=meta.get("mutation_operator"),
        mutation_payload=meta.get("mutation_payload"),
    )
    record = {
        "candidate_id": candidate_id,
        "primary_family": meta.get("primary_family"),
        "template_id": meta.get("template_id"),
        "entrypoint_filename": entrypoint_filename,
        "config_hash": fingerprints["config_hash"],
        "mutation_hash": fingerprints["mutation_hash"],
        "entrypoint_hash": fingerprints["entrypoint_hash"],
        "env_overrides_hash": fingerprints["env_overrides_hash"],
        "env_overrides": normalize_env_overrides(meta.get("env_overrides", {})),
        "mutation_operator": meta.get("mutation_operator"),
        "created_at": meta.get("created_at"),
    }
    return record


def rebuild_dedupe_index() -> dict[str, Any]:
    by_config_hash: dict[str, dict[str, Any]] = {}
    by_mutation_hash: dict[str, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    for cand_dir in sorted(path for path in candidates_root().iterdir() if path.is_dir()):
        record = candidate_record(cand_dir.name)
        if record is None:
            continue
        records.append(record)
        by_config_hash.setdefault(record["config_hash"], record)
        by_mutation_hash.setdefault(record["mutation_hash"], record)

    payload = {
        "version": 1,
        "last_updated": iso_utc(),
        "records": records,
        "by_config_hash": by_config_hash,
        "by_mutation_hash": by_mutation_hash,
    }
    dump_json(state_root() / "dedupe_index.json", payload)
    return payload


def load_dedupe_index(refresh: bool = False) -> dict[str, Any]:
    index_path = state_root() / "dedupe_index.json"
    if refresh or not index_path.is_file():
        return rebuild_dedupe_index()
    return load_json(index_path, default={"version": 1, "records": [], "by_config_hash": {}, "by_mutation_hash": {}})


def find_duplicate(
    *,
    config_hash: str,
    mutation_hash: str | None = None,
    refresh: bool = True,
    exclude_candidate_id: str | None = None,
) -> dict[str, Any] | None:
    index = load_dedupe_index(refresh=refresh)
    if exclude_candidate_id is None:
        duplicate = (index.get("by_config_hash") or {}).get(config_hash)
        if duplicate is not None:
            return duplicate
        if mutation_hash:
            return (index.get("by_mutation_hash") or {}).get(mutation_hash)
        return None

    for record in index.get("records", []):
        if record.get("candidate_id") == exclude_candidate_id:
            continue
        if record.get("config_hash") == config_hash:
            return record
    if mutation_hash:
        for record in index.get("records", []):
            if record.get("candidate_id") == exclude_candidate_id:
                continue
            if record.get("mutation_hash") == mutation_hash:
                return record
    return None
