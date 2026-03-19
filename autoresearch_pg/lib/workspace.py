from __future__ import annotations

import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def autoresearch_root() -> Path:
    return repo_root() / "autoresearch_pg"


def candidates_root() -> Path:
    return autoresearch_root() / "candidates"


def runs_root() -> Path:
    return autoresearch_root() / "runs"


def state_root() -> Path:
    return autoresearch_root() / "state"


def config_root() -> Path:
    return autoresearch_root() / "config"


def ensure_layout() -> None:
    for path in (candidates_root(), runs_root(), state_root()):
        path.mkdir(parents=True, exist_ok=True)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_stamp() -> str:
    return utc_now().strftime("%Y%m%d_%H%M%S")


def iso_utc() -> str:
    return utc_now().isoformat().replace("+00:00", "Z")


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    slug = slug.strip("._-")
    return slug or "candidate"


def load_json(path: Path, default: Any | None = None) -> Any:
    if not path.is_file():
        if default is None:
            raise FileNotFoundError(path)
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp_path, path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def candidate_dir(candidate_id: str) -> Path:
    return candidates_root() / candidate_id


def run_dir(candidate_id: str, tier: str, run_id: str) -> Path:
    return runs_root() / candidate_id / tier / run_id


def default_candidate_id(prefix: str = "cand") -> str:
    return f"{prefix}_{utc_stamp()}"


def relative_to_repo(path: Path) -> str:
    return str(path.resolve().relative_to(repo_root()))


def bootstrap_candidate(
    candidate_id: str,
    source_train_gpt: Path,
    parent_candidate_id: str | None = None,
    note: str | None = None,
) -> Path:
    ensure_layout()
    out_dir = candidate_dir(candidate_id)
    if out_dir.exists():
        raise FileExistsError(f"candidate already exists: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=False)

    entrypoint_filename = source_train_gpt.name
    train_dst = out_dir / entrypoint_filename
    shutil.copy2(source_train_gpt, train_dst)

    meta = {
        "candidate_id": candidate_id,
        "created_at": iso_utc(),
        "entrypoint_filename": entrypoint_filename,
        "parent_candidate_id": parent_candidate_id,
        "source_train_gpt": str(source_train_gpt.resolve()),
    }
    dump_json(out_dir / "meta.json", meta)

    notes = [
        f"# {candidate_id}",
        "",
        "1. Hypothesis",
        "",
        note or "Fill in the intended improvement.",
        "",
        "2. Expected upside",
        "",
        "3. Expected risk",
        "",
        "4. Exact knobs changed",
        "",
        "5. Promotion bar",
        ""
    ]
    (out_dir / "notes.md").write_text("\n".join(notes), encoding="utf-8")
    return out_dir


def iter_run_json_paths() -> list[Path]:
    return sorted(runs_root().rglob("run.json"))


def best_run_for_candidate(
    candidate_id: str,
    tier: str | None = None,
    require_valid: bool = False,
) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []

    if tier is None:
        json_paths = sorted((runs_root() / candidate_id).rglob("run.json"))
    else:
        json_paths = sorted((runs_root() / candidate_id / tier).rglob("run.json"))

    for path in json_paths:
        payload = load_json(path)
        if require_valid and not payload.get("objective", {}).get("valid"):
            continue
        candidates.append(payload)

    if not candidates:
        return None

    return min(candidates, key=lambda item: float(item.get("objective", {}).get("proxy_score", 1e9)))


def update_best_state(run_payload: dict[str, Any]) -> None:
    ensure_layout()
    best_path = state_root() / "best.json"
    best = load_json(best_path, default={"global": None, "by_tier": {}})

    summary = {
        "candidate_id": run_payload["candidate_id"],
        "tier": run_payload["tier"],
        "run_id": run_payload["run_id"],
        "run_dir": run_payload["run_dir"],
        "proxy_score": run_payload["objective"].get("proxy_score"),
        "post_quant_val_bpb": run_payload["objective"].get("post_quant_val_bpb"),
        "bytes_total": run_payload["objective"].get("bytes_total"),
        "valid": run_payload["objective"].get("valid")
    }

    current_global = best.get("global")
    if current_global is None or summary["proxy_score"] < current_global.get("proxy_score", 1e9):
        best["global"] = summary

    tier_key = run_payload["tier"]
    current_tier = best.get("by_tier", {}).get(tier_key)
    if current_tier is None or summary["proxy_score"] < current_tier.get("proxy_score", 1e9):
        best.setdefault("by_tier", {})[tier_key] = summary

    dump_json(best_path, best)
