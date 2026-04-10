#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

METRIC_NAMES = ("val_bpb", "cap_val_bpb", "diag_bpb", "sliding_bpb", "ngram9_bpb", "delta")
ERROR_MARKERS = (
    "Traceback",
    "ValueError",
    "OOM",
    "out of memory",
    "CUDA error",
    "ChildFailedError",
    "AssertionError",
    "RuntimeError",
    "failed",
    "Killed",
    "shard size mismatch",
)
PROMOTE_MARKERS = ("PROMOTE:", "Decision:", "promote", "PROMOTE")
TIMESTAMP_PATTERNS = (
    r"(20\d{2}-\d{2}-\d{2}(?:[ _T]\d{2}:\d{2}:\d{2})?)",
    r"(20\d{6}_\d{6})",
    r"(20\d{6})",
    r"(\d{8}_\d{6})",
)


@dataclass
class CandidateFile:
    path: Path
    category: str


def discover_repo_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if (parent / "experiments").exists() and (parent / "results").exists() and (parent / "logs").exists():
            return parent
    raise SystemExit("Could not locate repository root from script path.")


def iter_source_files(repo_root: Path) -> Iterable[CandidateFile]:
    experiments = repo_root / "experiments"
    results = repo_root / "results"
    logs = repo_root / "logs"

    def is_hub_path(path: Path) -> bool:
        return any(part.startswith("research_hub_") for part in path.parts)

    if experiments.exists():
        for path in experiments.rglob("*.md"):
            if path.is_file() and not is_hub_path(path):
                yield CandidateFile(path, infer_category(path))
        for suffix in ("*.log", "*.tsv", "*.env", "*.sh"):
            for path in experiments.rglob(suffix):
                if path.is_file() and "vast_tests" in path.parts and not is_hub_path(path):
                    yield CandidateFile(path, infer_category(path))

    if results.exists():
        for suffix in ("*.log", "*.txt", "*.tsv", "*.md"):
            for path in results.rglob(suffix):
                if path.is_file():
                    yield CandidateFile(path, infer_category(path))

    if logs.exists():
        for suffix in ("*.log", "*.txt", "*.tsv"):
            for path in logs.rglob(suffix):
                if path.is_file():
                    yield CandidateFile(path, infer_category(path))


def infer_category(path: Path) -> str:
    suffix = path.suffix.lower()
    name = path.name.lower()
    parts = [part.lower() for part in path.parts]
    in_vast = "vast_tests" in parts

    if suffix == ".sh":
        return "script"
    if suffix == ".env":
        return "env"
    if suffix == ".tsv":
        return "tsv_metric"
    if suffix == ".md":
        if "summary" in name:
            return "summary"
        if in_vast:
            return "report" if "report" in name else "summary"
        return "report"
    if suffix in {".log", ".txt"}:
        if "summary" in name:
            return "summary"
        return "run_log"
    return "other"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def extract_timestamp_hint(rel_path: str) -> str:
    for pattern in TIMESTAMP_PATTERNS:
        match = re.search(pattern, rel_path)
        if match:
            return match.group(1)
    return ""


def derive_run_tag(path: Path, rel_path: str) -> str:
    stem = path.stem
    parent = path.parent.name
    if parent in {"logs", "results"} and stem:
        return stem
    if "run" in stem.lower() or "ab" in stem.lower() or "sweep" in stem.lower():
        return stem
    if "vast_tests" in path.parts:
        return parent if parent != "vast_tests" else stem
    return stem


def derive_experiment_group(path: Path, repo_root: Path) -> str:
    rel = path.relative_to(repo_root)
    if not rel.parts:
        return ""
    root = rel.parts[0]
    if root == "experiments" and len(rel.parts) > 1:
        return rel.parts[1]
    if root in {"results", "logs"}:
        return root
    return root


def parse_metrics(text: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for line in text.splitlines():
        for name, value in re.findall(r"(?<![A-Za-z0-9_])(" + "|".join(METRIC_NAMES) + r")\s*[:=]\s*([+-]?\d+(?:\.\d+)?)", line, flags=re.I):
            metrics[name.lower()] = float(value)
        delta_match = re.search(r"(?i)\bdelta(?:[^=\n:]{0,40})[:=]\s*([+-]?\d+(?:\.\d+)?)", line)
        if delta_match:
            metrics["delta"] = float(delta_match.group(1))
    return metrics


def extract_notes(text: str) -> list[str]:
    notes: list[str] = []
    for marker in PROMOTE_MARKERS:
        for line in text.splitlines():
            if marker.lower() in line.lower():
                cleaned = line.strip()
                if cleaned and cleaned not in notes:
                    notes.append(cleaned)
    for marker in ERROR_MARKERS:
        for line in text.splitlines():
            if marker.lower() in line.lower():
                cleaned = line.strip()
                if cleaned and cleaned not in notes:
                    notes.append(cleaned)
    return notes[:8]


def derive_status(text: str, metrics: dict[str, float], notes: list[str]) -> str:
    lowered = text.lower()
    if any(marker.lower() in lowered for marker in ERROR_MARKERS):
        return "error"
    if notes or any(marker in text for marker in PROMOTE_MARKERS):
        return "warn"
    if metrics:
        return "ok"
    return "unknown"


def make_snippet(text: str, max_lines: int = 8, max_chars: int = 420) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    chosen: list[str] = []
    important = (
        "val_bpb",
        "cap_val_bpb",
        "diag_bpb",
        "sliding_bpb",
        "ngram9_bpb",
        "PROMOTE",
        "Decision:",
        "Traceback",
        "ValueError",
        "OOM",
        "failed",
    )
    for line in lines:
        if any(token.lower() in line.lower() for token in important):
            chosen.append(line)
            if len(chosen) >= max_lines:
                break
    if not chosen:
        chosen = lines[:max_lines]
    snippet = " | ".join(chosen)
    return snippet[:max_chars]


def file_record(candidate: CandidateFile, repo_root: Path) -> dict:
    path = candidate.path
    rel_path = path.relative_to(repo_root).as_posix()
    text = read_text(path)
    metrics = parse_metrics(text)
    notes = extract_notes(text)
    status = derive_status(text, metrics, notes)
    timestamp_hint = extract_timestamp_hint(rel_path)
    run_tag = derive_run_tag(path, rel_path)

    record_id = hashlib.sha1(rel_path.encode("utf-8")).hexdigest()[:16]
    return {
        "id": f"{candidate.category}:{record_id}",
        "path": str(path),
        "rel_path": rel_path,
        "category": candidate.category,
        "experiment_group": derive_experiment_group(path, repo_root),
        "run_tag": run_tag,
        "timestamp_hint": timestamp_hint,
        "metrics": metrics,
        "status": status,
        "notes": notes,
        "snippet": make_snippet(text),
    }


def count_by(records: list[dict], key: str) -> dict[str, int]:
    return dict(Counter((record.get(key) or "unknown") for record in records))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the research hub JSON index.")
    parser.add_argument("--repo-root", default="", help="Repository root. Auto-detected if omitted.")
    parser.add_argument("--out", default="", help="Output JSON path. Defaults to hub data/hub_index.json.")
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    repo_root = Path(args.repo_root).resolve() if args.repo_root else discover_repo_root(script_path.parent)
    hub_root = script_path.parent.parent
    out_path = Path(args.out).resolve() if args.out else hub_root / "data" / "hub_index.json"

    records = []
    seen = set()
    for candidate in iter_source_files(repo_root):
        rel = candidate.path.relative_to(repo_root).as_posix()
        if rel in seen:
            continue
        seen.add(rel)
        records.append(file_record(candidate, repo_root))

    records.sort(key=lambda item: (item["category"], item["experiment_group"], item["rel_path"]))

    counts = {
        "total_records": len(records),
        "by_category": count_by(records, "category"),
        "by_status": count_by(records, "status"),
        "with_metrics": sum(1 for record in records if record["metrics"]),
        "with_errors": sum(1 for record in records if record["status"] == "error"),
        "with_promote_notes": sum(
            1 for record in records if any("promote" in note.lower() or "decision:" in note.lower() for note in record["notes"])
        ),
    }

    payload = {
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "source_roots": ["experiments", "results", "logs"],
        "counts": counts,
        "records": records,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"Wrote {out_path} ({len(records)} records)")


if __name__ == "__main__":
    main()
