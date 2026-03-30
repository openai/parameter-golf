#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

METRIC_KEYS = ("val_bpb", "cap_val_bpb", "diag_bpb", "sliding_bpb", "ngram9_bpb", "base_model_bpb", "delta", "model_size_bytes")
LOWER_IS_BETTER = {"val_bpb", "cap_val_bpb", "diag_bpb", "sliding_bpb", "ngram9_bpb", "base_model_bpb", "delta", "model_size_bytes"}
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
NUMERIC_RE = r"([+-]?\d+(?:\.\d+)?)"


@dataclass
class CandidateFile:
    path: Path
    category: str


@dataclass
class RecordBundle:
    candidate: CandidateFile
    record: dict
    text: str


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
        # Include SOTA snapshot scripts so path-encoded base/ngram scores are indexed.
        sota_dir = experiments / "SOTA"
        if sota_dir.exists():
            for path in sota_dir.rglob("run.sh"):
                if path.is_file() and not is_hub_path(path):
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
    rel_lower = rel_path.lower()
    sota_match = re.search(r"experiments/sota/([^/]+)/", rel_lower)
    if sota_match:
        return sota_match.group(1)
    if parent in {"logs", "results"} and stem:
        return stem
    if any(token in stem.lower() for token in ("run", "ab", "sweep", "proxy", "summary")):
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


def first_number(pattern: str, text: str, flags: int = re.I) -> float | None:
    match = re.search(pattern, text, flags)
    return float(match.group(1)) if match else None


def parse_metrics(text: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        lower = line.lower()
        if not line:
            continue

        if "final_sliding_window_ngram" in lower:
            ng_order = first_number(r"final_sliding_window_ngram([0-9]+)", line)
            ng_val = first_number(r"\bval_bpb\b\s*[:=]\s*" + NUMERIC_RE, line)
            if ng_order is not None and ng_val is not None and int(ng_order) == 9:
                metrics["ngram9_bpb"] = ng_val
            continue

        cap = first_number(r"\bcap[ _]val_bpb\b\s*[:=]\s*" + NUMERIC_RE, line)
        if cap is not None:
            metrics["cap_val_bpb"] = cap

        diag = None
        if "diagnostic post_ema" in lower or "diag post_ema" in lower:
            diag = first_number(r"\bval_bpb\b\s*[:=]\s*" + NUMERIC_RE, line)
        if diag is not None:
            metrics["diag_bpb"] = diag

        sliding = None
        if ("final_sliding_window" in lower and "ngram" not in lower) or "sliding_bpb" in lower or "sliding_window" in lower:
            sliding = first_number(r"\b(?:sliding_bpb|val_bpb)\b\s*[:=]\s*" + NUMERIC_RE, line)
        if sliding is not None:
            metrics["sliding_bpb"] = sliding
            metrics.setdefault("base_model_bpb", sliding)

        ngram = first_number(r"\bngram9_bpb\b\s*[:=]\s*" + NUMERIC_RE, line)
        if ngram is not None:
            metrics["ngram9_bpb"] = ngram
        if "ngram9" in lower and "val_bpb" in lower:
            ngram_line = first_number(r"\bval_bpb\b\s*[:=]\s*" + NUMERIC_RE, line)
            if ngram_line is not None:
                metrics["ngram9_bpb"] = ngram_line

        delta = first_number(r"\bdelta(?:\([^\)]*\))?(?:[^:=\n]{0,30})[:=]\s*" + NUMERIC_RE, line)
        if delta is not None:
            metrics["delta"] = delta

        generic = None
        if "val_bpb" in lower and not any(token in lower for token in ("cap val_bpb", "cap_val_bpb", "diagnostic post_ema", "diag post_ema", "final_sliding_window", "sliding_bpb", "ngram")):
            generic = first_number(r"\bval_bpb\b\s*[:=]\s*" + NUMERIC_RE, line)
        if generic is not None:
            metrics["val_bpb"] = generic

        base_direct = first_number(r"\bbase(?:[_ ]model)?[ _]bpb(?:\s*\(sliding\))?\b[^0-9+-]*" + NUMERIC_RE, line)
        if base_direct is not None:
            metrics["base_model_bpb"] = base_direct

        model_size = first_number(r"\bserialized model:\s*([0-9]+)\s*bytes\b", line)
        if model_size is not None:
            metrics["model_size_bytes"] = model_size

    table_base_matches = re.findall(
        r"^\|\s*[^|\n]*base[^|\n]*\|\s*\*{0,2}([0-9]+\.[0-9]+)\*{0,2}\s*\|",
        text,
        flags=re.M | re.I,
    )
    table_base_values = [float(v) for v in table_base_matches if 0.9 < float(v) < 2.5]
    if table_base_values:
        metrics.setdefault("base_model_bpb", min(table_base_values))

    return metrics


def extract_notes(text: str) -> list[str]:
    notes: list[str] = []
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in lines:
        lowered = line.lower()
        if any(marker.lower() in lowered for marker in PROMOTE_MARKERS + ERROR_MARKERS) or lowered.startswith("interpretation:"):
            if line not in notes:
                notes.append(line)
    return notes[:10]


def derive_status(text: str, metrics: dict[str, float], notes: list[str]) -> str:
    lowered = text.lower()
    if any(marker.lower() in lowered for marker in ERROR_MARKERS):
        return "error"
    if notes or any(marker.lower() in lowered for marker in PROMOTE_MARKERS):
        return "warn"
    if metrics:
        return "ok"
    return "unknown"


def make_snippet(text: str, max_lines: int = 8, max_chars: int = 520) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    chosen: list[str] = []
    important = (
        "val_bpb",
        "cap_val_bpb",
        "cap val_bpb",
        "diag",
        "sliding",
        "PROMOTE",
        "Decision:",
        "Traceback",
        "ValueError",
        "OOM",
        "failed",
        "Interpretation:",
    )
    for line in lines:
        if any(token.lower() in line.lower() for token in important):
            chosen.append(line)
            if len(chosen) >= max_lines:
                break
    if not chosen:
        chosen = lines[:max_lines]
    return " | ".join(chosen)[:max_chars]


def extract_keywords(text: str) -> list[str]:
    lowered = text.lower()
    hits = []
    for marker in ("oom", "traceback", "shard size mismatch", "cuda error", "promote", "decision", "proxy", "warmdown", "swa", "illegal", "oracle"):
        if marker in lowered:
            hits.append(marker)
    return hits


def parse_path_snapshot_metrics(rel_path: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    snapshot = re.search(r"base_([0-9]+\.[0-9]+)_ngram_([0-9]+\.[0-9]+)", rel_path.lower())
    if snapshot:
        metrics["base_model_bpb"] = float(snapshot.group(1))
        metrics["ngram9_bpb"] = float(snapshot.group(2))
    return metrics


def is_illegal_record(text: str, rel_path: str, notes: list[str]) -> bool:
    lowered = text.lower()
    rel_lower = rel_path.lower()
    if "illegal oracle" in lowered:
        return True
    if "illegal" in lowered and "oracle" in lowered:
        return True
    if any("illegal" in note.lower() for note in notes):
        return True
    if "oracle" in rel_lower and "ngram" in rel_lower:
        return True
    return False


def build_record(candidate: CandidateFile, repo_root: Path, text: str) -> dict:
    path = candidate.path
    rel_path = path.relative_to(repo_root).as_posix()
    metrics = parse_metrics(text)
    metrics.update({k: v for k, v in parse_path_snapshot_metrics(rel_path).items() if k not in metrics})
    notes = extract_notes(text)
    status = derive_status(text, metrics, notes)
    if candidate.category == "script" and metrics:
        status = "ok"
    timestamp_hint = extract_timestamp_hint(rel_path)
    run_tag = derive_run_tag(path, rel_path)
    illegal_score = is_illegal_record(text, rel_path, notes)
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
        "keywords": extract_keywords(text),
        "illegal_score": illegal_score,
    }


def parse_fastab_summary(bundle: RecordBundle) -> dict | None:
    text = bundle.text
    if "A/B" not in text and "Delta (B - A)" not in text:
        return None

    a_line = next((line for line in text.splitlines() if line.strip().startswith("- A:")), "")
    b_line = next((line for line in text.splitlines() if line.strip().startswith("- B:")), "")
    a_cap = first_number(r"cap val_bpb\s+" + NUMERIC_RE, a_line)
    b_cap = first_number(r"cap val_bpb\s+" + NUMERIC_RE, b_line)
    a_diag = first_number(r"DIAGNOSTIC post_ema val_bpb\s+" + NUMERIC_RE, a_line)
    b_diag = first_number(r"DIAGNOSTIC post_ema val_bpb\s+" + NUMERIC_RE, b_line)
    cap_delta = first_number(r"cap val_bpb:\s*" + NUMERIC_RE, text)
    if a_cap is None or b_cap is None:
        return None

    delta = cap_delta if cap_delta is not None else (b_cap - a_cap)
    verdict = "promote challenger" if delta < 0 else "keep baseline"
    return {
        "id": f"ab:{bundle.record['id']}",
        "kind": "ab_pair",
        "title": "Rat Rod Fast A/B",
        "group": "ratrod_fastab",
        "experiment_group": bundle.record["experiment_group"],
        "source_rel_path": bundle.record["rel_path"],
        "timestamp_hint": bundle.record["timestamp_hint"],
        "primary_metric": "cap_val_bpb",
        "baseline_label": "A v1 baseline",
        "baseline_value": a_cap,
        "candidate_label": "B value_residual",
        "candidate_value": b_cap,
        "delta": delta,
        "verdict": verdict,
        "confidence": "medium",
        "summary": f"B improves cap_val_bpb by {delta:.4f} versus A in the economical A/B screen.",
        "rows": [
            {"label": "A v1 baseline", "metrics": {"cap_val_bpb": a_cap, "diag_bpb": a_diag}},
            {"label": "B value_residual", "metrics": {"cap_val_bpb": b_cap, "diag_bpb": b_diag}},
        ],
    }


def parse_proxy_ablation(bundle: RecordBundle) -> dict | None:
    lines = [line.strip() for line in bundle.text.splitlines() if line.strip()]
    table_rows = []
    capture = False
    for line in lines:
        if line.startswith("arm\tseed\tcap_step"):
            capture = True
            continue
        if capture:
            if "\t" not in line:
                break
            parts = line.split("\t")
            if len(parts) >= 6:
                try:
                    table_rows.append(
                        {
                            "label": parts[0],
                            "metrics": {
                                "cap_val_bpb": float(parts[3]),
                            },
                            "seed": parts[1],
                        }
                    )
                except ValueError:
                    pass
    if len(table_rows) < 2:
        return None

    delta = first_number(r"delta\([^\)]*\)=\s*" + NUMERIC_RE, bundle.text)
    baseline = table_rows[0]
    winner = min(table_rows, key=lambda row: row["metrics"]["cap_val_bpb"])
    candidate = winner if winner is not baseline else table_rows[1]
    effective_delta = delta if delta is not None else candidate["metrics"]["cap_val_bpb"] - baseline["metrics"]["cap_val_bpb"]
    verdict = "watch proxy" if effective_delta < 0 else "no signal"
    return {
        "id": f"ab:{bundle.record['id']}",
        "kind": "ab_pair",
        "title": f"{bundle.record['experiment_group']} proxy A/B",
        "group": bundle.record["run_tag"],
        "experiment_group": bundle.record["experiment_group"],
        "source_rel_path": bundle.record["rel_path"],
        "timestamp_hint": bundle.record["timestamp_hint"],
        "primary_metric": "cap_val_bpb",
        "baseline_label": baseline["label"],
        "baseline_value": baseline["metrics"]["cap_val_bpb"],
        "candidate_label": candidate["label"],
        "candidate_value": candidate["metrics"]["cap_val_bpb"],
        "delta": effective_delta,
        "verdict": verdict,
        "confidence": "low",
        "summary": f"Proxy run compares {baseline['label']} vs {candidate['label']} with delta {effective_delta:.4f} cap_val_bpb.",
        "rows": table_rows,
    }


def parse_tsv_ablation(bundle: RecordBundle) -> dict | None:
    text = bundle.text.strip()
    if not text or "\t" not in text:
        return None
    reader = csv.DictReader(text.splitlines(), delimiter="\t")
    rows = []
    for raw_row in reader:
        row = {key.strip(): (value or "").strip() for key, value in raw_row.items() if key}
        label = row.get("arm") or row.get("value") or row.get("sweep") or row.get("seed")
        metrics = {}
        for key in ("val_bpb", "cap_val_bpb", "diag_bpb", "sliding_bpb", "ngram9_bpb", "delta"):
            if row.get(key) and row[key] != "-":
                try:
                    metrics[key] = float(row[key])
                except ValueError:
                    pass
        if label and metrics:
            rows.append({"label": label, "metrics": metrics})
    if len(rows) < 2:
        return None

    primary_metric = next((key for key in ("cap_val_bpb", "diag_bpb", "val_bpb", "sliding_bpb", "ngram9_bpb") if all(key in row["metrics"] for row in rows[: min(3, len(rows))])), None)
    if primary_metric is None:
        primary_metric = next((key for key in ("cap_val_bpb", "diag_bpb", "val_bpb", "sliding_bpb", "ngram9_bpb") if any(key in row["metrics"] for row in rows)), None)
    if primary_metric is None:
        return None

    baseline = rows[0]
    winner = min((row for row in rows if primary_metric in row["metrics"]), key=lambda row: row["metrics"][primary_metric])
    delta = winner["metrics"][primary_metric] - baseline["metrics"].get(primary_metric, winner["metrics"][primary_metric])
    label_prefix = bundle.record["run_tag"].replace("_", " ")
    verdict = "improves baseline" if delta < 0 else "baseline remains best"
    return {
        "id": f"sweep:{bundle.record['id']}",
        "kind": "sweep",
        "title": label_prefix,
        "group": bundle.record["run_tag"],
        "experiment_group": bundle.record["experiment_group"],
        "source_rel_path": bundle.record["rel_path"],
        "timestamp_hint": bundle.record["timestamp_hint"],
        "primary_metric": primary_metric,
        "baseline_label": baseline["label"],
        "baseline_value": baseline["metrics"].get(primary_metric),
        "candidate_label": winner["label"],
        "candidate_value": winner["metrics"].get(primary_metric),
        "delta": delta,
        "verdict": verdict,
        "confidence": "medium",
        "summary": f"Best row is {winner['label']} at {winner['metrics'][primary_metric]:.4f} {primary_metric}; delta vs first row is {delta:.4f}.",
        "rows": rows[:24],
    }


def collect_ablations(bundles: list[RecordBundle]) -> list[dict]:
    ablations: list[dict] = []
    for bundle in bundles:
        if bundle.record["category"] == "summary":
            parsed = parse_fastab_summary(bundle)
            if parsed:
                ablations.append(parsed)
                continue
        if bundle.record["category"] == "run_log":
            parsed = parse_proxy_ablation(bundle)
            if parsed:
                ablations.append(parsed)
        if bundle.record["category"] == "tsv_metric":
            parsed = parse_tsv_ablation(bundle)
            if parsed:
                ablations.append(parsed)

    seen = set()
    deduped = []
    for item in ablations:
        key = (item["kind"], item["group"], item["source_rel_path"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    deduped.sort(key=lambda item: (abs(item.get("delta") or 0.0), item.get("timestamp_hint") or ""), reverse=True)
    return deduped


def to_iso_day(timestamp_hint: str) -> str:
    if not timestamp_hint:
        return "unknown"
    digits = re.sub(r"[^0-9]", "", timestamp_hint)
    if len(digits) >= 8:
        return f"{digits[:4]}-{digits[4:6]}-{digits[6:8]}"
    return "unknown"


def build_hypothesis(ablations: list[dict], records: list[dict]) -> dict:
    improvements = [item for item in ablations if isinstance(item.get("delta"), (int, float)) and item["delta"] < 0]
    regressions = [item for item in ablations if isinstance(item.get("delta"), (int, float)) and item["delta"] > 0]
    best = min(improvements, key=lambda item: item["delta"]) if improvements else None
    worst = max(regressions, key=lambda item: item["delta"]) if regressions else None
    failures = [record for record in records if record["status"] == "error" and record["category"] == "run_log"]
    top_failure = failures[0] if failures else None

    current = (
        f"Favor {best['candidate_label']} within {best['group']} and keep pushing the {best['experiment_group']} lane; it has the cleanest measured improvement on {best['primary_metric']}."
        if best
        else "Current evidence is fragmented; prioritize measurable A/B lanes over broad exploratory logs."
    )
    supporting = (
        f"{best['title']}: {best['candidate_label']} improved {best['primary_metric']} by {best['delta']:.4f} versus {best['baseline_label']}."
        if best
        else "No negative-delta ablation was parsed strongly enough to claim support."
    )
    contradictory = (
        f"{worst['title']}: {worst['candidate_label']} regressed {worst['primary_metric']} by +{worst['delta']:.4f} versus {worst['baseline_label']}."
        if worst
        else (f"Failure pressure remains high in {top_failure['run_tag']}: {top_failure['snippet']}" if top_failure else "No strong contradictory ablation was found.")
    )
    next_test = (
        f"Promote {best['candidate_label']} from {best['group']} into a longer canonical run, then cross-check it against the worst regression family before spending multi-GPU time."
        if best
        else "Create one economical A/B with a single decisive metric and stop carrying forward ambiguous proxy-only wins."
    )
    return {
        "current_hypothesis": current,
        "supporting_signal": supporting,
        "contradictory_signal": contradictory,
        "next_test": next_test,
    }


def build_independent_considerations(records: list[dict]) -> dict:
    def normalized_entry(record: dict, metric_key: str, value: float, label: str, metric_used: str | None = None) -> dict:
        return {
            "category": metric_key,
            "label": label,
            "metric_key": metric_key,
            "metric_used": metric_used or metric_key,
            "value": value,
            "run_tag": record["run_tag"],
            "experiment_group": record["experiment_group"],
            "status": record["status"],
            "rel_path": record["rel_path"],
            "id": record["id"],
        }

    def dedupe_rows(rows: list[dict]) -> list[dict]:
        seen = set()
        out = []
        for row in rows:
            sig = (
                round(float(row["value"]), 8) if isinstance(row.get("value"), (int, float)) else row.get("value"),
                row.get("metric_used"),
                row.get("experiment_group"),
                row.get("run_tag"),
            )
            if sig in seen:
                continue
            seen.add(sig)
            out.append(row)
        return out

    def is_base_family(record: dict) -> bool:
        rel = (record.get("rel_path") or "").lower()
        tag = (record.get("run_tag") or "").lower()
        return (
            "experiments/sota/" in rel
            or "a_wing_green" in rel
            or "awing_green" in rel
            or "rat_rod" in rel
            or "ratrod" in rel
            or "greenrod" in rel
            or "a_wing_green" in tag
            or "awing_green" in tag
            or "rat_rod" in tag
            or "ratrod" in tag
            or "_sota_" in tag
        )

    def is_experiment_evidence(record: dict) -> bool:
        return record.get("category") in {"run_log", "summary", "script", "tsv_metric"}

    def is_plausible_base(value: float | None) -> bool:
        return isinstance(value, (int, float)) and 0.85 <= float(value) <= 2.5

    valid_records = [
        record
        for record in records
        if record["status"] in {"ok", "warn"} and not record.get("illegal_score", False)
    ]

    evidence_records = [record for record in valid_records if is_experiment_evidence(record)]
    if not evidence_records:
        evidence_records = valid_records

    bpb_metric_priority = ("sliding_bpb", "ngram9_bpb", "cap_val_bpb", "diag_bpb", "val_bpb")

    bpb_candidates = []
    for record in evidence_records:
        values = []
        for metric in bpb_metric_priority:
            value = record["metrics"].get(metric)
            if isinstance(value, (int, float)) and value > 0:
                values.append((metric, value))
        if not values:
            continue
        metric_used, best_value = min(values, key=lambda item: item[1])
        bpb_candidates.append(normalized_entry(record, "best_bpb", best_value, "Best BPB", metric_used=metric_used))
    bpb_candidates.sort(key=lambda item: item["value"])
    bpb_candidates = dedupe_rows(bpb_candidates)

    tagged_base_pool = [record for record in evidence_records if is_base_family(record)]
    base_pool = tagged_base_pool or evidence_records

    base_candidates = []
    for record in base_pool:
        value = record["metrics"].get("base_model_bpb")
        metric_used = "base_model_bpb"
        if not is_plausible_base(value):
            value = record["metrics"].get("sliding_bpb")
            metric_used = "sliding_bpb_fallback"
        if is_plausible_base(value):
            base_candidates.append(normalized_entry(record, "best_base_model", float(value), "Best Base Model", metric_used=metric_used))
    base_candidates.sort(key=lambda item: item["value"])
    base_candidates = dedupe_rows(base_candidates)

    size_candidates = []
    for record in evidence_records:
        value = record["metrics"].get("model_size_bytes")
        if isinstance(value, (int, float)) and value > 0:
            size_candidates.append(normalized_entry(record, "lowest_file_size", value, "Lowest File Size", metric_used="model_size_bytes"))
    size_candidates.sort(key=lambda item: item["value"])
    size_candidates = dedupe_rows(size_candidates)

    def winner_or_placeholder(candidates: list[dict], category: str, label: str, metric_used: str) -> dict:
        if candidates:
            return candidates[0]
        return {
            "category": category,
            "label": label,
            "metric_key": category,
            "metric_used": metric_used,
            "value": None,
            "run_tag": "untracked",
            "experiment_group": "n/a",
            "status": "unknown",
            "rel_path": "",
            "id": "",
        }

    considerations = [
        winner_or_placeholder(bpb_candidates, "best_bpb", "Best BPB", "mixed_bpb"),
        winner_or_placeholder(base_candidates, "best_base_model", "Best Base Model", "base_model_bpb"),
        winner_or_placeholder(size_candidates, "lowest_file_size", "Lowest File Size", "model_size_bytes"),
    ]

    return {
        "considerations": considerations,
        "leaderboards": {
            "best_bpb": bpb_candidates[:20],
            "best_base_model": base_candidates[:20],
            "lowest_file_size": size_candidates[:20],
        },
    }


def build_chart_data(records: list[dict], ablations: list[dict]) -> dict:
    status_counts = Counter(record["status"] for record in records)
    category_counts = Counter(record["category"] for record in records)

    timeline_counts: dict[str, Counter] = defaultdict(Counter)
    for record in records:
        timeline_counts[to_iso_day(record.get("timestamp_hint") or "")][record["status"]] += 1

    top_ablations = []
    for item in ablations:
        delta = item.get("delta")
        if isinstance(delta, (int, float)):
            top_ablations.append(
                {
                    "label": f"{item['title']} -> {item['candidate_label']}",
                    "delta": delta,
                    "primary_metric": item.get("primary_metric"),
                }
            )
    top_ablations.sort(key=lambda item: abs(item["delta"]), reverse=True)

    return {
        "status_distribution": [{"name": key, "value": value} for key, value in status_counts.items()],
        "category_distribution": [{"name": key, "value": value} for key, value in category_counts.items()],
        "timeline": [
            {
                "day": day,
                "ok": counts.get("ok", 0),
                "warn": counts.get("warn", 0),
                "error": counts.get("error", 0),
                "unknown": counts.get("unknown", 0),
            }
            for day, counts in sorted(timeline_counts.items())
        ],
        "top_ablation_deltas": top_ablations[:10],
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

    bundles: list[RecordBundle] = []
    seen = set()
    for candidate in iter_source_files(repo_root):
        rel = candidate.path.relative_to(repo_root).as_posix()
        if rel in seen:
            continue
        seen.add(rel)
        text = read_text(candidate.path)
        record = build_record(candidate, repo_root, text)
        bundles.append(RecordBundle(candidate=candidate, record=record, text=text))

    records = [bundle.record for bundle in bundles]
    records.sort(key=lambda item: (item["category"], item["experiment_group"], item["rel_path"]))

    ablations = collect_ablations(bundles)
    hypothesis = build_hypothesis(ablations, records)
    chart_data = build_chart_data(records, ablations)

    counts = {
        "total_records": len(records),
        "by_category": count_by(records, "category"),
        "by_status": count_by(records, "status"),
        "with_metrics": sum(1 for record in records if record["metrics"]),
        "with_errors": sum(1 for record in records if record["status"] == "error"),
        "with_promote_notes": sum(
            1 for record in records if any("promote" in note.lower() or "decision:" in note.lower() for note in record["notes"])
        ),
        "ablations": len(ablations),
    }

    independent = build_independent_considerations(records)

    payload = {
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "source_roots": ["experiments", "results", "logs"],
        "counts": counts,
        "hypothesis": hypothesis,
        "favorite_considerations": independent["considerations"],
        "independent_rankings": independent["leaderboards"],
        "personal_sotas": independent["considerations"],
        "ablations": ablations,
        "charts": chart_data,
        "records": records,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"Wrote {out_path} ({len(records)} records, {len(ablations)} ablations)")


if __name__ == "__main__":
    main()
