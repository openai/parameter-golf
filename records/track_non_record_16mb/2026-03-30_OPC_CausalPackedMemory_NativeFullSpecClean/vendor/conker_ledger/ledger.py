from __future__ import annotations

import json
import math
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
import csv


DATE_SUFFIX_RE = re.compile(r"_\d{4}-\d{2}-\d{2}$")
FULL_EVAL_SUFFIX_RE = re.compile(r"_fullval_(?:train|test)_[a-z0-9]+$")
SEED_RE = re.compile(r"_seed(\d+)")


def _json_default(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def dumps_json(value: Any) -> str:
    return json.dumps(value, indent=2, default=_json_default)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


CLAIM_LEVELS = {
    0: "No justified claim yet",
    1: "Bridge metric only",
    2: "Fresh-process held-out replay confirmed",
    3: "Packed-artifact replay confirmed",
    4: "Structural audit passed",
    5: "Behavioral legality audit passed",
}
TIER3_PROMOTION_TRUST_LEVELS = {"traced", "strict"}
LIMITED_TIER3_SCOPES = {"prefix-only", "one_shot_runtime_handoff"}


def finite_or_none(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        if math.isfinite(float(value)):
            return float(value)
        return None
    return None


def _resolve_input_path(base_dir: Path, value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _resolve_output_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Bundle attachment destination must stay inside the bundle: {value}")
    return path


def _load_manifest_value(value: Any, base_dir: Path) -> Any:
    if value is None:
        return {}
    if isinstance(value, str):
        return load_json(_resolve_input_path(base_dir, value))
    return value


def _dict_has_payload(data: Any, keys: tuple[str, ...]) -> bool:
    if not isinstance(data, dict):
        return False
    for key in keys:
        if key in data and data[key] not in (None, "", {}, []):
            return True
    return False


def _audit_status(audits: Any, tier: str) -> str | None:
    if not isinstance(audits, dict):
        return None
    tier_data = audits.get(tier)
    if not isinstance(tier_data, dict):
        return None
    status = tier_data.get("status")
    return str(status) if status is not None else None


def infer_claim_level(claim: Any, metrics: Any, audits: Any) -> dict[str, Any]:
    level = 0
    if claim not in (None, "", {}, []) or metrics not in (None, "", {}, []):
        level = 1
    if _dict_has_payload(metrics, ("fresh_process_full", "fresh_process_replay", "held_out_replay")):
        level = max(level, 2)
    if _dict_has_payload(metrics, ("packed_artifact_full", "packed_artifact_replay", "packed_replay")):
        level = max(level, 3)
    if _audit_status(audits, "tier2") == "pass":
        level = max(level, 4)
    tier3_credit = _tier3_claim_credit(audits)
    if tier3_credit["credited"]:
        level = max(level, 5)
    label = CLAIM_LEVELS[level]
    if level == 5 and tier3_credit.get("trust_achieved") in TIER3_PROMOTION_TRUST_LEVELS:
        label = f"{label} ({tier3_credit['trust_achieved']})"
    return {
        "level": level,
        "label": label,
        "tier3_credit": tier3_credit,
        "notes": tier3_credit.get("notes", []),
    }


def _copy_attachment(base_dir: Path, out_dir: Path, spec: dict[str, Any]) -> dict[str, Any]:
    source_value = spec.get("source") or spec.get("path")
    if not isinstance(source_value, str):
        raise ValueError("Each attachment needs a string source/path")
    source = _resolve_input_path(base_dir, source_value)
    if not source.exists():
        raise FileNotFoundError(source)
    dest_rel = _resolve_output_path(spec.get("dest") or f"artifacts/{source.name}")
    dest = out_dir / dest_rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    resolved_dest = dest.resolve()
    resolved_out = out_dir.resolve()
    if not resolved_dest.is_relative_to(resolved_out):
        raise ValueError(f"Resolved attachment destination escapes bundle: {dest_rel} -> {resolved_dest}")
    if source.is_dir():
        shutil.copytree(source, dest, dirs_exist_ok=True)
        kind = "directory"
    else:
        shutil.copy2(source, dest)
        kind = "file"
    return {
        "source": str(source),
        "dest": str(dest_rel),
        "kind": kind,
    }


def render_validity_bundle_readme(
    *,
    bundle_id: str,
    claim: Any,
    metrics: Any,
    provenance: Any,
    audits: Any,
    claim_level: dict[str, Any],
    attachments: list[dict[str, Any]],
    detector_summaries: list[dict[str, Any]],
) -> str:
    requested_label = None
    if isinstance(claim, dict):
        requested_label = claim.get("requested_label") or claim.get("requested_claim")
    bridge_bpb = metrics.get("bridge", {}).get("bpb") if isinstance(metrics, dict) and isinstance(metrics.get("bridge"), dict) else None
    fresh_bpb = (
        metrics.get("fresh_process_full", {}).get("bpb")
        if isinstance(metrics, dict) and isinstance(metrics.get("fresh_process_full"), dict)
        else None
    )
    packed_bpb = (
        metrics.get("packed_artifact_full", {}).get("bpb")
        if isinstance(metrics, dict) and isinstance(metrics.get("packed_artifact_full"), dict)
        else None
    )
    provenance_rows: list[str] = []
    if isinstance(provenance, dict):
        for key in ("run_id", "family_id", "submission_pr", "source_repo", "source_root", "report_dir", "source_commit"):
            value = provenance.get(key)
            if value not in (None, "", [], {}):
                provenance_rows.append(f"- {key}: `{value}`")
    tier_lines = []
    for tier in ("tier1", "tier2", "tier3"):
        status = _audit_status(audits, tier) or "missing"
        tier_lines.append(f"- {tier}: `{status}`")
    tier3_details = _tier3_detail_lines(audits)
    attachment_lines = [f"- `{row['dest']}` <= `{row['source']}`" for row in attachments] or ["- none"]
    detector_lines: list[str] = []
    for row in detector_summaries:
        kind = row.get("kind", "detector")
        detector_lines.append(f"- `{row['dest']}` kind=`{kind}`")
        if kind == "legality":
            checks = ", ".join(f"{key}={value}" for key, value in row["checks"].items()) or "none"
            obligations = ", ".join(f"{key}={value}" for key, value in row["obligations"].items()) or "none"
            detector_lines.append(f"  profile: `{row['profile']}`")
            if row.get("trust_requested") is not None:
                detector_lines.append(
                    "  trust: "
                    f"requested=`{row.get('trust_requested')}`, "
                    f"achieved=`{row.get('trust_achieved')}`, "
                    f"satisfied=`{row.get('trust_satisfied')}`"
                )
            detector_lines.append(f"  checks: {checks}")
            detector_lines.append(f"  obligations: {obligations}")
        elif kind == "submission":
            checks = ", ".join(f"{key}={value}" for key, value in row["checks"].items()) or "none"
            detector_lines.append(f"  verdict: `{row['verdict']}`")
            detector_lines.append(f"  checks: {checks}")
        elif kind == "provenance":
            checks = ", ".join(f"{key}={value}" for key, value in row["checks"].items()) or "none"
            detector_lines.append(f"  verdict: `{row['verdict']}`")
            detector_lines.append(f"  selection: submitted_run_id=`{row.get('submitted_run_id')}`, selection_mode=`{row.get('selection_mode')}`")
            detector_lines.append(f"  checks: {checks}")
        elif kind == "replay":
            repeatability = row.get("repeatability", "unknown")
            mean_bpb = row.get("mean_bpb")
            detector_lines.append(f"  profile: `{row['profile']}`")
            detector_lines.append(f"  mean_bpb: `{mean_bpb}`")
            detector_lines.append(f"  repeatability: `{repeatability}`")
    metric_lines = []
    if bridge_bpb is not None:
        metric_lines.append(f"- bridge bpb: `{bridge_bpb}`")
    if fresh_bpb is not None:
        metric_lines.append(f"- fresh-process full bpb: `{fresh_bpb}`")
    if packed_bpb is not None:
        metric_lines.append(f"- packed-artifact full bpb: `{packed_bpb}`")
    if not metric_lines:
        if isinstance(metrics, dict) and metrics:
            for key, value in list(metrics.items())[:8]:
                if isinstance(value, dict):
                    fields = []
                    for subkey, subvalue in value.items():
                        if subvalue in (None, "", [], {}):
                            continue
                        fields.append(f"{subkey}={subvalue}")
                        if len(fields) == 3:
                            break
                    metric_lines.append(f"- {key}: " + (", ".join(fields) if fields else "object"))
                else:
                    metric_lines.append(f"- {key}: `{value}`")
        else:
            metric_lines.append("- no structured metric summary provided")
    lines = [
        "# Validity Bundle",
        "",
        f"- bundle id: `{bundle_id}`",
        f"- strongest justified claim: `Tier {claim_level['level']}: {claim_level['label']}`",
    ]
    for note in claim_level.get("notes", []):
        lines.append(f"- claim note: {note}")
    if requested_label:
        lines.append(f"- requested label: `{requested_label}`")
    lines.extend(
        [
            "",
            "## Audit Coverage",
            "",
            *tier_lines,
            *tier3_details,
            "",
            "## Metrics",
            "",
            *metric_lines,
            "",
            "## Provenance",
            "",
            *(provenance_rows or ["- no provenance summary provided"]),
            "",
            "## Attachments",
            "",
            *attachment_lines,
            "",
            "## Detector Summaries",
            "",
            *(detector_lines or ["- no detector JSON attachments summarized"]),
            "",
            "## Files",
            "",
            "- `claim.json`",
            "- `evidence/metrics.json`",
            "- `evidence/provenance.json`",
            "- `evidence/audits.json`",
            "- `bundle_manifest.json`",
            "- `report/README.md`",
        ]
    )
    return "\n".join(lines) + "\n"


def write_validity_bundle(manifest_path: Path, out_dir: Path) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError("Bundle manifest must be a JSON object")
    base_dir = manifest_path.parent.resolve()

    claim = _load_manifest_value(manifest.get("claim"), base_dir)
    metrics = _load_manifest_value(manifest.get("metrics"), base_dir)
    provenance = _load_manifest_value(manifest.get("provenance"), base_dir)
    audits = _load_manifest_value(manifest.get("audits"), base_dir)

    bundle_id = (
        manifest.get("bundle_id")
        or (claim.get("candidate_id") if isinstance(claim, dict) else None)
        or manifest_path.stem
    )
    claim_level = infer_claim_level(claim, metrics, audits)

    out_dir.mkdir(parents=True, exist_ok=True)
    attachments = [
        _copy_attachment(base_dir, out_dir, spec)
        for spec in manifest.get("attachments", [])
    ]
    detector_summaries = _collect_detector_attachment_summaries(out_dir, attachments)

    normalized_manifest = {
        "bundle_id": bundle_id,
        "claim": claim,
        "metrics": metrics,
        "provenance": provenance,
        "audits": audits,
        "attachments": attachments,
        "source_manifest": str(manifest_path.resolve()),
        "claim_level": claim_level,
    }

    (out_dir / "claim.json").write_text(dumps_json(claim) + "\n", encoding="utf-8")
    (out_dir / "evidence" / "metrics.json").parent.mkdir(parents=True, exist_ok=True)
    (out_dir / "evidence" / "metrics.json").write_text(dumps_json(metrics) + "\n", encoding="utf-8")
    (out_dir / "evidence" / "provenance.json").write_text(dumps_json(provenance) + "\n", encoding="utf-8")
    (out_dir / "evidence" / "audits.json").write_text(dumps_json(audits) + "\n", encoding="utf-8")
    (out_dir / "bundle_manifest.json").write_text(dumps_json(normalized_manifest) + "\n", encoding="utf-8")
    (out_dir / "report").mkdir(parents=True, exist_ok=True)
    (out_dir / "report" / "README.md").write_text(
        render_validity_bundle_readme(
            bundle_id=str(bundle_id),
            claim=claim,
            metrics=metrics,
            provenance=provenance,
            audits=audits,
            claim_level=claim_level,
            attachments=attachments,
            detector_summaries=detector_summaries,
        ),
        encoding="utf-8",
    )

    return {
        "bundle_id": str(bundle_id),
        "claim_level": claim_level,
        "attachment_count": len(attachments),
        "detector_attachment_count": len(detector_summaries),
        "legality_attachment_count": sum(1 for row in detector_summaries if row.get("kind") == "legality"),
        "out_dir": str(out_dir),
    }


def _collect_detector_attachment_summaries(
    out_dir: Path,
    attachments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in attachments:
        dest = spec.get("dest")
        if not isinstance(dest, str) or not dest.endswith(".json"):
            continue
        path = out_dir / dest
        if not path.is_file():
            continue
        try:
            data = load_json(path)
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        checks = data.get("checks")
        obligations = data.get("obligations")
        profile = data.get("profile")
        if isinstance(checks, dict) and isinstance(obligations, dict) and profile is not None:
            trust = data.get("trust", {}) if isinstance(data.get("trust"), dict) else {}
            rows.append(
                {
                    "kind": "legality",
                    "dest": dest,
                    "profile": str(profile),
                    "trust_requested": trust.get("requested"),
                    "trust_achieved": trust.get("achieved"),
                    "trust_satisfied": trust.get("satisfied"),
                    "checks": _flatten_legality_checks(checks),
                    "obligations": _flatten_legality_obligations(obligations),
                }
            )
            continue
        if isinstance(checks, dict) and "submission" in data and "verdict" in data:
            rows.append(
                {
                    "kind": "submission",
                    "dest": dest,
                    "verdict": str(data.get("verdict")),
                    "checks": _flatten_generic_checks(checks),
                }
            )
            continue
        if isinstance(checks, dict) and "provenance" in data and "verdict" in data:
            provenance = data.get("provenance", {})
            rows.append(
                {
                    "kind": "provenance",
                    "dest": dest,
                    "verdict": str(data.get("verdict")),
                    "submitted_run_id": provenance.get("submitted_run_id"),
                    "selection_mode": provenance.get("selection_mode"),
                    "checks": _flatten_generic_checks(checks),
                }
            )
            continue
        aggregate = data.get("aggregate")
        repeatability = data.get("repeatability")
        if profile is not None and isinstance(aggregate, dict) and isinstance(repeatability, dict):
            rows.append(
                {
                    "kind": "replay",
                    "dest": dest,
                    "profile": str(profile),
                    "mean_bpb": aggregate.get("mean_bpb"),
                    "repeatability": "pass" if repeatability.get("pass") is True else "fail" if repeatability.get("pass") is False else "unknown",
                }
            )
    return rows


def _flatten_legality_checks(checks: dict[str, Any]) -> dict[str, str]:
    rows: dict[str, str] = {}
    for key, value in checks.items():
        if isinstance(value, dict):
            covered = value.get("covered")
            passed = value.get("pass")
            if covered is False:
                rows[key] = "uncovered"
            elif passed is True:
                rows[key] = "pass"
            elif passed is False:
                rows[key] = "fail"
            else:
                rows[key] = "unknown"
        else:
            rows[key] = str(value)
    return rows


def _flatten_legality_obligations(obligations: dict[str, Any]) -> dict[str, str]:
    rows: dict[str, str] = {}
    for key, value in obligations.items():
        if isinstance(value, dict):
            rows[key] = str(value.get("status", "unknown"))
        else:
            rows[key] = str(value)
    return rows


def _flatten_generic_checks(checks: dict[str, Any]) -> dict[str, str]:
    rows: dict[str, str] = {}
    for key, value in checks.items():
        if isinstance(value, dict):
            passed = value.get("pass")
            if passed is True:
                rows[key] = "pass"
            elif passed is False:
                rows[key] = "fail"
            else:
                rows[key] = "unknown"
        else:
            rows[key] = str(value)
    return rows


def _tier3_claim_credit(audits: Any) -> dict[str, Any]:
    if not isinstance(audits, dict):
        return {"considered": False, "credited": False, "notes": []}
    tier3 = audits.get("tier3")
    if not isinstance(tier3, dict):
        return {"considered": False, "credited": False, "notes": []}
    status = str(tier3.get("status")) if tier3.get("status") is not None else None
    scope = tier3.get("scope")
    trust_achieved = tier3.get("trust_level_achieved")
    trust_satisfied = tier3.get("trust_satisfied")
    notes: list[str] = []
    credited = status == "pass"
    if status != "pass":
        return {
            "considered": True,
            "credited": False,
            "status": status,
            "scope": scope,
            "trust_achieved": trust_achieved,
            "trust_satisfied": trust_satisfied,
            "notes": notes,
        }
    if scope in LIMITED_TIER3_SCOPES:
        credited = False
        notes.append(f"Tier 3 was not promoted because its scope was limited: `{scope}`.")
    if trust_achieved is not None and str(trust_achieved) not in TIER3_PROMOTION_TRUST_LEVELS:
        credited = False
        notes.append(
            "Tier 3 was not promoted because the achieved legality trust level was "
            f"`{trust_achieved}`, below the promotion floor of `traced`."
        )
    return {
        "considered": True,
        "credited": credited,
        "status": status,
        "scope": scope,
        "trust_achieved": trust_achieved,
        "trust_satisfied": trust_satisfied,
        "notes": notes,
    }


def _tier3_detail_lines(audits: Any) -> list[str]:
    if not isinstance(audits, dict):
        return []
    tier3 = audits.get("tier3")
    if not isinstance(tier3, dict):
        return []
    lines: list[str] = []
    scope = tier3.get("scope")
    if scope not in (None, "", [], {}):
        lines.append(f"- tier3 scope: `{scope}`")
    trust_requested = tier3.get("trust_level_requested")
    trust_achieved = tier3.get("trust_level_achieved")
    trust_satisfied = tier3.get("trust_satisfied")
    if trust_requested not in (None, "", [], {}):
        lines.append(
            "- tier3 trust: "
            f"requested=`{trust_requested}`, "
            f"achieved=`{trust_achieved}`, "
            f"satisfied=`{trust_satisfied}`"
        )
    return lines


def infer_run_id_from_stem(stem: str) -> str:
    stem = FULL_EVAL_SUFFIX_RE.sub("", stem)
    stem = DATE_SUFFIX_RE.sub("", stem)
    return stem


def infer_family_id(run_id: str) -> str:
    run_id = re.sub(r"_seed\d+", "", run_id)
    run_id = re.sub(r"_save$", "", run_id)
    return run_id


def parse_bridge_record(path: Path, data: dict[str, Any]) -> dict[str, Any]:
    model = data.get("model", {}) if isinstance(data.get("model"), dict) else {}
    quant_rows = data.get("quantization", [])
    saved_state_path = model.get("saved_state_path")
    loaded_state_path = model.get("loaded_state_path")
    run_id = infer_run_id_from_stem(Path(saved_state_path).stem) if saved_state_path else infer_run_id_from_stem(path.stem)
    seed_match = SEED_RE.search(run_id)
    quant_by_bits: dict[str, float | None] = {}
    for row in quant_rows if isinstance(quant_rows, list) else []:
        bits = row.get("bits")
        key = f"int{int(bits)}" if isinstance(bits, (int, float)) else None
        if key:
            quant_by_bits[key] = finite_or_none(row.get("test_bpb"))
    return {
        "kind": "bridge",
        "path": str(path),
        "title": data.get("title"),
        "run_id": run_id,
        "family_id": infer_family_id(run_id),
        "seed": int(seed_match.group(1)) if seed_match else model.get("seed"),
        "bpb": finite_or_none(model.get("test_bpb")),
        "bits_per_token": finite_or_none(model.get("test_bits_per_token")),
        "loss": finite_or_none(model.get("test_eval_loss")),
        "train_time_sec": finite_or_none(model.get("train_time_sec")),
        "params": model.get("params"),
        "saved_state_path": saved_state_path,
        "loaded_state_path": loaded_state_path,
        "int4_bpb": quant_by_bits.get("int4"),
        "int6_bpb": quant_by_bits.get("int6"),
        "raw": {
            "preset": model.get("preset"),
            "variant": model.get("variant"),
            "scale": model.get("scale"),
            "learning_rate": model.get("learning_rate"),
        },
    }


def parse_full_eval_record(path: Path, data: dict[str, Any]) -> dict[str, Any]:
    state_npz = data.get("state_npz")
    run_id = infer_run_id_from_stem(Path(state_npz).stem) if isinstance(state_npz, str) else infer_run_id_from_stem(path.stem)
    seed_match = SEED_RE.search(run_id)
    quant_bits = int(data.get("quant_bits", 0) or 0)
    quant_label = "fp16" if quant_bits == 0 else f"int{quant_bits}"
    artifact_bytes = data.get("artifact_bytes_zlib")
    return {
        "kind": "full_eval",
        "path": str(path),
        "title": data.get("title"),
        "run_id": run_id,
        "family_id": infer_family_id(run_id),
        "seed": int(seed_match.group(1)) if seed_match else None,
        "quant_label": quant_label,
        "quant_bits": quant_bits,
        "bpb": finite_or_none(data.get("eval_bpb")),
        "bits_per_token": finite_or_none(data.get("eval_bits_per_token")),
        "loss": finite_or_none(data.get("eval_loss")),
        "eval_tokens": data.get("eval_tokens"),
        "artifact_bytes": int(artifact_bytes) if isinstance(artifact_bytes, (int, float)) and math.isfinite(float(artifact_bytes)) else None,
        "state_npz": state_npz,
        "summary_json": data.get("summary_json"),
    }


def parse_study_record(path: Path, data: dict[str, Any]) -> dict[str, Any]:
    variants = data.get("variants", [])
    models = data.get("models", [])
    best_label = None
    best_metric = None
    metric_name = None
    if isinstance(models, list):
        ranked: list[tuple[float, str]] = []
        for model in models:
            if not isinstance(model, dict):
                continue
            label = model.get("label")
            test_mean = finite_or_none(model.get("test_mean"))
            if label and test_mean is not None:
                ranked.append((test_mean, str(label)))
        if ranked:
            ranked.sort()
            best_metric, best_label = ranked[0]
            metric_name = "test_mean"
    return {
        "kind": "study",
        "path": str(path),
        "title": data.get("title"),
        "run_id": infer_run_id_from_stem(path.stem),
        "family_id": infer_family_id(infer_run_id_from_stem(path.stem)),
        "variant_count": len(variants) if isinstance(variants, list) else len(models) if isinstance(models, list) else 0,
        "best_label": best_label,
        "best_metric": best_metric,
        "metric_name": metric_name,
    }


def classify_record(path: Path, data: Any) -> dict[str, Any] | None:
    if not isinstance(data, dict):
        return None
    if "eval_bpb" in data:
        return parse_full_eval_record(path, data)
    model = data.get("model")
    if isinstance(model, dict) and "test_bpb" in model:
        return parse_bridge_record(path, data)
    if "variants" in data or "models" in data:
        return parse_study_record(path, data)
    return None


def scan_results(root: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    skipped: list[str] = []
    for path in sorted(root.glob("*.json")):
        try:
            data = load_json(path)
            record = classify_record(path, data)
        except Exception as exc:  # pragma: no cover - defensive scan path
            skipped.append(f"{path.name}: {exc}")
            continue
        if record is None:
            skipped.append(path.name)
            continue
        records.append(record)

    by_kind = Counter(record["kind"] for record in records)
    by_family = Counter(record["family_id"] for record in records)
    return {
        "root": str(root),
        "record_count": len(records),
        "by_kind": dict(by_kind),
        "family_count": len(by_family),
        "top_families": by_family.most_common(20),
        "records": records,
        "skipped": skipped,
    }


def sort_records(records: list[dict[str, Any]], metric: str, *, ascending: bool = True) -> list[dict[str, Any]]:
    def key_fn(record: dict[str, Any]) -> tuple[int, float]:
        value = record.get(metric)
        if value is None:
            return (1, float("inf"))
        try:
            return (0, float(value))
        except (TypeError, ValueError):
            return (1, float("inf"))

    return sorted(records, key=key_fn, reverse=not ascending)


def survival_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = defaultdict(lambda: {"bridge": None, "full": {}})
    for record in records:
        if record["kind"] == "bridge":
            grouped[record["run_id"]]["bridge"] = record
        elif record["kind"] == "full_eval":
            grouped[record["run_id"]]["full"][record.get("quant_label") or "unknown"] = record

    rows: list[dict[str, Any]] = []
    for run_id, group in sorted(grouped.items()):
        bridge = group["bridge"]
        full = group["full"]
        if bridge is None:
            continue
        bridge_bpb = bridge.get("bpb")
        bridge_int6 = bridge.get("int6_bpb")
        full_fp16 = full.get("fp16", {}).get("bpb") if "fp16" in full else None
        full_int6 = full.get("int6", {}).get("bpb") if "int6" in full else None
        status = "bridge_only"
        if full:
            if any(v.get("bpb") is None for v in full.values()):
                status = "full_eval_failed"
            else:
                status = "survived_full_eval"
        rows.append(
            {
                "run_id": run_id,
                "family_id": infer_family_id(run_id),
                "seed": bridge.get("seed"),
                "bridge_fp16": bridge_bpb,
                "bridge_int6": bridge_int6,
                "full_fp16": full_fp16,
                "full_int6": full_int6,
                "delta_fp16": None if bridge_bpb is None or full_fp16 is None else full_fp16 - bridge_bpb,
                "delta_int6": None if bridge_int6 is None or full_int6 is None else full_int6 - bridge_int6,
                "status": status,
                "bridge_path": bridge.get("path"),
                "full_paths": {k: v.get("path") for k, v in full.items()},
            }
        )
    return rows


def lineage_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        if record["kind"] != "bridge":
            continue
        loaded = record.get("loaded_state_path")
        saved = record.get("saved_state_path")
        if not loaded or not saved:
            continue
        parent_id = infer_run_id_from_stem(Path(loaded).stem)
        child_id = infer_run_id_from_stem(Path(saved).stem)
        rows.append(
            {
                "parent_run_id": parent_id,
                "child_run_id": child_id,
                "family_id": record["family_id"],
                "seed": record.get("seed"),
                "child_bpb": record.get("bpb"),
                "child_path": record.get("path"),
            }
        )
    return rows


def render_table(rows: list[dict[str, Any]], columns: list[str], top: int | None = None) -> str:
    if top is not None:
        rows = rows[:top]
    if not rows:
        return "(no rows)"
    widths = {col: max(len(col), *(len(str(row.get(col, ""))) for row in rows)) for col in columns}
    header = "  ".join(col.ljust(widths[col]) for col in columns)
    sep = "  ".join("-" * widths[col] for col in columns)
    body = [
        "  ".join(str(row.get(col, "")).ljust(widths[col]) for col in columns)
        for row in rows
    ]
    return "\n".join([header, sep, *body])


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


def _svg_escape(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


_PALETTE = [
    "#2f6fed", "#c23b22", "#2ca02c", "#9467bd", "#e377c2",
    "#8c564b", "#17becf", "#bcbd22", "#ff7f0e", "#7f7f7f",
    "#1f77b4", "#d62728", "#98df8a", "#aec7e8", "#ffbb78",
]


def _family_color(family_id: str) -> str:
    return _PALETTE[hash(family_id) % len(_PALETTE)]


def _truncate_label(label: str, max_chars: int = 32) -> str:
    if len(label) <= max_chars:
        return label
    return label[: max_chars - 1] + "\u2026"


def _nice_ticks(vmin: float, vmax: float, target_count: int = 5) -> list[float]:
    span = vmax - vmin
    if span <= 0:
        return [vmin]
    raw_step = span / max(target_count, 1)
    magnitude = 10 ** math.floor(math.log10(raw_step))
    for nice in [1, 2, 5, 10]:
        step = nice * magnitude
        if step >= raw_step:
            break
    if step <= 0:
        return [vmin, vmax]
    start = math.ceil(vmin / step) * step
    ticks: list[float] = []
    val = start
    while val <= vmax + step * 0.001:
        ticks.append(round(val, 10))
        val += step
    return ticks


def write_bar_svg(path: Path, title: str, labels: list[str], values: list[float], *, width: int = 960, height: int = 480) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not labels or not values:
        path.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="960" height="120"></svg>\n', encoding="utf-8")
        return
    margin_left = 260
    margin_right = 80
    margin_top = 50
    margin_bottom = 40
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    bar_gap = 6
    bar_height = max(8, (plot_height - bar_gap * (len(values) - 1)) // max(len(values), 1))
    vmax = max(max(values), 1e-12)
    ticks = _nice_ticks(0, vmax, 5)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<style>text{font-family:Menlo,Monaco,monospace;font-size:11px;fill:#333}'
        ' .title{font-size:16px;font-weight:700;fill:#111}'
        ' .axis{stroke:#888;stroke-width:1}'
        ' .grid{stroke:#ddd;stroke-width:1;stroke-dasharray:4,4}'
        ' .tick{font-size:10px;fill:#666}'
        ' .val{font-size:10px;fill:#333}</style>',
        f'<text class="title" x="{margin_left}" y="28">{_svg_escape(title)}</text>',
    ]
    # gridlines and tick labels
    for tick in ticks:
        x = margin_left + plot_width * (tick / vmax) if vmax > 0 else margin_left
        parts.append(f'<line class="grid" x1="{x:.1f}" y1="{margin_top}" x2="{x:.1f}" y2="{margin_top + plot_height}"/>')
        parts.append(f'<text class="tick" x="{x:.1f}" y="{margin_top + plot_height + 16}" text-anchor="middle">{tick:.4f}</text>')
    # bottom axis
    parts.append(f'<line class="axis" x1="{margin_left}" y1="{margin_top + plot_height}" x2="{width - margin_right}" y2="{margin_top + plot_height}"/>')
    # bars
    for idx, (label, value) in enumerate(zip(labels, values)):
        y = margin_top + idx * (bar_height + bar_gap)
        bar_w = plot_width * (value / vmax) if vmax > 0 else 0
        color = _family_color(label.split(":")[0])
        truncated = _svg_escape(_truncate_label(label, 36))
        parts.append(f'<text x="{margin_left - 8}" y="{y + bar_height - 2}" text-anchor="end">{truncated}</text>')
        parts.append(f'<rect fill="{color}" x="{margin_left}" y="{y}" width="{bar_w:.2f}" height="{bar_height}" rx="2"/>')
        parts.append(f'<text class="val" x="{margin_left + bar_w + 6:.2f}" y="{y + bar_height - 2}">{value:.4f}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_scatter_svg(
    path: Path,
    title: str,
    rows: list[dict[str, Any]],
    *,
    x_key: str,
    y_key: str,
    label_key: str,
    reference_line: bool = False,
    width: int = 960,
    height: int = 480,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    points = [(row.get(x_key), row.get(y_key), row.get(label_key), row.get("family_id", "")) for row in rows]
    points = [(float(x), float(y), str(label), str(fam)) for x, y, label, fam in points if x is not None and y is not None]
    if not points:
        path.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="960" height="120"></svg>\n', encoding="utf-8")
        return
    margin_left = 70
    margin_right = 40
    margin_top = 50
    margin_bottom = 50
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    if max_x == min_x:
        max_x += 1e-9
    if max_y == min_y:
        max_y += 1e-9
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<style>text{font-family:Menlo,Monaco,monospace;font-size:11px;fill:#333}'
        ' .title{font-size:16px;font-weight:700;fill:#111}'
        ' .axis{stroke:#888;stroke-width:1}'
        ' .grid{stroke:#ddd;stroke-width:1;stroke-dasharray:4,4}'
        ' .tick{font-size:10px;fill:#666}'
        ' .ref{stroke:#bbb;stroke-width:1;stroke-dasharray:6,3}</style>',
        f'<text class="title" x="{margin_left}" y="28">{_svg_escape(title)}</text>',
    ]
    # gridlines and ticks — x axis
    for tick in _nice_ticks(min_x, max_x, 5):
        px = margin_left + (tick - min_x) / (max_x - min_x) * plot_width
        parts.append(f'<line class="grid" x1="{px:.1f}" y1="{margin_top}" x2="{px:.1f}" y2="{margin_top + plot_height}"/>')
        parts.append(f'<text class="tick" x="{px:.1f}" y="{margin_top + plot_height + 16}" text-anchor="middle">{tick:.4f}</text>')
    # gridlines and ticks — y axis
    for tick in _nice_ticks(min_y, max_y, 5):
        py = margin_top + plot_height - (tick - min_y) / (max_y - min_y) * plot_height
        parts.append(f'<line class="grid" x1="{margin_left}" y1="{py:.1f}" x2="{width - margin_right}" y2="{py:.1f}"/>')
        parts.append(f'<text class="tick" x="{margin_left - 6}" y="{py + 4:.1f}" text-anchor="end">{tick:.4f}</text>')
    # axes
    parts.append(f'<line class="axis" x1="{margin_left}" y1="{margin_top + plot_height}" x2="{width - margin_right}" y2="{margin_top + plot_height}"/>')
    parts.append(f'<line class="axis" x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}"/>')
    # y=x reference line
    if reference_line:
        ref_min = max(min_x, min_y)
        ref_max = min(max_x, max_y)
        if ref_min < ref_max:
            rx1 = margin_left + (ref_min - min_x) / (max_x - min_x) * plot_width
            ry1 = margin_top + plot_height - (ref_min - min_y) / (max_y - min_y) * plot_height
            rx2 = margin_left + (ref_max - min_x) / (max_x - min_x) * plot_width
            ry2 = margin_top + plot_height - (ref_max - min_y) / (max_y - min_y) * plot_height
            parts.append(f'<line class="ref" x1="{rx1:.1f}" y1="{ry1:.1f}" x2="{rx2:.1f}" y2="{ry2:.1f}"/>')
    # points — collect y positions for collision offset
    rendered: list[float] = []
    for x, y, label, fam in points:
        px = margin_left + (x - min_x) / (max_x - min_x) * plot_width
        py = margin_top + plot_height - (y - min_y) / (max_y - min_y) * plot_height
        color = _family_color(fam)
        parts.append(f'<circle fill="{color}" opacity="0.8" cx="{px:.2f}" cy="{py:.2f}" r="5"/>')
        # offset label if it collides with a previous label
        label_y = py - 6
        for prev_y in rendered:
            if abs(label_y - prev_y) < 14:
                label_y = prev_y - 14
        rendered.append(label_y)
        truncated = _svg_escape(_truncate_label(label, 28))
        parts.append(f'<text x="{px + 7:.2f}" y="{label_y:.2f}">{truncated}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_pie_svg(
    path: Path,
    title: str,
    labels: list[str],
    values: list[float],
    colors: list[str],
    *,
    width: int = 480,
    height: int = 400,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not values or sum(values) == 0:
        path.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="480" height="120"></svg>\n', encoding="utf-8")
        return
    cx, cy = width // 2, height // 2 + 10
    r = min(cx, cy) - 60
    total = sum(values)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<style>text{font-family:Menlo,Monaco,monospace;font-size:12px;fill:#333}'
        ' .title{font-size:16px;font-weight:700;fill:#111}'
        ' .legend{font-size:11px}</style>',
        f'<text class="title" x="{cx}" y="24" text-anchor="middle">{_svg_escape(title)}</text>',
    ]
    angle = -math.pi / 2  # start at 12 o'clock
    for i, (label, value, color) in enumerate(zip(labels, values, colors)):
        frac = value / total
        sweep = frac * 2 * math.pi
        if len(values) == 1:
            # full circle
            parts.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{color}"/>')
        else:
            x1 = cx + r * math.cos(angle)
            y1 = cy + r * math.sin(angle)
            x2 = cx + r * math.cos(angle + sweep)
            y2 = cy + r * math.sin(angle + sweep)
            large = 1 if sweep > math.pi else 0
            parts.append(
                f'<path d="M{cx},{cy} L{x1:.2f},{y1:.2f} A{r},{r} 0 {large},1 {x2:.2f},{y2:.2f} Z" fill="{color}"/>'
            )
        # label at midpoint of arc
        mid_angle = angle + sweep / 2
        lx = cx + (r * 0.65) * math.cos(mid_angle)
        ly = cy + (r * 0.65) * math.sin(mid_angle)
        pct = f"{frac * 100:.0f}%"
        parts.append(f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="middle" fill="#fff" font-weight="700">{int(value)}</text>')
        # legend entry
        legend_y = height - 20 * (len(values) - i)
        parts.append(f'<rect x="10" y="{legend_y - 10}" width="12" height="12" fill="{color}"/>')
        parts.append(f'<text class="legend" x="28" y="{legend_y}">{_svg_escape(label)} ({pct})</text>')
        angle += sweep
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_histogram_svg(
    path: Path,
    title: str,
    values: list[float],
    *,
    bins: int = 10,
    width: int = 960,
    height: int = 400,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not values:
        path.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="960" height="120"></svg>\n', encoding="utf-8")
        return
    margin_left = 60
    margin_right = 40
    margin_top = 50
    margin_bottom = 50
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    vmin, vmax = min(values), max(values)
    if vmax == vmin:
        vmax = vmin + 1e-9
    bin_width = (vmax - vmin) / bins
    counts = [0] * bins
    for v in values:
        idx = min(int((v - vmin) / bin_width), bins - 1)
        counts[idx] += 1
    max_count = max(counts) if counts else 1
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<style>text{font-family:Menlo,Monaco,monospace;font-size:11px;fill:#333}'
        ' .title{font-size:16px;font-weight:700;fill:#111}'
        ' .axis{stroke:#888;stroke-width:1}'
        ' .grid{stroke:#ddd;stroke-width:1;stroke-dasharray:4,4}'
        ' .tick{font-size:10px;fill:#666}'
        ' .bar{fill:#2f6fed;opacity:0.85}</style>',
        f'<text class="title" x="{margin_left}" y="28">{_svg_escape(title)}</text>',
    ]
    # y-axis gridlines
    for tick in _nice_ticks(0, max_count, 4):
        py = margin_top + plot_height - (tick / max(max_count, 1)) * plot_height
        parts.append(f'<line class="grid" x1="{margin_left}" y1="{py:.1f}" x2="{width - margin_right}" y2="{py:.1f}"/>')
        parts.append(f'<text class="tick" x="{margin_left - 6}" y="{py + 4:.1f}" text-anchor="end">{int(tick)}</text>')
    # axes
    parts.append(f'<line class="axis" x1="{margin_left}" y1="{margin_top + plot_height}" x2="{width - margin_right}" y2="{margin_top + plot_height}"/>')
    parts.append(f'<line class="axis" x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}"/>')
    # bars
    rect_w = plot_width / bins - 2
    for i, count in enumerate(counts):
        x = margin_left + i * (plot_width / bins) + 1
        bar_h = (count / max(max_count, 1)) * plot_height
        y = margin_top + plot_height - bar_h
        parts.append(f'<rect class="bar" x="{x:.1f}" y="{y:.1f}" width="{rect_w:.1f}" height="{bar_h:.1f}" rx="1"/>')
    # x-axis tick labels
    for tick in _nice_ticks(vmin, vmax, 5):
        px = margin_left + (tick - vmin) / (vmax - vmin) * plot_width
        parts.append(f'<text class="tick" x="{px:.1f}" y="{margin_top + plot_height + 16}" text-anchor="middle">{tick:.4f}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_grouped_bar_svg(
    path: Path,
    title: str,
    rows: list[dict[str, Any]],
    *,
    key_a: str,
    key_b: str,
    label_key: str,
    width: int = 960,
    height: int = 480,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    filtered = [r for r in rows if r.get(key_a) is not None and r.get(key_b) is not None]
    if not filtered:
        path.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="960" height="120"></svg>\n', encoding="utf-8")
        return
    margin_left = 260
    margin_right = 80
    margin_top = 50
    margin_bottom = 40
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    group_gap = 10
    bar_gap = 2
    group_height = max(16, (plot_height - group_gap * (len(filtered) - 1)) // max(len(filtered), 1))
    sub_bar = (group_height - bar_gap) // 2
    all_vals = [r[key_a] for r in filtered] + [r[key_b] for r in filtered]
    vmax = max(max(all_vals), 1e-12) if all_vals else 1e-12
    ticks = _nice_ticks(0, vmax, 5)
    color_a = "#2f6fed"
    color_b = "#c23b22"
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<style>text{font-family:Menlo,Monaco,monospace;font-size:11px;fill:#333}'
        ' .title{font-size:16px;font-weight:700;fill:#111}'
        ' .axis{stroke:#888;stroke-width:1}'
        ' .grid{stroke:#ddd;stroke-width:1;stroke-dasharray:4,4}'
        ' .tick{font-size:10px;fill:#666}'
        ' .val{font-size:10px;fill:#333}'
        ' .legend{font-size:11px}</style>',
        f'<text class="title" x="{margin_left}" y="28">{_svg_escape(title)}</text>',
        # legend
        f'<rect x="{width - 200}" y="10" width="12" height="12" fill="{color_a}"/>',
        f'<text class="legend" x="{width - 184}" y="21">{_svg_escape(key_a)}</text>',
        f'<rect x="{width - 200}" y="28" width="12" height="12" fill="{color_b}"/>',
        f'<text class="legend" x="{width - 184}" y="39">{_svg_escape(key_b)}</text>',
    ]
    # gridlines
    for tick in ticks:
        x = margin_left + plot_width * (tick / vmax) if vmax > 0 else margin_left
        parts.append(f'<line class="grid" x1="{x:.1f}" y1="{margin_top}" x2="{x:.1f}" y2="{margin_top + plot_height}"/>')
        parts.append(f'<text class="tick" x="{x:.1f}" y="{margin_top + plot_height + 16}" text-anchor="middle">{tick:.4f}</text>')
    parts.append(f'<line class="axis" x1="{margin_left}" y1="{margin_top + plot_height}" x2="{width - margin_right}" y2="{margin_top + plot_height}"/>')
    for idx, row in enumerate(filtered):
        y = margin_top + idx * (group_height + group_gap)
        label = _svg_escape(_truncate_label(str(row.get(label_key, "")), 36))
        va, vb = float(row[key_a]), float(row[key_b])
        wa = plot_width * (va / vmax) if vmax > 0 else 0
        wb = plot_width * (vb / vmax) if vmax > 0 else 0
        parts.append(f'<text x="{margin_left - 8}" y="{y + group_height // 2 + 4}" text-anchor="end">{label}</text>')
        parts.append(f'<rect fill="{color_a}" x="{margin_left}" y="{y}" width="{wa:.2f}" height="{sub_bar}" rx="2"/>')
        parts.append(f'<text class="val" x="{margin_left + wa + 4:.2f}" y="{y + sub_bar - 2}">{va:.4f}</text>')
        parts.append(f'<rect fill="{color_b}" x="{margin_left}" y="{y + sub_bar + bar_gap}" width="{wb:.2f}" height="{sub_bar}" rx="2"/>')
        parts.append(f'<text class="val" x="{margin_left + wb + 4:.2f}" y="{y + group_height - 2}">{vb:.4f}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def _mermaid_id(run_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]", "_", run_id)


def render_lineage_mermaid(rows: list[dict[str, Any]], *, max_nodes: int = 30) -> str:
    if not rows:
        return "graph TD\n    empty[No lineage data]"
    # build adjacency and find longest chains
    children: dict[str, list[str]] = defaultdict(list)
    bpb_map: dict[str, float | None] = {}
    for row in rows:
        p, c = row["parent_run_id"], row["child_run_id"]
        children[p].append(c)
        bpb_map[c] = row.get("child_bpb")
    # collect all unique nodes up to max_nodes
    seen: set[str] = set()
    edges: list[tuple[str, str]] = []
    for row in rows:
        p, c = row["parent_run_id"], row["child_run_id"]
        new_nodes = {p, c} - seen
        if len(seen) + len(new_nodes) > max_nodes:
            break
        seen.update(new_nodes)
        edges.append((p, c))
    lines = ["graph TD"]
    for node in sorted(seen):
        mid = _mermaid_id(node)
        short = _truncate_label(node, 24)
        bpb = bpb_map.get(node)
        if bpb is not None:
            lines.append(f'    {mid}["{short}<br/>{bpb:.4f}"]')
        else:
            lines.append(f'    {mid}["{short}"]')
    for p, c in edges:
        lines.append(f"    {_mermaid_id(p)} --> {_mermaid_id(c)}")
    return "\n".join(lines)


def render_survival_mermaid(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "graph LR\n    empty[No survival data]"
    total = len(rows)
    survived = sum(1 for r in rows if r.get("status") == "survived_full_eval")
    failed = sum(1 for r in rows if r.get("status") == "full_eval_failed")
    bridge_only = sum(1 for r in rows if r.get("status") == "bridge_only")
    attempted = survived + failed
    lines = [
        "graph LR",
        f'    A["Bridge Runs<br/>{total}"]',
        f'    B["Full Eval Attempted<br/>{attempted}"]',
        f'    C["Survived<br/>{survived}"]',
        f'    D["Failed<br/>{failed}"]',
        f'    E["Bridge Only<br/>{bridge_only}"]',
        "    A --> B",
        "    A --> E",
        "    B --> C",
        "    B --> D",
        "    style C fill:#2ca02c,color:#fff",
        "    style D fill:#c23b22,color:#fff",
        "    style E fill:#7f7f7f,color:#fff",
    ]
    return "\n".join(lines)


def write_report_bundle(root: Path, out_dir: Path, *, top: int = 20) -> dict[str, Any]:
    scanned = scan_results(root)
    records = scanned["records"]
    top_full_eval = sort_records([r for r in records if r["kind"] == "full_eval"], "bpb")[:top]
    top_bridge = sort_records([r for r in records if r["kind"] == "bridge"], "bpb")[:top]
    top_study = sort_records([r for r in records if r["kind"] == "study"], "best_metric")[:top]
    survival = survival_rows(records)
    survival_non_bridge = [row for row in survival if row["status"] != "bridge_only"]
    failed = [row for row in survival if row["status"] == "full_eval_failed"]
    lineage = lineage_rows(records)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "scan_summary.json").write_text(
        dumps_json(
            {
                "root": scanned["root"],
                "record_count": scanned["record_count"],
                "by_kind": scanned["by_kind"],
                "family_count": scanned["family_count"],
                "top_families": scanned["top_families"],
                "skipped": scanned["skipped"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (out_dir / "top_full_eval.json").write_text(dumps_json(top_full_eval) + "\n", encoding="utf-8")
    (out_dir / "top_bridge.json").write_text(dumps_json(top_bridge) + "\n", encoding="utf-8")
    (out_dir / "top_study.json").write_text(dumps_json(top_study) + "\n", encoding="utf-8")
    (out_dir / "survival.json").write_text(dumps_json(survival_non_bridge) + "\n", encoding="utf-8")
    (out_dir / "failed_full_eval.json").write_text(dumps_json(failed) + "\n", encoding="utf-8")
    (out_dir / "lineage.json").write_text(dumps_json(lineage) + "\n", encoding="utf-8")

    write_csv(out_dir / "top_full_eval.csv", top_full_eval, ["family_id", "run_id", "seed", "quant_label", "bpb", "artifact_bytes", "path"])
    write_csv(out_dir / "top_study.csv", top_study, ["family_id", "run_id", "best_label", "best_metric", "metric_name", "variant_count", "path"])
    write_csv(out_dir / "survival.csv", survival_non_bridge, ["family_id", "run_id", "seed", "bridge_fp16", "full_fp16", "bridge_int6", "full_int6", "delta_fp16", "delta_int6", "status"])
    write_csv(out_dir / "failed_full_eval.csv", failed, ["family_id", "run_id", "seed", "bridge_fp16", "bridge_int6", "status", "bridge_path"])

    # --- existing SVG charts (improved) ---
    full_eval_with_bpb = [row for row in top_full_eval[:12] if row.get("bpb") is not None]
    full_labels = [f"{row['family_id']}:{row.get('quant_label')}" for row in full_eval_with_bpb]
    full_values = [row["bpb"] for row in full_eval_with_bpb]
    write_bar_svg(out_dir / "top_full_eval.svg", "Top Full-Eval Rows", full_labels, full_values)
    study_rows = [row for row in top_study if row.get("best_metric") is not None][:12]
    write_bar_svg(
        out_dir / "top_study.svg",
        "Top Study Rows",
        [f"{row['family_id']}:{row.get('best_label') or 'study'}" for row in study_rows],
        [float(row["best_metric"]) for row in study_rows],
    )

    gap_rows = [row for row in survival_non_bridge if row.get("bridge_fp16") is not None and row.get("full_fp16") is not None][:20]
    write_scatter_svg(
        out_dir / "bridge_vs_full_fp16.svg",
        "Bridge FP16 vs Full FP16",
        gap_rows,
        x_key="bridge_fp16",
        y_key="full_fp16",
        label_key="family_id",
        reference_line=True,
    )

    conker7_rows = [row for row in survival if str(row["family_id"]).startswith("conker7_")]
    conker7_with_bpb = [row for row in conker7_rows if row.get("bridge_fp16") is not None]
    write_bar_svg(
        out_dir / "conker7_bridge_fp16.svg",
        "Conker-7 Bridge FP16 Rows",
        [row["family_id"] for row in conker7_with_bpb],
        [row["bridge_fp16"] for row in conker7_with_bpb],
    )

    # --- new SVG charts ---
    # survival status pie
    survived_count = sum(1 for r in survival if r["status"] == "survived_full_eval")
    failed_count = len(failed)
    bridge_only_count = sum(1 for r in survival if r["status"] == "bridge_only")
    pie_labels = []
    pie_values: list[float] = []
    pie_colors = []
    if survived_count:
        pie_labels.append("survived_full_eval")
        pie_values.append(survived_count)
        pie_colors.append("#2ca02c")
    if failed_count:
        pie_labels.append("full_eval_failed")
        pie_values.append(failed_count)
        pie_colors.append("#c23b22")
    if bridge_only_count:
        pie_labels.append("bridge_only")
        pie_values.append(bridge_only_count)
        pie_colors.append("#7f7f7f")
    write_pie_svg(out_dir / "survival_status.svg", "Survival Status", pie_labels, pie_values, pie_colors)

    # delta histogram
    deltas = [row["delta_fp16"] for row in survival_non_bridge if row.get("delta_fp16") is not None]
    write_histogram_svg(out_dir / "delta_fp16_histogram.svg", "Bridge-to-Full Delta (FP16)", deltas)

    # grouped bar: bridge vs full by family
    family_best: dict[str, dict[str, Any]] = {}
    for row in survival_non_bridge:
        fid = row["family_id"]
        if row.get("bridge_fp16") is not None and row.get("full_fp16") is not None:
            if fid not in family_best or (row["full_fp16"] < family_best[fid]["full_fp16"]):
                family_best[fid] = row
    grouped_rows = sort_records(list(family_best.values()), "full_fp16")[:12]
    write_grouped_bar_svg(
        out_dir / "bridge_vs_full_grouped.svg",
        "Bridge vs Full-Eval by Family",
        grouped_rows,
        key_a="bridge_fp16",
        key_b="full_fp16",
        label_key="family_id",
    )

    # --- mermaid diagrams ---
    lineage_mermaid = render_lineage_mermaid(lineage)
    survival_mermaid = render_survival_mermaid(survival)

    # --- README ---
    summary_lines = [
        "# Public Backlog Report",
        "",
        f"- root: `{root}`",
        f"- normalized records: `{scanned['record_count']}`",
        f"- bridge rows: `{scanned['by_kind'].get('bridge', 0)}`",
        f"- full eval rows: `{scanned['by_kind'].get('full_eval', 0)}`",
        f"- study rows: `{scanned['by_kind'].get('study', 0)}`",
        f"- experiment families: `{scanned['family_count']}`",
        "",
        "## Headline",
        "",
    ]
    if top_full_eval and top_full_eval[0].get("bpb") is not None:
        best = top_full_eval[0]
        summary_lines.append(
            f"- best normalized full eval in this backlog: `{best['family_id']}` `{best['quant_label']}` at `{best['bpb']:.6f} bpb`"
        )
    elif top_study and top_study[0].get("best_metric") is not None:
        best = top_study[0]
        summary_lines.append(
            f"- best study quick-check in this backlog: `{best['family_id']}` `{best.get('best_label')}` at `{best['best_metric']:.6f}` `{best.get('metric_name') or 'metric'}`"
        )
    if failed:
        summary_lines.append(f"- full-eval failures detected after optimistic bridge results: `{len(failed)}`")
    summary_lines.extend(
        [
            "",
            "## Survival Pipeline",
            "",
            "```mermaid",
            survival_mermaid,
            "```",
            "",
            "## Lineage",
            "",
            "```mermaid",
            lineage_mermaid,
            "```",
            "",
            "## Files",
            "",
            "- `scan_summary.json`",
            "- `top_full_eval.json` / `top_full_eval.csv` / `top_full_eval.svg`",
            "- `top_bridge.json`",
            "- `top_study.json` / `top_study.csv` / `top_study.svg`",
            "- `survival.json` / `survival.csv` / `survival_status.svg`",
            "- `failed_full_eval.json` / `failed_full_eval.csv`",
            "- `lineage.json`",
            "- `bridge_vs_full_fp16.svg` / `bridge_vs_full_grouped.svg`",
            "- `delta_fp16_histogram.svg`",
            "- `conker7_bridge_fp16.svg`",
            "",
            "## Visuals",
            "",
            "### Top Study Rows",
            "",
            "![Top study rows](./top_study.svg)",
            "",
            "### Survival Status",
            "",
            "![Survival status](./survival_status.svg)",
            "",
            "### Top Full-Eval Rows",
            "",
            "![Top full eval rows](./top_full_eval.svg)",
            "",
            "### Bridge vs Full-Eval FP16",
            "",
            "![Bridge vs full fp16](./bridge_vs_full_fp16.svg)",
            "",
            "### Bridge vs Full-Eval by Family",
            "",
            "![Bridge vs full grouped](./bridge_vs_full_grouped.svg)",
            "",
            "### Delta Distribution (FP16)",
            "",
            "![Delta histogram](./delta_fp16_histogram.svg)",
            "",
            "### Conker-7 Bridge Rows",
            "",
            "![Conker-7 bridge rows](./conker7_bridge_fp16.svg)",
        ]
    )
    if failed:
        summary_lines.extend(["", "## Failed Full-Eval Rows", ""])
        for row in failed[:20]:
            summary_lines.append(
                f"- `{row['family_id']}` seed `{row.get('seed')}` bridge fp16 `{row.get('bridge_fp16')}` bridge int6 `{row.get('bridge_int6')}`"
            )
    (out_dir / "README.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    return {
        "scan_summary": {
            "record_count": scanned["record_count"],
            "by_kind": scanned["by_kind"],
            "family_count": scanned["family_count"],
        },
        "best_full_eval": top_full_eval[0] if top_full_eval else None,
        "failed_full_eval_count": len(failed),
        "report_dir": str(out_dir),
    }
