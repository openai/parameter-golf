from __future__ import annotations

from typing import Mapping


EXACT_SUFFIX = "_exact"
PRIMARY_SUBMISSION_LABELS = (
    "legal_ttt_exact",
    "final_int6_sliding_window_exact",
)
PRIMARY_SUBMISSION_LABELS_BY_TRACK = {
    "score_first_ttt": (
        "legal_ttt_exact",
        "final_ttt_exact",
        "final_sliding_window_exact",
        "final_int6_sliding_window_exact",
        "final_roundtrip_exact",
        "final_int8_zlib_roundtrip_exact",
    ),
    "prequant_ttt": (
        "prequant_ttt_exact",
        "legal_ttt_exact",
        "final_ttt_exact",
        "final_sliding_window_exact",
        "final_int6_sliding_window_exact",
        "final_roundtrip_exact",
    ),
    "fixed_predictor": (
        "final_sliding_window_exact",
        "final_int6_sliding_window_exact",
        "final_roundtrip_exact",
        "final_int8_zlib_roundtrip_exact",
    ),
}


def _base_metric_label(label: str) -> str:
    return label[: -len(EXACT_SUFFIX)] if label.endswith(EXACT_SUFFIX) else label


def _exact_metric_label(label: str) -> str:
    return label if label.endswith(EXACT_SUFFIX) else f"{label}{EXACT_SUFFIX}"


def _merge_metric_payloads(
    primary: dict[str, object] | None,
    fallback: dict[str, object] | None,
) -> dict[str, object] | None:
    if primary is None:
        return fallback
    if fallback is None:
        return primary
    merged = dict(fallback)
    merged.update(primary)
    return merged


def metric_payload_by_label(metrics: Mapping[str, object] | None, label: str) -> dict[str, object] | None:
    if not isinstance(metrics, Mapping):
        return None
    base_label = _base_metric_label(label)
    exact_candidate: dict[str, object] | None = None
    named_exact = metrics.get("named_evals_exact")
    if isinstance(named_exact, Mapping):
        for key in (label, base_label):
            candidate = named_exact.get(key)
            if isinstance(candidate, dict) and candidate.get("val_bpb") is not None:
                exact_candidate = candidate
                break
    named_candidate: dict[str, object] | None = None
    named = metrics.get("named_evals")
    if isinstance(named, Mapping):
        for key in (label, base_label):
            candidate = named.get(key)
            if isinstance(candidate, dict) and candidate.get("val_bpb") is not None:
                named_candidate = candidate
                break
    if exact_candidate is not None:
        return _merge_metric_payloads(exact_candidate, named_candidate)
    if named_candidate is not None:
        return named_candidate
    candidate = metrics.get(label)
    if isinstance(candidate, dict) and candidate.get("val_bpb") is not None:
        return candidate
    if base_label != label:
        candidate = metrics.get(base_label)
        if isinstance(candidate, dict) and candidate.get("val_bpb") is not None:
            return candidate
    return None


def _best_eval_from_mapping(
    evals: Mapping[str, object] | None,
    *,
    exact_labels: bool,
) -> tuple[str | None, dict[str, object] | None]:
    if not isinstance(evals, Mapping):
        return None, None
    best_label: str | None = None
    best_payload: dict[str, object] | None = None
    best_bpb: float | None = None
    for label, payload in evals.items():
        if not isinstance(payload, dict) or payload.get("val_bpb") is None:
            continue
        val_bpb = float(payload["val_bpb"])
        if best_bpb is None or val_bpb < best_bpb:
            best_label = _exact_metric_label(str(label)) if exact_labels else str(label)
            best_payload = payload
            best_bpb = val_bpb
    return best_label, best_payload


def _primary_labels_for_track(track: str | None) -> tuple[str, ...]:
    if track is None:
        return PRIMARY_SUBMISSION_LABELS
    return PRIMARY_SUBMISSION_LABELS_BY_TRACK.get(track, PRIMARY_SUBMISSION_LABELS)


def canonical_submission_eval(
    metrics: Mapping[str, object] | None,
    *,
    track: str | None = None,
) -> tuple[str | None, dict[str, object] | None]:
    if not isinstance(metrics, Mapping):
        return None, None
    primary_labels = _primary_labels_for_track(track)

    named_exact = metrics.get("named_evals_exact")
    if isinstance(named_exact, Mapping):
        for label in primary_labels:
            candidate = metric_payload_by_label(metrics, label)
            if candidate is not None:
                return label, candidate
        best_label, best_payload = _best_eval_from_mapping(named_exact, exact_labels=True)
        if best_payload is not None:
            return best_label, best_payload

    named = metrics.get("named_evals")
    if isinstance(named, Mapping):
        for label in primary_labels:
            candidate = metric_payload_by_label(metrics, label)
            if candidate is not None:
                return label, candidate
        best_label, best_payload = _best_eval_from_mapping(named, exact_labels=False)
        if best_payload is not None:
            return best_label, best_payload

    for legacy_label in ("final_roundtrip_exact", "final_roundtrip", "last_val"):
        candidate = metric_payload_by_label(metrics, legacy_label)
        if candidate is not None:
            return legacy_label, candidate
    return None, None


def canonical_submission_fields(
    metrics: Mapping[str, object] | None,
    *,
    track: str | None = None,
) -> dict[str, object]:
    return canonical_submission_fields_for_status(metrics, status="completed", track=track)


def canonical_submission_fields_for_status(
    metrics: Mapping[str, object] | None,
    *,
    status: str | None,
    track: str | None = None,
) -> dict[str, object]:
    if status != "completed":
        return {
            "final_submission_metric_label": None,
            "official_submission_metric_label": None,
            "final_submission_loss": None,
            "final_submission_bpb": None,
        }
    label, payload = canonical_submission_eval(metrics, track=track)
    if payload is None:
        return {
            "final_submission_metric_label": None,
            "official_submission_metric_label": None,
            "final_submission_loss": None,
            "final_submission_bpb": None,
        }
    return {
        "final_submission_metric_label": label,
        "official_submission_metric_label": label,
        "final_submission_loss": payload.get("val_loss"),
        "final_submission_bpb": payload.get("val_bpb"),
    }
