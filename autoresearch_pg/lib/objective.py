from __future__ import annotations

from pathlib import Path
from typing import Any

from autoresearch_pg.lib.train_log import parse_train_log


BYTE_CAP = 16_000_000
INVALID_SCORE = 1_000_000.0


def evaluate_metrics(parsed: dict[str, Any], byte_cap: int = BYTE_CAP) -> dict[str, Any]:
    artifact = parsed.get("artifact", {})
    final_exact = parsed.get("final_int8_exact", {})
    final_rounded = parsed.get("final_int8", {})
    last_train_val = parsed.get("last_train_val", {})

    post_quant_val_loss = final_exact.get("val_loss", final_rounded.get("val_loss"))
    post_quant_val_bpb = final_exact.get("val_bpb", final_rounded.get("val_bpb"))
    pre_quant_val_loss = last_train_val.get("val_loss")
    pre_quant_val_bpb = last_train_val.get("val_bpb")
    bytes_total = artifact.get("bytes_total")

    quant_gap_loss = None
    if post_quant_val_loss is not None and pre_quant_val_loss is not None:
        quant_gap_loss = post_quant_val_loss - pre_quant_val_loss

    quant_gap_bpb = None
    if post_quant_val_bpb is not None and pre_quant_val_bpb is not None:
        quant_gap_bpb = post_quant_val_bpb - pre_quant_val_bpb

    valid_metrics = post_quant_val_bpb is not None
    valid_bytes = bytes_total is not None and bytes_total <= byte_cap
    bytes_over = None if bytes_total is None else max(bytes_total - byte_cap, 0)
    valid = valid_metrics and valid_bytes

    reasons: list[str] = []
    if not valid_metrics:
        reasons.append("missing_post_quant_metric")
    if bytes_total is None:
        reasons.append("missing_total_bytes")
    elif bytes_total > byte_cap:
        reasons.append("artifact_over_cap")

    if post_quant_val_bpb is None:
        proxy_score = INVALID_SCORE
    else:
        proxy_score = float(post_quant_val_bpb)
        if bytes_over:
            proxy_score += 10.0 * (bytes_over / byte_cap)
        if not valid:
            proxy_score += 1.0

    return {
        "byte_cap": byte_cap,
        "valid": valid,
        "reasons": reasons,
        "post_quant_val_loss": post_quant_val_loss,
        "post_quant_val_bpb": post_quant_val_bpb,
        "pre_quant_val_loss": pre_quant_val_loss,
        "pre_quant_val_bpb": pre_quant_val_bpb,
        "quant_gap_loss": quant_gap_loss,
        "quant_gap_bpb": quant_gap_bpb,
        "bytes_total": bytes_total,
        "bytes_model_int8_zlib": artifact.get("model_int8_zlib_bytes"),
        "bytes_code": artifact.get("code_bytes"),
        "bytes_over": bytes_over,
        "model_params": parsed.get("model_params"),
        "step_avg_ms": parsed.get("timing", {}).get("step_avg_ms"),
        "train_time_ms": parsed.get("timing", {}).get("train_time_ms"),
        "proxy_score": proxy_score,
    }


def evaluate_train_log(path: Path, byte_cap: int = BYTE_CAP) -> dict[str, Any]:
    parsed = parse_train_log(path)
    return evaluate_metrics(parsed, byte_cap=byte_cap)
