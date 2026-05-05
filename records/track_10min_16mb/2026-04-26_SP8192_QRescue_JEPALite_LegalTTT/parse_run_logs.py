#!/usr/bin/env python3
import argparse
import json
import re
import sys
from pathlib import Path


ARTIFACT_LIMIT = 16_000_000


def _last_int(pattern, text, flags=re.I):
    vals = re.findall(pattern, text, flags)
    return int(vals[-1].replace(",", "")) if vals else None


def _last_float(pattern, text, flags=re.I):
    vals = re.findall(pattern, text, flags)
    return float(vals[-1]) if vals else None


def parse_log_text(text):
    summary = {
        "seed": None,
        "train_seconds": None,
        "eval_seconds": None,
        "artifact_bytes": None,
        "code_bytes": None,
        "model_bytes": None,
        "prequant_bpb": None,
        "quantized_bpb": None,
        "sliding_bpb": None,
        "ttt_bpb": None,
        "val_bpb": None,
        "ttt_enabled": False,
        "ttt_epochs": None,
        "score_first_ttt_logged": False,
        "artifact_under_16mb": False,
        "train_under_600s": False,
        "eval_under_600s": False,
    }

    summary["seed"] = _last_int(r"\bseed:\s*(\d+)", text)
    summary["artifact_bytes"] = _last_int(r"\bartifact_bytes:\s*([0-9,]+)", text)
    if summary["artifact_bytes"] is None:
        summary["artifact_bytes"] = _last_int(r"Total submission size[^:\n]*:\s*([0-9,]+)\s*bytes", text)
    summary["code_bytes"] = _last_int(r"\bcode_bytes:\s*([0-9,]+)", text)
    if summary["code_bytes"] is None:
        summary["code_bytes"] = _last_int(r"Code size:\s*([0-9,]+)\s*bytes", text)
    summary["model_bytes"] = _last_int(r"\bmodel_bytes:\s*([0-9,]+)", text)
    if summary["model_bytes"] is None:
        summary["model_bytes"] = _last_int(r"Serialized model quantized[^:\n]*:\s*([0-9,]+)\s*bytes", text)

    train_seconds = _last_float(r"\btrain_seconds:\s*([0-9.]+)", text)
    if train_seconds is None:
        train_minutes = re.findall(r"\btrain_time:\s*([0-9.]+)m(?!s)", text)
        train_ms = re.findall(r"\btrain_time:\s*([0-9.]+)ms", text)
        candidates = [float(x) * 60.0 for x in train_minutes]
        candidates += [float(x) / 1000.0 for x in train_ms]
        if candidates:
            train_seconds = max(candidates)
    summary["train_seconds"] = train_seconds

    eval_ms = []
    for label, bpb, ms in re.findall(r"([A-Za-z0-9_\- ]+?)\s+val_loss:\s*[0-9.eE+-]+\s+val_bpb:\s*([0-9.eE+-]+)\s+eval_time:\s*([0-9.]+)ms", text):
        key = label.strip().lower()
        bpb_value = float(bpb)
        if key.startswith("pre-quantization"):
            summary["prequant_bpb"] = bpb_value
            continue
        eval_ms.append(float(ms))
        if key == "quantized":
            summary["quantized_bpb"] = bpb_value
        elif "sliding" in key:
            summary["sliding_bpb"] = bpb_value
        elif "ttt" in key:
            summary["ttt_bpb"] = bpb_value
    if eval_ms:
        summary["eval_seconds"] = sum(eval_ms) / 1000.0

    if summary["ttt_bpb"] is not None:
        summary["val_bpb"] = summary["ttt_bpb"]
    elif summary["sliding_bpb"] is not None:
        summary["val_bpb"] = summary["sliding_bpb"]
    elif summary["quantized_bpb"] is not None:
        summary["val_bpb"] = summary["quantized_bpb"]
    else:
        summary["val_bpb"] = _last_float(r"\bval_bpb:\s*([0-9.]+)", text)

    ttt_enabled_match = re.findall(r"\bttt_enabled:\s*(True|False|0|1)", text, re.I)
    if ttt_enabled_match:
        summary["ttt_enabled"] = ttt_enabled_match[-1].lower() in ("true", "1")
    score_first_protocols = (
        "TTT protocol: score_first",
        "TTT protocol: chunkwise_score_first_full_sgd",
        "TTT protocol: document_batched_phased_score_first_lora",
        "TTT protocol: chunkwise_score_first_lora",
    )
    if any(p in text for p in score_first_protocols):
        summary["ttt_enabled"] = True
    summary["ttt_epochs"] = _last_int(r"TTT epochs:\s*(\d+)", text)
    if summary["ttt_epochs"] is None:
        summary["ttt_epochs"] = _last_int(r"\bttt_epochs:\s*(\d+)", text)
    summary["score_first_ttt_logged"] = (
        any(p in text for p in score_first_protocols)
        and "TTT score_before_update: true" in text
        and "TTT no_rescore: true" in text
    )

    if summary["artifact_bytes"] is not None:
        summary["artifact_under_16mb"] = summary["artifact_bytes"] <= ARTIFACT_LIMIT
    if summary["train_seconds"] is not None:
        summary["train_under_600s"] = summary["train_seconds"] <= 600.5
    if summary["eval_seconds"] is not None:
        summary["eval_under_600s"] = summary["eval_seconds"] <= 600.5
    return summary


def validate_required(summary, smoke=False):
    required = ["artifact_bytes"]
    if not smoke:
        required += ["train_seconds", "eval_seconds", "val_bpb"]
    missing = [k for k in required if summary.get(k) is None]
    if missing:
        return [f"missing required parsed value: {k}" for k in missing]
    if not smoke and summary.get("ttt_enabled"):
        if not summary.get("score_first_ttt_logged"):
            return ["TTT enabled but score-first protocol lines were not found"]
        if summary.get("ttt_epochs") is None:
            return ["TTT enabled but TTT epochs were not parsed"]
    return []


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--json", dest="json_path")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)

    text = Path(args.log).read_text(encoding="utf-8", errors="replace")
    summary = parse_log_text(text)
    errors = validate_required(summary, smoke=args.smoke)

    out = json.dumps(summary, indent=2, sort_keys=True)
    print(out)
    if args.json_path:
        Path(args.json_path).write_text(out + "\n", encoding="utf-8")
    if errors:
        for err in errors:
            print(f"ERROR: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
