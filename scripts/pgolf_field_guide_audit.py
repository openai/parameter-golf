#!/usr/bin/env python3
"""Static Field Guide audit helper for Parameter Golf submissions.

This is intentionally conservative: it does not prove legality by itself.
It decodes packed train_gpt.py wrappers, scans for common Issue #1017 risks,
and emits a small JSON report that can be included next to reproduction logs.
"""

from __future__ import annotations

import argparse
import base64
import json
import lzma
import re
from pathlib import Path


def load_source(path: Path) -> tuple[str, dict[str, object]]:
    raw = path.read_text(encoding="utf-8")
    meta: dict[str, object] = {"path": str(path), "packed_wrapper": False}
    marker = 'B.b85decode("'
    if marker not in raw:
        return raw, meta

    try:
        start = raw.index(marker) + len(marker)
        end = raw.index('"),format=L.FORMAT_RAW', start)
        payload = raw[start:end]
        decoded = lzma.decompress(
            base64.b85decode(payload),
            format=lzma.FORMAT_RAW,
            filters=[{"id": lzma.FILTER_LZMA2}],
        )
        meta.update(
            {
                "packed_wrapper": True,
                "packed_bytes": len(raw.encode("utf-8")),
                "decoded_bytes": len(decoded),
            }
        )
        return decoded.decode("utf-8"), meta
    except Exception as exc:  # pragma: no cover - diagnostic path
        meta["decode_error"] = repr(exc)
        return raw, meta


def extract_function(source: str, name: str) -> str:
    m = re.search(rf"^def {re.escape(name)}\(", source, re.M)
    if not m:
        return ""
    start = m.start()
    next_def = re.search(r"^def [A-Za-z_][A-Za-z0-9_]*\(", source[m.end() :], re.M)
    if not next_def:
        return source[start:]
    return source[start : m.end() + next_def.start()]


def getenv_default(source: str, key: str) -> str | None:
    m = re.search(rf'os\.environ\.get\("{re.escape(key)}",\s*([^)]+)\)', source)
    return m.group(1).strip() if m else None


def audit(source: str, meta: dict[str, object]) -> dict[str, object]:
    lower = source.lower()
    eval_ttt = extract_function(source, "eval_val_ttt")
    byte_luts = extract_function(source, "build_sentencepiece_luts")

    no_grad_pos = eval_ttt.find("torch.no_grad")
    score_pos = min(
        [p for p in [eval_ttt.find("loss_sum +="), eval_ttt.find("byte_count +=")] if p >= 0],
        default=-1,
    )
    step_pos = eval_ttt.find("optimizer.step")

    suspicious_terms = {
        "slot": "slot" in lower,
        "ngram": "ngram" in lower or "n-gram" in lower,
        "ppm": "ppm" in lower,
        "etlb": "etlb" in lower,
        "logit_bias": "logit_bias" in lower,
        "caseops_or_casefold": "caseops" in lower or "casefold" in lower,
    }

    checks = {
        "condition_1_causal_prefix_static": {
            "status": "pass"
            if "causal=True" in source or "is_causal=True" in source
            else "review",
            "evidence": "causal attention flag present; static audit cannot prove every data path",
        },
        "condition_2_full_distribution_static": {
            "status": "pass"
            if "F.cross_entropy" in source and not suspicious_terms["logit_bias"]
            else "review",
            "evidence": "uses F.cross_entropy on full logits; no logit_bias token-only path found"
            if not suspicious_terms["logit_bias"]
            else "logit_bias found; inspect full-vocab normalization manually",
        },
        "condition_3_score_before_update_static": {
            "status": "pass"
            if no_grad_pos >= 0 and score_pos >= 0 and step_pos >= 0 and no_grad_pos < score_pos < step_pos
            else "review",
            "evidence": {
                "torch_no_grad_pos": no_grad_pos,
                "score_accum_pos": score_pos,
                "optimizer_step_pos": step_pos,
            },
        },
        "condition_4_single_pass_static": {
            "status": "pass"
            if "for ci in range" in eval_ttt and "optimizer.step" in eval_ttt and "min(" not in eval_ttt[:500]
            else "review",
            "evidence": "eval_val_ttt iterates chunks once; no obvious min-over-runs pattern in function head",
        },
        "byte_accounting_static": {
            "status": "pass"
            if "base_bytes_np" in byte_luts
            and "has_leading_space" in byte_luts
            and "is_boundary_token" in byte_luts
            and "byte_count +=" in source
            else "review",
            "evidence": "SentencePiece byte LUT plus leading-space correction found",
        },
    }

    return {
        "metadata": meta,
        "defaults": {
            "TTT_ENABLED": getenv_default(source, "TTT_ENABLED"),
            "TTT_LR": getenv_default(source, "TTT_LR"),
            "TTT_EPOCHS": getenv_default(source, "TTT_EPOCHS"),
            "TTT_CHUNK_TOKENS": getenv_default(source, "TTT_CHUNK_TOKENS"),
            "MUON_WD": getenv_default(source, "MUON_WD"),
            "MUON_WD_MLP": getenv_default(source, "MUON_WD_MLP"),
            "SLIDING_WINDOW_ENABLED": getenv_default(source, "SLIDING_WINDOW_ENABLED"),
        },
        "suspicious_terms": suspicious_terms,
        "checks": checks,
        "summary": {
            "pass": sum(1 for item in checks.values() if item["status"] == "pass"),
            "review": sum(1 for item in checks.values() if item["status"] == "review"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("train_gpt", type=Path)
    parser.add_argument("--write-decoded", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    source, meta = load_source(args.train_gpt)
    if args.write_decoded:
        args.write_decoded.parent.mkdir(parents=True, exist_ok=True)
        args.write_decoded.write_text(source, encoding="utf-8")

    report = audit(source, meta)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
