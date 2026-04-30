#!/usr/bin/env python3
import argparse
import json
import re
import sys
from pathlib import Path

from parse_run_logs import ARTIFACT_LIMIT, parse_log_text


FORBIDDEN_PATTERNS = [
    r"\bsp1024\b",
    r"CaseOps",
    r"Scylla",
    r"TokenMonster",
    r"\bSLOT\b",
    r"\bETLB\b.*enabled",
    r"n-gram cache",
    r"ngram cache",
    r"score-after-update",
    r"retokeniz(?:e|ed|ation)",
]


def _pass(label, detail):
    print(f"[PASS] {label}: {detail}")
    return True


def _fail(label, detail):
    print(f"[FAIL] {label}: {detail}")
    return False


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--summary")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--expected-ttt-epochs", type=int, default=4)
    args = ap.parse_args(argv)

    log_path = Path(args.log)
    text = log_path.read_text(encoding="utf-8", errors="replace")
    if args.summary and Path(args.summary).is_file():
        summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    else:
        summary = parse_log_text(text)

    seed = summary.get("seed")
    print(f"VALIDATION SUMMARY seed={seed}")
    ok = True

    artifact = summary.get("artifact_bytes")
    if artifact is None:
        ok &= _fail("compressed artifact size line found", "missing artifact_bytes / Total submission size line")
    else:
        ok &= _pass("compressed artifact size line found", artifact)
        if artifact <= ARTIFACT_LIMIT:
            ok &= _pass("artifact <= 16,000,000 bytes", artifact)
        else:
            ok &= _fail("artifact <= 16,000,000 bytes", artifact)

    train_seconds = summary.get("train_seconds")
    if train_seconds is None:
        ok &= _pass("train <= 600s", "not required in smoke") if args.smoke else _fail("train <= 600s", "missing")
    elif train_seconds <= 600.5:
        ok &= _pass("train <= 600s", f"{train_seconds:.3f}s")
    else:
        ok &= _fail("train <= 600s", f"{train_seconds:.3f}s")

    eval_seconds = summary.get("eval_seconds")
    if eval_seconds is None:
        ok &= _pass("eval <= 600s", "not required in smoke") if args.smoke else _fail("eval <= 600s", "missing")
    elif eval_seconds <= 600.5:
        ok &= _pass("eval <= 600s", f"{eval_seconds:.3f}s")
    else:
        ok &= _fail("eval <= 600s", f"{eval_seconds:.3f}s")

    ttt_enabled = bool(summary.get("ttt_enabled"))
    if not args.smoke and not ttt_enabled:
        ok &= _fail("TTT enabled for final run", "TTT disabled or not parsed")
    if ttt_enabled:
        if summary.get("score_first_ttt_logged"):
            ok &= _pass("score-first TTT logged", "protocol, score_before_update, no_rescore")
        else:
            ok &= _fail("score-first TTT logged", "missing required TTT protocol lines")
        if summary.get("ttt_epochs") == args.expected_ttt_epochs:
            ok &= _pass(f"TTT_EPOCHS == {args.expected_ttt_epochs}", summary.get("ttt_epochs"))
        else:
            ok &= _fail(f"TTT_EPOCHS == {args.expected_ttt_epochs}", summary.get("ttt_epochs"))
    else:
        ok &= _pass("score-first TTT logged", "TTT disabled for this run")

    val_bpb = summary.get("val_bpb")
    if val_bpb is None:
        ok &= _pass("val_bpb found", "not required in smoke") if args.smoke else _fail("val_bpb found", "missing")
    else:
        ok &= _pass("val_bpb found", val_bpb)

    forbidden_hits = []
    for pat in FORBIDDEN_PATTERNS:
        if re.search(pat, text, re.I):
            forbidden_hits.append(pat)
    if forbidden_hits:
        ok &= _fail("no retokenization path used", ", ".join(forbidden_hits))
    else:
        ok &= _pass("no retokenization path used", "no forbidden log patterns")

    vocab_values = re.findall(r"\bvocab_size:\s*(\d+)", text, re.I)
    if any(v != "8192" for v in vocab_values):
        ok &= _fail("SP8192 tokenizer expected", f"vocab_size lines={vocab_values}")
    elif "fineweb10B_sp1024" in text:
        ok &= _fail("SP8192 tokenizer expected", "SP1024 path found")
    elif vocab_values or "fineweb10B_sp8192" in text or "fineweb_8192_bpe.model" in text:
        ok &= _pass("SP8192 tokenizer expected", "SP8192 log/path found")
    else:
        ok &= _pass("SP8192 tokenizer expected", "no contrary tokenizer path found") if args.smoke else _fail("SP8192 tokenizer expected", "missing SP8192 evidence")

    if not ok:
        return 1
    print("VALIDATION PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
