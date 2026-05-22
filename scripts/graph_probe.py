#!/usr/bin/env python3
"""
graph_probe.py — post-run analyzer for torch.compile / CUDA-graph behavior.

Reads one or more training log files (the .log output produced by run_experiment.sh)
and extracts:
  - which compile mode was selected at each site (train / eval / ttt)
  - how many times Dynamo recompiled (a recompile invalidates any captured CUDA graph)
  - how many graph breaks were recorded
  - the trained-step tok/s figures
  - val_bpb points (pre-quant, quantized, sliding, ttt)

Emits a compact TSV summary per file plus a final PASS/FAIL verdict against
user-supplied baseline numbers.

Usage:
  python3 scripts/graph_probe.py <baseline.log> <candidate.log>
  python3 scripts/graph_probe.py --json logs/sweep/graph_baseline_ctrl.log logs/sweep/graph_eval_ro.log

Intended to answer: did the `mode="reduce-overhead"` flag actually capture a graph,
how many recompiles invalidated it, and what is the signed tok/s delta vs baseline.
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path

RECOMPILE_RE   = re.compile(r"Recompiling function", re.IGNORECASE)
GRAPHBREAK_RE  = re.compile(r"Graph break", re.IGNORECASE)
COMPILE_MODE_RE= re.compile(r"compile:(train|eval|ttt)\s+mode=(\S+)")
TOK_S_RE       = re.compile(r"tok/s:\s*([\d.]+)")
VAL_BPB_RE     = re.compile(r"val_bpb:\s*([\d.]+)")
PRE_QUANT_RE   = re.compile(r"pre-quantization post-ema val_loss:\s*([\d.]+)\s+val_bpb:\s*([\d.]+)")
QUANT_RE       = re.compile(r"quantized val_loss:\s*([\d.]+)\s+val_bpb:\s*([\d.]+)")
QUANT_SW_RE    = re.compile(r"quantized_sliding_window val_loss:\s*([\d.]+)\s+val_bpb:\s*([\d.]+)")
QUANT_TTT_RE   = re.compile(r"quantized_ttt val_loss:\s*([\d.]+)\s+val_bpb:\s*([\d.]+)")
PEAK_MEM_RE    = re.compile(r"peak memory allocated:\s*(\d+)\s*MiB")


def analyze(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    modes = {}
    for role, mode in COMPILE_MODE_RE.findall(text):
        # keep first occurrence per role
        modes.setdefault(role, mode)
    tok_s_series = [float(x) for x in TOK_S_RE.findall(text)]
    last_tok_s = tok_s_series[-1] if tok_s_series else None

    def _m(rx):
        m = rx.search(text)
        return (float(m.group(1)), float(m.group(2))) if m else (None, None)

    pre_loss, pre_bpb   = _m(PRE_QUANT_RE)
    q_loss,   q_bpb     = _m(QUANT_RE)
    qsw_loss, qsw_bpb   = _m(QUANT_SW_RE)
    qttt_loss,qttt_bpb  = _m(QUANT_TTT_RE)
    peak_m = PEAK_MEM_RE.search(text)
    peak_mib = int(peak_m.group(1)) if peak_m else None

    return {
        "file": str(path),
        "compile_modes": modes,           # {'train': 'reduce-overhead', 'eval': 'reduce-overhead', ...}
        "recompiles": len(RECOMPILE_RE.findall(text)),
        "graph_breaks": len(GRAPHBREAK_RE.findall(text)),
        "tok_s_last": last_tok_s,
        "tok_s_samples": len(tok_s_series),
        "pre_quant_bpb": pre_bpb,
        "quant_bpb": q_bpb,
        "sliding_bpb": qsw_bpb,
        "ttt_bpb": qttt_bpb,
        "peak_mib": peak_mib,
    }


def fmt_row(d: dict) -> str:
    modes = ",".join(f"{k}={v}" for k, v in sorted(d["compile_modes"].items())) or "-"
    def f(x, p=2):
        return f"{x:.{p}f}" if isinstance(x, float) else "-"
    return (
        f"{Path(d['file']).name:<40}  "
        f"modes=[{modes}]  "
        f"recomp={d['recompiles']:>3}  gbreak={d['graph_breaks']:>3}  "
        f"tok/s={f(d['tok_s_last'],0):>6}  "
        f"prebpb={f(d['pre_quant_bpb'],4)}  qbpb={f(d['quant_bpb'],4)}  "
        f"swbpb={f(d['sliding_bpb'],4)}  tttbpb={f(d['ttt_bpb'],4)}  "
        f"peak={d['peak_mib'] or '-'}MiB"
    )


def verdict(baseline: dict, cand: dict) -> list[str]:
    out = []
    if baseline["tok_s_last"] and cand["tok_s_last"]:
        delta = (cand["tok_s_last"] - baseline["tok_s_last"]) / baseline["tok_s_last"] * 100
        out.append(f"tok/s delta: {delta:+.2f}%  ({baseline['tok_s_last']:.0f} -> {cand['tok_s_last']:.0f})")
    for k, label in [("pre_quant_bpb","pre-quant bpb"),("quant_bpb","quant bpb"),("sliding_bpb","sliding bpb"),("ttt_bpb","ttt bpb")]:
        b, c = baseline.get(k), cand.get(k)
        if b is not None and c is not None:
            out.append(f"{label} delta: {c-b:+.6f}  ({b:.6f} -> {c:.6f})")
    if cand["recompiles"] > baseline["recompiles"]:
        out.append(f"WARNING: candidate recompiled {cand['recompiles']} times vs baseline {baseline['recompiles']} (graph capture invalidated)")
    if cand["graph_breaks"] > baseline["graph_breaks"]:
        out.append(f"WARNING: candidate has {cand['graph_breaks']} graph breaks vs baseline {baseline['graph_breaks']}")
    return out


def main():
    ap = argparse.ArgumentParser(description="Analyze torch.compile behavior in training logs.")
    ap.add_argument("logs", nargs="+", type=Path, help="log files (first treated as baseline)")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = ap.parse_args()
    rows = [analyze(p) for p in args.logs]
    if args.json:
        print(json.dumps(rows, indent=2))
        return
    for r in rows:
        print(fmt_row(r))
    if len(rows) >= 2:
        print()
        print(f"-- verdict (baseline={Path(rows[0]['file']).name}) --")
        for cand in rows[1:]:
            print(f"\n[{Path(cand['file']).name}]")
            for line in verdict(rows[0], cand):
                print(f"  {line}")


if __name__ == "__main__":
    main()
