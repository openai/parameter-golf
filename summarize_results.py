#!/usr/bin/env python3
"""
summarize_results.py — Analyze experiment history from results.tsv.
Shows throughput, keep rate, category analysis, and recommendations.
"""
import sys
from collections import defaultdict


def load_results(path="results.tsv"):
    rows = []
    with open(path) as f:
        header = f.readline().strip().split("\t")
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            row = dict(zip(header, parts))
            rows.append(row)
    return rows


def categorize(desc: str) -> str:
    """Rough category from description text."""
    desc_lower = desc.lower()
    categories = {
        "lr": ["lr ", "learning rate", "embed_lr", "matrix_lr", "scalar_lr"],
        "schedule": ["warmdown", "warmup", "schedule", "cosine"],
        "optimizer": ["muon", "adam", "momentum", "ema", "swa"],
        "architecture": ["layer", "depth", "xsa", "attention", "mlp", "head",
                         "gqa", "rope", "leaky", "relu", "activation", "ve",
                         "value embed", "smeargate", "bigram", "u-net", "skip"],
        "quantization": ["quant", "int6", "int5", "int8", "gptq", "qat",
                         "pruning", "prune"],
        "compression": ["lzma", "zstd", "compress"],
        "training": ["batch", "seq_len", "tokens", "gradient", "accum"],
        "eval": ["sliding", "stride", "ttt", "test-time"],
    }
    for cat, keywords in categories.items():
        if any(kw in desc_lower for kw in keywords):
            return cat
    return "other"


def main():
    rows = load_results()
    if not rows:
        print("No experiments found in results.tsv")
        return

    total = len(rows)
    keeps = [r for r in rows if r.get("status") == "keep"]
    discards = [r for r in rows if r.get("status") == "discard"]
    crashes = [r for r in rows if r.get("status") in ("crash", "oversize")]

    print(f"{'='*60}")
    print(f"EXPERIMENT SUMMARY ({total} experiments)")
    print(f"{'='*60}")
    print(f"  Keeps:    {len(keeps)} ({100*len(keeps)/total:.0f}%)")
    print(f"  Discards: {len(discards)} ({100*len(discards)/total:.0f}%)")
    print(f"  Crashes:  {len(crashes)} ({100*len(crashes)/total:.0f}%)")
    print()

    # Best result
    valid = [r for r in rows if r.get("status") == "keep" and float(r.get("val_bpb", 999)) < 999]
    if valid:
        best = min(valid, key=lambda r: float(r["val_bpb"]))
        print(f"  Best BPB: {best['val_bpb']} ({best['description']})")
        print(f"  Artifact: {best.get('artifact_mb', '?')} MB")
    print()

    # Category analysis
    cat_stats = defaultdict(lambda: {"total": 0, "keeps": 0})
    for r in rows:
        cat = categorize(r.get("description", ""))
        cat_stats[cat]["total"] += 1
        if r.get("status") == "keep":
            cat_stats[cat]["keeps"] += 1

    print("CATEGORY BREAKDOWN:")
    print(f"  {'Category':<15} {'Total':>6} {'Keeps':>6} {'Rate':>8}")
    print(f"  {'-'*15} {'-'*6} {'-'*6} {'-'*8}")
    structural_count = 0
    tuning_count = 0
    for cat, stats in sorted(cat_stats.items(), key=lambda x: -x[1]["total"]):
        rate = f"{100*stats['keeps']/stats['total']:.0f}%" if stats["total"] > 0 else "N/A"
        print(f"  {cat:<15} {stats['total']:>6} {stats['keeps']:>6} {rate:>8}")
        if cat in ("architecture", "eval", "quantization"):
            structural_count += stats["total"]
        elif cat in ("lr", "schedule", "optimizer", "training"):
            tuning_count += stats["total"]

    print()
    if structural_count + tuning_count > 0:
        ratio = structural_count / max(tuning_count, 1)
        print(f"  Structural:Tuning ratio: {structural_count}:{tuning_count} ({ratio:.1f}:1)")
        if ratio < 2.0:
            print(f"  WARNING: Below 2:1 target. Try more structural changes.")
    print()

    # Consecutive discards at end
    consec_discards = 0
    for r in reversed(rows):
        if r.get("status") == "discard":
            consec_discards += 1
        else:
            break
    if consec_discards > 0:
        print(f"  Current streak: {consec_discards} consecutive discards")
        if consec_discards >= 5:
            print(f"  SUGGESTION: Go bolder. Try a fundamentally different approach.")
        elif consec_discards >= 3:
            print(f"  SUGGESTION: Current direction may be exhausted. Try a different category.")
    else:
        print(f"  Last experiment was a keep. Refine nearby.")
    print()

    # Estimated throughput
    print("THROUGHPUT:")
    print(f"  At ~6 min/experiment (DEV mode): ~{60//6} experiments/hour")
    print(f"  At ~12 min/experiment (FULL mode): ~{60//12} experiments/hour")
    print(f"  Prescreen failures save ~4.5 min each")
    prescreen_fails = [r for r in rows if r.get("status") == "prescreen_fail"]
    if prescreen_fails:
        saved_min = len(prescreen_fails) * 4.5
        print(f"  Prescreens saved: {len(prescreen_fails)} ({saved_min:.0f} min)")
    print()

    # BPB progression
    print("BPB PROGRESSION (keeps only):")
    for r in keeps:
        bpb = r.get("val_bpb", "?")
        desc = r.get("description", "")[:50]
        print(f"  {bpb:>8}  {desc}")


if __name__ == "__main__":
    main()
