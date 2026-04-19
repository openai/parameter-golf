"""ASCII Pareto-frontier visualization per tensor category.

Groups tensors by role (tok_emb, attn, mlp) and shows the Pareto frontier of
(MSE, compressed bytes) with PR-1493's reference config marked.

Usage:
    /tmp/torch_env/bin/python3 pareto_viz.py /tmp/sweep_results.csv
"""
import csv
import sys
from collections import defaultdict


def load(path):
    out = []
    with open(path) as f:
        for r in csv.DictReader(f):
            r["bits"] = int(r["bits"])
            r["k"] = float(r["k"])
            r["numel"] = int(r["numel"])
            r["rel_mse"] = float(r["rel_mse"])
            r["compressed_bytes"] = int(r["compressed_bytes"])
            r["bits_per_weight_effective"] = float(r["bits_per_weight_effective"])
            out.append(r)
    return out


def tensor_role(name: str) -> str:
    if "tok_emb" in name:
        return "tok_emb"
    if ".mlp.fc" in name:
        return "mlp_fc"
    if ".mlp.proj" in name:
        return "mlp_proj"
    if ".attn.c_q" in name:
        return "attn_q"
    if ".attn.c_k" in name:
        return "attn_k"
    if ".attn.c_v" in name:
        return "attn_v"
    if ".attn.proj" in name:
        return "attn_proj"
    return "other"


def frontier(pts):
    """Pareto frontier: minimize (compressed_bytes, rel_mse)."""
    srt = sorted(pts, key=lambda p: (p["compressed_bytes"], p["rel_mse"]))
    out = []
    best = float("inf")
    for p in srt:
        if p["rel_mse"] < best:
            out.append(p)
            best = p["rel_mse"]
    return out


def show_tensor(name, pts, ref_bits, ref_k):
    """Print a compact Pareto summary for a single tensor."""
    ref = next((p for p in pts if p["grid"] == "uniform" and p["bits"] == ref_bits and abs(p["k"] - ref_k) < 0.01), None)
    fr = frontier(pts)
    print(f"\n== {name}  (numel={pts[0]['numel']})")
    print(f"   Pareto frontier ({len(fr)} configs):")
    print(f"   {'config':<25}  {'rel_mse':>11}  {'bytes':>8}  {'bpw':>6}  {'marker':<6}")
    # Also include the reference config even if not on frontier, for comparison
    marked = set(id(p) for p in fr)
    if ref and id(ref) not in marked:
        fr_plus = sorted(fr + [ref], key=lambda p: p["compressed_bytes"])
    else:
        fr_plus = fr
    for p in fr_plus:
        cfg = f"{p['grid']}/b{p['bits']}/k{p['k']:.1f}" if p["grid"] == "uniform" else f"{p['grid']}/b{p['bits']}"
        marker = ""
        if ref is not None and p is ref:
            marker = "← PR-1493"
        elif id(p) in marked and p["bits_per_weight_effective"] < (ref["bits_per_weight_effective"] if ref else 99) and p["rel_mse"] <= (ref["rel_mse"] if ref else 99):
            marker = "*dominates*"
        print(f"   {cfg:<25}  {p['rel_mse']:>11.4e}  {p['compressed_bytes']:>8}  {p['bits_per_weight_effective']:>6.2f}  {marker:<6}")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/sweep_results.csv"
    rows = load(path)
    by_tensor = defaultdict(list)
    for r in rows:
        by_tensor[r["tensor"]].append(r)

    # Show one representative tensor per role — the largest of that role
    by_role = defaultdict(list)
    for name in by_tensor:
        by_role[tensor_role(name)].append(name)

    representatives = {}
    for role, names in by_role.items():
        if not names:
            continue
        # Pick the first one (deterministic order)
        representatives[role] = sorted(names)[0]

    # Role-specific reference configs
    for role, name in representatives.items():
        ref_bits = 8 if role == "tok_emb" else 6
        ref_k = 20.0 if role == "tok_emb" else 12.85
        show_tensor(name, by_tensor[name], ref_bits, ref_k)

    # Aggregate summary: for each role, sum savings across all tensors of that role
    print("\n\n=== Aggregate: PR-1493 vs best per tensor (same or better MSE) ===")
    print(f"{'role':<12} {'#tensors':>10} {'numel':>12} {'PR-1493 bytes':>16} {'best bytes':>12} {'savings':>10} {'pct':>6}")
    print("-" * 80)
    total_ref = 0
    total_best = 0
    for role, names in sorted(by_role.items()):
        role_ref = 0
        role_best = 0
        role_numel = 0
        for name in names:
            pts = by_tensor[name]
            ref_bits = 8 if role == "tok_emb" else 6
            ref_k = 20.0 if role == "tok_emb" else 12.85
            ref = next((p for p in pts if p["grid"] == "uniform" and p["bits"] == ref_bits and abs(p["k"] - ref_k) < 0.01), None)
            if ref is None:
                continue
            same_or_better = [p for p in pts if p["rel_mse"] <= ref["rel_mse"]]
            best = min(same_or_better, key=lambda p: p["compressed_bytes"])
            role_ref += ref["compressed_bytes"]
            role_best += best["compressed_bytes"]
            role_numel += ref["numel"]
        if role_ref == 0:
            continue
        pct = 100 * (role_ref - role_best) / role_ref
        print(f"{role:<12} {len(names):>10} {role_numel:>12} {role_ref:>16} {role_best:>12} {role_ref-role_best:>10} {pct:>5.1f}%")
        total_ref += role_ref
        total_best += role_best
    print("-" * 80)
    pct_total = 100 * (total_ref - total_best) / total_ref
    print(f"{'TOTAL':<12} {'':>10} {'':>12} {total_ref:>16} {total_best:>12} {total_ref-total_best:>10} {pct_total:>5.1f}%")


if __name__ == "__main__":
    main()
