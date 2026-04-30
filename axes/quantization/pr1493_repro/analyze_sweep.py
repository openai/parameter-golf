"""Analyze compression_sweep.py output.

Given sweep CSV (tensor × bits × k × grid → mse + compressed_bytes), answers:

1. What is the per-tensor Pareto frontier? (MSE vs compressed bytes)
2. Is PR-1493's (int6, k=12.85, uniform) configuration on that frontier?
3. Does NF universally dominate uniform int? Or only sometimes?
4. Is low-bit + high-k ALWAYS the winning compression strategy after Brotli?
5. If we allocate each tensor to its best (b, k, grid) independently,
   what's the total budget savings vs PR-1493's uniform choice?

Usage:
    /tmp/torch_env/bin/python3 analyze_sweep.py /tmp/sweep_results.csv
"""
import csv
import sys
from collections import defaultdict


def load_csv(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            r["bits"] = int(r["bits"])
            r["k"] = float(r["k"])
            r["numel"] = int(r["numel"])
            r["mse"] = float(r["mse"])
            r["rel_mse"] = float(r["rel_mse"])
            r["compressed_bytes"] = int(r["compressed_bytes"])
            r["bits_per_weight_effective"] = float(r["bits_per_weight_effective"])
            rows.append(r)
    return rows


def pareto_frontier(points):
    """Return the subset of points on the Pareto frontier (minimize both mse and bytes).

    points: iterable of dicts with "rel_mse" and "compressed_bytes".
    """
    sorted_by_size = sorted(points, key=lambda p: (p["compressed_bytes"], p["rel_mse"]))
    frontier = []
    best_mse = float("inf")
    for p in sorted_by_size:
        if p["rel_mse"] < best_mse:
            frontier.append(p)
            best_mse = p["rel_mse"]
    return frontier


def summarize_per_tensor(rows):
    """For each tensor, find the Pareto frontier and key reference points."""
    by_tensor = defaultdict(list)
    for r in rows:
        by_tensor[r["tensor"]].append(r)

    print(f"{'tensor':<45} {'numel':>8}  {'PR-1493 (int6 k=12.85)':>30}  {'best-matched-size':>25}  {'|frontier|':>10}")
    print("-" * 130)

    total_pr1493 = 0
    total_best = 0
    total_numel = 0

    savings_rows = []
    for name in sorted(by_tensor):
        points = by_tensor[name]
        tok_emb_like = name == "tok_emb.weight"

        # PR-1493 reference config
        if tok_emb_like:
            ref = next((p for p in points if p["grid"] == "uniform" and p["bits"] == 8 and abs(p["k"] - 20.0) < 0.01), None)
        else:
            ref = next((p for p in points if p["grid"] == "uniform" and p["bits"] == 6 and abs(p["k"] - 12.85) < 0.01), None)
        if ref is None:
            continue

        # Find smallest config with MSE <= ref MSE (same quality, minimum bytes)
        same_quality = [p for p in points if p["rel_mse"] <= ref["rel_mse"]]
        same_quality.sort(key=lambda p: p["compressed_bytes"])
        best = same_quality[0] if same_quality else ref

        frontier = pareto_frontier(points)
        savings = ref["compressed_bytes"] - best["compressed_bytes"]
        savings_rows.append({
            "tensor": name,
            "numel": ref["numel"],
            "ref_bytes": ref["compressed_bytes"],
            "best_bytes": best["compressed_bytes"],
            "best_config": f"{best['grid']}/b{best['bits']}/k{best['k']:.1f}",
            "savings_bytes": savings,
            "savings_pct": 100 * savings / ref["compressed_bytes"],
        })

        total_pr1493 += ref["compressed_bytes"]
        total_best += best["compressed_bytes"]
        total_numel += ref["numel"]

        ref_label = f"{ref['compressed_bytes']/1024:.1f}KB ({ref['bits_per_weight_effective']:.2f}bpw)"
        best_label = f"{best['grid']}/b{best['bits']}/k{best['k']:.1f} → {best['compressed_bytes']/1024:.1f}KB"
        print(f"{name:<45} {ref['numel']:>8}  {ref_label:>30}  {best_label:>25}  {len(frontier):>10}")

    print("-" * 130)
    print(f"{'TOTAL':<45} {total_numel:>8}  "
          f"PR-1493: {total_pr1493/1024:>8.1f}KB ({total_pr1493*8/total_numel:.2f}bpw)    "
          f"Best:    {total_best/1024:>8.1f}KB ({total_best*8/total_numel:.2f}bpw)    "
          f"Savings: {(total_pr1493-total_best)/1024:>7.1f}KB "
          f"({100*(total_pr1493-total_best)/total_pr1493:.1f}%)")

    return savings_rows


def grid_comparison(rows):
    """Compare uniform-best-k vs NF at each bit width."""
    by_tensor_bits = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_tensor_bits[r["tensor"]][r["bits"]].append(r)

    print("\n=== Uniform (best k) vs NF at each bit width (aggregated over tensors) ===")
    print(f"{'bits':>4}  {'uni MSE':>12}  {'uni bytes':>12}  {'NF MSE':>12}  {'NF bytes':>12}  {'NF_MSE/uni_MSE':>15}  {'NF_bytes/uni_bytes':>20}")
    print("-" * 110)

    for bits in sorted({r["bits"] for r in rows}):
        # For each tensor at this bit width, find the uniform config that matches
        # NF MSE most closely (or just the smallest uniform byte count with MSE <= NF MSE)
        tot_uni_mse = 0.0
        tot_uni_bytes = 0
        tot_nf_mse = 0.0
        tot_nf_bytes = 0
        tot_numel = 0

        for tname, bdict in by_tensor_bits.items():
            configs = bdict.get(bits, [])
            nf = next((p for p in configs if p["grid"] == "nf"), None)
            if nf is None:
                continue
            # Best uniform config at the same bit width: min (bytes, mse) — use Pareto knee
            unis = [p for p in configs if p["grid"] == "uniform"]
            # For aggregate, pick the uniform config with MSE closest-but-no-worse-than NF,
            # failing that pick smallest-MSE uniform.
            no_worse = [p for p in unis if p["rel_mse"] <= nf["rel_mse"]]
            if no_worse:
                best_uni = min(no_worse, key=lambda p: p["compressed_bytes"])
            else:
                best_uni = min(unis, key=lambda p: p["rel_mse"])

            n = nf["numel"]
            tot_uni_mse += best_uni["rel_mse"] * n
            tot_uni_bytes += best_uni["compressed_bytes"]
            tot_nf_mse += nf["rel_mse"] * n
            tot_nf_bytes += nf["compressed_bytes"]
            tot_numel += n

        if tot_numel == 0:
            continue
        avg_uni_mse = tot_uni_mse / tot_numel
        avg_nf_mse = tot_nf_mse / tot_numel
        mse_ratio = avg_nf_mse / avg_uni_mse if avg_uni_mse > 0 else float("inf")
        bytes_ratio = tot_nf_bytes / tot_uni_bytes if tot_uni_bytes > 0 else float("inf")
        print(f"{bits:>4}  {avg_uni_mse:>12.4e}  {tot_uni_bytes/1024:>10.1f}KB  "
              f"{avg_nf_mse:>12.4e}  {tot_nf_bytes/1024:>10.1f}KB  "
              f"{mse_ratio:>15.3f}  {bytes_ratio:>20.3f}")


def high_k_low_bits_check(rows):
    """Is 'few bins + high k' universally the compression winner?"""
    print("\n=== Per-tensor optimal configs (minimum bytes for rel_mse <= 0.02) ===")
    by_tensor = defaultdict(list)
    for r in rows:
        by_tensor[r["tensor"]].append(r)

    # Histogram of winning (bits, grid) choices
    winner_hist = defaultdict(int)
    for name, points in by_tensor.items():
        good = [p for p in points if p["rel_mse"] <= 0.02]
        if not good:
            continue
        best = min(good, key=lambda p: p["compressed_bytes"])
        winner_hist[(best["bits"], best["grid"])] += 1
    print(f"{'(bits, grid)':<15}  {'# tensors where this wins':<30}")
    for (b, g), n in sorted(winner_hist.items(), key=lambda x: -x[1]):
        print(f"  b={b} {g:<8}  {n}")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/sweep_results.csv"
    rows = load_csv(path)
    print(f"Loaded {len(rows)} sweep rows from {path}\n")
    print("=== Per-tensor: PR-1493 reference vs best-matched-MSE config ===\n")
    summarize_per_tensor(rows)
    grid_comparison(rows)
    high_k_low_bits_check(rows)


if __name__ == "__main__":
    main()
