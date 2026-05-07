"""
proof_fineweb_causal.py — Causal-only FineWeb benchmark

NO training pre-fill. Cache built incrementally from validation data only,
strictly causal (score position t, then update cache with token t).

This is the regime where concentration matters most — early positions
have very few counts, so the smoothing parameter determines quality.
"""

import math
import numpy as np
import time
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from binding_ctw import BindingCTW


def load_fineweb_tokens(path: str) -> np.ndarray:
    header = np.fromfile(path, dtype=np.int32, count=256)
    assert header[0] == 20240520, f"Bad magic: {header[0]}"
    n_tokens = int(header[2])
    with open(path, "rb") as f:
        f.seek(256 * 4)
        tokens = np.frombuffer(f.read(n_tokens * 2), dtype=np.uint16)
    return tokens.copy()


def run():
    print("=" * 70)
    print("FINEWEB CAUSAL BENCHMARK: No training pre-fill")
    print("=" * 70)

    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            "data", "datasets", "fineweb10B_sp1024")
    train_path = os.path.join(data_dir, "fineweb_train_000000.bin")
    val_path = os.path.join(data_dir, "fineweb_val_000000.bin")

    print("\n[1] Loading data...")
    val_tokens = load_fineweb_tokens(val_path)
    # Load training just for IDF (token frequencies), NOT for n-gram cache
    train_tokens = load_fineweb_tokens(train_path)
    print(f"    Val: {len(val_tokens):,} tokens")
    print(f"    Train: {len(train_tokens):,} tokens (IDF only, no cache pre-fill)")

    vocab_size = 1024
    freq = np.bincount(train_tokens.astype(np.int32),
                       minlength=vocab_size).astype(np.float64)

    # Score in windows, updating cache after each window (causal)
    eval_size = 100_000  # score first 100K val tokens
    window_size = 1024   # update cache every 1024 tokens
    max_order = 9
    num_buckets = 65536

    configs = [
        ("Fixed c=5.0", 5.0, 0.0),
        ("Fixed c=2.0", 2.0, 0.0),
        ("Fixed c=1.0", 1.0, 0.0),
        ("Fixed c=0.5", 0.5, 0.0),
        ("Binding (c=5, β=1)", 5.0, 1.0),
        ("Binding (c=5, β=2)", 5.0, 2.0),
        ("Binding (c=5, β=3)", 5.0, 3.0),
        ("Binding (c=3, β=2)", 3.0, 2.0),
        ("Binding (c=3, β=3)", 3.0, 3.0),
    ]

    results = []

    for name, c_base, beta in configs:
        print(f"\n[2] {name}")

        cache = BindingCTW(
            max_order=max_order, min_order=2,
            num_buckets=num_buckets, min_count=1,  # min_count=1 for sparse regime
            c_base=c_base, beta=beta, vocab_size=vocab_size)

        # Only warm IDF for binding energy — NO n-gram cache pre-fill
        if beta > 0:
            cache.warm_from_training(freq, len(train_tokens))

        t0 = time.time()
        all_probs = []

        # Causal scoring: score window, then update cache
        for start in range(0, eval_size, window_size):
            end = min(start + window_size, eval_size)
            seg_len = end - start
            base_p = np.full(seg_len, 1.0 / vocab_size)

            if beta == 0:
                probs = cache.lookup_hierarchical_fixed(
                    val_tokens, start, end, base_p, concentration=c_base)
            else:
                probs = cache.lookup_hierarchical_binding(
                    val_tokens, start, end, base_p, context_len=8)

            all_probs.append(probs)

            # Update cache with scored tokens (causal — already scored)
            cache.update(val_tokens, start, end)

        t1 = time.time()

        all_probs = np.concatenate(all_probs)
        all_probs = np.clip(all_probs, 1e-15, 1.0)
        bpt = float(-np.log2(all_probs).mean())

        # Also compute early vs late performance
        early = all_probs[:10_000]
        late = all_probs[50_000:]
        bpt_early = float(-np.log2(np.clip(early, 1e-15, 1.0)).mean())
        bpt_late = float(-np.log2(np.clip(late, 1e-15, 1.0)).mean())

        print(f"    All:   {bpt:.6f} bpt")
        print(f"    Early: {bpt_early:.6f} bpt (first 10K, sparse cache)")
        print(f"    Late:  {bpt_late:.6f} bpt (after 50K, warmer cache)")
        print(f"    Time:  {t1-t0:.1f}s")

        results.append({
            "name": name, "c_base": c_base, "beta": beta,
            "bpt": bpt, "bpt_early": bpt_early, "bpt_late": bpt_late,
            "time": t1 - t0,
        })

    # Summary
    print(f"\n{'='*70}")
    print(f"RESULTS — Causal scoring, no training pre-fill")
    print(f"{'='*70}")
    print(f"{'Method':<30} {'All':>10} {'Early':>10} {'Late':>10}")
    print(f"{'-'*62}")

    best_fixed = min(r["bpt"] for r in results if r["beta"] == 0)
    best_binding = min(r["bpt"] for r in results if r["beta"] > 0)
    best_overall = min(r["bpt"] for r in results)

    for r in results:
        marker = " *" if r["bpt"] == best_overall else ""
        print(f"{r['name']:<30} {r['bpt']:>10.6f} {r['bpt_early']:>10.6f} {r['bpt_late']:>10.6f}{marker}")

    delta = best_fixed - best_binding
    print(f"\n{'='*70}")
    print(f"Best fixed:   {best_fixed:.6f}")
    print(f"Best binding: {best_binding:.6f}")
    print(f"Delta:        {delta:+.6f} ({100*delta/best_fixed:+.2f}%)")
    if delta > 0:
        print(f"BINDING WINS")
    else:
        print(f"FIXED WINS")

    # Early-only comparison (where concentration matters most)
    best_fixed_early = min(r["bpt_early"] for r in results if r["beta"] == 0)
    best_binding_early = min(r["bpt_early"] for r in results if r["beta"] > 0)
    delta_early = best_fixed_early - best_binding_early
    print(f"\nEarly positions (first 10K, sparse cache):")
    print(f"  Best fixed:   {best_fixed_early:.6f}")
    print(f"  Best binding: {best_binding_early:.6f}")
    print(f"  Delta:        {delta_early:+.6f} ({100*delta_early/best_fixed_early:+.2f}%)")

    out = {
        "mode": "causal_no_prefill",
        "eval_tokens": eval_size,
        "window_size": window_size,
        "max_order": max_order,
        "results": results,
        "best_fixed": best_fixed,
        "best_binding": best_binding,
        "delta": delta,
        "delta_early": delta_early,
    }
    out_path = os.path.join(os.path.dirname(__file__), "proof_fineweb_causal_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    run()
