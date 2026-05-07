"""
Data Order Experiment: Which shards best match the validation distribution?

Training consumes ~37.7 shards (7185 steps × 524288 tokens / 100M per shard).
Goal: select and order shards to maximize val_bpb.

Methods (after bug fixes):
  M1. N-gram cosine similarity (unigram + bigram joint dist)
  M2. Jensen-Shannon Divergence (unigram)
  M3. Moore-Lewis cross-entropy difference (bigram LMs, all-train general LM)
  M5. Val-trained LM perplexity (raw CE — separate signal, not averaged)
  M6. Conditional bigram embedding cosine
  M8. Importance weighting (density ratio p_val/p_train, all-train denominator)

Dropped: M4 (classifier, 51% accuracy = useless), M7 (Wasserstein on unordered categoricals)
"""

import numpy as np
import json
from pathlib import Path
import time

DATA_DIR = Path("data/datasets/fineweb10B_sp1024")
VOCAB_SIZE = 1024
OUTPUT_DIR = Path("experiments/data_order/stage1_shard_level")
N_SELECT = 38  # shards to select for training


def load_shard_tokens(path: Path) -> np.ndarray:
    header = np.fromfile(path, dtype="<i4", count=256)
    assert int(header[0]) == 20240520 and int(header[1]) == 1
    num_tokens = int(header[2])
    offset = 256 * np.dtype("<i4").itemsize
    return np.fromfile(path, dtype="<u2", count=num_tokens, offset=offset)


def unigram_dist(tokens):
    counts = np.bincount(tokens, minlength=VOCAB_SIZE).astype(np.float64)
    return counts / counts.sum()


def bigram_counts_flat(tokens):
    prev = tokens[:-1].astype(np.int64)
    curr = tokens[1:].astype(np.int64)
    return np.bincount(prev * VOCAB_SIZE + curr, minlength=VOCAB_SIZE * VOCAB_SIZE).astype(np.float64)


def bigram_dist(tokens):
    counts = bigram_counts_flat(tokens)
    return counts / counts.sum()


def cosine_sim(a, b):
    dot = np.dot(a, b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(dot / (na * nb))


def kl_div(p, q, eps=1e-12):
    p, q = p + eps, q + eps
    p, q = p / p.sum(), q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def jsd(p, q):
    m = 0.5 * (p + q)
    return 0.5 * kl_div(p, m) + 0.5 * kl_div(q, m)


def make_lm(bigram_2d, alpha=0.01):
    c = bigram_2d + alpha
    return np.log2(c / c.sum(axis=1, keepdims=True))


def bigram_ce_from_counts(lm_log2, bi_flat):
    total = bi_flat.sum()
    if total == 0:
        return float("inf")
    return float(-np.dot(bi_flat, lm_log2.flatten()) / total)


def cond_bigram_embedding(bigram_2d):
    rs = bigram_2d.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1
    return (bigram_2d / rs).flatten()


if __name__ == "__main__":
    t_start = time.time()

    val_path = DATA_DIR / "fineweb_val_000000.bin"
    shard_paths = sorted(DATA_DIR.glob("fineweb_train_*.bin"))
    print(f"Train shards: {len(shard_paths)}, selecting top {N_SELECT}")

    # ── Load val ──────────────────────────────────────────────────────
    print("Loading val tokens...")
    val_tokens = load_shard_tokens(val_path)
    val_uni = unigram_dist(val_tokens)
    val_bi = bigram_dist(val_tokens)
    val_bi_counts = bigram_counts_flat(val_tokens)
    val_lm = make_lm(val_bi_counts.reshape(VOCAB_SIZE, VOCAB_SIZE))
    val_cond = cond_bigram_embedding(val_bi_counts.reshape(VOCAB_SIZE, VOCAB_SIZE))
    print(f"Val: {len(val_tokens):,} tokens")

    # ── Precompute per-shard stats (single pass) ──────────────────────
    print("Precomputing shard statistics...")
    shard_data = []
    all_train_bi = np.zeros(VOCAB_SIZE * VOCAB_SIZE, dtype=np.float64)

    for j, path in enumerate(shard_paths):
        tokens = load_shard_tokens(path)
        uni = unigram_dist(tokens)
        bi_flat = bigram_counts_flat(tokens)
        bi_norm = bi_flat / bi_flat.sum()
        bi_2d = bi_flat.reshape(VOCAB_SIZE, VOCAB_SIZE)
        all_train_bi += bi_flat
        shard_data.append({
            "name": path.stem,
            "uni": uni,
            "bi_flat": bi_flat,
            "bi_norm": bi_norm,
            "bi_2d": bi_2d,
        })
        if (j + 1) % 10 == 0:
            print(f"  {j+1}/{len(shard_paths)}")

    # ── FIX: All-train general LM (no bias to specific shards) ────────
    all_train_lm = make_lm(all_train_bi.reshape(VOCAB_SIZE, VOCAB_SIZE))
    print(f"Precomputation: {time.time()-t_start:.1f}s")

    # ── Score all methods ─────────────────────────────────────────────
    results = []
    for sd in shard_data:
        name = sd["name"]
        s_uni = sd["uni"]
        s_bi_norm = sd["bi_norm"]
        s_bi_flat = sd["bi_flat"]
        s_bi_2d = sd["bi_2d"]

        # M1: unigram + bigram joint cosine
        m1 = (cosine_sim(val_uni, s_uni) + cosine_sim(val_bi, s_bi_norm)) / 2

        # M2: JSD on unigrams (lower = more similar, negate for ranking)
        m2 = -jsd(val_uni, s_uni)

        # M3: Moore-Lewis (CE_general - CE_val). Uses full shard counts.
        ce_val = bigram_ce_from_counts(val_lm, s_bi_flat)
        ce_gen = bigram_ce_from_counts(all_train_lm, s_bi_flat)
        m3 = ce_gen - ce_val

        # M5: raw CE under val LM (lower = more predictable by val patterns)
        # Uses full shard bigram counts (consistent with validation)
        m5 = -ce_val  # negate so higher = more similar

        # M6: conditional bigram embedding cosine
        s_cond = cond_bigram_embedding(s_bi_2d)
        m6 = cosine_sim(val_cond, s_cond)

        # M8: importance weighting E_shard[log(p_val/p_train)]
        log_ratio = val_lm.flatten() - all_train_lm.flatten()
        s_bi_n = s_bi_flat / (s_bi_flat.sum() + 1e-12)
        m8 = float(np.dot(s_bi_n, log_ratio))

        results.append({
            "shard": name,
            "m1_ngram_cosine": round(m1, 6),
            "m2_neg_jsd": round(m2, 6),
            "m3_moore_lewis": round(m3, 6),
            "m5_neg_val_ce": round(m5, 4),
            "m6_embed_cosine": round(m6, 6),
            "m8_importance": round(m8, 6),
        })

    # ── Rank by each method ───────────────────────────────────────────
    # Distributional methods (averaged for composite): M1, M2, M3, M6, M8
    dist_methods = ["m1_ngram_cosine", "m2_neg_jsd", "m3_moore_lewis", "m6_embed_cosine", "m8_importance"]

    ranks = {}
    for m in dist_methods + ["m5_neg_val_ce"]:
        sorted_by_m = sorted(results, key=lambda x: x[m], reverse=True)
        ranks[m] = {r["shard"]: i for i, r in enumerate(sorted_by_m)}

    # Print per-method top/bottom 10
    for m in dist_methods + ["m5_neg_val_ce"]:
        label = m.replace("neg_", "").replace("_", " ").upper()
        sorted_r = sorted(results, key=lambda x: x[m], reverse=True)
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")
        print("  Top 10:")
        for i, r in enumerate(sorted_r[:10]):
            print(f"    {i+1:2d}. {r['shard']}: {r[m]}")
        print("  Bottom 5:")
        for i, r in enumerate(sorted_r[-5:]):
            print(f"    {len(sorted_r)-4+i}. {r['shard']}: {r[m]}")

    # ── Composite ranking (distributional methods only) ───────────────
    print(f"\n{'='*60}")
    print(f"  COMPOSITE RANKING (5 distributional methods)")
    print(f"{'='*60}")

    composite = []
    for r in results:
        name = r["shard"]
        method_ranks = {m: ranks[m][name] for m in dist_methods}
        avg_rank = sum(method_ranks.values()) / len(dist_methods)
        composite.append({
            "shard": name,
            "avg_rank": round(avg_rank, 2),
            **{f"rank_{m}": method_ranks[m] for m in dist_methods},
            "rank_m5_valce": ranks["m5_neg_val_ce"][name],
        })

    composite.sort(key=lambda x: x["avg_rank"])

    print(f"\nAll {len(composite)} shards (most val-similar first):")
    print(f"  {'#':>3s} {'shard':>25s} {'avg':>6s}  m1  m2  m3  m6  m8  (m5)")
    for i, c in enumerate(composite):
        selected = ">>>" if i < N_SELECT else "   "
        print(f"  {i+1:3d} {c['shard']:>25s} {c['avg_rank']:6.2f}  "
              f"{c['rank_m1_ngram_cosine']:2d}  {c['rank_m2_neg_jsd']:2d}  "
              f"{c['rank_m3_moore_lewis']:2d}  {c['rank_m6_embed_cosine']:2d}  "
              f"{c['rank_m8_importance']:2d}  ({c['rank_m5_valce']:2d}) {selected}")

    # ── Save results ──────────────────────────────────────────────────
    selected_shards = [c["shard"] for c in composite[:N_SELECT]]
    excluded_shards = [c["shard"] for c in composite[N_SELECT:]]

    # Training order: least similar of selected first, most similar last
    training_order = list(reversed(selected_shards))

    print(f"\n{'='*60}")
    print(f"  SELECTED {N_SELECT} SHARDS (training order: least -> most val-similar)")
    print(f"{'='*60}")
    for i, name in enumerate(training_order):
        print(f"  Step {i+1:2d}: {name}")

    print(f"\n  EXCLUDED {len(excluded_shards)} shards:")
    for name in excluded_shards:
        print(f"    {name}")

    # Save everything
    with open(OUTPUT_DIR / "shard_analysis.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(OUTPUT_DIR / "shard_ranking.json", "w") as f:
        json.dump(composite, f, indent=2)

    with open(OUTPUT_DIR / "composite_selection.json", "w") as f:
        json.dump({
            "n_selected": N_SELECT,
            "training_order": training_order,
            "excluded": excluded_shards,
            "method": "composite of M1(ngram cosine), M2(JSD), M3(Moore-Lewis), M6(embed cosine), M8(importance weight)",
        }, f, indent=2)

    print(f"\nTotal time: {time.time()-t_start:.1f}s")
    print(f"Results saved to {OUTPUT_DIR}/")
