"""
Validate shard-ranking methods via holdout on the val set.

Precompute all per-shard statistics ONCE, then run 10 random val splits
cheaply using only the precomputed data.
"""

import numpy as np
import json
from pathlib import Path
import time

DATA_DIR = Path("data/datasets/fineweb10B_sp1024")
VOCAB_SIZE = 1024
OUTPUT_DIR = Path("experiments/data_order/stage1_shard_level")


def load_shard_tokens(path):
    header = np.fromfile(path, dtype="<i4", count=256)
    assert int(header[0]) == 20240520 and int(header[1]) == 1
    num_tokens = int(header[2])
    offset = 256 * np.dtype("<i4").itemsize
    return np.fromfile(path, dtype="<u2", count=num_tokens, offset=offset)


def unigram_counts(tokens):
    return np.bincount(tokens, minlength=VOCAB_SIZE).astype(np.float64)


def bigram_counts_flat(tokens):
    prev = tokens[:-1].astype(np.int64)
    curr = tokens[1:].astype(np.int64)
    return np.bincount(prev * VOCAB_SIZE + curr, minlength=VOCAB_SIZE * VOCAB_SIZE).astype(np.float64)


def normalize(counts):
    s = counts.sum()
    return counts / s if s > 0 else counts


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


def make_lm(bigram_counts_2d, alpha=0.01):
    c = bigram_counts_2d + alpha
    return np.log2(c / c.sum(axis=1, keepdims=True))


def bigram_ce_from_counts(lm_log2, shard_bigram_flat):
    """CE = -sum(count * log2(prob)) / total_bigrams. No need for raw tokens."""
    flat_lm = lm_log2.flatten()
    total = shard_bigram_flat.sum()
    if total == 0:
        return float("inf")
    return float(-np.dot(shard_bigram_flat, flat_lm) / total)


def cond_bigram_embedding(bigram_2d):
    rs = bigram_2d.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1
    return (bigram_2d / rs).flatten()


def spearman(x, y):
    n = len(x)
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    d = rx - ry
    return 1.0 - 6.0 * np.sum(d ** 2) / (n * (n ** 2 - 1))


def score_all_methods(ref_uni, ref_bi_flat, ref_lm, ref_cond_emb, gen_lm, all_train_lm,
                      shard_uni_list, shard_bi_flat_list, shard_bi_2d_list):
    """Score all shards against a reference (val half). Uses precomputed shard stats."""
    n_shards = len(shard_uni_list)
    ref_uni_norm = normalize(ref_uni)
    ref_bi_norm = normalize(ref_bi_flat)

    scores = {
        "m1_ngram_cosine": np.zeros(n_shards),
        "m2_jsd": np.zeros(n_shards),
        "m3_moore_lewis": np.zeros(n_shards),
        "m5_val_ce": np.zeros(n_shards),
        "m6_embed_cosine": np.zeros(n_shards),
        "m8_importance": np.zeros(n_shards),
    }

    for i in range(n_shards):
        s_uni = normalize(shard_uni_list[i])
        s_bi = normalize(shard_bi_flat_list[i])

        # M1: combined unigram+bigram cosine
        scores["m1_ngram_cosine"][i] = (cosine_sim(ref_uni_norm, s_uni) + cosine_sim(ref_bi_norm, s_bi)) / 2

        # M2: JSD (negate -> higher = more similar)
        scores["m2_jsd"][i] = -jsd(ref_uni_norm, s_uni)

        # M3: Moore-Lewis (CE_gen - CE_ref)
        ce_ref = bigram_ce_from_counts(ref_lm, shard_bi_flat_list[i])
        ce_gen = bigram_ce_from_counts(gen_lm, shard_bi_flat_list[i])
        scores["m3_moore_lewis"][i] = ce_gen - ce_ref

        # M5: raw CE under ref LM (negate -> higher = more similar)
        scores["m5_val_ce"][i] = -ce_ref

        # M6: conditional bigram embedding cosine
        s_cond = cond_bigram_embedding(shard_bi_2d_list[i])
        scores["m6_embed_cosine"][i] = cosine_sim(ref_cond_emb, s_cond)

        # M8: importance weighting = avg log(p_ref/p_train) over shard bigrams
        ref_flat = ref_lm.flatten()
        train_flat = all_train_lm.flatten()
        log_ratio = ref_flat - train_flat  # log2(p_ref / p_train) per bigram
        # Weight by shard's bigram distribution
        s_bi_n = normalize(shard_bi_flat_list[i])
        scores["m8_importance"][i] = float(np.dot(s_bi_n, log_ratio))

    return scores


if __name__ == "__main__":
    t_start = time.time()

    val_path = DATA_DIR / "fineweb_val_000000.bin"
    shard_paths = sorted(DATA_DIR.glob("fineweb_train_*.bin"))

    print("Loading val tokens...")
    val_tokens = load_shard_tokens(val_path)
    print(f"Val: {len(val_tokens):,} tokens")

    # ── Precompute per-shard statistics (ONE pass over all data) ──────
    print("Precomputing per-shard statistics...")
    shard_uni_list = []
    shard_bi_flat_list = []
    shard_bi_2d_list = []
    total_bi = np.zeros(VOCAB_SIZE * VOCAB_SIZE, dtype=np.float64)

    for j, path in enumerate(shard_paths):
        tokens = load_shard_tokens(path)
        uni = unigram_counts(tokens)
        bi = bigram_counts_flat(tokens)
        shard_uni_list.append(uni)
        shard_bi_flat_list.append(bi)
        shard_bi_2d_list.append(bi.reshape(VOCAB_SIZE, VOCAB_SIZE).copy())
        total_bi += bi
        if (j + 1) % 10 == 0:
            print(f"  {j+1}/{len(shard_paths)} shards processed")

    # All-train LM
    all_train_lm = make_lm(total_bi.reshape(VOCAB_SIZE, VOCAB_SIZE))

    # General LM (all-train aggregate — NOT biased 3-shard subset)
    gen_lm = all_train_lm

    print(f"Precomputation done in {time.time()-t_start:.1f}s")

    # ── Run splits ────────────────────────────────────────────────────
    n_splits = 10
    methods = ["m1_ngram_cosine", "m2_jsd", "m3_moore_lewis",
               "m5_val_ce", "m6_embed_cosine", "m8_importance"]
    correlations = {m: [] for m in methods}

    rng = np.random.RandomState(42)

    for split_i in range(n_splits):
        t0 = time.time()
        # FIX: contiguous splits to preserve sequential structure (bigram patterns)
        # Randomly choose a split point, take two contiguous halves
        half = len(val_tokens) // 2
        split_point = rng.randint(half // 2, len(val_tokens) - half // 2)
        val_A = val_tokens[split_point - half // 2 : split_point + half // 2]
        val_B_start = (split_point + half // 2) % len(val_tokens)
        # Wrap around if needed
        if val_B_start + half <= len(val_tokens):
            val_B = val_tokens[val_B_start : val_B_start + half]
        else:
            val_B = np.concatenate([val_tokens[val_B_start:], val_tokens[:half - (len(val_tokens) - val_B_start)]])

        # Compute val_A stats
        a_uni = unigram_counts(val_A)
        a_bi = bigram_counts_flat(val_A)
        a_lm = make_lm(a_bi.reshape(VOCAB_SIZE, VOCAB_SIZE))
        a_cond = cond_bigram_embedding(a_bi.reshape(VOCAB_SIZE, VOCAB_SIZE))

        # Compute val_B stats
        b_uni = unigram_counts(val_B)
        b_bi = bigram_counts_flat(val_B)
        b_lm = make_lm(b_bi.reshape(VOCAB_SIZE, VOCAB_SIZE))
        b_cond = cond_bigram_embedding(b_bi.reshape(VOCAB_SIZE, VOCAB_SIZE))

        scores_A = score_all_methods(a_uni, a_bi, a_lm, a_cond, gen_lm, all_train_lm,
                                     shard_uni_list, shard_bi_flat_list, shard_bi_2d_list)
        scores_B = score_all_methods(b_uni, b_bi, b_lm, b_cond, gen_lm, all_train_lm,
                                     shard_uni_list, shard_bi_flat_list, shard_bi_2d_list)

        print(f"Split {split_i+1}/{n_splits} ({time.time()-t0:.1f}s):", end="")
        for m in methods:
            rho = spearman(scores_A[m], scores_B[m])
            correlations[m].append(rho)
            print(f"  {m.split('_',1)[1][:6]}={rho:.3f}", end="")
        print()

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("VALIDATION: Spearman rank correlation (val_A ranking -> val_B ranking)")
    print("Higher = method produces more stable, reliable shard rankings")
    print("=" * 70)

    summary = {}
    for m in sorted(methods, key=lambda m: -np.mean(correlations[m])):
        vals = correlations[m]
        mean_rho = np.mean(vals)
        std_rho = np.std(vals)
        summary[m] = {"mean_rho": round(float(mean_rho), 4), "std_rho": round(float(std_rho), 4),
                       "all_rhos": [round(float(v), 4) for v in vals]}
        bar = "#" * int(mean_rho * 50)
        print(f"  {m:20s}: rho = {mean_rho:.4f} +/- {std_rho:.4f}  {bar}")

    with open(OUTPUT_DIR / "method_validation.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nTotal time: {time.time()-t_start:.1f}s")
    print(f"Saved to {OUTPUT_DIR / 'method_validation.json'}")
