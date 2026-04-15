"""
Extended analysis — compares:

(1) PER-DOC cache (resets at each document boundary — what kill_switch measured)
(2) GLOBAL cache (accumulates across all docs — closer to what eval_val_ttt
    actually does, since the val stream is a single concatenated sequence)

Plus: an alpha sweep simulation using a FROZEN BIGRAM proxy for the "neural"
model. This is a cheap approximation — it tells us RELATIVE gain (ngram vs no
ngram for the same model), not absolute BPB. Gives us an alpha-sensitivity
curve without training anything.

Metric: measured BPB reduction from adding the n-gram on top of bigram.

This is a LOCAL, ZERO-COST experiment. Running overnight.
"""
from __future__ import annotations
import argparse
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


# ---------- data loading ----------

def load_val_tokens(path: Path):
    header_bytes = 256 * 4
    tokens = np.fromfile(path, dtype='<u2', offset=header_bytes)
    return tokens


def segment_documents(tokens, bos_id: int = 1):
    boundaries = np.nonzero(tokens == bos_id)[0]
    docs = []
    start = 0
    for b in boundaries:
        if b > start:
            docs.append(tokens[start:b])
        start = b + 1
    if start < len(tokens):
        docs.append(tokens[start:])
    return docs


# ---------- bigram "neural" proxy ----------

class BigramLM:
    """Simple add-1 bigram LM trained ONCE on the val set itself before
    evaluation. This is NOT legal for a real submission — it's only a stand-in
    for the "neural" model so we can measure the RELATIVE gain of the n-gram
    addition.

    We then evaluate its BPB WITH and WITHOUT an additive n-gram contribution.
    """

    def __init__(self, vocab_size: int):
        self.V = vocab_size
        self.counts = defaultdict(Counter)  # prev_token -> Counter(next_token)
        self.totals = defaultdict(int)
        self._log_probs = None  # (V, V) tensor after fit

    def fit(self, tokens: np.ndarray):
        """Fit unconditionally on all tokens. For a fair comparison this is a
        cheat (it sees the val tokens), but we're only using the DIFFERENCE
        with and without n-gram, so the absolute BPB doesn't matter."""
        for i in range(len(tokens) - 1):
            prev = int(tokens[i])
            nxt = int(tokens[i + 1])
            self.counts[prev][nxt] += 1
            self.totals[prev] += 1
        # Precompute log-prob matrix with add-1 smoothing
        self._log_probs = np.full((self.V, self.V), -math.log(self.V),
                                    dtype=np.float32)
        for prev, counter in self.counts.items():
            total = self.totals[prev]
            denom = total + self.V
            for tok in range(self.V):
                c = counter.get(tok, 0)
                self._log_probs[prev, tok] = math.log((c + 1) / denom)

    def log_probs(self, prev_token: int) -> np.ndarray:
        return self._log_probs[prev_token]


# ---------- n-gram cache (for BOTH per-doc and global modes) ----------

class ExactNGramCache:
    """Exact counts with add-delta smoothing and order-K backoff."""

    def __init__(self, vocab_size: int, order: int, delta: float = 0.5,
                 min_ctx: int = 2):
        self.V = vocab_size
        self.K = order
        self.delta = delta
        self.min_ctx = min_ctx
        self.counts = {k: defaultdict(Counter) for k in range(1, order + 1)}
        self.totals = {k: defaultdict(int) for k in range(1, order + 1)}

    def add(self, history: list, tok: int):
        for k in range(1, self.K + 1):
            ctx = tuple(history[-(k - 1):]) if k > 1 else ()
            if k > 1 and len(history) < k - 1:
                continue
            self.counts[k][ctx][tok] += 1
            self.totals[k][ctx] += 1

    def clear(self):
        self.counts = {k: defaultdict(Counter) for k in range(1, self.K + 1)}
        self.totals = {k: defaultdict(int) for k in range(1, self.K + 1)}

    def log_probs(self, history: list) -> np.ndarray:
        """Full-vocab log-prob vector via backoff from K -> 1."""
        for k in range(self.K, 0, -1):
            ctx_len = k - 1
            if ctx_len == 0:
                ctx = ()
            elif len(history) < ctx_len:
                continue
            else:
                ctx = tuple(history[-ctx_len:])
            total = self.totals[k].get(ctx, 0)
            if total >= self.min_ctx:
                counter = self.counts[k].get(ctx)
                denom = total + self.delta * self.V
                vec = np.full(self.V, self.delta / denom, dtype=np.float32)
                if counter:
                    for tok, c in counter.items():
                        vec[tok] = (c + self.delta) / denom
                return np.log(vec)
        return np.full(self.V, -math.log(self.V), dtype=np.float32)


# ---------- experiments ----------

def simulate_bpb(tokens: np.ndarray, bigram: BigramLM,
                 ngram: ExactNGramCache | None,
                 alpha: float,
                 mode: str = "per_doc",
                 doc_boundaries: list | None = None,
                 update_after_score: bool = True,
                 verbose_every: int = 0) -> dict:
    """Measure per-token NLL under the blend `bigram + alpha * ngram`.

    Mode:
        "per_doc": reset the n-gram cache at each document boundary
        "global": never reset
        "none": no n-gram, just bigram

    Returns a dict with nll_sum, token_count, and derived mean loss + BPB.
    Note: since we're working on tokens not bytes, "BPB" here is actually
    bits-per-TOKEN. Useful for relative comparison only.
    """
    nll_sum = 0.0
    token_count = 0

    running_hist = []  # rolling context for n-gram lookups
    if ngram is not None:
        ngram.clear()

    N = len(tokens)
    for t in range(1, N):
        prev = int(tokens[t - 1])
        tgt = int(tokens[t])

        # Reset n-gram cache on doc boundary (per_doc mode)
        if mode == "per_doc" and doc_boundaries is not None and t in doc_boundaries:
            ngram.clear() if ngram is not None else None
            running_hist = []

        # Compute log-prob of target under the blend
        log_p_bigram = bigram.log_probs(prev)
        log_p_bigram_shifted = log_p_bigram - log_p_bigram.max()  # for stability
        if ngram is not None and alpha != 0.0:
            log_p_ng = ngram.log_probs(running_hist)
            # Blend as ADDITIVE LOGITS: logit = log_p_bigram + alpha*log_p_ngram
            # then softmax. We approximate logits ≈ log_p since bigram already
            # outputs log-probs.
            blended = log_p_bigram_shifted + alpha * log_p_ng
            # Softmax to get normalized distribution
            blended -= blended.max()
            e = np.exp(blended)
            p = e / e.sum()
            nll = -math.log(max(p[tgt], 1e-30))
        else:
            nll = -log_p_bigram[tgt]

        nll_sum += nll
        token_count += 1

        # Update the n-gram AFTER scoring (respects C3)
        if ngram is not None and update_after_score:
            ngram.add(running_hist, tgt)

        running_hist.append(tgt)
        if len(running_hist) > ngram.K - 1 if ngram is not None else 0:
            running_hist = running_hist[-(ngram.K - 1):]

        if verbose_every and token_count % verbose_every == 0:
            current = nll_sum / token_count / math.log(2)
            print(f"    ... {token_count}/{N - 1}  bits/tok={current:.4f}",
                  file=sys.stderr)

    mean_nll = nll_sum / max(token_count, 1)
    return {
        "nll_sum": nll_sum,
        "token_count": token_count,
        "mean_nll": mean_nll,
        "bits_per_tok": mean_nll / math.log(2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", type=Path,
                    default=Path("data/datasets/fineweb10B_sp1024/fineweb_val_000000.bin"))
    ap.add_argument("--max-tokens", type=int, default=2_000_000,
                    help="Limit tokens for speed (2M default, full run is 62M)")
    ap.add_argument("--orders", type=str, default="4,5")
    ap.add_argument("--alphas", type=str, default="0,0.1,0.3,0.5,1.0,1.5")
    args = ap.parse_args()

    print(f"Loading {args.val}...", file=sys.stderr)
    tokens = load_val_tokens(args.val)
    if args.max_tokens and args.max_tokens < len(tokens):
        tokens = tokens[:args.max_tokens]
    print(f"  using {len(tokens):,} tokens", file=sys.stderr)

    # Segment doc boundaries (indices into tokens where a new doc starts)
    bos = 1
    doc_starts = set()
    for i, t in enumerate(tokens):
        if t == bos:
            doc_starts.add(i + 1)

    vocab_size = 1024
    print("Fitting bigram baseline...", file=sys.stderr)
    t0 = time.time()
    bg = BigramLM(vocab_size=vocab_size)
    bg.fit(tokens)
    print(f"  bigram fit in {time.time() - t0:.1f}s", file=sys.stderr)

    # Baseline: bigram only, no n-gram contribution
    print("\n=== BASELINE: bigram only (no n-gram) ===", file=sys.stderr)
    t0 = time.time()
    base = simulate_bpb(tokens, bg, ngram=None, alpha=0.0, mode="none",
                         verbose_every=500_000)
    print(f"  time {time.time() - t0:.1f}s  bits/tok={base['bits_per_tok']:.5f}")
    baseline_bits = base["bits_per_tok"]

    orders = [int(x) for x in args.orders.split(",")]
    alphas = [float(x) for x in args.alphas.split(",")]

    print("\n=== PER-DOC CACHE vs GLOBAL CACHE — alpha sweep ===")
    rows = []
    rows.append(f"{'order':>5} {'mode':>8} {'alpha':>6} {'bits/tok':>10} {'delta':>10}")
    rows.append("-" * 50)
    for order in orders:
        for mode in ["per_doc", "global"]:
            for alpha in alphas:
                if alpha == 0.0 and mode != "global":
                    continue  # alpha=0 is mode-independent
                ng = ExactNGramCache(vocab_size=vocab_size, order=order,
                                      delta=0.5, min_ctx=2)
                t0 = time.time()
                res = simulate_bpb(tokens, bg, ngram=ng, alpha=alpha,
                                    mode=mode, doc_boundaries=doc_starts,
                                    update_after_score=True, verbose_every=0)
                dt = time.time() - t0
                delta = res['bits_per_tok'] - baseline_bits
                rows.append(f"{order:>5} {mode:>8} {alpha:>6.2f} {res['bits_per_tok']:>10.5f} {delta:>+10.5f}")
                print(f"  order={order} {mode} alpha={alpha:.2f}  "
                      f"bits/tok={res['bits_per_tok']:.5f}  delta={delta:+.5f}  ({dt:.0f}s)",
                      file=sys.stderr)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
