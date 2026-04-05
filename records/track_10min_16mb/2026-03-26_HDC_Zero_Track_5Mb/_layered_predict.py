"""Layered Prediction Pipeline — 5-layer HDC attention analog.

Provides the full layered_predict() function that fires when the rolling-hash
table has low confidence (table miss or count < threshold).

Architecture mirrors transformer layers but uses only HDC operations:

    Layer 0: Table lookup (caller handles — returns immediately if conf > 0.9)
    Layer 1: Direct S[p] WHT query + butterfly consistency check
    Layer 2: HDC attention (sem_bwd as key, sem_fwd as value)
    Layer 3: Multi-hop forward composition (3 hops, top-4 candidates)
    Layer 4: Backward validation on top-8 candidates
    Layer 5: Self-consistency check on top-3 after backward filtering

Compute budget per ambiguous token:
    WHT (any layer):          ~10K ops  → ~0.1 μs
    Attention key scan:       ~16K ops  → ~0.16 μs
    One hop (top-4 compose):  ~4K ops   → ~0.04 μs
    Backward validation (k=8):~8K ops   → ~0.08 μs
    Self-consistency (k=3):   ~48 ops   → ~0.0003 μs
    Full 5-layer pipeline:    ~50K ops  → ~0.5 μs/token

For 20M token eval with ~30% ambiguous tokens:
    6M × 0.5 μs = 3 seconds total — well within budget.

Key design invariants:
    - Layers only activate on table miss — zero overhead on confident predictions
    - Each layer's output is the next layer's input (residual refinement)
    - Suffix grammar score is an optional final morphological gate
    - All operations are O(vocab × W) or better — no quadratic terms
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np

# Import SRH utilities (WHT, bipolar, butterfly_consistency)
try:
    from _semantic_rolling_hash import (
        SemanticRollingHash,
        bipolar,
        wht_vectorised as wht,
        hamming_sim,
    )
    _SRH_AVAILABLE = True
except ImportError:
    _SRH_AVAILABLE = False
    SemanticRollingHash = None

    def bipolar(hv):
        bits = np.unpackbits(hv.view(np.uint8))
        return bits.astype(np.float32) * 2.0 - 1.0

    def wht(x):
        x = x.copy()
        n = len(x)
        h = 1
        while h < n:
            x_r = x.reshape(-1, 2 * h)
            u = x_r[:, :h].copy()
            v = x_r[:, h:].copy()
            x_r[:, :h] = u + v
            x_r[:, h:] = u - v
            x = x_r.reshape(-1)
            h *= 2
        return x

    def hamming_sim(a, b):
        xor = a ^ b
        bits = int(np.unpackbits(xor.view(np.uint8)).sum())
        return 1.0 - bits / (len(a) * 64)


# ---------------------------------------------------------------------------
# Softmax utility
# ---------------------------------------------------------------------------

def _softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Numerically stable softmax."""
    x = np.asarray(x, dtype=np.float32) / temperature
    x = x - x.max()
    e = np.exp(x)
    return e / (e.sum() + 1e-10)


# ---------------------------------------------------------------------------
# Majority vote (weighted binary bundling)
# ---------------------------------------------------------------------------

def majority_vote(
    vecs: List[np.ndarray],
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Weighted majority vote over a list of uint64 hypervectors.

    Each bit in the output is set if the weighted sum of that bit across
    all input vectors is positive (majority of weight voted 1).

    Parameters
    ----------
    vecs    : list of (W_UINT64,) uint64 — input hypervectors
    weights : (len(vecs),) float — vote weights (uniform if None)

    Returns
    -------
    (W_UINT64,) uint64 — majority-vote result
    """
    if not vecs:
        raise ValueError("majority_vote requires at least one vector")

    W = len(vecs[0])
    dim = W * 64

    if weights is None:
        weights = np.ones(len(vecs), dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)

    # Accumulate weighted bipolar votes
    accumulator = np.zeros(dim, dtype=np.float32)
    for vec, w in zip(vecs, weights):
        bits = np.unpackbits(vec.view(np.uint8)).astype(np.float32)
        bipolar_bits = bits * 2.0 - 1.0   # 0→-1, 1→+1
        accumulator += w * bipolar_bits

    # Majority vote: set bit where accumulator > 0
    result = np.zeros(W, dtype=np.uint64)
    positive = (accumulator > 0).reshape(W, 64)
    for block in range(W):
        for bit in range(64):
            if positive[block, bit]:
                result[block] |= np.uint64(1) << np.uint64(bit)

    return result


def majority_vote_fast(
    vecs: List[np.ndarray],
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Faster majority vote using numpy unpackbits/packbits.

    Equivalent to majority_vote() but avoids the inner Python loop.
    """
    if not vecs:
        raise ValueError("majority_vote_fast requires at least one vector")

    W = len(vecs[0])
    dim = W * 64

    if weights is None:
        weights = np.ones(len(vecs), dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)

    # Stack all vectors and unpack bits: (n_vecs, dim)
    stacked = np.stack([np.unpackbits(v.view(np.uint8)) for v in vecs], axis=0)
    bipolar_stacked = stacked.astype(np.float32) * 2.0 - 1.0   # (n_vecs, dim)

    # Weighted sum: (dim,)
    accumulator = (weights[:, None] * bipolar_stacked).sum(axis=0)

    # Pack back to uint64
    positive_bits = (accumulator > 0).astype(np.uint8)
    packed = np.packbits(positive_bits)   # (dim//8,) uint8
    return packed.view(np.uint64)         # (W_UINT64,) uint64


# ---------------------------------------------------------------------------
# Layer 1: Direct WHT prediction
# ---------------------------------------------------------------------------

def wht_predict_with_check(
    query: np.ndarray,
    codebook: np.ndarray,
    srh: Optional[SemanticRollingHash] = None,
    depth: int = 100,
) -> Tuple[int, float, str]:
    """WHT prediction with butterfly consistency check and regime classification.

    Parameters
    ----------
    query    : (W_UINT64,) uint64 — query hypervector (S[p] or refined query)
    codebook : (vocab_size, W_UINT64) uint64
    srh      : SemanticRollingHash — for butterfly_consistency and classify_regime
    depth    : int — accumulation depth (for SNR-scaled threshold)

    Returns
    -------
    (winner, confidence, regime)
    regime: 'clean', 'ambiguous', or 'noise'
    """
    vocab_size = codebook.shape[0]
    bipolar_q = bipolar(query)
    correlations = wht(bipolar_q)[:vocab_size] / float(len(bipolar_q))

    winner = int(np.argmax(correlations))
    winner_corr = float(correlations[winner])

    if srh is not None:
        regime, winner, confidence = srh.classify_regime(correlations, depth=depth)
    else:
        # Simple threshold without SRH
        consistency = _simple_butterfly_consistency(correlations, winner)
        if winner_corr > 0.1 and consistency > 0.7:
            regime, confidence = 'clean', winner_corr * consistency
        elif winner_corr > 0.05:
            regime, confidence = 'ambiguous', winner_corr * 0.5
        else:
            regime, confidence = 'noise', 0.0

    return winner, confidence, regime


def _simple_butterfly_consistency(correlations: np.ndarray, winner: int,
                                   n_levels: int = 10) -> float:
    """Butterfly consistency without SRH dependency."""
    vocab_size = len(correlations)
    winner_corr = abs(float(correlations[winner]))
    if winner_corr < 1e-8:
        return 0.0
    ratios = []
    for k in range(n_levels):
        partner = winner ^ (1 << k)
        if 0 <= partner < vocab_size:
            ratios.append(abs(float(correlations[partner])) / (winner_corr + 1e-8))
    if not ratios:
        return 1.0
    return max(0.0, 1.0 - max(ratios))


# ---------------------------------------------------------------------------
# Layer 2: HDC Attention (sem_bwd as key, sem_fwd as value)
# ---------------------------------------------------------------------------

def hdc_attention(
    query: np.ndarray,
    sem_fwd_matrix: np.ndarray,
    sem_bwd_matrix: np.ndarray,
    codebook: np.ndarray,
    vocab_size: int,
    W_UINT64: int,
    top_k: int = 4,
) -> np.ndarray:
    """HDC analog of transformer attention.

    Transformer: attention(q, K, V) = softmax(q @ K.T) @ V
    HDC analog:
        Query  = current semantic state (S[p] or refined query)
        Key    = sem_bwd[t] — "what contexts typically precede token t?"
        Value  = sem_fwd[t] — "what typically follows token t?"

    High key similarity → token t is consistent with this context.
    Value bundle → what the attended tokens predict next.

    Parameters
    ----------
    query          : (W_UINT64,) uint64 — current query state
    sem_fwd_matrix : (vocab_size, W_UINT64) uint64
    sem_bwd_matrix : (vocab_size, W_UINT64) uint64
    codebook       : (vocab_size, W_UINT64) uint64
    vocab_size     : int
    W_UINT64       : int
    top_k          : int — number of tokens to attend to

    Returns
    -------
    (W_UINT64,) uint64 — attention-refined query
    """
    # Step 1: key matching — which tokens' backward profiles match current context?
    # sem_bwd[t] encodes "what contexts typically precede t"
    # High similarity to query → t is consistent with this context
    key_sims = np.array([
        hamming_sim(query, sem_bwd_matrix[t])
        for t in range(vocab_size)
    ], dtype=np.float32)   # (vocab_size,)

    # Step 2: top-k attention weights
    top_k_indices = np.argsort(key_sims)[-top_k:]
    attn_weights = _softmax(key_sims[top_k_indices])

    # Step 3: weighted value aggregation
    # value = sem_fwd[t] — what each attended token predicts next
    value_vecs = [sem_fwd_matrix[t] for t in top_k_indices]
    value_bundle = majority_vote_fast(value_vecs, weights=attn_weights)

    # Step 4: residual correction — XOR in the delta between attention output and query
    # Acts like: query = query + alpha * (attention_output - query)
    correction = value_bundle ^ query
    # Apply correction with a confidence mask: only flip bits where attention is confident
    # Use the top attention weight as a proxy for confidence
    top_attn_conf = float(attn_weights.max())
    if top_attn_conf > 0.4:
        # Strong attention signal — apply full correction
        refined = value_bundle
    elif top_attn_conf > 0.2:
        # Moderate signal — blend via majority vote
        refined = majority_vote_fast([query, value_bundle], weights=np.array([0.7, 0.3]))
    else:
        # Weak signal — keep original query
        refined = query

    return refined


# ---------------------------------------------------------------------------
# Layer 3: Multi-hop forward composition
# ---------------------------------------------------------------------------

def multihop_layer(
    query: np.ndarray,
    sem_fwd_matrix: np.ndarray,
    codebook: np.ndarray,
    vocab_size: int,
    W_UINT64: int,
    n_hops: int = 3,
    top_k: int = 4,
    srh: Optional[SemanticRollingHash] = None,
) -> np.ndarray:
    """Multi-hop forward composition.

    Each hop asks: "what would typically follow the tokens my current query
    points to?" This disambiguates polysemous tokens by reasoning about
    what would logically follow each sense.

    Example: token "bank" is ambiguous between financial and river senses.
    - sem_fwd["bank" (financial)] → points toward "account", "loan", "interest"
    - sem_fwd["bank" (river)]     → points toward "river", "water", "fish"
    After 2 hops, the query has implicitly resolved which sense fits the context.

    Parameters
    ----------
    query          : (W_UINT64,) uint64 — current query state
    sem_fwd_matrix : (vocab_size, W_UINT64) uint64
    codebook       : (vocab_size, W_UINT64) uint64
    vocab_size     : int
    W_UINT64       : int
    n_hops         : int — number of forward hops
    top_k          : int — candidates per hop
    srh            : SemanticRollingHash — for WHT (optional)

    Returns
    -------
    (W_UINT64,) uint64 — multi-hop refined query
    """
    current_query = query.copy()

    for hop in range(n_hops):
        # Find current top-k predictions via WHT
        bipolar_q = bipolar(current_query)
        correlations = wht(bipolar_q)[:vocab_size] / float(len(bipolar_q))
        candidates = np.argsort(correlations)[-top_k:]
        candidate_weights = _softmax(correlations[candidates])

        # Compose: next query = weighted majority vote of sem_fwd[candidates]
        hop_vecs = [sem_fwd_matrix[t] for t in candidates]
        next_query = majority_vote_fast(hop_vecs, weights=candidate_weights)

        # Residual blend: 70% current, 30% hop result
        # Acts like: query = query + 0.3 * (next_query - query)
        current_query = majority_vote_fast(
            [current_query, next_query],
            weights=np.array([0.7, 0.3], dtype=np.float32)
        )

    return current_query


# ---------------------------------------------------------------------------
# Layer 4: Backward validation
# ---------------------------------------------------------------------------

def backward_validation(
    candidates: np.ndarray,
    S_p: np.ndarray,
    sem_bwd_matrix: np.ndarray,
    W_UINT64: int,
) -> np.ndarray:
    """Backward validation: check each candidate against accumulated history.

    For each candidate token t, checks:
        "Does t's backward profile match the current semantic state S_p?"

    High match → t was genuinely predicted by this context.
    Low match  → t is a noise artifact, even if sem_fwd points to it.

    This is the HDC analog of BERT's bidirectional attention.
    Cost: O(k × W) = O(4 × 16) = 64 ops for k=4.

    Parameters
    ----------
    candidates     : (k,) int — candidate token IDs
    S_p            : (W_UINT64,) uint64 — current semantic state
    sem_bwd_matrix : (vocab_size, W_UINT64) uint64
    W_UINT64       : int

    Returns
    -------
    (k,) float32 — softmax-normalised backward consistency scores
    """
    scores = np.array([
        hamming_sim(sem_bwd_matrix[int(t)], S_p)
        for t in candidates
    ], dtype=np.float32)

    return _softmax(scores)


# ---------------------------------------------------------------------------
# Layer 5: Self-consistency check (hypothetical forward chaining)
# ---------------------------------------------------------------------------

def self_consistency_check(
    candidate: int,
    S_p: np.ndarray,
    sem_fwd_matrix: np.ndarray,
    key: np.uint64,
    W_UINT64: int,
    codebook: np.ndarray,
    srh: Optional[SemanticRollingHash] = None,
) -> float:
    """Self-consistency: does predicting this candidate lead to a coherent next state?

    Hypothesis: if we predict token `candidate` here, the next semantic state
    would be S_hyp = S_p XOR (sem_fwd[candidate] * key).

    Check: does S_hyp produce a coherent WHT spectrum (peaked, not flat)?

    A coherent prediction produces a peaked WHT spectrum.
    Noise produces a flat spectrum.

    This is the MLP analog — reasoning about consequences without materialising them.
    Cost: O(W) for state update + O(vocab * log(vocab)) for WHT = ~10K ops.

    Parameters
    ----------
    candidate      : int — token ID to check
    S_p            : (W_UINT64,) uint64 — current semantic state
    sem_fwd_matrix : (vocab_size, W_UINT64) uint64
    key            : np.uint64 — HADAMARD_KEY[p]
    W_UINT64       : int
    codebook       : (vocab_size, W_UINT64) uint64
    srh            : SemanticRollingHash — for butterfly_consistency

    Returns
    -------
    float in [0, 1] — consistency score (1.0 = coherent, 0.0 = noise)
    """
    vocab_size = codebook.shape[0]

    # Hypothetical next state
    with np.errstate(over='ignore'):
        binding = sem_fwd_matrix[candidate] * key
    S_hyp = S_p ^ binding

    # WHT over hypothetical state
    bipolar_hyp = bipolar(S_hyp)
    hyp_correlations = wht(bipolar_hyp)[:vocab_size] / float(len(bipolar_hyp))

    hyp_winner = int(np.argmax(hyp_correlations))

    if srh is not None:
        consistency = srh.butterfly_consistency(hyp_correlations, hyp_winner)
    else:
        consistency = _simple_butterfly_consistency(hyp_correlations, hyp_winner)

    return consistency


# ---------------------------------------------------------------------------
# Full 5-layer pipeline
# ---------------------------------------------------------------------------

def layered_predict(
    S_p: np.ndarray,
    sem_fwd_matrix: np.ndarray,
    sem_bwd_matrix: np.ndarray,
    codebook: np.ndarray,
    keys: np.ndarray,
    p: int,
    vocab_size: int,
    W_UINT64: int,
    srh: Optional[SemanticRollingHash] = None,
    suffix_grammar=None,
    depth: int = 100,
    top_k_l4: int = 8,
    top_k_l5: int = 3,
) -> Tuple[int, float]:
    """Full 5-layer HDC prediction pipeline.

    Only called when the rolling-hash table has low confidence.
    High-confidence table hits return immediately from Layer 0 (caller handles).

    Parameters
    ----------
    S_p            : (W_UINT64,) uint64 — semantic state at position p
    sem_fwd_matrix : (vocab_size, W_UINT64) uint64
    sem_bwd_matrix : (vocab_size, W_UINT64) uint64
    codebook       : (vocab_size, W_UINT64) uint64
    keys           : (N,) uint64 — HADAMARD_KEY array
    p              : int — current position
    vocab_size     : int
    W_UINT64       : int
    srh            : SemanticRollingHash — for WHT/butterfly (optional but recommended)
    suffix_grammar : SuffixGrammarTable — morphological gate (optional)
    depth          : int — accumulation depth for SNR-scaled threshold
    top_k_l4       : int — candidates for backward validation (Layer 4)
    top_k_l5       : int — candidates for self-consistency (Layer 5)

    Returns
    -------
    (winner_token, confidence)
    confidence = 0.0 means noise — fall through to next layer in waterfall
    """
    key = keys[p] if p < len(keys) else np.uint64(0x9E3779B97F4A7C15)

    # ── Layer 1: Direct S[p] WHT query ──────────────────────────────────────
    query = S_p.copy()
    winner_L1, conf_L1, regime_L1 = wht_predict_with_check(
        query, codebook, srh=srh, depth=depth
    )

    if regime_L1 == 'clean' and conf_L1 > 0.5:
        # Strong clean signal — return immediately
        return winner_L1, conf_L1

    # ── Layer 2: HDC attention refinement ───────────────────────────────────
    try:
        query = hdc_attention(
            query, sem_fwd_matrix, sem_bwd_matrix, codebook,
            vocab_size, W_UINT64, top_k=4
        )
        winner_L2, conf_L2, regime_L2 = wht_predict_with_check(
            query, codebook, srh=srh, depth=depth
        )
    except Exception:
        winner_L2, conf_L2, regime_L2 = winner_L1, conf_L1, regime_L1

    if regime_L2 == 'clean' and conf_L2 > 0.4:
        return winner_L2, conf_L2

    # ── Layer 3: Multi-hop forward composition ───────────────────────────────
    try:
        query = multihop_layer(
            query, sem_fwd_matrix, codebook, vocab_size, W_UINT64,
            n_hops=3, top_k=4, srh=srh
        )
        winner_L3, conf_L3, regime_L3 = wht_predict_with_check(
            query, codebook, srh=srh, depth=depth
        )
    except Exception:
        winner_L3, conf_L3, regime_L3 = winner_L2, conf_L2, regime_L2

    # ── Layer 4: Backward validation on top-k candidates ────────────────────
    try:
        bipolar_q = bipolar(query)
        all_correlations = wht(bipolar_q)[:vocab_size] / float(len(bipolar_q))
        top_candidates = np.argsort(all_correlations)[-top_k_l4:]

        bwd_scores = backward_validation(top_candidates, S_p, sem_bwd_matrix, W_UINT64)
        fwd_scores = _softmax(all_correlations[top_candidates])

        # Geometric mean: both directions must agree
        combined_L4 = np.sqrt(np.maximum(fwd_scores * bwd_scores, 1e-10))
        best_L4_idx = int(np.argmax(combined_L4))
        winner_L4   = int(top_candidates[best_L4_idx])
        conf_L4     = float(combined_L4[best_L4_idx])

        # Top-3 after backward filtering for Layer 5
        top3_idx = np.argsort(combined_L4)[-top_k_l5:]
        final_candidates = top_candidates[top3_idx]
    except Exception:
        winner_L4 = winner_L3
        conf_L4   = conf_L3
        final_candidates = np.array([winner_L3])

    # ── Layer 5: Self-consistency check ─────────────────────────────────────
    try:
        consistency_scores = np.array([
            self_consistency_check(
                int(t), S_p, sem_fwd_matrix, key, W_UINT64, codebook, srh=srh
            )
            for t in final_candidates
        ], dtype=np.float32)

        # Re-score: backward score × consistency × forward correlation
        bipolar_q = bipolar(query)
        all_corr = wht(bipolar_q)[:vocab_size] / float(len(bipolar_q))

        final_scores = np.zeros(len(final_candidates), dtype=np.float32)
        for i, t in enumerate(final_candidates):
            fwd_s = max(0.0, float(all_corr[t]))
            bwd_s = hamming_sim(sem_bwd_matrix[int(t)], S_p)
            con_s = consistency_scores[i]
            final_scores[i] = fwd_s * bwd_s * con_s

        best_final_idx = int(np.argmax(final_scores))
        winner_L5      = int(final_candidates[best_final_idx])
        conf_L5        = float(final_scores[best_final_idx])
    except Exception:
        winner_L5 = winner_L4
        conf_L5   = conf_L4

    # ── Optional: suffix grammar gate ───────────────────────────────────────
    if suffix_grammar is not None and conf_L5 > 0.0:
        try:
            gram_score = suffix_grammar.suffix_grammar_score(winner_L5, S_p)
            # Blend: grammar score modulates confidence but doesn't override
            conf_L5 = conf_L5 * (0.5 + 0.5 * gram_score)
        except Exception:
            pass

    return winner_L5, conf_L5


# ---------------------------------------------------------------------------
# Diagonal-aware prediction (XOR orbit diagonals)
# ---------------------------------------------------------------------------

def diagonal_prediction(
    S_p: np.ndarray,
    R: np.ndarray,
    codebook: np.ndarray,
    recent_token: int,
    vocab_size: int,
    threshold: float = 0.55,
) -> Tuple[Optional[int], float]:
    """Predict using XOR orbit diagonal table R[k].

    R[k] encodes: "what semantic jump does XOR offset k represent?"
    For a BPE tokenizer with regularities in ID assignment, R[k] for small k
    encodes structured semantic relationships (morphological variants, etc.).

    At query time: which XOR diagonal is the current context traveling along?
    Prediction: recent_token XOR winning_k

    Parameters
    ----------
    S_p          : (W_UINT64,) uint64 — current semantic state
    R            : (vocab_size, W_UINT64) uint64 — XOR orbit diagonal table
    codebook     : (vocab_size, W_UINT64) uint64
    recent_token : int — most recent token (for XOR prediction)
    vocab_size   : int
    threshold    : float — minimum similarity to trust diagonal prediction

    Returns
    -------
    (candidate_token, confidence) or (None, 0.0) if below threshold
    """
    # Which XOR offset does the current semantic state best align with?
    diag_sims = np.array([
        hamming_sim(S_p, R[k]) for k in range(vocab_size)
    ], dtype=np.float32)

    winning_k = int(np.argmax(diag_sims))
    diag_conf = float(diag_sims[winning_k])

    if diag_conf > threshold:
        candidate = recent_token ^ winning_k
        if 0 <= candidate < vocab_size:
            return candidate, diag_conf

    return None, 0.0


# ---------------------------------------------------------------------------
# Skip-bigram diagonal prediction (lag-2 to lag-5)
# ---------------------------------------------------------------------------

def skip_bigram_predict(
    context_tokens: np.ndarray,
    sem_fwd_lags: Dict[int, np.ndarray],
    codebook: np.ndarray,
    vocab_size: int,
    W_UINT64: int,
    srh: Optional[SemanticRollingHash] = None,
    min_conf: float = 0.15,
) -> Tuple[Optional[int], float]:
    """Predict using skip-bigram lag vectors (lag-2 to lag-5).

    Each lag-k vector encodes: "what token typically follows the token
    k positions back?" Lag-2 often outperforms lag-1 for ambiguous tokens
    because it skips over function words and captures phrase-level structure.

    Cross-lag consensus: if multiple lags agree on the same candidate,
    confidence is boosted logarithmically.

    Parameters
    ----------
    context_tokens : (ctx_len,) int — recent context tokens (most recent last)
    sem_fwd_lags   : Dict[int, np.ndarray] — {lag: (vocab_size, W_UINT64) uint64}
    codebook       : (vocab_size, W_UINT64) uint64
    vocab_size     : int
    W_UINT64       : int
    srh            : SemanticRollingHash — for WHT (optional)
    min_conf       : float — minimum per-lag confidence to count as a vote

    Returns
    -------
    (winner_token, confidence) or (None, 0.0) if no confident prediction
    """
    votes: Dict[int, float] = {}
    ctx_len = len(context_tokens)

    for lag, lag_matrix in sem_fwd_lags.items():
        if lag > ctx_len:
            continue

        ctx_tok = int(context_tokens[-(lag)])
        if ctx_tok < 0 or ctx_tok >= vocab_size:
            continue

        lag_vec = lag_matrix[ctx_tok]   # (W_UINT64,) uint64

        # WHT prediction over lag vector
        bipolar_lag = bipolar(lag_vec)
        correlations = wht(bipolar_lag)[:vocab_size] / float(len(bipolar_lag))
        candidate = int(np.argmax(correlations))
        conf = float(correlations[candidate])

        # Weight by 1/lag: recent lags trusted more than distant ones
        weighted_conf = conf / float(lag)
        if weighted_conf > min_conf:
            votes[candidate] = votes.get(candidate, 0.0) + weighted_conf

    if not votes:
        return None, 0.0

    winner = max(votes, key=lambda k: votes[k])
    n_agreeing = sum(1 for v in votes.values() if v > min_conf * 0.5)
    # Logarithmic confidence boost for cross-lag consensus
    confidence = votes[winner] * float(np.log1p(n_agreeing))

    return winner, confidence


# ---------------------------------------------------------------------------
# Combined diagonal-aware prediction
# ---------------------------------------------------------------------------

def diagonal_aware_predict(
    S_p: np.ndarray,
    context_tokens: np.ndarray,
    sem_fwd_lags: Dict[int, np.ndarray],
    R: Optional[np.ndarray],
    codebook: np.ndarray,
    vocab_size: int,
    W_UINT64: int,
    srh: Optional[SemanticRollingHash] = None,
) -> Tuple[Optional[int], float]:
    """Combined diagonal-aware prediction using skip-bigrams and XOR orbit diagonals.

    Collects evidence from:
    1. Skip-bigram lags 2-5 (phrase-level structure)
    2. XOR orbit diagonal table R[k] (relationship-type prediction)

    Cross-diagonal consensus boosts confidence logarithmically.

    Parameters
    ----------
    S_p            : (W_UINT64,) uint64 — current semantic state
    context_tokens : (ctx_len,) int — recent context tokens (most recent last)
    sem_fwd_lags   : Dict[int, np.ndarray] — {lag: (vocab_size, W_UINT64) uint64}
    R              : (vocab_size, W_UINT64) uint64 — XOR orbit table (or None)
    codebook       : (vocab_size, W_UINT64) uint64
    vocab_size     : int
    W_UINT64       : int
    srh            : SemanticRollingHash — for WHT (optional)

    Returns
    -------
    (winner_token, confidence) or (None, 0.0)
    """
    votes: Dict[int, float] = {}

    # Skip-bigram votes
    if sem_fwd_lags:
        skip_winner, skip_conf = skip_bigram_predict(
            context_tokens, sem_fwd_lags, codebook, vocab_size, W_UINT64, srh=srh
        )
        if skip_winner is not None and skip_conf > 0.0:
            votes[skip_winner] = votes.get(skip_winner, 0.0) + skip_conf

    # XOR orbit diagonal vote
    if R is not None and len(context_tokens) > 0:
        recent_token = int(context_tokens[-1])
        orbit_winner, orbit_conf = diagonal_prediction(
            S_p, R, codebook, recent_token, vocab_size
        )
        if orbit_winner is not None:
            votes[orbit_winner] = votes.get(orbit_winner, 0.0) + orbit_conf

    if not votes:
        return None, 0.0

    winner = max(votes, key=lambda k: votes[k])
    n_agreeing = sum(1 for v in votes.values() if v > 0.1)
    confidence = votes[winner] * float(np.log1p(n_agreeing))

    return winner, confidence