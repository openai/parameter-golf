"""Trigram side-memory pack & lookup.

Built once at the end of training from a slice of the train shard. Lives as a
set of buffers on the GPT module, riding through int8 serialization. At
inference time, model.forward blends model log-probs with a trigram-with-bigram-
backoff predictor in log-prob space.

Storage layout (all int / fp32; integers go through quantize_state_dict_int8's
passthrough path):

  trigram_keys             (n_ctx,) int32, sorted ascending
                           key = prev2 * VOCAB + prev1
  trigram_offsets          (n_ctx + 1,) int32 — CSR-style offsets
  trigram_next             (n_total,) int16 — next-token id per top-K entry
  trigram_log2p_quant      (n_total,) int8 — full blended log-prob for that
                           (ctx, next_tok), including backoff*p_b[next_tok]
  trigram_log2p_scale      scalar fp32
  trigram_log2p_offset     scalar fp32
  trigram_log2_backoff_quant   (n_ctx,) int8 — log2(backoff_mass) per matched ctx
  trigram_log2_backoff_scale   scalar fp32
  trigram_log2_backoff_offset  scalar fp32
  bigram_log2p_quant       (VOCAB, VOCAB) int8 — log2 P(next | prev1)
  bigram_log2p_scales      (VOCAB,) fp32 — per-row scale
  bigram_log2p_offsets     (VOCAB,) fp32 — per-row offset
  unigram_log2p            (VOCAB,) fp32 — fallback for t == 0
  trigram_blend_alpha      scalar fp32 — model weight for blend

Forward blend math (per position (b, t)):
  if t == 0:               trigram_lp = unigram_log2p
  if t == 1:               trigram_lp = bigram_log2p[x[b, 0], :]
  if t >= 2:
    key = x[b, t-2] * V + x[b, t-1]
    idx = searchsorted(trigram_keys, key)
    if matched (key == trigram_keys[idx]):
      trigram_lp = bigram_log2p[x[b, t-1], :] + log2_backoff[idx]
      for each (next_tok, log2p_full) in entries[idx]:
        trigram_lp[next_tok] = log2p_full
    else:
      trigram_lp = bigram_log2p[x[b, t-1], :]

  log_blend = logsumexp([log2(alpha) + model_log2p,
                         log2(1 - alpha) + trigram_lp])
  return cross_entropy on log_blend (i.e. -log_blend[b, t, target])
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable

import numpy as np
import torch
from torch import Tensor


# ---------- Pack build ----------

def _quantize_to_int8(arr: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Affine quantize float array to int8 via min/max.

    int8 range [-127, 127] (matches train_gpt's clipping convention).
    Returns (q, scale, offset) where dequant_value = q * scale + offset.
    """
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        return arr.astype(np.int8), 1.0, 0.0
    lo = float(arr.min())
    hi = float(arr.max())
    if hi - lo < 1e-12:
        # constant array
        return np.zeros_like(arr, dtype=np.int8), 1.0, lo
    scale = (hi - lo) / 254.0
    offset = lo + 127.0 * scale  # dequant: q*scale + offset gives lo at q=-127, hi at q=+127
    q = np.clip(np.round((arr - offset) / scale), -127, 127).astype(np.int8)
    return q, float(scale), float(offset)


def _quantize_per_row_int8(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-row affine int8 quantization. mat shape (R, C)."""
    R, C = mat.shape
    q = np.zeros((R, C), dtype=np.int8)
    scales = np.ones((R,), dtype=np.float32)
    offsets = np.zeros((R,), dtype=np.float32)
    for r in range(R):
        row = mat[r].astype(np.float32)
        lo = float(row.min())
        hi = float(row.max())
        if hi - lo < 1e-12:
            q[r] = 0
            scales[r] = 1.0
            offsets[r] = lo
            continue
        scale = (hi - lo) / 254.0
        offset = lo + 127.0 * scale
        q[r] = np.clip(np.round((row - offset) / scale), -127, 127).astype(np.int8)
        scales[r] = scale
        offsets[r] = offset
    return q, scales, offsets


def build_trigram_pack(
    train_tokens: np.ndarray,
    vocab_size: int,
    top_k: int,
    min_count: int,
    bigram_alpha: float = 0.01,
    discount: float = 0.5,
    K: int = 3,
    top_n_ctx: int = 0,
    per_context_alpha: bool = False,
    alpha_tau: float = 0.5,
    alpha_thresh: float = 5.0,
    alpha_min: float = 0.5,
    alpha_max: float = 0.95,
) -> dict[str, Tensor]:
    """Build K-gram + bigram + unigram tables from a token stream.

    Args:
        train_tokens: 1-D int32/int64 numpy array of token ids.
        vocab_size: vocabulary size V.
        top_k: keep top-K next tokens per matched context.
        min_count: prune contexts whose total count < min_count.
        bigram_alpha: add-alpha smoothing for bigram fallback.
        discount: stupid-backoff discount d for K-gram counts.
        K: K-gram order. K=3 is the original trigram path (key = p2*V + p1);
            K=4 is the 4-gram path (key = p3*V^2 + p2*V + p1). For K=3 with
            top_n_ctx=0 the build is byte-identical to the original K=3 code.
        top_n_ctx: keep only the top-N contexts by total count (descending).
            0 = keep all (original behavior).
        per_context_alpha: 0074. When True, also computes a per-context α
            weight (the model weight in the inference blend) from each
            context's empirical entropy `H = -sum(p_top log2 p_top)
            + (1 - sum p_top) * log2(V)` (top-K + uniform-residual approx).
            α = clip(sigmoid(alpha_tau * (H - alpha_thresh)), alpha_min, alpha_max).
            Adds 3 buffers to the output dict: trigram_alpha_quant (int8,
            n_ctx), trigram_alpha_scale, trigram_alpha_offset (scalars).
            When False (default), no α buffers are emitted; pack is
            byte-identical to the original (0067/0068/0069) build.
        alpha_tau, alpha_thresh, alpha_min, alpha_max: per-context α tuning
            (only used when per_context_alpha=True).

    Returns: dict of cpu tensors ready to register as model buffers.
    """
    train = np.asarray(train_tokens, dtype=np.int64)
    V = int(vocab_size)
    N = train.size
    if K not in (3, 4):
        raise ValueError(f"K must be 3 or 4, got {K}")
    print(
        f"[trigram] building from {N:,} tokens, V={V}, K={K}, top_k={top_k}, "
        f"min_count={min_count}, top_n_ctx={top_n_ctx}",
        flush=True,
    )

    # Unigram (with add-1 smoothing for t=0 fallback)
    unigram_counts = np.bincount(train, minlength=V).astype(np.float64)
    unigram_total = float(unigram_counts.sum())
    unigram_p = (unigram_counts + 1.0) / (unigram_total + V)
    unigram_log2p = np.log2(np.maximum(unigram_p, 1e-12)).astype(np.float32)

    # Bigram (counts → add-alpha smoothed, log2)
    bigram_counts = np.zeros((V, V), dtype=np.float64)
    # Vectorized count via 1-D indexing trick
    flat = train[:-1] * V + train[1:]
    bg_flat_counts = np.bincount(flat, minlength=V * V).astype(np.float64)
    bigram_counts = bg_flat_counts.reshape(V, V)
    row_totals = bigram_counts.sum(axis=1, keepdims=True)
    bigram_p = (bigram_counts + bigram_alpha) / (row_totals + bigram_alpha * V)
    bigram_log2p = np.log2(np.maximum(bigram_p, 1e-12)).astype(np.float32)
    print(f"[trigram] bigram done (mean log2p={bigram_log2p.mean():.3f})", flush=True)

    # K-gram counts: build per-context next-token distributions.
    # For K=3: key = prev2 * V + prev1.   Max key = V^2 - 1 = 2^20 - 1, fits in int32.
    # For K=4: key = prev3 * V^2 + prev2 * V + prev1.  Max key = V^3 - 1 = 2^30 - 1, fits in int32.
    print(f"[trigram] counting {K}-grams ...", flush=True)
    if K == 3:
        p2 = train[:-2]
        p1 = train[1:-1]
        nxt = train[2:]
        keys = (p2.astype(np.int64) * V + p1.astype(np.int64))
    else:  # K == 4
        p3 = train[:-3]
        p2 = train[1:-2]
        p1 = train[2:-1]
        nxt = train[3:]
        keys = (
            p3.astype(np.int64) * (V * V)
            + p2.astype(np.int64) * V
            + p1.astype(np.int64)
        )
    # Group by key. Sort once, then split.
    order = np.argsort(keys, kind="stable")
    keys_sorted = keys[order]
    nxt_sorted = nxt[order]
    # Find boundaries
    # diff != 0 marks new run
    if keys_sorted.size == 0:
        boundaries = np.array([0], dtype=np.int64)
    else:
        change = np.concatenate(([True], keys_sorted[1:] != keys_sorted[:-1]))
        boundaries = np.flatnonzero(change)
    n_unique_ctx_raw = boundaries.size
    print(f"[trigram] unique {K}-gram contexts (raw): {n_unique_ctx_raw:,}", flush=True)

    # Walk runs to compute per-context totals (for top-N pruning) and starts/ends.
    starts = boundaries
    ends = np.concatenate((boundaries[1:], [keys_sorted.size]))
    totals_per_ctx = (ends - starts).astype(np.int64)

    # top-N context pruning (by total count desc). When top_n_ctx == 0, keep all.
    if top_n_ctx > 0 and top_n_ctx < n_unique_ctx_raw:
        # argsort by -count (stable for tie-determinism). Take first top_n_ctx.
        order_topn = np.argsort(-totals_per_ctx, kind="stable")[:top_n_ctx]
        kept_mask = np.zeros(n_unique_ctx_raw, dtype=bool)
        kept_mask[order_topn] = True
        n_pruned_topn = int(n_unique_ctx_raw - top_n_ctx)
        print(
            f"[trigram] top-N pruning: keeping top {top_n_ctx:,} contexts by count "
            f"(pruned {n_pruned_topn:,} below threshold)",
            flush=True,
        )
    else:
        kept_mask = np.ones(n_unique_ctx_raw, dtype=bool)

    kept_keys: list[int] = []
    kept_offsets: list[int] = [0]
    kept_next: list[int] = []
    kept_log2p_full: list[float] = []
    kept_log2_backoff: list[float] = []
    kept_alpha: list[float] = []  # per-context α weight (model weight); only populated when per_context_alpha=True
    cumulative = 0
    n_pruned = 0
    log2_V = math.log2(float(V))
    for ci, (s, e) in enumerate(zip(starts, ends)):
        if not kept_mask[ci]:
            n_pruned += 1
            continue
        run_nxt = nxt_sorted[s:e]
        total = int(e - s)
        if total < min_count:
            n_pruned += 1
            continue
        # Count next tokens in this run
        run_counts = np.bincount(run_nxt, minlength=V)
        # Pick top-K
        if top_k >= V:
            top_idx = np.argsort(-run_counts)
            top_idx = top_idx[run_counts[top_idx] > 0]
        else:
            # argpartition for top-K
            top_idx_unsorted = np.argpartition(-run_counts, min(top_k, V - 1))[:top_k]
            # sort by descending count among top-K, drop zero counts
            top_idx = top_idx_unsorted[np.argsort(-run_counts[top_idx_unsorted])]
            top_idx = top_idx[run_counts[top_idx] > 0]
        if top_idx.size == 0:
            n_pruned += 1
            continue
        key = int(keys_sorted[s])
        # Decode prev1 (most-recent context token) from key — same low-V residue
        # for both K=3 (key % V) and K=4 (key % V).
        prev1 = key % V
        # Compute discounted top-K probs
        top_counts = run_counts[top_idx].astype(np.float64)
        sum_p3 = float(np.maximum(top_counts - discount, 0.0).sum() / total)
        backoff = max(1.0 - sum_p3, 1e-12)
        log2_backoff = math.log2(backoff)
        # Bigram backoff probs for these top-K next-toks
        p_b_top = (bigram_counts[prev1, top_idx] + bigram_alpha) / (row_totals[prev1, 0] + bigram_alpha * V)
        # Full blended log-prob per top-K entry
        p_3 = np.maximum(top_counts - discount, 0.0) / total
        p_full = p_3 + backoff * p_b_top
        log2p_full = np.log2(np.maximum(p_full, 1e-12))
        # Write
        kept_keys.append(key)
        for j in range(top_idx.size):
            kept_next.append(int(top_idx[j]))
            kept_log2p_full.append(float(log2p_full[j]))
        cumulative += int(top_idx.size)
        kept_offsets.append(cumulative)
        kept_log2_backoff.append(log2_backoff)
        if per_context_alpha:
            # Per-context entropy from top-K kept probabilities + uniform residual
            # over the rest of the vocabulary. This matches the offline reference
            # in scratch/blend_probe/poe_blend.py.
            sum_top = float(p_full.sum())
            residual = max(1.0 - sum_top, 0.0)
            h_top = float(-(p_full * np.log2(np.maximum(p_full, 1e-12))).sum())
            h_residual = residual * log2_V if residual > 0 else 0.0
            h_total = h_top + h_residual
            # alpha = sigmoid(tau * (entropy - thresh)), clipped to [alpha_min, alpha_max].
            # Higher entropy (uncertain trigram) → higher α (trust model more).
            x = alpha_tau * (h_total - alpha_thresh)
            sig = 1.0 / (1.0 + math.exp(-x))
            alpha_val = max(alpha_min, min(alpha_max, sig))
            kept_alpha.append(float(alpha_val))
    n_ctx = len(kept_keys)
    print(
        f"[trigram] kept {n_ctx:,} contexts ({n_pruned:,} pruned), "
        f"total entries: {cumulative:,}",
        flush=True,
    )

    # kept_keys is already sorted ascending because we iterated runs in
    # keys_sorted order and the top-N mask preserves that order.

    # Quantize trigram log2p (full) and log2 backoff
    log2p_arr = np.asarray(kept_log2p_full, dtype=np.float32)
    log2p_q, log2p_scale, log2p_offset = _quantize_to_int8(log2p_arr)
    backoff_arr = np.asarray(kept_log2_backoff, dtype=np.float32)
    backoff_q, backoff_scale, backoff_offset = _quantize_to_int8(backoff_arr)

    # Per-row int8 quantize the bigram log2p table
    bigram_q, bigram_scales, bigram_offsets = _quantize_per_row_int8(bigram_log2p)

    # Pack as torch tensors, all on CPU. dtype matters for the int8 passthrough.
    pack: dict[str, Tensor] = {
        "trigram_keys": torch.from_numpy(np.asarray(kept_keys, dtype=np.int32)).contiguous(),
        "trigram_offsets": torch.from_numpy(np.asarray(kept_offsets, dtype=np.int32)).contiguous(),
        "trigram_next": torch.from_numpy(np.asarray(kept_next, dtype=np.int16)).contiguous(),
        "trigram_log2p_quant": torch.from_numpy(log2p_q).contiguous(),
        "trigram_log2p_scale": torch.tensor(log2p_scale, dtype=torch.float32),
        "trigram_log2p_offset": torch.tensor(log2p_offset, dtype=torch.float32),
        "trigram_log2_backoff_quant": torch.from_numpy(backoff_q).contiguous(),
        "trigram_log2_backoff_scale": torch.tensor(backoff_scale, dtype=torch.float32),
        "trigram_log2_backoff_offset": torch.tensor(backoff_offset, dtype=torch.float32),
        "bigram_log2p_quant": torch.from_numpy(bigram_q).contiguous(),
        "bigram_log2p_scales": torch.from_numpy(bigram_scales).contiguous(),
        "bigram_log2p_offsets": torch.from_numpy(bigram_offsets).contiguous(),
        "unigram_log2p": torch.from_numpy(unigram_log2p).contiguous(),
    }
    if per_context_alpha:
        # Quantize per-context α to int8. Tiny pack: n_ctx ≤ 200K → ~200 KB.
        alpha_arr = np.asarray(kept_alpha, dtype=np.float32)
        alpha_q, alpha_scale, alpha_offset = _quantize_to_int8(alpha_arr)
        pack["trigram_alpha_quant"] = torch.from_numpy(alpha_q).contiguous()
        pack["trigram_alpha_scale"] = torch.tensor(alpha_scale, dtype=torch.float32)
        pack["trigram_alpha_offset"] = torch.tensor(alpha_offset, dtype=torch.float32)
        print(
            f"[trigram] per-context α: n={len(kept_alpha):,} "
            f"min={alpha_arr.min():.3f} max={alpha_arr.max():.3f} mean={alpha_arr.mean():.3f}",
            flush=True,
        )
    return pack


def pack_byte_size(pack: dict[str, Tensor]) -> int:
    return sum(int(t.numel()) * int(t.element_size()) for t in pack.values())


def estimate_brotli_size(pack: dict[str, Tensor]) -> int:
    """Estimate brotli compressed size by actually brotli-compressing a torch.save dump."""
    import io
    import brotli
    buf = io.BytesIO()
    torch.save(pack, buf)
    return len(brotli.compress(buf.getvalue(), quality=11))


# ---------- Forward blend (used by GPT.forward) ----------

def trigram_blend_loss(
    model_log_softmax: Tensor,  # (B, L, V) — natural log
    target_ids: Tensor,          # (B, L) int64
    input_ids: Tensor,           # (B, L) int64
    *,
    trigram_keys: Tensor,        # (n_ctx,) int32
    trigram_offsets: Tensor,     # (n_ctx+1,) int32
    trigram_next: Tensor,        # (n_total,) int16
    trigram_log2p_quant: Tensor, # (n_total,) int8
    trigram_log2p_scale: Tensor, # scalar fp32
    trigram_log2p_offset: Tensor,# scalar fp32
    trigram_log2_backoff_quant: Tensor,  # (n_ctx,) int8
    trigram_log2_backoff_scale: Tensor,  # scalar
    trigram_log2_backoff_offset: Tensor, # scalar
    bigram_log2p_quant: Tensor,  # (V, V) int8
    bigram_log2p_scales: Tensor, # (V,) fp32
    bigram_log2p_offsets: Tensor,# (V,) fp32
    unigram_log2p: Tensor,       # (V,) fp32
    blend_alpha: float,
    vocab_size: int,
    K: int = 3,
    conf_gate_threshold: float = -1e9,
) -> Tensor:
    """Compute mean blended cross-entropy loss.

    K=3 is the original trigram path (key = prev2 * V + prev1, applied at
    positions t >= 2). K=4 is the 4-gram path (key = prev3 * V^2 + prev2 * V
    + prev1, applied at t >= 3). For positions where the full K-context isn't
    available (t < K-1), the bigram fallback (or unigram for t == 0) is used.

    0076 confidence-gated blend (`conf_gate_threshold`): when the model's
    max log2-prob at a position is ABOVE the threshold (model is confident),
    skip the blend at that position and use model log-probs alone. When below
    (model is uncertain), use the existing blend. Default -1e9 disables
    gating (always blend), which is byte-identical to the parent 0074 path.

    Returns a scalar Tensor (natural log units, to match F.cross_entropy convention).
    """
    LN2 = math.log(2.0)
    device = model_log_softmax.device
    B, L, V = model_log_softmax.shape
    assert V == vocab_size
    if K not in (3, 4):
        raise ValueError(f"K must be 3 or 4, got {K}")

    # Build the (B, L, V) trigram log2-prob tensor.
    #
    # Indexing convention (matches scratch/blend_probe/ offline analysis):
    #   target_ids[b, t] is the token to predict at logical position t in the
    #   batch. The available "input window" preceding that prediction is
    #   input_ids[b, 0..t]. The most recent input is input_ids[b, t]; the
    #   token before that is input_ids[b, t-1].
    #   For target_ids[b, t]:
    #     t == 0: only one input token (input_ids[b, 0]) — use unigram fallback
    #             (matches offline convention where prev1 < 0 -> unigram).
    #     t == 1: prev1 = input_ids[b, 1] (most recent) — use bigram only
    #             (matches offline convention where prev2 < 0 -> bigram only).
    #     K=3: t >= 2: prev2 = input_ids[b, t-1], prev1 = input_ids[b, t] — full trigram.
    #     K=4: t == 2: bigram only (prev3 < 0 → fall back).
    #          t >= 3: prev3 = input_ids[b, t-2], prev2 = input_ids[b, t-1], prev1 = input_ids[b, t].
    #
    # Decode bigram fully into fp32, then index. (V, V) int8 -> fp32 = 1024*1024*4 = 4 MB temp;
    # acceptable on MPS for one-shot eval.
    bigram_dq = bigram_log2p_quant.to(torch.float32) * bigram_log2p_scales.unsqueeze(1) + bigram_log2p_offsets.unsqueeze(1)
    # (V, V) — row r is log2 P(next | prev1=r)

    # Trigram dq pieces (small — keep as fp32 buffers in scope)
    tri_log2p_full = trigram_log2p_quant.to(torch.float32) * trigram_log2p_scale + trigram_log2p_offset  # (n_total,)
    tri_log2_backoff = trigram_log2_backoff_quant.to(torch.float32) * trigram_log2_backoff_scale + trigram_log2_backoff_offset  # (n_ctx,)

    # Initialize trigram_lp[b, t, :]:
    #   - For t >= 1: bigram_dq[input_ids[b, t], :]
    #   - For t == 0: unigram_log2p (overwritten below)
    # We index the bigram with input_ids[b, t] directly (this is "prev1" for
    # predicting target_ids[b, t]).
    trigram_lp = bigram_dq[input_ids.long()]  # (B, L, V)
    # Overwrite t == 0 with unigram fallback
    if L >= 1:
        trigram_lp[:, 0, :] = unigram_log2p

    # K-gram lookup positions: t >= K-1 (i.e. K-1 prior input tokens available).
    # For K=3: positions t in [2, L). For K=4: positions t in [3, L).
    t_start = K - 1
    if L > t_start:
        n_pos = L - t_start  # number of output positions covered by the lookup
        if K == 3:
            # For position t in [2, L): prev2 = input_ids[t-1], prev1 = input_ids[t].
            prev2_slice = input_ids[:, 1:-1].long()      # input[1..L-2] -- prev2 at output positions 2..L-1
            prev1_slice = input_ids[:, 2:].long()        # input[2..L-1] -- prev1 at output positions 2..L-1
            keys = prev2_slice * vocab_size + prev1_slice   # int64 (B, L-2)
        else:  # K == 4
            # For position t in [3, L): prev3 = input_ids[t-2], prev2 = input_ids[t-1], prev1 = input_ids[t].
            prev3_slice = input_ids[:, 1:-2].long()     # input[1..L-3] -- prev3 at output positions 3..L-1
            prev2_slice = input_ids[:, 2:-1].long()     # input[2..L-2] -- prev2 at output positions 3..L-1
            prev1_slice = input_ids[:, 3:].long()       # input[3..L-1] -- prev1 at output positions 3..L-1
            keys = (
                prev3_slice * (vocab_size * vocab_size)
                + prev2_slice * vocab_size
                + prev1_slice
            )  # int64 (B, L-3)
        keys_flat = keys.reshape(-1).to(torch.int32)    # match trigram_keys dtype

        # searchsorted
        idx = torch.searchsorted(trigram_keys, keys_flat)  # int64 (Bn,)
        idx = torch.clamp(idx, max=int(trigram_keys.numel()) - 1)
        matched = trigram_keys[idx] == keys_flat  # bool (Bn,)

        # For matched positions, add log2_backoff to bigram
        log2_backoff_per_pos = torch.where(
            matched,
            tri_log2_backoff[idx.long()],
            torch.zeros((), dtype=torch.float32, device=device),
        )  # (Bn,)
        # log2_backoff shape (Bn,) -> reshape to (B, n_pos) and add to trigram_lp[:, t_start:, :]
        log2_backoff_per_pos_2d = log2_backoff_per_pos.reshape(B, n_pos)
        trigram_lp[:, t_start:, :] = trigram_lp[:, t_start:, :] + log2_backoff_per_pos_2d.unsqueeze(-1)

        # For matched positions, overwrite the top-K next-tok slots with the full log2p.
        # Gather entries: for each matched position, we need range entries[idx]:entries[idx+1].
        # This is jagged; do it as a flat scatter using global entry indices.
        Bn = idx.numel()
        # Per-position slice [start, end)
        start_per_pos = trigram_offsets[idx.long()].long()        # (Bn,)
        end_per_pos = trigram_offsets[(idx + 1).long()].long()    # (Bn,)
        # Build per-position entry indices as a flat list, with a parallel "pos_idx" tag.
        seg_lens = torch.where(matched, end_per_pos - start_per_pos, torch.zeros_like(end_per_pos))  # (Bn,)
        total_entries = int(seg_lens.sum().item())
        if total_entries > 0:
            # pos_id_per_entry: which (b, t) flat index this entry belongs to
            pos_id_per_entry = torch.repeat_interleave(
                torch.arange(Bn, device=device, dtype=torch.long), seg_lens
            )
            # entry_global_idx: for each pos, expand range(start, end). Compute via cumulative offsets.
            cum_lens = torch.cumsum(seg_lens, dim=0)  # (Bn,)
            seg_start_in_flat = cum_lens - seg_lens  # (Bn,)
            within = torch.arange(total_entries, device=device, dtype=torch.long) - seg_start_in_flat[pos_id_per_entry]
            entry_global_idx = start_per_pos[pos_id_per_entry] + within  # (total_entries,)
            # Read values
            entry_next = trigram_next[entry_global_idx].long()  # (total_entries,)
            entry_log2p = tri_log2p_full[entry_global_idx]      # (total_entries,)
            # Scatter into trigram_lp[:, t_start:, :]. pos_id was constructed over (B, n_pos)
            # in row-major order, so:
            pos_b = pos_id_per_entry // n_pos
            pos_t_off = pos_id_per_entry % n_pos
            trigram_lp[pos_b, pos_t_off + t_start, entry_next] = entry_log2p

    # Now blend in log-prob space.
    # model_log_softmax is in NATS. Convert to log2.
    model_log2p = model_log_softmax / LN2  # (B, L, V) log2
    log2_alpha = math.log2(blend_alpha) if blend_alpha > 0 else -1e30
    log2_one_minus_alpha = math.log2(1.0 - blend_alpha) if blend_alpha < 1.0 else -1e30
    # logsumexp in log2 space: log2(2^a + 2^b) = max(a,b) + log2(1 + 2^{-|a-b|})
    a = log2_alpha + model_log2p
    b = log2_one_minus_alpha + trigram_lp
    m = torch.maximum(a, b)
    log2_blend = m + torch.log2(torch.exp2(a - m) + torch.exp2(b - m))  # (B, L, V)
    # Cross-entropy on the blended distribution: -log P(target). Need natural log.
    nat_log_blend = log2_blend * LN2  # (B, L, V)
    # Gather targets — log-prob of true target at each position.
    nat_log_blend_at_target = nat_log_blend.gather(-1, target_ids.long().unsqueeze(-1)).squeeze(-1)  # (B, L)

    # 0076: confidence gate. When model max log2p > threshold (model confident),
    # use model log-prob at target alone; otherwise keep the blended one.
    if conf_gate_threshold > -1e8:
        model_max_log2p = model_log2p.max(dim=-1).values  # (B, L)
        gate_mask = model_max_log2p < conf_gate_threshold  # (B, L) bool — True = uncertain → blend
        nat_log_model_at_target = (model_log_softmax.gather(-1, target_ids.long().unsqueeze(-1)).squeeze(-1))  # (B, L) nats
        nat_log_at_target = torch.where(gate_mask, nat_log_blend_at_target, nat_log_model_at_target)
    else:
        nat_log_at_target = nat_log_blend_at_target

    nll = -nat_log_at_target  # (B, L)
    return nll.mean()


# ---------- Multi-K helpers (0069) ----------

def kgram_log2p_per_position(
    input_ids: Tensor,           # (B, L) int64
    *,
    trigram_keys: Tensor,        # (n_ctx,) int32
    trigram_offsets: Tensor,     # (n_ctx+1,) int32
    trigram_next: Tensor,        # (n_total,) int16
    trigram_log2p_quant: Tensor, # (n_total,) int8
    trigram_log2p_scale: Tensor, # scalar fp32
    trigram_log2p_offset: Tensor,# scalar fp32
    trigram_log2_backoff_quant: Tensor,  # (n_ctx,) int8
    trigram_log2_backoff_scale: Tensor,  # scalar
    trigram_log2_backoff_offset: Tensor, # scalar
    bigram_dq: Tensor,           # (V, V) fp32 — pre-dequantized bigram log2-prob
    unigram_log2p: Tensor,       # (V,) fp32
    vocab_size: int,
    K: int,
    return_match_info: bool = False,
    alpha_quant: Tensor = None,  # (n_ctx,) int8 — per-context α, optional
    alpha_scale: Tensor = None,  # scalar fp32
    alpha_offset: Tensor = None, # scalar fp32
):
    """Compute the per-position K-gram log2-prob tensor (B, L, V).

    Identical lookup logic to `trigram_blend_loss` (factored out so multiple K's
    can share the bigram fallback and we can blend them together).

    When `return_match_info=True`, returns (kgram_lp, matched_2d, alpha_2d) where:
      - matched_2d (B, L) bool — True at positions where the K-gram lookup matched
        a context (False at fallback positions, including t < K-1).
      - alpha_2d (B, L) fp32 — per-context α at matched positions (0.0 elsewhere).
        Only filled when alpha_quant is provided; else returned as None.
    """
    device = input_ids.device
    B, L = input_ids.shape
    if K not in (3, 4):
        raise ValueError(f"K must be 3 or 4, got {K}")

    tri_log2p_full = trigram_log2p_quant.to(torch.float32) * trigram_log2p_scale + trigram_log2p_offset  # (n_total,)
    tri_log2_backoff = trigram_log2_backoff_quant.to(torch.float32) * trigram_log2_backoff_scale + trigram_log2_backoff_offset  # (n_ctx,)
    if alpha_quant is not None:
        tri_alpha = alpha_quant.to(torch.float32) * alpha_scale + alpha_offset  # (n_ctx,)
    else:
        tri_alpha = None

    # Bigram fallback for t >= 1; unigram for t == 0.
    kgram_lp = bigram_dq[input_ids.long()]  # (B, L, V)
    if L >= 1:
        kgram_lp[:, 0, :] = unigram_log2p

    matched_2d = torch.zeros(B, L, dtype=torch.bool, device=device) if return_match_info else None
    alpha_2d = torch.zeros(B, L, dtype=torch.float32, device=device) if (return_match_info and tri_alpha is not None) else None

    t_start = K - 1
    if L > t_start:
        n_pos = L - t_start
        if K == 3:
            prev2_slice = input_ids[:, 1:-1].long()
            prev1_slice = input_ids[:, 2:].long()
            keys = prev2_slice * vocab_size + prev1_slice
        else:  # K == 4
            prev3_slice = input_ids[:, 1:-2].long()
            prev2_slice = input_ids[:, 2:-1].long()
            prev1_slice = input_ids[:, 3:].long()
            keys = (
                prev3_slice * (vocab_size * vocab_size)
                + prev2_slice * vocab_size
                + prev1_slice
            )
        keys_flat = keys.reshape(-1).to(torch.int32)

        idx = torch.searchsorted(trigram_keys, keys_flat)
        idx = torch.clamp(idx, max=int(trigram_keys.numel()) - 1)
        matched = trigram_keys[idx] == keys_flat

        log2_backoff_per_pos = torch.where(
            matched,
            tri_log2_backoff[idx.long()],
            torch.zeros((), dtype=torch.float32, device=device),
        )
        log2_backoff_per_pos_2d = log2_backoff_per_pos.reshape(B, n_pos)
        kgram_lp[:, t_start:, :] = kgram_lp[:, t_start:, :] + log2_backoff_per_pos_2d.unsqueeze(-1)

        if return_match_info:
            matched_2d[:, t_start:] = matched.reshape(B, n_pos)
            if tri_alpha is not None:
                alpha_per_pos = torch.where(
                    matched,
                    tri_alpha[idx.long()],
                    torch.zeros((), dtype=torch.float32, device=device),
                )
                alpha_2d[:, t_start:] = alpha_per_pos.reshape(B, n_pos)

        Bn = idx.numel()
        start_per_pos = trigram_offsets[idx.long()].long()
        end_per_pos = trigram_offsets[(idx + 1).long()].long()
        seg_lens = torch.where(matched, end_per_pos - start_per_pos, torch.zeros_like(end_per_pos))
        total_entries = int(seg_lens.sum().item())
        if total_entries > 0:
            pos_id_per_entry = torch.repeat_interleave(
                torch.arange(Bn, device=device, dtype=torch.long), seg_lens
            )
            cum_lens = torch.cumsum(seg_lens, dim=0)
            seg_start_in_flat = cum_lens - seg_lens
            within = torch.arange(total_entries, device=device, dtype=torch.long) - seg_start_in_flat[pos_id_per_entry]
            entry_global_idx = start_per_pos[pos_id_per_entry] + within
            entry_next = trigram_next[entry_global_idx].long()
            entry_log2p = tri_log2p_full[entry_global_idx]
            pos_b = pos_id_per_entry // n_pos
            pos_t_off = pos_id_per_entry % n_pos
            kgram_lp[pos_b, pos_t_off + t_start, entry_next] = entry_log2p

    if return_match_info:
        return kgram_lp, matched_2d, alpha_2d
    return kgram_lp  # (B, L, V) log2


def trigram_blend_loss_multi_K(
    model_log_softmax: Tensor,   # (B, L, V) — natural log
    target_ids: Tensor,           # (B, L) int64
    input_ids: Tensor,            # (B, L) int64
    *,
    packs_by_K: dict,             # {K: pack_dict_of_tensors_for_that_K}
    bigram_log2p_quant: Tensor,   # (V, V) int8 — shared across K's
    bigram_log2p_scales: Tensor,  # (V,) fp32 — shared
    bigram_log2p_offsets: Tensor, # (V,) fp32 — shared
    unigram_log2p: Tensor,        # (V,) fp32 — shared
    blend_weights: list,          # [w_model, w_K1, w_K2, ...] — must sum to 1.0
    vocab_size: int,
    K_order: list,                # ordered list of K's matching blend_weights[1:]
    per_context_alpha: bool = False,
    alpha_max_default: float = 0.95,
    conf_gate_threshold: float = -1e9,
) -> Tensor:
    """Multi-K blended cross-entropy.

    blend_weights has the model weight first, then one weight per K (in
    `K_order`'s order). Computes per-position K-gram log2-probs for each K and
    blends with the model in log-prob space via N-way logsumexp:
        log P_blend = logsumexp([log(w_m) + log P_model, log(w_K) + log P_K, ...])

    When `per_context_alpha=True`, each pack must contain `alpha_quant`,
    `alpha_scale`, `alpha_offset` buffers (per-context α). The model weight is
    replaced per-token by an entropy-derived α: K-largest matched K wins (we
    iterate K_order and prefer later/larger K's), with α_max_default at
    positions where no K matched. Per-K weights are renormalized:
        β_K_i = blend_weights[1+i] / sum(blend_weights[1:])
    then per-token weights are (α_m, β_3 * (1-α_m), β_4 * (1-α_m)).

    Returns mean cross-entropy in nats.
    """
    LN2 = math.log(2.0)
    B, L, V = model_log_softmax.shape
    assert V == vocab_size
    assert len(blend_weights) == 1 + len(K_order), (
        f"blend_weights must have model weight + one per K (got {len(blend_weights)} weights, {len(K_order)} K's)"
    )
    weight_sum = sum(blend_weights)
    if abs(weight_sum - 1.0) > 1e-4:
        raise ValueError(f"blend_weights must sum to 1.0, got {weight_sum:.6f} ({blend_weights})")

    # Decode bigram once (shared across K's). Same path as trigram_blend_loss.
    bigram_dq = bigram_log2p_quant.to(torch.float32) * bigram_log2p_scales.unsqueeze(1) + bigram_log2p_offsets.unsqueeze(1)

    # Build per-K log2-prob tensors. When per_context_alpha is enabled, also
    # collect matched / α info per K so we can derive the per-token model weight.
    kgram_lps = []
    matched_per_K: list = []
    alpha_per_K: list = []
    for K in K_order:
        pack = packs_by_K[K]
        if per_context_alpha:
            lp, matched_2d, alpha_2d = kgram_log2p_per_position(
                input_ids,
                trigram_keys=pack["keys"],
                trigram_offsets=pack["offsets"],
                trigram_next=pack["next"],
                trigram_log2p_quant=pack["log2p_quant"],
                trigram_log2p_scale=pack["log2p_scale"],
                trigram_log2p_offset=pack["log2p_offset"],
                trigram_log2_backoff_quant=pack["log2_backoff_quant"],
                trigram_log2_backoff_scale=pack["log2_backoff_scale"],
                trigram_log2_backoff_offset=pack["log2_backoff_offset"],
                bigram_dq=bigram_dq,
                unigram_log2p=unigram_log2p,
                vocab_size=vocab_size,
                K=K,
                return_match_info=True,
                alpha_quant=pack["alpha_quant"],
                alpha_scale=pack["alpha_scale"],
                alpha_offset=pack["alpha_offset"],
            )
            matched_per_K.append(matched_2d)
            alpha_per_K.append(alpha_2d)
        else:
            lp = kgram_log2p_per_position(
                input_ids,
                trigram_keys=pack["keys"],
                trigram_offsets=pack["offsets"],
                trigram_next=pack["next"],
                trigram_log2p_quant=pack["log2p_quant"],
                trigram_log2p_scale=pack["log2p_scale"],
                trigram_log2p_offset=pack["log2p_offset"],
                trigram_log2_backoff_quant=pack["log2_backoff_quant"],
                trigram_log2_backoff_scale=pack["log2_backoff_scale"],
                trigram_log2_backoff_offset=pack["log2_backoff_offset"],
                bigram_dq=bigram_dq,
                unigram_log2p=unigram_log2p,
                vocab_size=vocab_size,
                K=K,
            )
        kgram_lps.append(lp)

    # Blend in log2-prob space via N-way logsumexp. Numerically stable with
    # subtract-max.
    model_log2p = model_log_softmax / LN2  # (B, L, V) log2

    def _w_log2(w: float) -> float:
        return math.log2(w) if w > 0 else -1e30

    if per_context_alpha:
        # Per-token model weight α_m. Priority: later K's in K_order win when
        # matched; positions with no match → α_max_default. Conventionally,
        # K_order is [3, 4] so K=4 dominates K=3 when both match.
        device = model_log_softmax.device
        alpha_m = torch.full((B, L), float(alpha_max_default), dtype=torch.float32, device=device)
        for K, matched_2d, alpha_2d in zip(K_order, matched_per_K, alpha_per_K):
            # Where this K matched, override (later K's overwrite earlier ones).
            alpha_m = torch.where(matched_2d, alpha_2d, alpha_m)
        # Safety clamp: never let α_m hit exactly 0 or 1 (log2 would explode).
        alpha_m = torch.clamp(alpha_m, min=1e-6, max=1.0 - 1e-6)
        one_minus = 1.0 - alpha_m  # (B, L)

        # Renormalized per-K β weights (constant across batch).
        kgram_w_sum = sum(blend_weights[1:])
        if kgram_w_sum <= 0:
            raise ValueError(f"per_context_alpha requires positive K-gram weights, got {blend_weights}")
        betas = [w / kgram_w_sum for w in blend_weights[1:]]

        # Build log-parts: model uses log2(α_m); each K uses log2(β_K * (1-α_m)) = log2(β_K) + log2(1-α_m).
        log2_alpha_m = torch.log2(alpha_m).unsqueeze(-1)            # (B, L, 1)
        log2_one_minus = torch.log2(one_minus).unsqueeze(-1)         # (B, L, 1)
        log_parts = [log2_alpha_m + model_log2p]
        for beta, lp in zip(betas, kgram_lps):
            log_parts.append(_w_log2(beta) + log2_one_minus + lp)
    else:
        log_parts = [_w_log2(blend_weights[0]) + model_log2p]
        for w, lp in zip(blend_weights[1:], kgram_lps):
            log_parts.append(_w_log2(w) + lp)

    # Compute elementwise max for stability
    m = log_parts[0]
    for x in log_parts[1:]:
        m = torch.maximum(m, x)
    # log2(sum 2^(x_i - m))
    s = torch.zeros_like(m)
    for x in log_parts:
        s = s + torch.exp2(x - m)
    log2_blend = m + torch.log2(s)

    nat_log_blend = log2_blend * LN2
    nat_log_blend_at_target = nat_log_blend.gather(-1, target_ids.long().unsqueeze(-1)).squeeze(-1)  # (B, L)

    # 0076: confidence gate. When model max log2p > threshold (model confident),
    # use model log-prob at target alone; otherwise keep the blended one.
    if conf_gate_threshold > -1e8:
        model_max_log2p = model_log_softmax.max(dim=-1).values / LN2  # (B, L)
        gate_mask = model_max_log2p < conf_gate_threshold  # True = uncertain → blend
        nat_log_model_at_target = model_log_softmax.gather(-1, target_ids.long().unsqueeze(-1)).squeeze(-1)
        nat_log_at_target = torch.where(gate_mask, nat_log_blend_at_target, nat_log_model_at_target)
    else:
        nat_log_at_target = nat_log_blend_at_target

    nll = -nat_log_at_target
    return nll.mean()
