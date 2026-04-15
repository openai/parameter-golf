"""
Causal N-gram Eval Integration for #1493 stack.

Provides `eval_val_ttt_with_ngram` — a drop-in replacement for `eval_val_ttt`
that injects a causal n-gram cache as an additive-logit contribution to the
neural model's output.

LEGALITY (matches causal_ngram.py module docstring):
  C1 strict causal: n-gram state at scoring time t reflects only tokens < t.
  C2 full normalized: blend is `softmax(logits_neural + alpha * log_p_ngram)`
     over full vocab. Normalization holds over actual tokens.
  C3 score-before-update: cache is frozen at chunk start, scored under
     inference_mode, updated only after all windows in the chunk have been
     scored.
  C4 single pass: one left-to-right traversal, no rescoring.

INTEGRATION POINT: after `compiled_logits(x_batch)` and before
`F.cross_entropy`, we compute `log_p_ngram` for every (b, t) position and add
`alpha * log_p_ngram` to the neural logits. The softmax inside cross-entropy
then produces a valid normalized distribution.

PERFORMANCE:
  - Prototype path: pure Python context-tuple lookup, slow but correct. Used
    for local prototype and small-model tests.
  - Fast path (TODO for A40/H100): pre-compute per-unique-context log-prob
    tensors and gather. Only rebuild when cache is updated (between chunks).
"""
from __future__ import annotations
import math
import os
import sys
import time
from collections import Counter, defaultdict
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

# Same module-local CausalNGram class. To keep the record-submission inlining
# simple we keep everything in one file.


class CausalNGram:
    """Exact non-hashed causal n-gram with backoff.

    State model: two count tables, `counts` (live) and `frozen_counts`
    (immutable snapshot used for lookups). `freeze()` snapshots live -> frozen.
    Lookups always read from frozen. Updates always write to live.

    Legal usage pattern for eval_val_ttt:
        ng = CausalNGram(vocab_size, order=5, delta=0.5)
        ng.freeze()  # initial empty frozen state
        for chunk in chunks:
            # Score the chunk against the CURRENT frozen state
            score_chunk(chunk, ng)
            # After scoring, add the chunk's scored tokens to live counts
            ng.add_many(chunk_history, chunk_tokens)
            # Re-freeze live into frozen for the NEXT chunk
            ng.freeze()
    """

    def __init__(self, vocab_size: int, order: int = 5, delta: float = 0.5,
                 min_context_count: int = 2):
        assert order >= 1 and vocab_size > 0
        self.V = vocab_size
        self.K = order
        self.delta = delta
        self.min_ctx = min_context_count
        # Live counts
        self.counts = {k: defaultdict(Counter) for k in range(1, order + 1)}
        self.totals = {k: defaultdict(int) for k in range(1, order + 1)}
        # Frozen snapshot (None until first freeze())
        self._frozen_counts = None
        self._frozen_totals = None
        # Log-prob vector cache (torch tensor per context tuple), invalidated
        # on every freeze().
        self._lp_cache: dict = {}

    def add_token(self, history_tail: tuple, token: int) -> None:
        """Update live counts. history_tail is the last K-1 tokens (as tuple).
        If history_tail is shorter than K-1, shorter orders still update."""
        for k in range(1, self.K + 1):
            ctx_len = k - 1
            if ctx_len == 0:
                ctx = ()
            else:
                if len(history_tail) < ctx_len:
                    continue
                ctx = history_tail[-ctx_len:]
            self.counts[k][ctx][token] += 1
            self.totals[k][ctx] += 1

    def add_many(self, tokens: list[int], history_prefix: tuple = ()) -> None:
        """Update live counts with a whole subsequence. `history_prefix` is the
        tokens that came before tokens[0] (for context-lookup on the first few
        positions). Typical usage: the context from the window's prefix."""
        running = list(history_prefix)[-(self.K - 1):] if self.K > 1 else []
        for tok in tokens:
            self.add_token(tuple(running), int(tok))
            running.append(int(tok))
            if len(running) > (self.K - 1):
                running = running[-(self.K - 1):]

    def freeze(self) -> None:
        """Snapshot live counts as the immutable frozen state. Invalidates the
        log-prob cache (since the frozen state has changed)."""
        self._frozen_counts = {k: {ctx: Counter(c) for ctx, c in d.items()}
                                for k, d in self.counts.items()}
        self._frozen_totals = {k: dict(d) for k, d in self.totals.items()}
        self._lp_cache.clear()

    def _lookup_log_probs(self, ctx_tail: tuple) -> np.ndarray:
        """Walk backoff from order K down. Return full-vocab log-prob vector.
        Reads ONLY the frozen snapshot.

        IMPORTANT: we now back off only to order >= 2 (bigram). If even bigram
        has no observation for the context, we return a FLAT uniform vector.
        This is important because a flat uniform contribution is a logit
        SHIFT, which softmax is invariant to — meaning positions with no real
        cache hit get zero effective n-gram contribution, avoiding the small
        positive drag observed in the localized-delta analysis.

        The min_bigram_for_hit threshold (backoff stops if order 2 has < this
        many observations) is a principled way to require a "real hit" before
        contributing anything.
        """
        if ctx_tail in self._lp_cache:
            return self._lp_cache[ctx_tail]
        src = self._frozen_counts
        tot = self._frozen_totals
        V = self.V
        uniform = np.full(V, -math.log(V), dtype=np.float32)

        if src is None:
            self._lp_cache[ctx_tail] = uniform
            return uniform

        log_p = None
        # Walk K -> 2 (NOT down to unigram — unigram is no-op vs neural)
        min_k = 2
        for k in range(self.K, min_k - 1, -1):
            ctx_len = k - 1
            if ctx_len == 0:
                ctx = ()
            elif len(ctx_tail) < ctx_len:
                continue
            else:
                ctx = ctx_tail[-ctx_len:]
            total = tot[k].get(ctx, 0)
            if total >= self.min_ctx:
                counter = src[k].get(ctx)
                denom = total + self.delta * V
                vec = np.full(V, self.delta / denom, dtype=np.float32)
                if counter:
                    for tok, c in counter.items():
                        vec[tok] = (c + self.delta) / denom
                log_p = np.log(vec)
                break
        if log_p is None:
            # No bigram-or-higher hit → flat uniform → softmax-invariant,
            # zero effective contribution to the blended distribution.
            log_p = uniform
        self._lp_cache[ctx_tail] = log_p
        return log_p

    def batch_log_probs_torch(self, x_batch: torch.Tensor) -> torch.Tensor:
        """Given x_batch of shape (B, T), return (B, T, V) log-probs from the
        frozen cache.

        Performance notes:
          - Builds a CPU numpy (B,T,V) buffer in one pass via bulk fills,
            then does ONE CPU->device transfer at the end (not B*T transfers).
          - Unique-context caching: many adjacent positions share the same
            context tuple — we collect unique contexts first, look up each
            once, then scatter into the output.
        """
        B, T = x_batch.shape
        V = self.V
        x_cpu = x_batch.detach().cpu().numpy().astype(np.int32)
        Ksub = self.K - 1  # context length (number of previous tokens)

        # Build a CPU buffer of shape (B, T, V) filled with per-position log-probs.
        # Use float32 numpy for speed, then transfer once.
        out_np = np.empty((B, T, V), dtype=np.float32)

        # Collect (b, t) positions grouped by context tuple, so we only look
        # up each unique context once per batch.
        groups: dict = {}
        for b in range(B):
            row = x_cpu[b]
            for t in range(T):
                start = max(0, t - Ksub + 1)
                ctx_tail = tuple(int(x) for x in row[start:t + 1])
                if ctx_tail in groups:
                    groups[ctx_tail].append((b, t))
                else:
                    groups[ctx_tail] = [(b, t)]

        # Lookup each unique context once, then scatter
        for ctx_tail, positions in groups.items():
            lp = self._lookup_log_probs(ctx_tail)  # numpy (V,)
            for b, t in positions:
                out_np[b, t] = lp

        # Single transfer to target device
        return torch.from_numpy(out_np).to(device=x_batch.device)

    # --- stats ---
    def unique_contexts(self) -> dict:
        return {k: len(self.counts[k]) for k in range(1, self.K + 1)}


def eval_val_ttt_with_ngram(h, device, val_data, base_model,
                             ngram: CausalNGram,
                             alpha: float,
                             batch_seqs: int = 32,
                             enable_ttt: bool = True):
    """Drop-in replacement for eval_val_ttt that additively blends a causal
    n-gram log-prob contribution into the neural logits at scoring time, then
    updates the n-gram with the scored tokens after each chunk.

    Args:
        h: Hyperparameters (same as #1493).
        device: torch device.
        val_data: ValidationData (with base_bytes_lut etc.)
        base_model: the compiled neural model (must expose forward_logits).
        ngram: CausalNGram instance. Should be fresh (empty) at call time.
        alpha: fixed scalar blend weight on log_p_ngram. Baked into the
            artifact — NOT eval-token dependent.
        batch_seqs: batch size for window scoring.
        enable_ttt: whether to also run SGD TTT in addition to n-gram.
    """
    import torch.distributed as dist
    rank = h.rank
    world_size = h.world_size
    seq_len = h.eval_seq_len
    stride = h.eval_stride
    total_tokens = val_data.val_tokens.numel() - 1
    ttt_chunk = h.ttt_chunk_tokens
    context_size = seq_len - stride

    # Pre-compute window starts and chunk assignment (same as #1493)
    window_starts = [ws for ws in range(0, total_tokens, stride)
                     if ws + context_size < total_tokens]
    num_chunks = (total_tokens + ttt_chunk - 1) // ttt_chunk
    chunk_windows = [[] for _ in range(num_chunks)]
    for ws in window_starts:
        wlen = min(ws + seq_len, total_tokens) - ws
        s = 0 if ws == 0 else context_size
        scored_start = ws + s
        ci = min(scored_start // ttt_chunk, num_chunks - 1)
        chunk_windows[ci].append(ws)

    print(f"ngram_ttt:start chunks={num_chunks} alpha={alpha} order={ngram.K}",
          file=sys.stderr)

    compiled_logits = torch.compile(base_model.forward_logits, dynamic=False, fullgraph=True) \
        if device.type == 'cuda' else base_model.forward_logits

    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)

    ttt_params = [p for p in base_model.parameters()]
    if enable_ttt:
        for p in ttt_params:
            p.requires_grad_(True)
        optimizer = torch.optim.SGD(ttt_params, lr=h.ttt_lr, momentum=h.ttt_momentum)
    else:
        optimizer = None

    # Initial freeze: empty cache → uniform log-probs everywhere
    ngram.freeze()

    for ci in range(num_chunks):
        windows = chunk_windows[ci]
        if not windows:
            continue
        chunk_start = ci * ttt_chunk
        chunk_end = min((ci + 1) * ttt_chunk, total_tokens)
        my_s = len(windows) * rank // world_size
        my_e = len(windows) * (rank + 1) // world_size
        my_windows = windows[my_s:my_e]
        base_model.eval()

        # Track which tokens get scored in this chunk (for n-gram update)
        chunk_scored_positions = []  # list of (global_position, token_id)

        with torch.no_grad():
            for bi in range(0, len(my_windows), batch_seqs):
                batch_ws = my_windows[bi:bi + batch_seqs]
                bsz = len(batch_ws)
                x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
                y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
                wlens = []
                for i, ws in enumerate(batch_ws):
                    we = min(ws + seq_len, total_tokens)
                    wlen = we - ws
                    wlens.append(wlen)
                    chunk_tok = val_data.val_tokens[ws:we + 1].to(dtype=torch.int64, device=device)
                    x_batch[i, :wlen] = chunk_tok[:-1]
                    y_batch[i, :wlen] = chunk_tok[1:]

                # 1. Compute neural logits
                if device.type == 'cuda':
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        logits = compiled_logits(x_batch)
                else:
                    logits = compiled_logits(x_batch)

                # 2. Compute n-gram log-probs (frozen cache). CPU-based lookup.
                #    Shape: (bsz, seq_len, V), same dtype as logits
                if alpha != 0.0:
                    ngram_log_p = ngram.batch_log_probs_torch(x_batch).to(logits.dtype)
                    # 3. Additive logit blend (legal: softmax produces a valid
                    #    normalized distribution over Σ, independent of x_t)
                    blended_logits = logits + alpha * ngram_log_p
                else:
                    blended_logits = logits

                # 4. Compute nll from blended logits
                nll = F.cross_entropy(
                    blended_logits.reshape(-1, blended_logits.size(-1)).float(),
                    y_batch.reshape(-1), reduction='none'
                ).reshape(bsz, seq_len)

                # 5. Score + byte counting (verbatim from #1493)
                for i, ws in enumerate(batch_ws):
                    wlen = wlens[i]
                    s = 0 if ws == 0 else context_size
                    scored_nll = nll[i, s:wlen].to(torch.float64)
                    loss_sum += scored_nll.sum()
                    token_count += float(wlen - s)
                    tgt = y_batch[i, s:wlen]
                    prev = x_batch[i, s:wlen]
                    tb = val_data.base_bytes_lut[tgt].to(torch.float64)
                    tb += (val_data.has_leading_space_lut[tgt] & ~val_data.is_boundary_token_lut[prev]).to(torch.float64)
                    byte_count += tb.sum()

                    # Record scored tokens for post-chunk n-gram update.
                    # The scored tokens are y_batch[i, s:wlen] at global
                    # positions (ws+s .. ws+wlen-1). Their contexts are
                    # x_batch[i, :s] (window prefix that leads up to s).
                    scored_toks = y_batch[i, s:wlen].cpu().numpy().astype(np.int64)
                    context_prefix = x_batch[i, :s].cpu().numpy().astype(np.int64)
                    # We record absolute positions so the update step is
                    # deterministic regardless of parallelism.
                    chunk_scored_positions.append(
                        (int(ws + s), context_prefix, scored_toks)
                    )

        # --- End of scoring window loop for this chunk ---
        # 6. N-GRAM UPDATE (after all scoring is complete for this chunk).
        #    This is the update-after-score discipline. Sort by global position
        #    to maintain a left-to-right update order.
        chunk_scored_positions.sort(key=lambda t: t[0])
        for gpos, ctx_prefix, toks in chunk_scored_positions:
            # Rolling context while updating. Start from the last K-1 tokens
            # of ctx_prefix (which came from the window prefix, already
            # previously scored in earlier windows/chunks).
            running = list(int(x) for x in ctx_prefix[-(ngram.K - 1):]) if ngram.K > 1 else []
            for tok in toks:
                ngram.add_token(tuple(running), int(tok))
                if ngram.K > 1:
                    running.append(int(tok))
                    if len(running) > ngram.K - 1:
                        running = running[-(ngram.K - 1):]
        # Re-freeze: live -> frozen, for use by the NEXT chunk
        ngram.freeze()

        # --- Optional SGD TTT (same as #1493) ---
        is_last_chunk = ci == num_chunks - 1
        if enable_ttt and not is_last_chunk and h.ttt_epochs > 0 and optimizer is not None:
            base_model.train()
            chunk_seqs = (chunk_end - chunk_start) // seq_len
            if chunk_seqs > 0:
                cos_lr = h.ttt_lr * 0.5 * (1.0 + math.cos(math.pi * ci / max(num_chunks - 1, 1)))
                for pg in optimizer.param_groups:
                    pg['lr'] = cos_lr
                my_seq_s = chunk_seqs * rank // world_size
                my_seq_e = chunk_seqs * (rank + 1) // world_size
                my_chunk_seqs = my_seq_e - my_seq_s
                for _ep in range(h.ttt_epochs):
                    for bs in range(0, my_chunk_seqs, batch_seqs):
                        be = min(bs + batch_seqs, my_chunk_seqs)
                        actual_bs = my_seq_s + bs
                        start_tok = chunk_start + actual_bs * seq_len
                        end_tok = chunk_start + (my_seq_s + be) * seq_len + 1
                        if end_tok > val_data.val_tokens.numel():
                            continue
                        local = val_data.val_tokens[start_tok:end_tok].to(device=device, dtype=torch.int64)
                        x = local[:-1].reshape(-1, seq_len)
                        y = local[1:].reshape(-1, seq_len)
                        optimizer.zero_grad(set_to_none=True)
                        if device.type == 'cuda':
                            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                                loss = base_model(x, y)
                        else:
                            loss = base_model(x, y)
                        loss.backward()
                        if world_size > 1:
                            for p in ttt_params:
                                if p.grad is not None:
                                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
                        torch.nn.utils.clip_grad_norm_(ttt_params, 1.0)
                        optimizer.step()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)
    if enable_ttt:
        for p in base_model.parameters():
            p.requires_grad_(True)
    base_model.eval()

    val_loss = (loss_sum / token_count).item()
    val_bpb = val_loss / math.log(2.0) * (token_count.item() / byte_count.item())
    return val_loss, val_bpb
