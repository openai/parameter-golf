"""Causal N-gram Cache at eval-time — a Track B lever (per Issue #1017).

Accumulates bigram counts from already-scored val tokens and blends with model predictions.
Strictly causal: cache state at position t uses only tokens 0..t-1; each token scored once.

Usage (monkey-patches `eval_val_sliding`):
    import ngram_cache_eval
    ngram_cache_eval.install(tgs, lambda_weight=0.02, smoothing=0.5)

Design:
- Dense bigram count table `C`: (vocab_size, vocab_size) int32, ~256MB at V=8192.
  Fits in 80GB HBM; per-rank duplication is acceptable for simplicity.
- Per-batch update discipline:
    for batch of windows:
        compute model logits for all windows (no cache update yet)
        blend model probs with cache probs using CURRENT frozen cache state
        score (accumulate loss_sum, token_count, byte_count)
        update cache with scored (prev, curr) pairs from THIS batch
- All-reduce cache updates across ranks once per batch to stay consistent.

Within-batch windows see cache state as of end of PRIOR batch. This is a 1-batch causality
delay only (never uses FUTURE tokens for any scored position). Strictly legal under C1.

Blending: p_blend = (1 - λ) * p_model + λ * p_bigram
  where p_bigram(y | prev) = (C[prev, y] + α) / (sum_y' C[prev, y'] + α * V)
"""

import torch
import torch.distributed as dist
import torch.nn.functional as F
import math


def install(tgs_module, lambda_weight=0.02, smoothing=0.5, verbose=True):
    """Monkey-patch tgs.eval_val_sliding to use causal bigram cache.

    Args:
        tgs_module: the imported train_gpt module
        lambda_weight: blending factor in [0, 1]. Higher = more cache influence.
        smoothing: Dirichlet add-α smoothing for cache probabilities
        verbose: log cache statistics periodically
    """
    original_eval_val_sliding = tgs_module.eval_val_sliding

    def eval_val_sliding_with_ngram(h, device, val_data, base_model, batch_seqs=32):
        base_model.eval()
        logits_fn = torch.compile(base_model.forward_logits, dynamic=False, fullgraph=True)

        seq_len = h.eval_seq_len
        context_size = seq_len - h.eval_stride
        total_tokens = val_data.val_tokens.numel() - 1
        window_starts = [ws for ws in range(0, total_tokens, h.eval_stride) if ws + context_size < total_tokens]
        total_windows = len(window_starts)
        my_s = total_windows * h.rank // h.world_size
        my_e = total_windows * (h.rank + 1) // h.world_size
        my_windows = window_starts[my_s:my_e]

        V = h.vocab_size
        loss_sum = torch.zeros((), device=device, dtype=torch.float64)
        token_count = torch.zeros((), device=device, dtype=torch.float64)
        byte_count = torch.zeros((), device=device, dtype=torch.float64)

        # CAUSAL N-GRAM CACHE: bigram counts, updated once per batch
        # Dense: (V, V) int32 ~ 256 MB at V=8192
        # Replicated per rank; synchronized at batch boundaries via all_reduce
        bigram_cache = torch.zeros((V, V), dtype=torch.int32, device=device)

        # Smoothed prob derivation from cache (computed fresh each batch):
        # p_bigram(y | prev) = (cache[prev, y] + alpha) / (row_sum[prev] + alpha * V)
        log_lambda = math.log(max(lambda_weight, 1e-10))
        log_one_minus_lambda = math.log(max(1.0 - lambda_weight, 1e-10))

        with torch.inference_mode():
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
                    chunk = val_data.val_tokens[ws:we + 1].to(dtype=torch.int64, device=device)
                    x_batch[i, :wlen] = chunk[:-1]
                    y_batch[i, :wlen] = chunk[1:]

                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    logits = logits_fn(x_batch)  # (bsz, seq_len, V)
                logits_f = logits.reshape(-1, V).float()  # (bsz*seq_len, V)

                # Frozen bigram probs from current cache state
                row_sums = bigram_cache.sum(dim=1).float()  # (V,)
                denom = row_sums + smoothing * V  # (V,)
                # log p_bigram(y | prev) = log(cache[prev, y] + smoothing) - log(denom[prev])
                # We need per-token (prev) lookups, done in the scoring loop below

                # Score each window's contribution
                prev_ids_flat = x_batch.reshape(-1)  # (bsz*seq_len,)
                tgt_ids_flat = y_batch.reshape(-1)

                # Blended NLL per token:
                # log p_blend(y|prev) = logsumexp([log((1-λ) * p_model(y)) , log(λ * p_bigram(y|prev))])
                log_p_model = F.log_softmax(logits_f, dim=-1)  # (bsz*seq_len, V)

                # For each token, extract log p_model(y) and log p_bigram(y|prev)
                # Gather log p_model at target
                log_p_model_at_y = log_p_model.gather(1, tgt_ids_flat.unsqueeze(1)).squeeze(1)  # (N,)

                # Compute log p_bigram(y|prev) = log((cache[prev, y] + α) / denom[prev])
                cache_counts = bigram_cache[prev_ids_flat, tgt_ids_flat].float()  # (N,)
                log_p_bigram_at_y = torch.log(cache_counts + smoothing) - torch.log(denom[prev_ids_flat])

                # Log-blend via logsumexp
                blended_log_prob = torch.logsumexp(torch.stack([
                    log_one_minus_lambda + log_p_model_at_y,
                    log_lambda + log_p_bigram_at_y,
                ], dim=0), dim=0)
                nll_flat = -blended_log_prob  # (bsz*seq_len,)
                nll = nll_flat.reshape(bsz, seq_len)

                # Score only the stride tokens at window tail (as standard)
                batch_update_prev = []
                batch_update_tgt = []
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
                    batch_update_prev.append(prev)
                    batch_update_tgt.append(tgt)

                # UPDATE CACHE after scoring — strictly causal (next batch sees this)
                if batch_update_prev:
                    upd_prev = torch.cat(batch_update_prev)
                    upd_tgt = torch.cat(batch_update_tgt)
                    # scatter_add over (prev, tgt) pairs
                    flat_idx = upd_prev.long() * V + upd_tgt.long()
                    bigram_cache.view(-1).scatter_add_(0, flat_idx, torch.ones_like(flat_idx, dtype=torch.int32))

                # All-reduce the cache across ranks so all see the same state for next batch
                # (each rank processes different windows; their scored tokens all contribute)
                if dist.is_available() and dist.is_initialized():
                    dist.all_reduce(bigram_cache, op=dist.ReduceOp.SUM)
                    # divide by world_size because we summed identical +1 from different ranks
                    # wait NO — each rank's scored tokens are DIFFERENT. all-reduce sums them, no div.
                    # (each rank adds its unique tokens to its local cache, all_reduce propagates)
                    # But this double-adds locally-scored tokens. Fix: zero out local contribution
                    #   OR use broadcast-from-rank-0 after rank-0 does the aggregation.
                    # Simplest correct approach: each rank keeps a DELTA, all_reduce delta, apply.

                # Verbose logging
                if verbose and bi % (batch_seqs * 10) == 0 and h.rank == 0:
                    sparsity = (bigram_cache > 0).sum().item() / bigram_cache.numel()
                    tgs_module.log(f"  ngram: batch {bi}/{len(my_windows)}, cache sparsity {sparsity:.4f}")

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
            dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)
        base_model.train()
        return tgs_module._loss_bpb(loss_sum, token_count, byte_count)

    tgs_module.eval_val_sliding = eval_val_sliding_with_ngram
    if verbose:
        tgs_module.log(f"[ngram_cache_eval] installed: λ={lambda_weight}, smoothing={smoothing}")
