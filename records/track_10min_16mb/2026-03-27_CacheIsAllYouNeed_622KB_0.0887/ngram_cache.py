"""The Cache Is All You Need — eval-time n-gram + phrase cache stack.

Drop-in eval enhancement for any language model. Builds backward-looking
n-gram and long-phrase hash tables from already-scored tokens, then blends
cache predictions with model logits using order-adaptive entropy gating.

Score-first legal: caches are updated ONLY after scoring each chunk.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor


class NgramEvalCache:
    """Multi-order n-gram backoff (orders 2-12) with order-adaptive entropy gating."""
    PRIMES = np.array([36313, 27191, 51647, 81929, 131071, 196613, 262147], dtype=np.uint64)

    def __init__(self, max_order=12, buckets=4_194_304, min_count=1,
                 alpha_low=0.05, alpha_high=0.95, entropy_thresh=4.0):
        self.max_order = max_order
        self.buckets = buckets
        self.min_count = min_count
        self.alpha_low = alpha_low
        self.alpha_high = alpha_high
        self.entropy_thresh = entropy_thresh
        self.mask = np.uint64(buckets - 1)
        self.ctx_tables = {n: np.zeros(buckets, dtype=np.uint32) for n in range(2, max_order + 1)}
        self.full_tables = {n: np.zeros(buckets, dtype=np.uint32) for n in range(2, max_order + 1)}

    def lookup(self, val_np, target_pos, targets):
        seg_len = len(target_pos)
        best_p = np.zeros(seg_len, dtype=np.float64)
        has_match = np.zeros(seg_len, dtype=bool)
        match_orders = np.zeros(seg_len, dtype=np.int32)
        tgt_u64 = targets.astype(np.uint64)
        n_primes = len(self.PRIMES)
        for n in range(self.max_order, 1, -1):
            ctx_w = n - 1
            eligible = (target_pos >= ctx_w) & ~has_match
            if not eligible.any():
                continue
            idx = np.where(eligible)[0]
            pos, tgt = target_pos[idx], tgt_u64[idx]
            ctx_hash = np.zeros(len(idx), dtype=np.uint64)
            for k in range(ctx_w):
                ctx_hash ^= val_np[pos - ctx_w + k].astype(np.uint64) * self.PRIMES[k % n_primes]
            ctx_key = (ctx_hash & self.mask).astype(np.intp)
            ctx_counts = self.ctx_tables[n][ctx_key]
            sufficient = ctx_counts >= self.min_count
            if not sufficient.any():
                continue
            s_idx = idx[sufficient]
            s_ctx = ctx_counts[sufficient].astype(np.float64)
            full_key = ((ctx_hash[sufficient] ^ (tgt[sufficient] * self.PRIMES[ctx_w % n_primes])) & self.mask).astype(np.intp)
            s_full = self.full_tables[n][full_key].astype(np.float64)
            has_target = s_full > 0
            if has_target.any():
                pi = s_idx[has_target]
                p_ng = np.minimum(s_full[has_target], s_ctx[has_target]) / np.maximum(s_ctx[has_target], 1.0)
                best_p[pi] = np.clip(p_ng, 0.0, 1.0)
                match_orders[pi] = n
                has_match[pi] = True
        return best_p, has_match, match_orders

    def get_alpha(self, entropy, match_orders):
        """Order-adaptive alpha: high orders trusted more, low orders suppressed."""
        order_frac = (match_orders - 2).astype(np.float64) / max(self.max_order - 2, 1)
        thresh_high = self.entropy_thresh + 1.0
        thresh_low = max(self.entropy_thresh - 2.0, 1.5)
        per_order_thresh = thresh_high - order_frac * (thresh_high - thresh_low)
        sig = 1.0 / (1.0 + np.exp(-2.0 * (entropy - per_order_thresh)))
        base_alpha = self.alpha_low + (self.alpha_high - self.alpha_low) * sig
        mult = 0.3 + order_frac * 1.7  # order 2 → 0.3×, order max → 2.0×
        return np.clip(base_alpha * mult, 0.0, 0.99)

    def update(self, val_np, start, end):
        n_primes = len(self.PRIMES)
        for n in range(2, self.max_order + 1):
            ctx_w = n - 1
            first = max(start, ctx_w)
            if first > end:
                continue
            positions = np.arange(first, end + 1)
            tgt = val_np[positions].astype(np.uint64)
            ctx_hash = np.zeros(len(positions), dtype=np.uint64)
            for k in range(ctx_w):
                ctx_hash ^= val_np[positions - ctx_w + k].astype(np.uint64) * self.PRIMES[k % n_primes]
            ctx_key = (ctx_hash & self.mask).astype(np.intp)
            full_key = ((ctx_hash ^ (tgt * self.PRIMES[ctx_w % n_primes])) & self.mask).astype(np.intp)
            np.add.at(self.ctx_tables[n], ctx_key, 1)
            np.add.at(self.full_tables[n], full_key, 1)


class LongPhraseCache:
    """Long-phrase suffix matcher — same as n-gram but at lengths 16-64."""
    PRIMES = np.array([36313, 27191, 51647, 81929, 131071, 196613, 262147,
                       393241, 524309, 655373, 786433, 917521, 1048583,
                       1179653, 1310729, 1441801, 1572871, 1703939,
                       1835017, 1966093, 2097169, 2228243, 2359321,
                       2490377, 2621447, 2752523, 2883593, 3014657,
                       3145739, 3276811, 3407879, 3538961, 3670037,
                       3801131, 3932203, 4063267, 4194319, 4325381,
                       4456441, 4587503, 4718579, 4849651, 4980719,
                       5111789, 5242877, 5373953, 5505023, 5636089], dtype=np.uint64)
    PROBE_LENGTHS = [64, 56, 48, 36, 28, 20, 16]

    def __init__(self, buckets=4_194_304, min_count=1, base_alpha=0.90):
        self.buckets = buckets
        self.min_count = min_count
        self.base_alpha = base_alpha
        self.mask = np.uint64(buckets - 1)
        self.ctx_table = np.zeros(buckets, dtype=np.uint32)
        self.full_table = np.zeros(buckets, dtype=np.uint32)

    def _hash(self, val_np, positions, L):
        n_primes = len(self.PRIMES)
        h = np.zeros(len(positions), dtype=np.uint64)
        for k in range(L):
            h ^= val_np[positions - L + k].astype(np.uint64) * self.PRIMES[k % n_primes]
        return h

    def lookup(self, val_np, target_pos, targets):
        seg_len = len(target_pos)
        best_p = np.zeros(seg_len, dtype=np.float64)
        has_match = np.zeros(seg_len, dtype=bool)
        match_lengths = np.zeros(seg_len, dtype=np.int32)
        tgt_u64 = targets.astype(np.uint64)
        for L in self.PROBE_LENGTHS:
            eligible = (target_pos >= L) & ~has_match
            if not eligible.any():
                continue
            idx = np.where(eligible)[0]
            pos, tgt = target_pos[idx], tgt_u64[idx]
            ctx_hash = self._hash(val_np, pos, L)
            ctx_key = (ctx_hash & self.mask).astype(np.intp)
            ctx_counts = self.ctx_table[ctx_key]
            sufficient = ctx_counts >= self.min_count
            if not sufficient.any():
                continue
            si = idx[sufficient]
            sc = ctx_counts[sufficient].astype(np.float64)
            fk = ((ctx_hash[sufficient] ^ (tgt[sufficient] * self.PRIMES[L % len(self.PRIMES)])) & self.mask).astype(np.intp)
            sf = self.full_table[fk].astype(np.float64)
            ht = sf > 0
            if ht.any():
                pi = si[ht]
                best_p[pi] = np.clip(np.minimum(sf[ht], sc[ht]) / np.maximum(sc[ht], 1.0), 0.0, 1.0)
                match_lengths[pi] = L
                has_match[pi] = True
        return best_p, has_match, match_lengths

    def get_alpha(self, match_lengths, entropy):
        len_factor = self.base_alpha + (0.99 - self.base_alpha) * (match_lengths - 16) / 32
        ent_factor = 1.0 / (1.0 + np.exp(-2.0 * (entropy - 2.5)))
        return np.clip(len_factor * (0.5 + 0.5 * ent_factor), 0.0, 0.99)

    def update(self, val_np, start, end):
        n_primes = len(self.PRIMES)
        for L in self.PROBE_LENGTHS:
            first = max(start, L)
            if first > end:
                continue
            positions = np.arange(first, end + 1)
            tgt = val_np[positions].astype(np.uint64)
            ctx_hash = self._hash(val_np, positions, L)
            ctx_key = (ctx_hash & self.mask).astype(np.intp)
            fk = ((ctx_hash ^ (tgt * self.PRIMES[L % n_primes])) & self.mask).astype(np.intp)
            np.add.at(self.ctx_table, ctx_key, 1)
            np.add.at(self.full_table, fk, 1)


def eval_val_with_cache(
    args, base_model, rank, world_size, device, val_tokens,
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
    stride=64, batch_seqs=32, ttt_chunk_tokens=131072,
):
    """Sliding window eval with n-gram + phrase cache. Score-first legal."""
    seq_len = args.train_seq_len
    total_tokens = val_tokens.numel() - 1
    val_np = val_tokens.cpu().numpy().astype(np.int64)

    ngram = NgramEvalCache(max_order=12, alpha_high=0.95, min_count=1)
    phrase = LongPhraseCache(base_alpha=0.90)

    window_starts = [ws for ws in range(0, total_tokens, stride)
                     if min(ws + seq_len, total_tokens) - ws >= stride or ws == 0]
    num_chunks = (total_tokens + ttt_chunk_tokens - 1) // ttt_chunk_tokens
    chunk_windows = [[] for _ in range(num_chunks)]
    for ws in window_starts:
        end = min(ws + seq_len, total_tokens)
        wlen = end - ws
        s = 0 if ws == 0 else max(wlen - stride, 0)
        ci = min((ws + s) // ttt_chunk_tokens, num_chunks - 1)
        chunk_windows[ci].append(ws)

    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    base_model.eval()
    t0 = time.perf_counter()

    for ci in range(num_chunks):
        windows = chunk_windows[ci]
        if not windows:
            continue
        my_s = (len(windows) * rank) // world_size
        my_e = (len(windows) * (rank + 1)) // world_size
        my_windows = windows[my_s:my_e]

        with torch.inference_mode():
            for bi in range(0, len(my_windows), batch_seqs):
                batch_ws = my_windows[bi:bi + batch_seqs]
                bsz = len(batch_ws)
                x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
                y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
                wlens = []
                for i, ws in enumerate(batch_ws):
                    end = min(ws + seq_len, total_tokens)
                    wlen = end - ws
                    wlens.append(wlen)
                    chunk = val_tokens[ws:end + 1].to(dtype=torch.int64, device=device)
                    x_batch[i, :wlen] = chunk[:-1]
                    y_batch[i, :wlen] = chunk[1:]
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = base_model.forward_logits(x_batch)
                logits_f = logits.float() / 0.85  # temperature sharpening
                nll = F.cross_entropy(
                    logits_f.reshape(-1, logits_f.size(-1)),
                    y_batch.reshape(-1), reduction="none",
                ).reshape(bsz, seq_len)
                lp = F.log_softmax(logits_f, dim=-1)
                entropy_batch = -(lp.exp() * lp).sum(-1)

                for i, ws in enumerate(batch_ws):
                    wlen = wlens[i]
                    s = 0 if ws == 0 else max(wlen - stride, 0)
                    seg_len_i = wlen - s
                    if seg_len_i <= 0:
                        continue
                    p_model = torch.exp(-nll[i, s:wlen]).cpu().numpy().astype(np.float64)
                    ent = entropy_batch[i, s:wlen].cpu().numpy().astype(np.float64)
                    tgt_pos = np.arange(ws + s + 1, ws + wlen + 1)
                    tgt_toks = val_np[tgt_pos]

                    # N-gram blending
                    p_ng, ng_match, ng_orders = ngram.lookup(val_np, tgt_pos, tgt_toks)
                    if ng_match.any():
                        alpha = ngram.get_alpha(ent, ng_orders)
                        p_model = np.where(ng_match, (1 - alpha) * p_model + alpha * p_ng, p_model)

                    # Long phrase blending (on top)
                    p_ph, ph_match, ph_lens = phrase.lookup(val_np, tgt_pos, tgt_toks)
                    if ph_match.any():
                        pa = phrase.get_alpha(ph_lens, ent)
                        p_model = np.where(ph_match, (1 - pa) * p_model + pa * p_ph, p_model)

                    scored_nll = torch.from_numpy(-np.log(np.clip(p_model, 1e-12, 1.0))).to(
                        device=device, dtype=torch.float64)
                    loss_sum += scored_nll.sum()
                    token_count += float(seg_len_i)
                    tgt = y_batch[i, s:wlen]
                    prev = x_batch[i, s:wlen]
                    tb = base_bytes_lut[tgt].to(torch.float64)
                    tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
                    byte_count += tb.sum()

        # Update caches with ALL chunk tokens (full-chunk sharing across ranks)
        cs = ci * ttt_chunk_tokens
        ce = min((ci + 1) * ttt_chunk_tokens, total_tokens)
        ngram.update(val_np, cs, ce)
        phrase.update(val_np, cs, ce)

        if rank == 0 and (ci % 10 == 0 or ci == num_chunks - 1):
            elapsed = time.perf_counter() - t0
            bpb = (loss_sum.item() / max(token_count.item(), 1)) / math.log(2) * \
                  (token_count.item() / max(byte_count.item(), 1))
            print(f"  cache_eval [{ci+1}/{num_chunks}] bpb={bpb:.6f} time={elapsed:.1f}s", flush=True)

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)
    val_loss = (loss_sum / token_count).item()
    val_bpb = val_loss / math.log(2) * (token_count.item() / byte_count.item())
    if rank == 0:
        print(f"cache_eval:done val_loss={val_loss:.6f} val_bpb={val_bpb:.6f} "
              f"elapsed={time.perf_counter() - t0:.1f}s")
    return val_loss, val_bpb
