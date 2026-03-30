#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F


_BASE_PATH = Path(__file__).with_name("train_gpt.py")
_SPEC = importlib.util.spec_from_file_location("rascal_sota_base", _BASE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Failed to load base trainer from {_BASE_PATH}")
_BASE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BASE)


def _parse_sparse_gap_patterns(raw: str) -> list[tuple[int, ...]]:
    """Parse semicolon-separated sparse context gap patterns.
    Example: '1,3;1,3,5;1,2,4,8'
    """
    if not raw.strip():
        return []
    out: list[tuple[int, ...]] = []
    for item in raw.split(";"):
        item = item.strip()
        if not item:
            continue
        try:
            gaps = tuple(int(x.strip()) for x in item.split(",") if x.strip())
        except ValueError:
            continue
        if not gaps or any(g <= 0 for g in gaps):
            continue
        out.append(gaps)
    return out


def _ngram_bulk_update_sparse(
    val_np,
    start,
    end,
    ctx_tables,
    full_tables,
    min_order,
    max_order,
    primes,
    mask,
    sparse_patterns_by_order: dict[int, list[tuple[int, ...]]] | None = None,
):
    """Bulk update with standard contiguous + sparse skip-gram contexts."""
    t = val_np[start:end].astype(np.uint64)
    n = len(t)
    for order in range(min_order, max_order + 1):
        ctx_width = order - 1
        gap_patterns = [tuple(range(ctx_width, 0, -1))]
        if sparse_patterns_by_order is not None:
            gap_patterns.extend(sparse_patterns_by_order.get(order, []))
        for gaps in gap_patterns:
            if len(gaps) != ctx_width:
                continue
            max_gap = max(gaps)
            if n <= max_gap:
                continue
            tgt = t[max_gap:]
            ctx_hash = np.zeros(tgt.shape[0], dtype=np.uint64)
            for k, gap in enumerate(gaps):
                ctx_hash ^= t[max_gap - gap:n - gap] * primes[k % len(primes)]
            ctx_key = (ctx_hash & mask).astype(np.int64)
            full_key = ((ctx_hash ^ (tgt * primes[ctx_width % len(primes)])) & mask).astype(np.int64)
            ctx_tables[order] += np.bincount(ctx_key, minlength=len(ctx_tables[order])).astype(np.uint32)
            full_tables[order] += np.bincount(full_key, minlength=len(full_tables[order])).astype(np.uint32)


def eval_val_sliding_hashed_ngram_sparse(
    args,
    base_model,
    rank,
    world_size,
    device,
    val_tokens,
    base_bytes_lut,
    has_leading_space_lut,
    is_boundary_token_lut,
    stride,
    order,
    alpha,
    min_count,
    buckets,
    max_seconds: float = 0.0,
    batch_seqs: int = 128,
    eval_seq_len: int | None = None,
):
    """Ablation: hashed n-gram eval with additional sparse skip-gram contexts."""
    min_order = max(args.ngram_eval_min_order, 2)
    max_order = max(order, min_order)
    adaptive = args.ngram_eval_adaptive
    alpha_min = args.ngram_eval_alpha_min
    alpha_max = args.ngram_eval_alpha_max
    ent_center = args.ngram_eval_entropy_center
    ent_scale = args.ngram_eval_entropy_scale

    fixed_order_mults = None
    if args.ngram_order_mults_str:
        fixed_order_mults = np.array([float(x) for x in args.ngram_order_mults_str.split(",")], dtype=np.float64)

    seq_len = eval_seq_len or args.train_seq_len
    total_tokens = val_tokens.numel() - 1
    all_window_starts = [ws for ws in range(0, total_tokens, stride) if min(ws + seq_len, total_tokens) - ws >= 1]
    total_scored_tokens = 0.0
    for ws in all_window_starts:
        end = min(ws + seq_len, total_tokens)
        wlen = end - ws
        s = 0 if ws == 0 else max(wlen - stride, 0)
        total_scored_tokens += float(max(wlen - s, 0))

    chunk_tokens = int(os.environ.get("NGRAM_CHUNK_TOKENS", "1048576"))
    num_chunks = (total_tokens + chunk_tokens - 1) // chunk_tokens
    chunk_windows: list[list[int]] = [[] for _ in range(num_chunks)]
    for ws in all_window_starts:
        end = min(ws + seq_len, total_tokens)
        wlen = end - ws
        s = 0 if ws == 0 else max(wlen - stride, 0)
        scored_start = ws + s
        ci = min(scored_start // chunk_tokens, num_chunks - 1)
        chunk_windows[ci].append(ws)

    val_np = val_tokens.numpy()
    ctx_tables = {n: np.zeros((buckets,), dtype=np.uint32) for n in range(min_order, max_order + 1)}
    full_tables = {n: np.zeros((buckets,), dtype=np.uint32) for n in range(min_order, max_order + 1)}
    mask = np.uint64(buckets - 1)
    primes = np.array(
        [
            np.uint64(36313),
            np.uint64(27191),
            np.uint64(51647),
            np.uint64(81929),
            np.uint64(131071),
            np.uint64(174763),
            np.uint64(233017),
        ],
        dtype=np.uint64,
    )

    sparse_patterns = _parse_sparse_gap_patterns(os.environ.get("NGRAM_SPARSE_PATTERNS", ""))
    sparse_by_order: dict[int, list[tuple[int, ...]]] = {
        n: [g for g in sparse_patterns if len(g) == (n - 1)] for n in range(min_order, max_order + 1)
    }

    loss_sum = 0.0
    token_count = 0.0
    byte_count = 0.0

    num_ent_bins = 3
    num_cnt_bins = 3
    ent_edges = np.array([ent_center - 1.0, ent_center + 1.0])
    cnt_edges = np.array([5.0, 50.0])
    total_cells = num_ent_bins * num_cnt_bins
    cc = getattr(args, "cubric_cadence", 0)
    cubric_on = cc > 0
    cubric_fired = 0
    if cubric_on:
        warm = {2: 0.45, 3: 0.30, 4: 0.45, 5: 1.88, 6: 2.00, 7: 2.00, 8: 2.00, 9: 2.00}
        c_alpha_mult = {n: [warm.get(n, 1.0)] * total_cells for n in range(min_order, max_order + 1)}
        c_hits = {n: [0] * total_cells for n in range(min_order, max_order + 1)}
        c_beats = {n: [0] * total_cells for n in range(min_order, max_order + 1)}

    base_model.eval()
    compiled_logits = _BASE.maybe_compile(base_model.forward_logits, enabled=args.compile_enabled, fullgraph=False)
    t0 = _BASE.time.perf_counter()
    deadline = (t0 + max_seconds) if max_seconds > 0.0 else None
    cutoff_hit = False

    if rank == 0:
        sparse_total = sum(len(v) for v in sparse_by_order.values())
        print(
            f"ngram_eval_sparse:chunks={num_chunks} chunk_tokens={chunk_tokens} "
            f"windows={len(all_window_starts)} sparse_patterns={sparse_total}",
            flush=True,
        )

    with torch.inference_mode():
        for ci in range(num_chunks):
            if deadline is not None and _BASE.time.perf_counter() >= deadline:
                cutoff_hit = True
                break

            windows = chunk_windows[ci]
            if not windows:
                continue
            my_s = (len(windows) * rank) // world_size
            my_e = (len(windows) * (rank + 1)) // world_size
            my_windows = windows[my_s:my_e]

            for bi in range(0, len(my_windows), batch_seqs):
                batch_ws = my_windows[bi : bi + batch_seqs]
                bsz = len(batch_ws)
                x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
                y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
                wlens: list[int] = []
                for i, ws in enumerate(batch_ws):
                    end = min(ws + seq_len, total_tokens)
                    wlen = end - ws
                    wlens.append(wlen)
                    chunk = val_tokens[ws : end + 1].to(dtype=torch.int64, device=device)
                    x_batch[i, :wlen] = chunk[:-1]
                    y_batch[i, :wlen] = chunk[1:]

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = compiled_logits(x_batch)
                logits_f = logits.float()
                nll = F.cross_entropy(
                    logits_f.reshape(-1, logits_f.size(-1)),
                    y_batch.reshape(-1),
                    reduction="none",
                ).reshape(bsz, seq_len)

                for i, ws in enumerate(batch_ws):
                    wlen = wlens[i]
                    s = 0 if ws == 0 else max(wlen - stride, 0)
                    seg_len = wlen - s
                    if seg_len <= 0:
                        continue

                    seg_nll = nll[i, s:wlen].to(torch.float64).cpu().numpy()
                    seg_model_p = np.exp(-seg_nll)

                    if adaptive:
                        log_probs = F.log_softmax(logits_f[i, s:wlen], dim=-1)
                        probs_a = log_probs.exp()
                        entropy = -(probs_a * log_probs).sum(dim=-1).cpu().numpy()
                        sig = 1.0 / (1.0 + np.exp(-ent_scale * (entropy - ent_center)))
                        per_token_alpha = alpha_min + (alpha_max - alpha_min) * sig
                        ent_bins = np.digitize(entropy, ent_edges).astype(np.int32)
                    else:
                        per_token_alpha = np.full(seg_len, alpha)
                        ent_bins = np.ones(seg_len, dtype=np.int32)

                    global_j = np.arange(ws + s + 1, ws + wlen + 1, dtype=np.int64)
                    p_ng = np.zeros(seg_len, dtype=np.float64)
                    ng_matched = np.zeros(seg_len, dtype=np.bool_)
                    ng_ord = np.zeros(seg_len, dtype=np.int32)
                    ng_ctx_count = np.zeros(seg_len, dtype=np.float64)
                    tgt_np = val_np[global_j].astype(np.uint64)

                    for n in range(max_order, min_order - 1, -1):
                        ctx_width = n - 1
                        gap_patterns = [tuple(range(ctx_width, 0, -1))]
                        gap_patterns.extend(sparse_by_order.get(n, []))
                        valid = (global_j >= 1) & (~ng_matched)
                        if not valid.any():
                            continue
                        best_p = np.full(seg_len, -1.0, dtype=np.float64)
                        best_ctx = np.zeros(seg_len, dtype=np.float64)
                        for gaps in gap_patterns:
                            if len(gaps) != ctx_width:
                                continue
                            max_gap = max(gaps)
                            valid_g = (global_j >= max_gap) & (~ng_matched)
                            if not valid_g.any():
                                continue
                            v_idx = np.nonzero(valid_g)[0]
                            jv = global_j[v_idx]
                            ctx_hash = np.zeros(len(jv), dtype=np.uint64)
                            for k, gap in enumerate(gaps):
                                tok = val_np[jv - gap].astype(np.uint64)
                                ctx_hash ^= tok * primes[k % len(primes)]
                            ctx_key = (ctx_hash & mask).astype(np.int64)
                            full_key = ((ctx_hash ^ (tgt_np[v_idx] * primes[ctx_width % len(primes)])) & mask).astype(
                                np.int64
                            )
                            ctx_counts = ctx_tables[n][ctx_key].astype(np.float64)
                            full_counts = full_tables[n][full_key].astype(np.float64)
                            has_data = ctx_counts >= float(min_count)
                            if not has_data.any():
                                continue
                            p = np.minimum(full_counts, ctx_counts) / np.maximum(ctx_counts, 1.0)
                            p = np.clip(p, 0.0, 1.0)
                            hit_idx = v_idx[has_data]
                            p_hit = p[has_data]
                            ctx_hit = ctx_counts[has_data]
                            better = p_hit > best_p[hit_idx]
                            if better.any():
                                chosen = hit_idx[better]
                                best_p[chosen] = p_hit[better]
                                best_ctx[chosen] = ctx_hit[better]
                        has_best = best_p >= 0.0
                        if has_best.any():
                            p_ng[has_best] = best_p[has_best]
                            ng_matched[has_best] = True
                            ng_ord[has_best] = n
                            ng_ctx_count[has_best] = best_ctx[has_best]

                    if ng_matched.any():
                        m_idx = np.nonzero(ng_matched)[0]
                        if adaptive and args.ngram_entropy_shift:
                            matched_ords = ng_ord[m_idx].astype(np.float64)
                            shifted_centers = ent_center - 0.25 * (matched_ords - float(min_order))
                            shifted_sig = 1.0 / (1.0 + np.exp(-ent_scale * (entropy[m_idx] - shifted_centers)))
                            per_token_alpha[m_idx] = alpha_min + (alpha_max - alpha_min) * shifted_sig

                        if fixed_order_mults is not None:
                            a = per_token_alpha[m_idx].copy()
                            mult_indices = ng_ord[m_idx] - min_order
                            mult_indices = np.clip(mult_indices, 0, len(fixed_order_mults) - 1)
                            a *= fixed_order_mults[mult_indices]
                            np.clip(a, 0.0, 0.95, out=a)
                        elif cubric_on:
                            a = per_token_alpha[m_idx].copy()
                            m_ent_bins = ent_bins[m_idx]
                            m_cnt_bins = np.digitize(ng_ctx_count[m_idx], cnt_edges).astype(np.int32)
                            for n in range(min_order, max_order + 1):
                                om = ng_ord[m_idx] == n
                                if not om.any():
                                    continue
                                for eb in range(num_ent_bins):
                                    for cb in range(num_cnt_bins):
                                        cell = eb * num_cnt_bins + cb
                                        mask_ecb = om & (m_ent_bins == eb) & (m_cnt_bins == cb)
                                        if mask_ecb.any():
                                            c_hits[n][cell] += int(mask_ecb.sum())
                                            c_beats[n][cell] += int(
                                                (p_ng[m_idx[mask_ecb]] > seg_model_p[m_idx[mask_ecb]]).sum()
                                            )
                                            a[mask_ecb] *= c_alpha_mult[n][cell]
                            np.clip(a, 0.0, 0.95, out=a)
                        else:
                            a = per_token_alpha[m_idx]
                        seg_model_p[m_idx] = (1.0 - a) * seg_model_p[m_idx] + a * p_ng[m_idx]

                    seg_nll = -np.log(np.clip(seg_model_p, 1e-12, 1.0))
                    loss_sum += float(seg_nll.sum())
                    token_count += float(seg_len)
                    tgt = y_batch[i, s:wlen]
                    prev = x_batch[i, s:wlen]
                    tb = base_bytes_lut[tgt].to(torch.float64)
                    tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
                    byte_count += float(tb.sum().item())

            chunk_start = ci * chunk_tokens
            chunk_end = min((ci + 1) * chunk_tokens, total_tokens)
            _ngram_bulk_update_sparse(
                val_np,
                chunk_start,
                chunk_end + 1,
                ctx_tables,
                full_tables,
                min_order,
                max_order,
                primes,
                mask,
                sparse_patterns_by_order=sparse_by_order,
            )

            if cubric_on:
                all_rates = []
                for n in range(min_order, max_order + 1):
                    for cell in range(total_cells):
                        if c_hits[n][cell] >= 8:
                            all_rates.append(c_beats[n][cell] / c_hits[n][cell])
                if len(all_rates) >= 4:
                    avg_rate = sum(all_rates) / len(all_rates)
                    for n in range(min_order, max_order + 1):
                        for cell in range(total_cells):
                            if c_hits[n][cell] >= 8:
                                rate = c_beats[n][cell] / c_hits[n][cell]
                                if rate > avg_rate + 0.05:
                                    c_alpha_mult[n][cell] = min(c_alpha_mult[n][cell] * 1.03, 2.0)
                                elif rate < avg_rate - 0.05:
                                    c_alpha_mult[n][cell] = max(c_alpha_mult[n][cell] * 0.97, 0.3)
                cubric_fired += 1
                c_hits = {n: [0] * total_cells for n in range(min_order, max_order + 1)}
                c_beats = {n: [0] * total_cells for n in range(min_order, max_order + 1)}

    loss_t = torch.tensor(loss_sum, device=device, dtype=torch.float64)
    toks_t = torch.tensor(token_count, device=device, dtype=torch.float64)
    bytes_t = torch.tensor(byte_count, device=device, dtype=torch.float64)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(toks_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(bytes_t, op=dist.ReduceOp.SUM)
    loss_sum = loss_t.item()
    token_count = toks_t.item()
    byte_count = bytes_t.item()

    coverage = token_count / max(total_scored_tokens, 1.0)
    if cutoff_hit:
        elapsed = _BASE.time.perf_counter() - t0
        print(
            f"ngram_eval_sparse:cutoff max_seconds={max_seconds:.1f} "
            f"coverage={coverage*100:.2f}% elapsed={elapsed:.0f}s",
            flush=True,
        )
    val_loss = loss_sum / max(token_count, 1.0)
    val_bpb = val_loss / math.log(2.0) * (token_count / max(byte_count, 1.0))
    base_model.train()
    return val_loss, val_bpb, coverage


def main() -> None:
    # Match SOTA profile defaults, but force signal-ablation settings unless user overrides.
    os.environ.setdefault("ITERATIONS", "2200")
    os.environ.setdefault("WARMDOWN_ITERS", "0")
    os.environ.setdefault("MAX_WALLCLOCK_SECONDS", "0")
    os.environ.setdefault("SKIP_GPTQ", "1")
    os.environ.setdefault("LOADER_MODE", "coprime")
    os.environ.setdefault("COPRIME_MAX_LOADED_SHARDS", "1")
    os.environ.setdefault("COPRIME_SHARDS_PER_BATCH", "1")
    os.environ.setdefault("COPRIME_SHARD_HOLD_STEPS", "64")
    os.environ.setdefault("COMPLEMENT_ALPHA", "0")
    os.environ.setdefault("XSA_LAST_N", "11")
    os.environ.setdefault("BIGRAM_VOCAB_SIZE", "2048")
    os.environ.setdefault("ROPE_DIMS", "16")
    os.environ.setdefault("SWA_EVERY", "50")
    os.environ.setdefault("MTP_NUM_HEADS", "0")
    os.environ.setdefault("TRIGRAM", "0")
    os.environ.setdefault("NGRAM_ENTROPY_SHIFT", "0")
    # Turn on hashed n-gram eval to test sparse skip-gram concept.
    os.environ.setdefault("SKIP_FINAL_EVAL", "0")
    os.environ.setdefault("NGRAM_EVAL_ORDER", "7")
    os.environ.setdefault("NGRAM_EVAL_MIN_ORDER", "2")
    os.environ.setdefault("NGRAM_EVAL_ALPHA", "0.30")
    os.environ.setdefault("NGRAM_EVAL_ADAPTIVE", "1")
    os.environ.setdefault("NGRAM_EVAL_MAX_SECONDS", "180")
    os.environ.setdefault(
        "NGRAM_SPARSE_PATTERNS",
        (
            "1,3;1,2;"
            "1,3,5;1,2,4;"
            "1,3,5,7;1,2,4,8;"
            "1,3,5,7,9;1,2,4,8,16;"
            "1,3,5,7,9,11;1,2,4,8,16,32"
        ),
    )

    _BASE._ngram_bulk_update = _ngram_bulk_update_sparse
    _BASE.eval_val_sliding_hashed_ngram = eval_val_sliding_hashed_ngram_sparse

    if int(os.environ.get("RANK", "0")) == 0:
        print(
            "ablation:sparse_skipgram_eval enabled "
            f"NGRAM_SPARSE_PATTERNS={os.environ.get('NGRAM_SPARSE_PATTERNS', '')}",
            flush=True,
        )
    _BASE.main()


if __name__ == "__main__":
    main()

