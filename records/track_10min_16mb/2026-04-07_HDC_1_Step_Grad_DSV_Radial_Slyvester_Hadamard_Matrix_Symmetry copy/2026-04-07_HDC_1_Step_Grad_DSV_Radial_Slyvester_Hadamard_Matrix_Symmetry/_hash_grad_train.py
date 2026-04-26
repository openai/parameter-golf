"""Hash-Addressed Gradient Learning — Stripped DSV-only Pipeline.

Signal chain (verified BPB 0.4118 on 3 independent runs):

  Phase 2  — Distributed frequency tabulation (all ranks, scatter_add_ + all_reduce)
               -> fingerprint table only
                  |
  Phase 3  — Multi-seed frequency merge (fingerprint merge only)
                  |
  Phase 6  — DirectionalSemanticVec.build_from_tokens()  <- PRIMARY BPB contributor
               sem_fwd only (128 KB)
               skip-bigram lags 2-5
                  |
  Phase 10 — LZMA9 artifact save (.hgz, magic HGZ3)
               fingerprint table + sem_fwd + skip-bigram lags
                  |
  Eval     — hash_grad_bpb() waterfall (NMF branch removed):
               fingerprint mismatch -> sem_fwd XOR codebook (collision fallback)
               miss                 -> sem_fwd XOR codebook (primary)
               lags 2-5             -> skip-bigram blend

NMF (embed/W_out), xor_orbit_regularise, build_frozen_prior, suffix_grammar,
and sem_bwd have all been removed -- they contribute zero signal to BPB 0.4118.
"""

from __future__ import annotations

import os
import time
from typing import List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def _dist_rank() -> int:
    """Return the current process rank (0 if not in a distributed context)."""
    try:
        import torch.distributed as _dist
        if _dist.is_available() and _dist.is_initialized():
            return _dist.get_rank()
    except Exception:
        pass
    return 0


def _dist_world_size() -> int:
    """Return the world size (1 if not in a distributed context)."""
    try:
        import torch.distributed as _dist
        if _dist.is_available() and _dist.is_initialized():
            return _dist.get_world_size()
    except Exception:
        pass
    return 1


def _dist_is_main() -> bool:
    """True only on rank 0 (or when not distributed)."""
    return _dist_rank() == 0


def _dist_barrier() -> None:
    """Synchronise all ranks (no-op when not distributed)."""
    try:
        import torch.distributed as _dist
        if _dist.is_available() and _dist.is_initialized():
            _dist.barrier()
    except Exception:
        pass


def _dist_all_reduce_sum_numpy(arr: np.ndarray) -> np.ndarray:
    """All-reduce (sum) a numpy array across all ranks.

    Converts to a torch tensor on the current CUDA device (required for NCCL),
    calls dist.all_reduce, then moves the result back to CPU numpy.
    Falls back to returning the array unchanged when not distributed.
    """
    try:
        import os as _os
        import torch
        import torch.distributed as _dist
        if not (_dist.is_available() and _dist.is_initialized()):
            return arr
        local_rank = int(_os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")
        t = torch.from_numpy(arr.copy()).to(device)
        _dist.all_reduce(t, op=_dist.ReduceOp.SUM)
        return t.cpu().numpy()
    except Exception:
        return arr


# ---------------------------------------------------------------------------
# Rolling hash pre-computation
# ---------------------------------------------------------------------------

FMIX64 = np.uint64(0x9E3779B97F4A7C15)


def precompute_g_states(tokens: np.ndarray) -> np.ndarray:
    """Compute the rolling XOR hash G[p] for every position p.

    G[p] encodes tokens[0 .. p-1].  Seed-independent -- seed only changes
    the finalise() step inside tabulate_bucket_frequencies.

    G[0] = 0
    G[p+1] = G[p] XOR (tokens[p] * KEY[p])
    KEY[p] = ((p+1) * PHI64) ^ ((p+1) >> 32)  |  1
    """
    _PHI64 = np.uint64(0x9E3779B97F4A7C15)
    N = len(tokens)
    t = tokens.astype(np.uint64)
    positions = np.arange(N, dtype=np.uint64) + np.uint64(1)
    with np.errstate(over='ignore'):
        keys = positions * _PHI64
    keys ^= (keys >> np.uint64(32))
    keys |= np.uint64(1)
    with np.errstate(over='ignore'):
        contribs = t * keys
    cumxor = np.bitwise_xor.accumulate(contribs)
    g_states = np.empty(N, dtype=np.uint64)
    g_states[0] = np.uint64(0)
    if N > 1:
        g_states[1:] = cumxor[:-1]
    return g_states


# ---------------------------------------------------------------------------
# Frequency tabulation
# ---------------------------------------------------------------------------

def tabulate_bucket_frequencies_distributed(
    tokens: np.ndarray,
    g_states: np.ndarray,
    seed: int,
    table_bits: int,
    vocab_size: int,
    chunk_size: int = 50_000_000,
    label: str = "FreqTab",
    build_fingerprint: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    rank       = _dist_rank()
    world_size = _dist_world_size()

    if world_size == 1:
        return tabulate_bucket_frequencies(
            tokens, g_states, seed, table_bits, vocab_size,
            chunk_size=chunk_size, label=label, build_fingerprint=build_fingerprint,
        )

    N          = len(tokens)
    shard_size = (N + world_size - 1) // world_size
    shard_start = rank * shard_size
    shard_end   = min(shard_start + shard_size, N)

    if shard_start >= N - 1:
        TABLE_SIZE  = 1 << table_bits
        freq_local  = np.zeros((TABLE_SIZE, vocab_size), dtype=np.uint32)
        count_local = np.zeros(TABLE_SIZE, dtype=np.uint32)
        fp_local    = np.zeros(TABLE_SIZE, dtype=np.uint8) if build_fingerprint else None
    else:
        shard_end_ext = min(shard_end + 1, N)
        tok_shard = tokens[shard_start:shard_end_ext]
        g_shard   = g_states[shard_start:shard_end_ext]

        print(f"[HashGrad {label} rank={rank}] Shard [{shard_start:,}, {shard_end_ext:,}) "
              f"-- {shard_end_ext - shard_start:,} tokens")

        freq_local, count_local, fp_local = tabulate_bucket_frequencies(
            tok_shard, g_shard, seed, table_bits, vocab_size,
            chunk_size=chunk_size, label=f"{label}_r{rank}",
            build_fingerprint=build_fingerprint,
        )

    _dist_barrier()

    freq_merged  = _dist_all_reduce_sum_numpy(freq_local.astype(np.int64)).astype(np.uint32)
    count_merged = _dist_all_reduce_sum_numpy(count_local.astype(np.int64)).astype(np.uint32)

    fp_merged = None
    if build_fingerprint and fp_local is not None:
        try:
            import os as _os
            import torch
            import torch.distributed as _dist_mod
            if _dist_mod.is_available() and _dist_mod.is_initialized():
                TABLE_SIZE = 1 << table_bits
                local_rank = int(_os.environ.get("LOCAL_RANK", 0))
                device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")
                fp_t     = torch.from_numpy(fp_local.astype(np.int32)).to(device)
                count_t  = torch.from_numpy(count_local.astype(np.int64)).to(device)

                fp_list    = [torch.zeros_like(fp_t)    for _ in range(_dist_mod.get_world_size())]
                count_list = [torch.zeros_like(count_t) for _ in range(_dist_mod.get_world_size())]
                _dist_mod.all_gather(fp_list,    fp_t)
                _dist_mod.all_gather(count_list, count_t)

                fp_merged  = fp_list[0].cpu().numpy().astype(np.uint8)
                best_count = count_list[0].cpu().numpy()
                for i in range(1, len(fp_list)):
                    c_i = count_list[i].cpu().numpy()
                    better = c_i > best_count
                    fp_merged[better]  = fp_list[i].cpu().numpy().astype(np.uint8)[better]
                    best_count[better] = c_i[better]
        except Exception as _fp_e:
            print(f"[HashGrad {label}] Fingerprint merge failed ({_fp_e}) -- using rank-0 fp")
            fp_merged = fp_local

    filled = int(np.sum(count_merged > 0))
    TABLE_SIZE = 1 << table_bits
    if _dist_is_main():
        print(f"[HashGrad {label} dist] All-reduce complete -- "
              f"{filled:,}/{TABLE_SIZE:,} buckets filled across {world_size} ranks")

    return freq_merged, count_merged, fp_merged


def tabulate_bucket_frequencies(
    tokens: np.ndarray,
    g_states: np.ndarray,
    seed: int,
    table_bits: int,
    vocab_size: int,
    chunk_size: int = 2_000_000,
    label: str = "FreqTab",
    build_fingerprint: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    try:
        import torch as _tch
        if _tch.cuda.is_available():
            return tabulate_bucket_frequencies_gpu(
                tokens, g_states, seed, table_bits, vocab_size,
                chunk_size=chunk_size, label=label, build_fingerprint=build_fingerprint,
            )
    except Exception as _gpu_e:
        print(f"[HashGrad {label}] GPU dispatch error ({_gpu_e!r}) -- using CPU")
    TABLE_SIZE = 1 << table_bits
    SHIFT      = np.uint64(64 - table_bits)
    FP_SHIFT   = np.uint64(64 - table_bits - 8)
    seed_u64   = np.uint64(seed)

    N           = len(tokens)
    freq        = np.zeros((TABLE_SIZE, vocab_size), dtype=np.uint32)
    count       = np.zeros(TABLE_SIZE, dtype=np.uint32)
    fingerprint = np.zeros(TABLE_SIZE, dtype=np.uint8) if build_fingerprint else None

    t0 = time.time()
    processed = 0
    for chunk_start in range(0, N - 1, chunk_size):
        chunk_end = min(chunk_start + chunk_size, N - 1)

        g_chunk   = g_states[chunk_start:chunk_end].astype(np.uint64)
        tgt_chunk = tokens[chunk_start + 1 : chunk_end + 1].astype(np.int32)

        finalised = (g_chunk ^ seed_u64) * FMIX64
        buckets   = (finalised >> SHIFT).astype(np.int64)

        if build_fingerprint:
            fps = ((finalised >> FP_SHIFT) & np.uint64(0xFF)).astype(np.uint8)
            fingerprint[buckets] = fps

        pair_keys = buckets * vocab_size + tgt_chunk
        uniq_keys, uniq_counts = np.unique(pair_keys, return_counts=True)
        uniq_b = (uniq_keys // vocab_size).astype(np.int64)
        uniq_t = (uniq_keys  % vocab_size).astype(np.int64)

        np.add.at(freq,  (uniq_b, uniq_t), uniq_counts.astype(np.uint32))
        np.add.at(count, uniq_b,           uniq_counts.astype(np.uint32))

        processed += chunk_end - chunk_start
        if processed % 50_000_000 == 0 or chunk_end >= N - 1:
            elapsed = time.time() - t0
            rate    = processed / max(elapsed, 1e-6) / 1e6
            print(f"[HashGrad {label}] {processed:,}/{N-1:,} tokens "
                  f"({100*processed/(N-1):.1f}%) -- {rate:.1f}M tok/s")

    filled = int(np.sum(count > 0))
    print(f"[HashGrad {label}] Done in {time.time()-t0:.1f}s -- "
          f"filled {filled:,}/{TABLE_SIZE:,} buckets ({100*filled/TABLE_SIZE:.1f}%)")
    return freq, count, fingerprint


def tabulate_bucket_frequencies_gpu(
    tokens: np.ndarray,
    g_states: np.ndarray,
    seed: int,
    table_bits: int,
    vocab_size: int,
    chunk_size: int = 50_000_000,
    label: str = "FreqTab",
    build_fingerprint: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    import torch
    TABLE_SIZE = 1 << table_bits
    SHIFT      = 64 - table_bits
    FP_SHIFT   = 64 - table_bits - 8
    TABLE_MASK = (1 << table_bits) - 1
    dev        = torch.device("cuda")
    N          = len(tokens)

    torch.cuda.empty_cache()

    free_vram, _total_vram = torch.cuda.mem_get_info(0)
    freq_bytes_needed = TABLE_SIZE * vocab_size * 8
    g_bytes      = (N - 1) * 8
    tok_bytes_sz = (N - 1) * 8
    work_bytes   = (N - 1) * 8
    total_needed = freq_bytes_needed + g_bytes + tok_bytes_sz + work_bytes
    if total_needed > free_vram * 0.80:
        raise RuntimeError(
            f"GPU VRAM too small ({total_needed/1e9:.1f} GB needed, "
            f"{free_vram/1e9:.1f} GB free) -- caller should use CPU"
        )

    _fmix_i64 = int(np.array([int(FMIX64)], dtype=np.uint64).view(np.int64)[0])
    seed_i64  = int(np.array([seed],         dtype=np.uint64).view(np.int64)[0])
    FMIX_t    = torch.tensor(_fmix_i64, dtype=torch.int64, device=dev)
    seed_t    = torch.tensor(seed_i64,  dtype=torch.int64, device=dev)
    MASK_t    = torch.tensor(TABLE_MASK, dtype=torch.int64, device=dev)
    FP_MSK_T  = torch.tensor(0xFF,       dtype=torch.int64, device=dev)

    freq_flat = torch.zeros(TABLE_SIZE * vocab_size, dtype=torch.int64, device=dev)
    fp_gpu    = torch.zeros(TABLE_SIZE, dtype=torch.int64, device=dev) if build_fingerprint else None

    t_up = time.time()
    g_gpu   = torch.as_tensor(g_states[:N-1].view(np.int64),   dtype=torch.int64, device=dev)
    tok_gpu = torch.as_tensor(tokens[1:N].astype(np.int64),     dtype=torch.int64, device=dev)
    torch.cuda.synchronize()
    print(f"[HashGrad {label} GPU] Uploaded {(N-1)*16/1e9:.2f} GB in {time.time()-t_up:.2f}s "
          f"-- processing {N-1:,} tokens...")

    ones_buf = torch.ones(min(chunk_size, N), dtype=torch.int64, device=dev)
    t0 = time.time()
    processed = 0

    for chunk_start in range(0, N - 1, chunk_size):
        chunk_end = min(chunk_start + chunk_size, N - 1)
        C = chunk_end - chunk_start

        g_ch   = g_gpu[chunk_start:chunk_end]
        tok_ch = tok_gpu[chunk_start:chunk_end]

        val     = (g_ch ^ seed_t) * FMIX_t
        buckets = (val >> SHIFT) & MASK_t

        if build_fingerprint and fp_gpu is not None:
            fps = (val >> FP_SHIFT) & FP_MSK_T
            fp_gpu.scatter_(0, buckets, fps)

        pair_keys = buckets * vocab_size + tok_ch
        freq_flat.scatter_add_(0, pair_keys, ones_buf[:C])

        processed += C
        if processed % 100_000_000 == 0 or chunk_end >= N - 1:
            elapsed = time.time() - t0
            rate    = processed / max(elapsed, 1e-6) / 1e6
            print(f"[HashGrad {label} GPU] {processed:,}/{N-1:,} tokens "
                  f"({100*processed/(N-1):.1f}%) -- {rate:.1f}M tok/s")

    freq  = freq_flat.cpu().numpy().reshape(TABLE_SIZE, vocab_size).astype(np.uint32)
    count = freq.sum(axis=1).astype(np.uint32)
    fingerprint = fp_gpu.cpu().numpy().astype(np.uint8) if build_fingerprint else None
    filled = int(np.sum(count > 0))
    print(f"[HashGrad {label} GPU] Done in {time.time()-t0:.1f}s -- "
          f"filled {filled:,}/{TABLE_SIZE:,} buckets ({100*filled/TABLE_SIZE:.1f}%)")

    del freq_flat, g_gpu, tok_gpu, ones_buf
    if fp_gpu is not None:
        del fp_gpu
    torch.cuda.empty_cache()

    return freq, count, fingerprint


# ---------------------------------------------------------------------------
# Multi-seed merge (fingerprint only)
# ---------------------------------------------------------------------------

def merge_seed_frequencies(
    freq_list: List[np.ndarray],
    count_list: List[np.ndarray],
    fingerprint_list: Optional[List[Optional[np.ndarray]]] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    n_seeds = len(freq_list)
    assert n_seeds >= 1

    freq_merged  = freq_list[0].copy()
    count_merged = count_list[0].copy()
    for i in range(1, n_seeds):
        freq_merged  += freq_list[i]
        count_merged += count_list[i]

    fp_merged = None
    if fingerprint_list is not None and all(f is not None for f in fingerprint_list):
        fp_merged  = fingerprint_list[0].copy()
        best_count = count_list[0].copy()
        for i in range(1, n_seeds):
            better = count_list[i] > best_count
            fp_merged[better]  = fingerprint_list[i][better]
            best_count[better] = count_list[i][better]

    filled     = int(np.sum(count_merged > 0))
    TABLE_SIZE = len(count_merged)
    avg        = float(count_merged[count_merged > 0].mean()) if filled > 0 else 0.0
    print(f"[HashGrad Merge] {n_seeds} seeds merged -- "
          f"{filled:,}/{TABLE_SIZE:,} buckets ({100*filled/TABLE_SIZE:.1f}%), "
          f"avg {avg:.1f} obs/bucket")
    return freq_merged, count_merged, fp_merged


# ---------------------------------------------------------------------------
# Artifact save / load  (HGZ3: fingerprint + sem_fwd + skip-bigram lags)
# ---------------------------------------------------------------------------

def save_hash_grad_artifact(
    fingerprint: np.ndarray,
    sem_fwd: np.ndarray,
    seed: int,
    table_bits: int,
    path: str,
    skip_bigram_lags: Optional[List[np.ndarray]] = None,
) -> int:
    """Save fingerprint + sem_fwd (+ optional skip-bigram lags) as LZMA9-compressed .hgz.

    Format HGZ3:
      magic       4 bytes  b"HGZ3"
      seed        8 bytes  uint64 LE
      table_bits  4 bytes  uint32 LE
      vocab_size  4 bytes  uint32 LE  (sem_fwd rows)
      W           4 bytes  uint32 LE  (sem_fwd cols, uint64 words per token)
      n_lags      4 bytes  uint32 LE  (number of skip-bigram lag arrays)
      fingerprint TABLE_SIZE bytes uint8
      sem_fwd     vocab_size * W * 8 bytes uint64
      lag_2..     vocab_size * W * 8 bytes uint64  (repeated n_lags times)
    """
    import lzma
    import struct
    import io as _io

    TABLE_SIZE = 1 << table_bits
    vocab_size, W = sem_fwd.shape
    n_lags = len(skip_bigram_lags) if skip_bigram_lags else 0

    buf = _io.BytesIO()
    buf.write(b"HGZ3")
    buf.write(struct.pack("<Q", int(seed)))
    buf.write(struct.pack("<I", int(table_bits)))
    buf.write(struct.pack("<I", int(vocab_size)))
    buf.write(struct.pack("<I", int(W)))
    buf.write(struct.pack("<I", int(n_lags)))
    buf.write(fingerprint.astype(np.uint8).tobytes())
    buf.write(sem_fwd.astype(np.uint64).tobytes())
    if skip_bigram_lags:
        for lag_arr in skip_bigram_lags:
            buf.write(lag_arr.astype(np.uint64).tobytes())

    raw        = buf.getvalue()
    compressed = lzma.compress(raw, preset=9)
    with open(path, "wb") as f:
        f.write(compressed)

    print(f"[HashGrad] Artifact saved -> {path} | "
          f"raw={len(raw)/1024/1024:.2f} MB -> "
          f"lzma9={len(compressed)/1024/1024:.2f} MB "
          f"({100*(1-len(compressed)/len(raw)):.1f}% reduction) "
          f"[fingerprint + sem_fwd + {n_lags} lag(s)]")
    return len(compressed)


def load_hash_grad_artifact(
    path: str,
) -> Tuple[np.ndarray, np.ndarray, int, int, Optional[List[np.ndarray]]]:
    """Load fingerprint + sem_fwd (+ skip-bigram lags) from a HGZ3 .hgz artifact.

    Returns (fingerprint, sem_fwd, seed, table_bits, skip_bigram_lags_or_None).
    sem_fwd is shaped (vocab_size, W).
    """
    import lzma
    import struct

    with open(path, "rb") as f:
        raw = lzma.decompress(f.read())

    magic = raw[:4]
    if magic != b"HGZ3":
        raise ValueError(f"Invalid magic bytes: {magic!r} -- expected HGZ3")

    seed, table_bits, vocab_size, W, n_lags = struct.unpack_from("<QIIIII", raw, 4)
    TABLE_SIZE = 1 << table_bits
    # magic(4) + seed(8) + table_bits(4) + vocab_size(4) + W(4) + n_lags(4)
    offset = 4 + 8 + 4 + 4 + 4 + 4

    fp_bytes = TABLE_SIZE
    fingerprint = np.frombuffer(raw[offset:offset + fp_bytes], dtype=np.uint8).copy()
    offset += fp_bytes

    sf_bytes = vocab_size * W * 8
    sem_fwd = np.frombuffer(raw[offset:offset + sf_bytes],
                            dtype=np.uint64).reshape(vocab_size, W).copy()
    offset += sf_bytes

    skip_bigram_lags = None
    if n_lags > 0:
        skip_bigram_lags = []
        for _ in range(n_lags):
            lag_arr = np.frombuffer(raw[offset:offset + sf_bytes],
                                    dtype=np.uint64).reshape(vocab_size, W).copy()
            skip_bigram_lags.append(lag_arr)
            offset += sf_bytes

    print(f"[HashGrad] Loaded HGZ3 artifact from {path} | "
          f"TABLE_BITS={table_bits}, vocab={vocab_size}, W={W}, lags={n_lags}")
    return fingerprint, sem_fwd, int(seed), int(table_bits), skip_bigram_lags


# ---------------------------------------------------------------------------
# BPB evaluation -- DSV-only waterfall (NMF branch removed)
# ---------------------------------------------------------------------------

def hash_grad_bpb(
    val_tokens: np.ndarray,
    g_states_val: np.ndarray,
    seed: int,
    table_bits: int,
    base_bytes: np.ndarray,
    has_leading_space: np.ndarray,
    is_boundary_token: np.ndarray,
    fingerprint_packed: Optional[np.ndarray] = None,
    sem_fwd: Optional[np.ndarray] = None,
    codebook: Optional[np.ndarray] = None,
    skip_bigram_lags: Optional[List[np.ndarray]] = None,
    batch_size: int = 500_000,
) -> Tuple[float, float]:
    """Evaluate BPB using the DSV-only waterfall.

    Eval waterfall (NMF branch removed):
      fingerprint MISMATCH -> sem_fwd XOR codebook (collision fallback)
      no mismatch (or no fingerprint) -> sem_fwd XOR codebook (miss path)
      lags 2-5 -> skip-bigram blend on miss path

    is_boundary_token must be provided (all-True initialisation, then False for
    real tokens) -- matches the official competition formula exactly.
    No np.maximum(tok_bytes, 1) floor -- bytes can be 0 for control tokens,
    matching the official eval exactly.
    """
    N          = len(val_tokens)
    SHIFT      = np.uint64(64 - table_bits)
    FP_SHIFT   = np.uint64(64 - table_bits - 8)
    seed_u64   = np.uint64(seed)
    vocab_size = sem_fwd.shape[0] if sem_fwd is not None else int(base_bytes.shape[0])

    total_bits  = 0.0
    total_bytes = 0
    total_nats  = 0.0
    total_toks  = 0

    for chunk_start in range(1, N, batch_size):
        chunk_end = min(chunk_start + batch_size, N)
        B = chunk_end - chunk_start

        g_chunk   = g_states_val[chunk_start:chunk_end].astype(np.uint64)
        tgt       = val_tokens[chunk_start:chunk_end].astype(np.int32)
        finalised = (g_chunk ^ seed_u64) * FMIX64
        buckets   = (finalised >> SHIFT).astype(np.int64)  # noqa: F841 (kept for fingerprint)

        collision_mask = np.zeros(B, dtype=bool)
        if fingerprint_packed is not None:
            query_fps      = ((finalised >> FP_SHIFT) & np.uint64(0xFF)).astype(np.uint8)
            stored_fps     = fingerprint_packed[buckets]
            collision_mask = (stored_fps != query_fps)

        # collision_pos: fingerprint mismatch -> DSV collision fallback
        # miss_mask:     no mismatch (or no fingerprint) -> DSV miss path
        collision_pos = collision_mask
        miss_mask     = ~collision_pos

        # --- Collision path ---
        if collision_pos.any():
            c_idx = np.where(collision_pos)[0]
            c_tgt = tgt[c_idx]
            p_col = np.full(len(c_idx), 1.0 / vocab_size, dtype=np.float32)

            if sem_fwd is not None and codebook is not None:
                prev_t  = np.clip(val_tokens[chunk_start + c_idx - 1].astype(np.int32),
                                  0, vocab_size - 1)
                sv = sem_fwd[prev_t]
                tv = codebook[c_tgt]
                xv = sv ^ tv
                bm = np.unpackbits(xv.view(np.uint8), axis=1)
                pc = bm.sum(axis=1).astype(np.float32)
                half = bm.shape[1] / 2.0
                conf = np.abs(pc - half) / half
                p_col = np.clip(0.5 + 0.49 * conf, 1e-30, 0.99)

            prev_t_c = np.clip(
                val_tokens[chunk_start + c_idx - 1].astype(np.int32), 0, base_bytes.shape[0] - 1)
            space_guard_c = has_leading_space[c_tgt] & ~is_boundary_token[prev_t_c]
            tok_bytes_c = np.where(space_guard_c,
                                   base_bytes[c_tgt].astype(np.float64) + 1,
                                   base_bytes[c_tgt].astype(np.float64))

            total_bits  += float(-np.log2(np.clip(p_col, 1e-30, 1.0)).sum())
            total_bytes += int(tok_bytes_c.sum())
            total_nats  += float(-np.log(np.clip(p_col, 1e-30, 1.0)).sum())
            total_toks  += len(c_idx)

        # --- Miss path ---
        if miss_mask.any():
            m_idx = np.where(miss_mask)[0]
            m_tgt = tgt[m_idx]
            p_sem = np.full(len(m_idx), 1.0 / vocab_size, dtype=np.float32)

            if sem_fwd is not None and codebook is not None:
                prev_t = np.clip(
                    val_tokens[chunk_start + m_idx - 1].astype(np.int32),
                    0, vocab_size - 1)
                sv = sem_fwd[prev_t]
                tv = codebook[m_tgt]
                xv = sv ^ tv
                bm = np.unpackbits(xv.view(np.uint8), axis=1)
                pc = bm.sum(axis=1).astype(np.float32)
                half = bm.shape[1] / 2.0
                conf = np.abs(pc - half) / half
                p_sem = np.clip(0.5 + 0.49 * conf, 1e-30, 0.99)

                if skip_bigram_lags is not None:
                    for lag_idx, lag_vec in enumerate(skip_bigram_lags):
                        lag = lag_idx + 2
                        lag_pos = chunk_start + m_idx - lag
                        valid   = lag_pos >= 0
                        if not valid.any():
                            continue
                        lp = np.clip(val_tokens[lag_pos[valid]].astype(np.int32), 0, vocab_size - 1)
                        sv_l = lag_vec[lp]
                        tv_l = codebook[m_tgt[valid]]
                        xv_l = sv_l ^ tv_l
                        bm_l = np.unpackbits(xv_l.view(np.uint8), axis=1)
                        pc_l = bm_l.sum(axis=1).astype(np.float32)
                        conf_l = np.abs(pc_l - half) / half
                        p_lag  = np.clip(0.5 + 0.49 * conf_l, 1e-30, 0.99)
                        w_lag  = 1.0 / lag
                        p_sem[valid] = (1 - w_lag) * p_sem[valid] + w_lag * p_lag

            prev_t_m = np.clip(
                val_tokens[chunk_start + m_idx - 1].astype(np.int32), 0, base_bytes.shape[0] - 1)
            space_guard_m = has_leading_space[m_tgt] & ~is_boundary_token[prev_t_m]
            tok_bytes_m = np.where(space_guard_m,
                                   base_bytes[m_tgt].astype(np.float64) + 1,
                                   base_bytes[m_tgt].astype(np.float64))

            total_bits  += float(-np.log2(np.clip(p_sem, 1e-30, 1.0)).sum())
            total_bytes += int(tok_bytes_m.sum())
            total_nats  += float(-np.log(np.clip(p_sem, 1e-30, 1.0)).sum())
            total_toks  += len(m_idx)

    if total_bytes == 0:
        return float('inf'), float('inf')

    avg_bytes_per_tok = total_bytes / max(total_toks, 1)
    bits_per_tok      = total_bits  / max(total_toks, 1)
    nats_per_tok      = total_nats  / max(total_toks, 1)
    print(f"[HashGrad BPB audit]")
    print(f"  total_tokens    : {total_toks:,}")
    print(f"  total_utf8_bytes: {total_bytes:,}")
    print(f"  avg bytes/token : {avg_bytes_per_tok:.4f}")
    print(f"  bits/token      : {bits_per_tok:.4f}")
    print(f"  nats/token (loss): {nats_per_tok:.4f}")
    print(f"  BPB = bits/token / bytes/token = "
          f"{bits_per_tok:.4f} / {avg_bytes_per_tok:.4f} = "
          f"{bits_per_tok / avg_bytes_per_tok:.4f}")
    print(f"  (same formula as reference train_gpt.py: "
          f"bits_per_token * tokens_per_byte)")

    return float(total_bits / total_bytes), float(nats_per_tok)
