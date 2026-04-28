"""Smoke check for the 0069 combined K=3 + K=4 side-memory pack + 3-way blend.

Verifies that the production pack-build + production multi-K lookup logic, when
fed the 0051 winner's per-token model log2-probs, reproduces the offline
analysis blended BPB (target -0.045 BPB at K=3 top_N=100K + K=4 top_N=200K with
weights (0.7, 0.10, 0.20), expected blended BPB ~1.9504) within +/- 0.005.

Also reports raw + estimated brotli'd pack sizes (per-K and total) to gate
against the 16 MB cap.

Run:
    /Users/tonyliu/Desktop/projects/parameter-golf-ssm/.venv/bin/python \
        experiments/0069_combined_k3_k4_side_memory/_combined_smoke.py
"""
from __future__ import annotations

import io
import math
import os
import sys
from pathlib import Path

import brotli
import numpy as np
import torch

REPO = Path("/Users/tonyliu/Desktop/projects/parameter-golf-ssm")
EXP_DIR = REPO / "experiments" / "0069_combined_k3_k4_side_memory"
DATA = REPO / "data" / "datasets" / "fineweb10B_sp1024"
sys.path.insert(0, str(EXP_DIR))

from modules.trigram_side_memory import (  # noqa: E402
    build_trigram_pack,
    pack_byte_size,
    estimate_brotli_size,
    trigram_blend_loss_multi_K,
)

VOCAB = 1024
SEQ_LEN = 1024
VAL_CAP = 16384  # tokens
K_LIST = [3, 4]
TOP_K = int(os.environ.get("TRIGRAM_TOP_K", 2))
TOP_N_CTX_K3 = int(os.environ.get("TRIGRAM_TOP_N_CTX_K3", 100_000))
TOP_N_CTX_K4 = int(os.environ.get("TRIGRAM_TOP_N_CTX_K4", 200_000))
# Default weights match the offline reference (combined_aggressive_K3.py):
# (model, K=3, K=4) = (0.7, 0.10, 0.20).
_W = os.environ.get("TRIGRAM_BLEND_WEIGHTS", "0.7,0.10,0.20")
BLEND_WEIGHTS = [float(x.strip()) for x in _W.split(",") if x.strip()]
BUILD_TOKENS = int(os.environ.get("TRIGRAM_BUILD_TOKENS", 100_000_000))
MIN_COUNT = int(os.environ.get("TRIGRAM_MIN_COUNT", 2))

# Expected reference (from scratch/blend_probe/combined_aggressive_K3.py with
# K=3 top_N=100K + K=4 top_N=200K, weights (0.7, 0.10, 0.20)):
# blended BPB 1.9504.
EXPECTED_BLEND_BPB_OFFLINE = 1.9504
TOLERANCE = 0.005

# 0064_brotli_swap final_model.int8.ptz size (parent artifact).
PARENT_ARTIFACT_MB = 13.44
SAFETY_CAP_MB = 15.9
HARD_CAP_MB = 16.0


def load_shard(path: Path) -> np.ndarray:
    h = np.fromfile(path, dtype="<i4", count=256)
    assert h[0] == 20240520 and h[1] == 1, f"bad header in {path}"
    n = int(h[2])
    return np.fromfile(path, dtype="<u2", count=n, offset=256 * 4).astype(np.int32)


def load_byte_lengths(vocab: int) -> np.ndarray:
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.load(str(REPO / "data" / "tokenizers" / "fineweb_1024_bpe.model"))
    out = np.zeros(vocab, dtype=np.int32)
    for tok in range(vocab):
        s = sp.id_to_piece(tok).replace("▁", " ")
        out[tok] = len(s.encode("utf-8"))
    return out


def fail(msg: str) -> int:
    print(f"\nSMOKE FAIL: {msg}", flush=True)
    return 1


def kgram_log2p_per_target_np(
    pack: dict,
    val_tokens_np: np.ndarray,
    targets_offline: np.ndarray,
    n_batches_off: int,
    K: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the per-target K-gram log2-prob array via raw numpy lookup of
    the production pack (matches _fourgram_smoke.py's reference loop).

    Returns (log2p, targets_check).
    """
    keys_arr = pack["trigram_keys"].numpy()
    offsets_arr = pack["trigram_offsets"].numpy()
    next_arr = pack["trigram_next"].numpy().astype(np.int64)
    log2p_q_arr = pack["trigram_log2p_quant"].numpy().astype(np.int64)
    log2p_scale = float(pack["trigram_log2p_scale"].item())
    log2p_offset = float(pack["trigram_log2p_offset"].item())
    backoff_q_arr = pack["trigram_log2_backoff_quant"].numpy().astype(np.int64)
    backoff_scale = float(pack["trigram_log2_backoff_scale"].item())
    backoff_offset = float(pack["trigram_log2_backoff_offset"].item())
    bigram_q_arr = pack["bigram_log2p_quant"].numpy().astype(np.float32)
    bigram_scales_arr = pack["bigram_log2p_scales"].numpy()
    bigram_offsets_arr = pack["bigram_log2p_offsets"].numpy()
    unigram_arr = pack["unigram_log2p"].numpy()

    bigram_dq = bigram_q_arr * bigram_scales_arr[:, None] + bigram_offsets_arr[:, None]

    n = targets_offline.shape[0]
    n_per_seq = SEQ_LEN
    kgram_log2p = np.empty(n, dtype=np.float64)
    targets_check = np.empty(n, dtype=np.int64)
    t_min = K - 1
    for b in range(n_batches_off):
        bs = b * SEQ_LEN
        for t in range(n_per_seq):
            i = b * SEQ_LEN + t
            tgt = int(val_tokens_np[bs + t + 1])
            targets_check[i] = tgt
            if t == 0:
                kgram_log2p[i] = unigram_arr[tgt]
                continue
            p1 = int(val_tokens_np[bs + t])
            if t < t_min:
                kgram_log2p[i] = bigram_dq[p1, tgt]
                continue
            if K == 3:
                p2 = int(val_tokens_np[bs + t - 1])
                key = p2 * VOCAB + p1
            else:  # K == 4
                p3 = int(val_tokens_np[bs + t - 2])
                p2 = int(val_tokens_np[bs + t - 1])
                key = p3 * VOCAB * VOCAB + p2 * VOCAB + p1
            idx = int(np.searchsorted(keys_arr, key))
            if idx < keys_arr.size and int(keys_arr[idx]) == key:
                base = float(bigram_dq[p1, tgt])
                log2_backoff = backoff_q_arr[idx] * backoff_scale + backoff_offset
                base += log2_backoff
                s = int(offsets_arr[idx])
                e = int(offsets_arr[idx + 1])
                for j in range(s, e):
                    if int(next_arr[j]) == tgt:
                        base = log2p_q_arr[j] * log2p_scale + log2p_offset
                        break
                kgram_log2p[i] = base
            else:
                kgram_log2p[i] = bigram_dq[p1, tgt]
    return kgram_log2p, targets_check


def compute_brotli_for_pack_subset(pack: dict, keys_to_include: list[str]) -> int:
    """Brotli-compress a torch.save dump of the named subset of pack tensors."""
    sub = {k: pack[k] for k in keys_to_include}
    buf = io.BytesIO()
    torch.save(sub, buf)
    return len(brotli.compress(buf.getvalue(), quality=11))


def main() -> int:
    print("\n=== 0069 combined K=3 + K=4 smoke check ===", flush=True)
    print(
        f"  K_LIST={K_LIST}  TOP_K={TOP_K}  "
        f"TOP_N_K3={TOP_N_CTX_K3:,}  TOP_N_K4={TOP_N_CTX_K4:,}  "
        f"WEIGHTS={BLEND_WEIGHTS}  BUILD_TOKENS={BUILD_TOKENS:,}  MIN_COUNT={MIN_COUNT}",
        flush=True,
    )
    if len(BLEND_WEIGHTS) != 1 + len(K_LIST):
        return fail(
            f"weights length mismatch: got {len(BLEND_WEIGHTS)} weights, expected "
            f"{1 + len(K_LIST)} (model + per-K)"
        )
    if abs(sum(BLEND_WEIGHTS) - 1.0) > 1e-4:
        return fail(f"weights must sum to 1.0, got {sum(BLEND_WEIGHTS):.6f}")

    # --- Load model log2p + targets from offline-analysis cache ---
    model_log2p = np.load(REPO / "scratch" / "blend_probe" / "model_log2p.npy").astype(np.float64)
    targets_offline = np.load(REPO / "scratch" / "blend_probe" / "val_targets.npy").astype(np.int64)
    n = model_log2p.shape[0]
    n_batches_off = n // SEQ_LEN
    assert n_batches_off * SEQ_LEN == n, "offline cache must be multiple of SEQ_LEN"
    print(f"  loaded model_log2p (n={n}, n_batches={n_batches_off})", flush=True)

    # --- Load val tokens for K-gram context lookup ---
    val_full = load_shard(DATA / "fineweb_val_000000.bin")
    cap = ((VAL_CAP - 1) // SEQ_LEN) * SEQ_LEN + 1
    val_tokens_np = val_full[:cap]
    n_pairs = val_tokens_np.size - 1
    assert n_pairs == n, f"target count mismatch: model_log2p has {n}, val gives {n_pairs}"

    byte_lens = load_byte_lengths(VOCAB)
    total_target_bytes = int(byte_lens[targets_offline].sum())
    model_bpb = -model_log2p.mean() * n / total_target_bytes
    print(f"  model BPB (implied): {model_bpb:.4f}  (offline cache reference: 1.9956)", flush=True)

    # --- Build packs EXACTLY as production will (one per K) ---
    train = load_shard(DATA / "fineweb_train_000000.bin")
    train_slice = train[:BUILD_TOKENS]
    print(f"  built train slice: {train_slice.size:,} tokens", flush=True)

    K_to_top_n = {3: TOP_N_CTX_K3, 4: TOP_N_CTX_K4}
    packs = {}
    for K in K_LIST:
        print(f"\n  building K={K} pack (top_n={K_to_top_n[K]:,}) ...", flush=True)
        packs[K] = build_trigram_pack(
            train_slice,
            vocab_size=VOCAB,
            top_k=TOP_K,
            min_count=MIN_COUNT,
            K=K,
            top_n_ctx=K_to_top_n[K],
        )

    # Verify shared (bigram + unigram) tensors are byte-identical across K's.
    # The build is deterministic so this should hold; worth asserting because
    # production assumes it can install them once.
    shared_keys = (
        "bigram_log2p_quant",
        "bigram_log2p_scales",
        "bigram_log2p_offsets",
        "unigram_log2p",
    )
    first_K = K_LIST[0]
    for K in K_LIST[1:]:
        for key in shared_keys:
            if not torch.equal(packs[first_K][key], packs[K][key]):
                return fail(
                    f"shared buffer {key} differs between K={first_K} and K={K}; "
                    f"production assumes bigram/unigram are identical across K's"
                )
    print(f"\n  shared bigram/unigram verified byte-identical across K's", flush=True)

    # --- Per-K + total pack size accounting ---
    print("\n  --- pack sizes ---", flush=True)
    K_specific_keys = (
        "trigram_keys",
        "trigram_offsets",
        "trigram_next",
        "trigram_log2p_quant",
        "trigram_log2p_scale",
        "trigram_log2p_offset",
        "trigram_log2_backoff_quant",
        "trigram_log2_backoff_scale",
        "trigram_log2_backoff_offset",
    )
    total_raw = 0
    total_brotli = 0
    # Bigram + unigram (shared, counted once)
    shared_pack = {k: packs[first_K][k] for k in shared_keys}
    shared_raw = pack_byte_size(shared_pack)
    shared_brotli = estimate_brotli_size(shared_pack)
    print(
        f"  shared (bigram+unigram):  raw={shared_raw:>10,} ({shared_raw / 1e6:.2f} MB)  "
        f"brotli={shared_brotli:>10,} ({shared_brotli / 1e6:.2f} MB)",
        flush=True,
    )
    total_raw += shared_raw
    total_brotli += shared_brotli
    for K in K_LIST:
        kspec_pack = {k: packs[K][k] for k in K_specific_keys}
        kraw = pack_byte_size(kspec_pack)
        kbrotli = estimate_brotli_size(kspec_pack)
        total_raw += kraw
        total_brotli += kbrotli
        print(
            f"  K={K} side-memory:        raw={kraw:>10,} ({kraw / 1e6:.2f} MB)  "
            f"brotli={kbrotli:>10,} ({kbrotli / 1e6:.2f} MB)",
            flush=True,
        )
    print(
        f"  TOTAL side-memory:        raw={total_raw:>10,} ({total_raw / 1e6:.2f} MB)  "
        f"brotli={total_brotli:>10,} ({total_brotli / 1e6:.2f} MB)",
        flush=True,
    )
    projected_artifact_mb = PARENT_ARTIFACT_MB + total_brotli / 1e6
    print(
        f"  parent (0064) artifact: {PARENT_ARTIFACT_MB:.2f} MB; "
        f"projected total: {projected_artifact_mb:.2f} MB "
        f"(safety target {SAFETY_CAP_MB}, hard cap {HARD_CAP_MB})",
        flush=True,
    )
    cap_ok = projected_artifact_mb <= SAFETY_CAP_MB

    # --- Per-K K-gram log2p via numpy lookup (production-pack data layout) ---
    print(
        "\n  computing per-K log2p via production pack (np-lookup) ...",
        flush=True,
    )
    per_K_log2p = {}
    for K in K_LIST:
        log2p, targets_check = kgram_log2p_per_target_np(
            packs[K], val_tokens_np, targets_offline, n_batches_off, K
        )
        if not np.array_equal(targets_check, targets_offline):
            return fail(
                f"target alignment mismatch for K={K} — production lookup uses "
                f"different val indexing than offline cache"
            )
        per_K_log2p[K] = log2p
        kgram_bpb = -log2p.mean() * n / total_target_bytes
        print(f"  K={K} alone BPB:        {kgram_bpb:.4f}", flush=True)

    # --- 3-way blend (np reference) ---
    log_parts = [math.log2(BLEND_WEIGHTS[0]) + model_log2p]
    for w, K in zip(BLEND_WEIGHTS[1:], K_LIST):
        log_parts.append(math.log2(w) + per_K_log2p[K])
    m = log_parts[0].copy()
    for x in log_parts[1:]:
        m = np.maximum(m, x)
    s = np.zeros_like(m)
    for x in log_parts:
        s = s + np.exp2(x - m)
    blend_log2p = m + np.log2(s)
    blend_bpb = -blend_log2p.mean() * n / total_target_bytes
    print(f"\n  BLENDED BPB (weights={BLEND_WEIGHTS}): {blend_bpb:.4f}", flush=True)
    delta = blend_bpb - model_bpb
    print(f"  delta vs model:        {delta:+.4f}", flush=True)
    print(
        f"  offline reference:     {EXPECTED_BLEND_BPB_OFFLINE:.4f} "
        f"(combined_aggressive_K3.py: K=3 top_N=100K + K=4 top_N=200K, weights {BLEND_WEIGHTS})",
        flush=True,
    )
    diff = blend_bpb - EXPECTED_BLEND_BPB_OFFLINE
    print(f"  smoke vs offline diff: {diff:+.4f}  (must be within +/- {TOLERANCE})", flush=True)
    smoke_ok = abs(diff) <= TOLERANCE

    # --- ALSO verify the production tensor-based forward path (multi-K) ---
    print("\n  verifying production tensor-based forward path (multi-K) ...", flush=True)
    LN2 = math.log(2.0)
    val_t = torch.from_numpy(val_tokens_np[: n_batches_off * SEQ_LEN + 1].astype(np.int64))
    input_ids_full = val_t[:-1].reshape(n_batches_off, SEQ_LEN)
    target_ids_full = val_t[1:].reshape(n_batches_off, SEQ_LEN)
    fake_model_log_softmax = torch.full(
        (n_batches_off, SEQ_LEN, VOCAB), -1e9, dtype=torch.float32
    )
    model_log2p_per_pos = torch.from_numpy(
        model_log2p.reshape(n_batches_off, SEQ_LEN).astype(np.float32)
    )
    fake_model_log_softmax.scatter_(
        2, target_ids_full.unsqueeze(-1), (model_log2p_per_pos * LN2).unsqueeze(-1)
    )

    # Build the packs_by_K dict in the layout multi-K expects ({"keys": ...
    # without the trigram_ prefix}).
    packs_by_K = {}
    for K in K_LIST:
        p = packs[K]
        packs_by_K[K] = {
            "keys": p["trigram_keys"],
            "offsets": p["trigram_offsets"],
            "next": p["trigram_next"],
            "log2p_quant": p["trigram_log2p_quant"],
            "log2p_scale": p["trigram_log2p_scale"],
            "log2p_offset": p["trigram_log2p_offset"],
            "log2_backoff_quant": p["trigram_log2_backoff_quant"],
            "log2_backoff_scale": p["trigram_log2_backoff_scale"],
            "log2_backoff_offset": p["trigram_log2_backoff_offset"],
        }
    loss = trigram_blend_loss_multi_K(
        fake_model_log_softmax,
        target_ids_full,
        input_ids_full,
        packs_by_K=packs_by_K,
        bigram_log2p_quant=packs[first_K]["bigram_log2p_quant"],
        bigram_log2p_scales=packs[first_K]["bigram_log2p_scales"],
        bigram_log2p_offsets=packs[first_K]["bigram_log2p_offsets"],
        unigram_log2p=packs[first_K]["unigram_log2p"],
        blend_weights=BLEND_WEIGHTS,
        vocab_size=VOCAB,
        K_order=K_LIST,
    )
    nat_loss = float(loss.item())
    log2_loss = nat_loss / LN2
    forward_blend_bpb = log2_loss * n / total_target_bytes
    fwd_diff = forward_blend_bpb - blend_bpb
    print(f"  forward-path blended BPB: {forward_blend_bpb:.4f}", flush=True)
    print(
        f"  forward vs np-lookup diff: {fwd_diff:+.6f}  (must be within +/- 0.001)",
        flush=True,
    )
    forward_ok = abs(fwd_diff) <= 0.001
    smoke_ok = smoke_ok and forward_ok

    # --- Summary ---
    print("\n=== summary ===", flush=True)
    print(
        f"  blended BPB:          {blend_bpb:.4f}  "
        f"(target ~{EXPECTED_BLEND_BPB_OFFLINE:.4f}, +/-{TOLERANCE})",
        flush=True,
    )
    print(f"  total raw:            {total_raw / 1e6:.2f} MB", flush=True)
    print(f"  total brotli est:     {total_brotli / 1e6:.2f} MB", flush=True)
    print(
        f"  projected total:      {projected_artifact_mb:.2f} MB "
        f"(cap {HARD_CAP_MB}; safety {SAFETY_CAP_MB})",
        flush=True,
    )
    print(f"  smoke OK:             {smoke_ok}", flush=True)
    print(f"  cap OK:               {cap_ok}", flush=True)

    if smoke_ok and cap_ok:
        print("\nSMOKE OK", flush=True)
        return 0
    msgs = []
    if not smoke_ok:
        if not forward_ok:
            msgs.append(
                f"forward-path BPB {forward_blend_bpb:.4f} differs from np-lookup "
                f"{blend_bpb:.4f} by {fwd_diff:+.6f}"
            )
        if abs(diff) > TOLERANCE:
            msgs.append(
                f"blend_bpb {blend_bpb:.4f} differs from offline ref "
                f"{EXPECTED_BLEND_BPB_OFFLINE:.4f} by {diff:+.4f} "
                f"(tol +/-{TOLERANCE})"
            )
    if not cap_ok:
        msgs.append(
            f"projected total {projected_artifact_mb:.2f} MB > {SAFETY_CAP_MB} MB safety target"
        )
    return fail("; ".join(msgs))


if __name__ == "__main__":
    sys.exit(main())
