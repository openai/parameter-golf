"""Smoke check for 0076 confidence-gated blend.

Three independent gates, each must pass for SMOKE OK:

1. PRODUCTION SHAPE on MPS — instantiates the full GPT model with the 0076
   recurrent triple-parallel topology (inherited from 0051/0074), registers
   per-context α buffers (parent 0074 settings), and runs forward+loss at
   (B=3, L=1024) on MPS with gate ON and gate OFF. This is the gate 0071
   missed: its MPS bounds bug only fired at production shape on MPS, not at
   small shape on CPU.

2. BYTE-IDENTITY w/ parent 0074 — with CONF_GATE_THRESHOLD=-1e9 (default),
   `trigram_blend_loss_multi_K` must return the SAME loss as the parent 0074
   path on the same input (same per-context α settings, no gating).

3. BPB on cached val — using the cached model log-probs from the 0064
   winner (scratch/blend_probe/model_log2p.npy) and the offline targets,
   blended BPB with CONF_GATE_THRESHOLD=-1.0 must land within ±0.005 of the
   offline reference 1.9378 (per-context α + gate -1.0, from
   scratch/blend_probe/conf_gate_on_per_ctx.py).

Run:
    /Users/tonyliu/Desktop/projects/parameter-golf-ssm/.venv/bin/python \
        experiments/0076_confidence_gated_blend/_gate_smoke.py
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path("/Users/tonyliu/Desktop/projects/parameter-golf-ssm")
EXP_DIR = REPO / "experiments" / "0076_confidence_gated_blend"
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
VAL_CAP = 16384
K_LIST = [3, 4]
TOP_K = 2
TOP_N_CTX_K3 = 100_000
TOP_N_CTX_K4 = 200_000
BLEND_WEIGHTS = [0.7, 0.10, 0.20]
BUILD_TOKENS = int(os.environ.get("TRIGRAM_BUILD_TOKENS", 100_000_000))
MIN_COUNT = 2

# Per-context α params — matches 0074 env.sh tuned values.
ALPHA_TAU = float(os.environ.get("ALPHA_TAU", 0.5))
ALPHA_THRESH = float(os.environ.get("ALPHA_THRESH", 3.0))
ALPHA_MIN = float(os.environ.get("ALPHA_MIN", 0.30))
ALPHA_MAX = float(os.environ.get("ALPHA_MAX", 0.85))

# 0076 confidence-gate threshold (best from offline sweep).
CONF_GATE_THRESHOLD = float(os.environ.get("CONF_GATE_THRESHOLD", -1.0))

# Reference BPBs (offline).
PER_CONTEXT_ALPHA_BPB = 1.9416     # per-context α alone (no gate)
GATED_OFFLINE_REF_BPB = 1.9378     # per-context α + gate -1.0 (target for this smoke)
GATED_TOLERANCE = 0.005            # ±0.005 vs offline ref


def fail(msg: str) -> int:
    print(f"\nSMOKE FAIL: {msg}", flush=True)
    return 1


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


def packs_to_packs_by_K(packs: dict, include_alpha: bool) -> dict:
    out: dict = {}
    for K, p in packs.items():
        d = {
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
        if include_alpha:
            d["alpha_quant"] = p["trigram_alpha_quant"]
            d["alpha_scale"] = p["trigram_alpha_scale"]
            d["alpha_offset"] = p["trigram_alpha_offset"]
        out[K] = d
    return out


def main() -> int:
    print("\n=== 0076 confidence-gated blend smoke check ===", flush=True)
    print(
        f"  K_LIST={K_LIST}  TOP_K={TOP_K}  TOP_N_K3={TOP_N_CTX_K3:,}  "
        f"TOP_N_K4={TOP_N_CTX_K4:,}  WEIGHTS={BLEND_WEIGHTS}  BUILD_TOKENS={BUILD_TOKENS:,}",
        flush=True,
    )
    print(
        f"  per-ctx α: tau={ALPHA_TAU}  thresh={ALPHA_THRESH}  "
        f"clip [{ALPHA_MIN}, {ALPHA_MAX}]  conf_gate={CONF_GATE_THRESHOLD}",
        flush=True,
    )

    # --- Build packs WITH per-context α (parent 0074 settings) ---
    train = load_shard(DATA / "fineweb_train_000000.bin")
    train_slice = train[:BUILD_TOKENS]
    print(f"\n  train slice: {train_slice.size:,} tokens", flush=True)

    K_to_top_n = {3: TOP_N_CTX_K3, 4: TOP_N_CTX_K4}
    packs_with_alpha: dict = {}
    for K in K_LIST:
        print(f"\n  building K={K} pack with per-context α (top_n={K_to_top_n[K]:,}) ...", flush=True)
        packs_with_alpha[K] = build_trigram_pack(
            train_slice,
            vocab_size=VOCAB,
            top_k=TOP_K,
            min_count=MIN_COUNT,
            K=K,
            top_n_ctx=K_to_top_n[K],
            per_context_alpha=True,
            alpha_tau=ALPHA_TAU,
            alpha_thresh=ALPHA_THRESH,
            alpha_min=ALPHA_MIN,
            alpha_max=ALPHA_MAX,
        )

    first_K = K_LIST[0]

    # --- Pack-size accounting (gate is inference-time only — no extra cap) ---
    print("\n  --- pack sizes (with per-context α) ---", flush=True)
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
        "trigram_alpha_quant",
        "trigram_alpha_scale",
        "trigram_alpha_offset",
    )
    shared_keys = (
        "bigram_log2p_quant",
        "bigram_log2p_scales",
        "bigram_log2p_offsets",
        "unigram_log2p",
    )
    shared_pack = {k: packs_with_alpha[first_K][k] for k in shared_keys}
    shared_raw = pack_byte_size(shared_pack)
    shared_brotli = estimate_brotli_size(shared_pack)
    print(
        f"  shared (bigram+unigram):  raw={shared_raw:>10,} ({shared_raw / 1e6:.2f} MB)  "
        f"brotli={shared_brotli:>10,} ({shared_brotli / 1e6:.2f} MB)",
        flush=True,
    )
    total_raw = shared_raw
    total_brotli = shared_brotli
    for K in K_LIST:
        kspec_pack = {k: packs_with_alpha[K][k] for k in K_specific_keys}
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
    # Gate is inference-time, no extra buffers vs 0074. Pack accounting unchanged.
    print(
        "  gate is inference-time → no additional buffers vs parent 0074 "
        "(pack size unchanged)",
        flush=True,
    )

    # --- Load val tokens + offline cached model log-probs (CPU reference) ---
    val_full = load_shard(DATA / "fineweb_val_000000.bin")
    cap = ((VAL_CAP - 1) // SEQ_LEN) * SEQ_LEN + 1
    val_tokens_np = val_full[:cap]
    n_pairs = val_tokens_np.size - 1
    model_log2p_np = np.load(REPO / "scratch" / "blend_probe" / "model_log2p.npy").astype(np.float64)
    targets_offline = np.load(REPO / "scratch" / "blend_probe" / "val_targets.npy").astype(np.int64)
    n = model_log2p_np.shape[0]
    n_batches_off = n // SEQ_LEN
    assert n_batches_off * SEQ_LEN == n
    assert n == n_pairs, f"target count mismatch: cache {n}, val {n_pairs}"
    byte_lens = load_byte_lengths(VOCAB)
    total_target_bytes = int(byte_lens[targets_offline].sum())
    model_bpb = -model_log2p_np.mean() * n / total_target_bytes
    print(f"\n  cached model BPB (impl): {model_bpb:.4f}  (offline ref 1.9956)", flush=True)

    # --- Build (B, L, V) fake log_softmax ---
    # The cached model_log2p is the log-prob at the TRUE TARGET only. To get the
    # "max log2p" needed by the gate, we need the full softmax. Since the offline
    # reference treats the target's log-prob AS IF it were the max (which it is
    # for confident tokens — that's the whole point of the gate), we approximate
    # by putting the cached log-prob at the target slot and small junk elsewhere.
    # But for the gate, we need max_log2p computed from the full distribution,
    # and the offline reference uses model_log2p (which = log P(target)) as the
    # "confidence" measure (`use_model_only = model_log2p > gate_thresh`).
    # So to make the in-implementation gate pick the SAME positions as offline,
    # we set the target slot to model_log2p and all other slots to a value LESS
    # than model_log2p, so max(log_softmax) == model_log2p[target]. Setting all
    # other slots to -1e9 is safe: max over vocab equals the target slot.
    LN2 = math.log(2.0)
    val_t = torch.from_numpy(val_tokens_np[: n_batches_off * SEQ_LEN + 1].astype(np.int64))
    input_ids_full = val_t[:-1].reshape(n_batches_off, SEQ_LEN)
    target_ids_full = val_t[1:].reshape(n_batches_off, SEQ_LEN)
    fake_model_log_softmax = torch.full(
        (n_batches_off, SEQ_LEN, VOCAB), -1e9, dtype=torch.float32
    )
    model_log2p_per_pos = torch.from_numpy(
        model_log2p_np.reshape(n_batches_off, SEQ_LEN).astype(np.float32)
    )
    fake_model_log_softmax.scatter_(
        2, target_ids_full.unsqueeze(-1), (model_log2p_per_pos * LN2).unsqueeze(-1)
    )

    packs_by_K_with_alpha_cpu = packs_to_packs_by_K(packs_with_alpha, include_alpha=True)

    # --- GATE 2: byte-identity vs parent 0074 (CONF_GATE_THRESHOLD=-1e9) ---
    print("\n  --- GATE 2: byte-identity vs parent 0074 (CONF_GATE_THRESHOLD=-1e9) ---", flush=True)
    parent_loss = trigram_blend_loss_multi_K(
        fake_model_log_softmax,
        target_ids_full,
        input_ids_full,
        packs_by_K=packs_by_K_with_alpha_cpu,
        bigram_log2p_quant=packs_with_alpha[first_K]["bigram_log2p_quant"],
        bigram_log2p_scales=packs_with_alpha[first_K]["bigram_log2p_scales"],
        bigram_log2p_offsets=packs_with_alpha[first_K]["bigram_log2p_offsets"],
        unigram_log2p=packs_with_alpha[first_K]["unigram_log2p"],
        blend_weights=BLEND_WEIGHTS,
        vocab_size=VOCAB,
        K_order=K_LIST,
        per_context_alpha=True,
        alpha_max_default=ALPHA_MAX,
        # gate disabled (default)
    )
    gate_off_loss = trigram_blend_loss_multi_K(
        fake_model_log_softmax,
        target_ids_full,
        input_ids_full,
        packs_by_K=packs_by_K_with_alpha_cpu,
        bigram_log2p_quant=packs_with_alpha[first_K]["bigram_log2p_quant"],
        bigram_log2p_scales=packs_with_alpha[first_K]["bigram_log2p_scales"],
        bigram_log2p_offsets=packs_with_alpha[first_K]["bigram_log2p_offsets"],
        unigram_log2p=packs_with_alpha[first_K]["unigram_log2p"],
        blend_weights=BLEND_WEIGHTS,
        vocab_size=VOCAB,
        K_order=K_LIST,
        per_context_alpha=True,
        alpha_max_default=ALPHA_MAX,
        conf_gate_threshold=-1e9,
    )
    parent_bpb = float(parent_loss.item()) / LN2 * n / total_target_bytes
    gate_off_bpb = float(gate_off_loss.item()) / LN2 * n / total_target_bytes
    same_loss_diff = abs(float(parent_loss.item()) - float(gate_off_loss.item()))
    print(f"  parent (default kw) BPB:                    {parent_bpb:.4f}", flush=True)
    print(f"  gate-OFF (CONF_GATE_THRESHOLD=-1e9) BPB:    {gate_off_bpb:.4f}", flush=True)
    print(f"  byte-identity diff:                         {same_loss_diff:.2e}", flush=True)
    same_loss_ok = same_loss_diff < 1e-7
    parent_ref_diff = abs(parent_bpb - PER_CONTEXT_ALPHA_BPB)
    print(
        f"  parent vs offline per-ctx α ref ({PER_CONTEXT_ALPHA_BPB:.4f}): "
        f"{parent_bpb - PER_CONTEXT_ALPHA_BPB:+.4f}  (must be within +/- 0.005)",
        flush=True,
    )
    parent_match_ok = parent_ref_diff <= 0.005

    # --- GATE 3: gated BPB (CONF_GATE_THRESHOLD=-1.0) ---
    print(
        f"\n  --- GATE 3: gated blended BPB "
        f"(CONF_GATE_THRESHOLD={CONF_GATE_THRESHOLD}) ---",
        flush=True,
    )
    gated_loss = trigram_blend_loss_multi_K(
        fake_model_log_softmax,
        target_ids_full,
        input_ids_full,
        packs_by_K=packs_by_K_with_alpha_cpu,
        bigram_log2p_quant=packs_with_alpha[first_K]["bigram_log2p_quant"],
        bigram_log2p_scales=packs_with_alpha[first_K]["bigram_log2p_scales"],
        bigram_log2p_offsets=packs_with_alpha[first_K]["bigram_log2p_offsets"],
        unigram_log2p=packs_with_alpha[first_K]["unigram_log2p"],
        blend_weights=BLEND_WEIGHTS,
        vocab_size=VOCAB,
        K_order=K_LIST,
        per_context_alpha=True,
        alpha_max_default=ALPHA_MAX,
        conf_gate_threshold=CONF_GATE_THRESHOLD,
    )
    gated_bpb = float(gated_loss.item()) / LN2 * n / total_target_bytes
    print(f"  gated blended BPB: {gated_bpb:.4f}", flush=True)
    print(
        f"  vs offline ref ({GATED_OFFLINE_REF_BPB:.4f}): "
        f"{gated_bpb - GATED_OFFLINE_REF_BPB:+.4f}  "
        f"(must be within +/- {GATED_TOLERANCE})",
        flush=True,
    )
    print(
        f"  vs parent per-ctx α ({PER_CONTEXT_ALPHA_BPB:.4f}): "
        f"{gated_bpb - PER_CONTEXT_ALPHA_BPB:+.4f}",
        flush=True,
    )
    bpb_ok = abs(gated_bpb - GATED_OFFLINE_REF_BPB) <= GATED_TOLERANCE

    # --- GATE 1: PRODUCTION SHAPE on MPS (catches 0071-style bugs) ---
    print("\n  --- GATE 1: production shape (B=3, L=1024) on MPS ---", flush=True)
    if not torch.backends.mps.is_available():
        return fail("MPS not available — this smoke MUST run on MPS to catch the 0071 trap")
    device = torch.device("mps")

    # Build the full GPT model with the 0076 env (recurrent triple-parallel
    # topology). We don't train it — we just want to exercise the forward path
    # at production shape with the per-context α blend + gate installed.
    os.environ.setdefault("VOCAB_SIZE", "1024")
    os.environ.setdefault("NUM_UNIQUE_LAYERS", "3")
    os.environ.setdefault("NUM_LOOPS", "3")
    os.environ.setdefault("MLP_MULT", "8")
    os.environ.setdefault("PARALLEL_LAYER_POSITIONS", "0,1,2")
    os.environ.setdefault("PARALLEL_SSM_TYPE", "mamba2_kill")
    os.environ.setdefault("MAMBA2_KILL_SELECTIVITY", "1")
    os.environ.setdefault("BIGRAM_VOCAB_SIZE", "0")
    os.environ.setdefault("BIGRAM_DIM", "64")
    os.environ.setdefault("ATTN_LAYER_POSITIONS", "")
    os.environ.setdefault("MAMBA2_LAYER_POSITIONS", "")
    # Side-memory is OFF in env: we install buffers manually below.
    os.environ.pop("TRIGRAM_SIDE_MEMORY", None)

    # Import after env setup so Hyperparameters() picks them up.
    import importlib
    import train_gpt as tg  # type: ignore
    importlib.reload(tg)

    args = tg.Hyperparameters()
    print(
        f"  building GPT: NUM_UNIQUE_LAYERS={args.num_unique_layers} "
        f"NUM_LOOPS={args.num_loops} MLP_MULT={args.mlp_mult} "
        f"PARALLEL={args.parallel_layer_positions} "
        f"SSM_TYPE={os.environ.get('PARALLEL_SSM_TYPE')}",
        flush=True,
    )

    parallel_positions = (
        set(int(x) for x in args.parallel_layer_positions.split(",") if x.strip())
        if args.parallel_layer_positions else set()
    )
    attn_positions = (
        set(int(x) for x in args.attn_layer_positions.split(",") if x.strip())
        if args.attn_layer_positions else set()
    )
    mamba2_positions = (
        set(int(x) for x in args.mamba2_layer_positions.split(",") if x.strip())
        if args.mamba2_layer_positions else set()
    )

    model = tg.GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        num_unique_layers=args.num_unique_layers,
        num_loops=args.num_loops,
        attn_layer_positions=attn_positions,
        mamba2_layer_positions=mamba2_positions,
        parallel_layer_positions=parallel_positions,
        bigram_vocab_size=args.bigram_vocab_size,
        bigram_dim=args.bigram_dim,
    ).to(device)
    model.eval()
    print("  model on MPS", flush=True)

    # Install per-context α buffers (multi-K K=3+K=4) onto the model.
    shared_keys_inst = (
        "bigram_log2p_quant",
        "bigram_log2p_scales",
        "bigram_log2p_offsets",
        "unigram_log2p",
    )
    for key in shared_keys_inst:
        model.register_buffer(key, packs_with_alpha[first_K][key].to(device), persistent=True)
    K_specific_inst = (
        "trigram_keys",
        "trigram_offsets",
        "trigram_next",
        "trigram_log2p_quant",
        "trigram_log2p_scale",
        "trigram_log2p_offset",
        "trigram_log2_backoff_quant",
        "trigram_log2_backoff_scale",
        "trigram_log2_backoff_offset",
        "trigram_alpha_quant",
        "trigram_alpha_scale",
        "trigram_alpha_offset",
    )
    for K in K_LIST:
        for key in K_specific_inst:
            new_name = key.replace("trigram_", f"trigram{K}_", 1)
            model.register_buffer(new_name, packs_with_alpha[K][key].to(device), persistent=True)
    model._use_trigram_blend = True
    model._trigram_blend_alpha = 0.7
    model._trigram_vocab_size = VOCAB
    model._trigram_K = K_LIST[0]
    model._trigram_K_list = list(K_LIST)
    model._trigram_blend_weights = list(BLEND_WEIGHTS)
    model._per_context_alpha = True
    model._alpha_max_default = ALPHA_MAX
    print("  buffers installed on MPS model with per_context_alpha=True", flush=True)

    # Production-shape forward at B=3, L=1024.
    B_PROD = 3
    L_PROD = 1024
    torch.manual_seed(0)
    input_ids_prod = torch.randint(0, VOCAB, (B_PROD, L_PROD), dtype=torch.int64, device=device)
    target_ids_prod = torch.randint(0, VOCAB, (B_PROD, L_PROD), dtype=torch.int64, device=device)

    # Gate OFF (default -1e9) — should be byte-identical to parent 0074 path.
    model._conf_gate_threshold = -1e9
    print(f"  running forward+loss at B={B_PROD}, L={L_PROD} on MPS  (gate OFF) ...", flush=True)
    with torch.no_grad():
        loss_off = model(input_ids_prod, target_ids_prod)
    loss_off_val = float(loss_off.detach().cpu().item())
    print(f"  forward OK (gate OFF) — loss = {loss_off_val:.4f} (nats)", flush=True)
    prod_off_ok = math.isfinite(loss_off_val) and loss_off_val > 0

    # Gate ON.
    model._conf_gate_threshold = CONF_GATE_THRESHOLD
    print(f"  running forward+loss at B={B_PROD}, L={L_PROD} on MPS  (gate ON, thresh={CONF_GATE_THRESHOLD}) ...", flush=True)
    with torch.no_grad():
        loss_on = model(input_ids_prod, target_ids_prod)
    loss_on_val = float(loss_on.detach().cpu().item())
    print(f"  forward OK (gate ON)  — loss = {loss_on_val:.4f} (nats)", flush=True)
    prod_on_ok = math.isfinite(loss_on_val) and loss_on_val > 0
    prod_ok = prod_off_ok and prod_on_ok

    # Byte-identity check on MPS too: gate OFF (-1e9) should reproduce parent
    # 0074 numerically (we re-compare loss tensors on the same input).
    model._conf_gate_threshold = -1e9
    with torch.no_grad():
        loss_off_repeat = model(input_ids_prod, target_ids_prod)
    mps_byte_diff = abs(float(loss_off_repeat.detach().cpu().item()) - loss_off_val)
    print(f"  MPS gate-OFF reproducibility diff: {mps_byte_diff:.2e}", flush=True)
    mps_byte_ok = mps_byte_diff < 1e-6

    # --- Summary ---
    print("\n=== summary ===", flush=True)
    print(f"  GATE 1 production shape MPS forward OK:    {prod_ok}", flush=True)
    print(f"    (gate OFF loss {loss_off_val:.4f}, gate ON loss {loss_on_val:.4f})", flush=True)
    print(f"  GATE 1b MPS gate-OFF reproducible:         {mps_byte_ok} (diff {mps_byte_diff:.2e})", flush=True)
    print(f"  GATE 2 byte-identity vs parent 0074:       {parent_match_ok and same_loss_ok}", flush=True)
    print(f"    (parent BPB {parent_bpb:.4f} vs offline {PER_CONTEXT_ALPHA_BPB:.4f}; gate-OFF byte-identity diff {same_loss_diff:.2e})", flush=True)
    print(
        f"  GATE 3 gated BPB within +/- {GATED_TOLERANCE} of {GATED_OFFLINE_REF_BPB}: "
        f"{bpb_ok} (got {gated_bpb:.4f}, Δ {gated_bpb - GATED_OFFLINE_REF_BPB:+.4f})",
        flush=True,
    )

    all_ok = prod_ok and mps_byte_ok and parent_match_ok and same_loss_ok and bpb_ok
    if all_ok:
        print("\nSMOKE OK", flush=True)
        return 0
    msgs = []
    if not prod_ok:
        msgs.append(f"production-shape MPS forward failed (off {loss_off_val}, on {loss_on_val})")
    if not mps_byte_ok:
        msgs.append(f"MPS gate-OFF reproducibility differs by {mps_byte_diff:.2e}")
    if not parent_match_ok:
        msgs.append(
            f"parent BPB {parent_bpb:.4f} differs from offline ref "
            f"{PER_CONTEXT_ALPHA_BPB:.4f} by {parent_bpb - PER_CONTEXT_ALPHA_BPB:+.4f}"
        )
    if not same_loss_ok:
        msgs.append(
            f"gate-OFF (CONF_GATE_THRESHOLD=-1e9) loss differs from parent default by {same_loss_diff:.2e}"
        )
    if not bpb_ok:
        msgs.append(
            f"gated BPB {gated_bpb:.4f} not within +/- {GATED_TOLERANCE} of {GATED_OFFLINE_REF_BPB}"
        )
    return fail("; ".join(msgs))


if __name__ == "__main__":
    sys.exit(main())
