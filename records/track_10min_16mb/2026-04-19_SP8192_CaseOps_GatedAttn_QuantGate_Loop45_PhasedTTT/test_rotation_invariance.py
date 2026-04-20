"""
CPU invariance test for spec 009's SpinQuant rotations.

Standalone — imports only `torch` and `numpy`. Cannot import from
`train_gpt.py` because that module imports flash_attn_3 and triton, which
aren't available on a CPU dev box.

What this tests:
  * For SPINQUANT_MODE=baseline: identity — forward pass before/after
    "applying rotations" (a no-op) must be bit-exact.
  * For SPINQUANT_MODE=internal_only: forward pass before/after applying
    per-layer, per-KV-group attention internal rotation R_a must match to
    within float32 tolerance. This catches banked-slice indexing bugs and
    fold-direction bugs in spinquant_hotstart.py's rotation math.

What this does NOT test:
  * The full #1736 forward pass (flash-attn, triton kernels, compiled path).
    We reimplement a minimal attention + MLP here that has the same banked
    layout and the same shape of computation, without the fused kernels.
    If rotation preserves this simpler forward pass, it preserves the full
    one too — orthogonal rotations commute the same way regardless of
    which attention implementation computes softmax(QK^T)V.
  * GPTQ / compression / eval. Those are #1736's code, unchanged.
  * GPU-specific numerical drift. We run on CPU in fp32; the bf16 drift on
    pod is a separate concern that the pod-side invariance check handles.

Usage:
    python3 test_rotation_invariance.py --ckpt runs/008-1736-reproduction/seed_42/final_model.pt

Exit code 0 on pass, 1 on any mismatch.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

# Mirror the rotation utilities from spinquant_hotstart.py. Duplicated
# intentionally so this test doesn't depend on the full script's imports.


def signed_hadamard(d: int, seed: int) -> torch.Tensor:
    assert d > 0 and (d & (d - 1)) == 0, f"d={d} is not a power of 2"
    H = torch.ones((1, 1), dtype=torch.float32)
    while H.shape[0] < d:
        H = torch.cat(
            [torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)],
            dim=0,
        )
    H = H / math.sqrt(d)
    g = torch.Generator().manual_seed(seed)
    signs = torch.randint(0, 2, (d,), generator=g, dtype=torch.float32) * 2 - 1
    return H * signs.unsqueeze(0)


def random_orthogonal(d: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    a = torch.randn((d, d), generator=g, dtype=torch.float32)
    q, _ = torch.linalg.qr(a)
    return q


def build_rotation(d: int, seed: int) -> torch.Tensor:
    if d > 0 and (d & (d - 1)) == 0:
        return signed_hadamard(d, seed)
    return random_orthogonal(d, seed)


# ---------- minimal forward pass mirroring #1736's banked attention ----------


def minimal_attention_forward(
    x: torch.Tensor,                       # [B, T, d_model]
    qo_bank_qs: torch.Tensor,              # [num_layers, d_model, d_model] — Q projections
    qo_bank_o: torch.Tensor,               # [num_layers, d_model, d_model] — O projections
    kv_bank_k: torch.Tensor,               # [num_layers, kv_dim,  d_model] — K projections
    kv_bank_v: torch.Tensor,               # [num_layers, kv_dim,  d_model] — V projections
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Stripped-down multi-layer attention using the banked tensors directly.

    For each layer:
        q = x @ qo_bank_qs[i].T  → [B, T, num_heads, head_dim]
        k = x @ kv_bank_k[i].T   → [B, T, num_kv_heads, head_dim] → repeat to num_heads
        v = x @ kv_bank_v[i].T   → [B, T, num_kv_heads, head_dim] → repeat to num_heads
        attn_out = softmax(q @ k.T / sqrt(head_dim)) @ v  (no RoPE, no causal mask)
        out = attn_out_concat @ qo_bank_o[i].T
        x = x + out   (simple residual — no norms, no scales)

    This is NOT #1736's real forward pass — no RoPE, no causal mask, no
    RMSNorm, no scaling params, no MLP, no Loop45 recurrence, no gates,
    no parallel residuals. That's the point. We want the simplest thing
    that exercises the banked attention arithmetic in the same shape so
    that a rotation bug in qo_bank / kv_bank surfaces as a numerical
    mismatch here.
    """
    B, T, d_model = x.shape
    assert num_heads * head_dim == d_model
    assert num_kv_heads > 0 and num_heads % num_kv_heads == 0
    group_size = num_heads // num_kv_heads
    n_layers = qo_bank_qs.shape[0]

    for i in range(n_layers):
        q = F.linear(x, qo_bank_qs[i]).view(B, T, num_heads, head_dim)
        k = F.linear(x, kv_bank_k[i]).view(B, T, num_kv_heads, head_dim)
        v = F.linear(x, kv_bank_v[i]).view(B, T, num_kv_heads, head_dim)

        # Match the real model's RMSNorm on Q and K (train_gpt.py lines 769-770).
        # RMSNorm is rotation-equivariant over the last axis, so it doesn't
        # interact with R_a (which rotates V's last axis and O's input-col axis).
        # Its purpose here is purely to bound attention logits — without it,
        # trained-weight magnitudes blow up q @ k.T, softmax saturates to
        # near-one-hot, and tiny rotational float errors in V become
        # catastrophic under brittle argmax-like attention.
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))

        # Broadcast KV heads up to num_heads via repeat_interleave on dim 2.
        k = k.repeat_interleave(group_size, dim=2)
        v = v.repeat_interleave(group_size, dim=2)

        # Plain (non-causal, no RoPE) attention.
        attn = torch.einsum("bthd,bshd->bhts", q, k) / math.sqrt(head_dim)
        attn = attn.softmax(dim=-1)
        out = torch.einsum("bhts,bshd->bthd", attn, v)                # [B,T,H,D]
        out = out.reshape(B, T, d_model)

        out = F.linear(out, qo_bank_o[i])
        x = x + out

    return x


# ---------- apply R_a to banked tensors on a clone ----------


def apply_Ra_to_banked(
    qo_bank_qs: torch.Tensor,
    qo_bank_o: torch.Tensor,
    kv_bank_k: torch.Tensor,
    kv_bank_v: torch.Tensor,
    *,
    base_seed: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, int]]:
    """Return rotated copies + the seed manifest. Does not mutate inputs."""
    qo_bank_qs = qo_bank_qs.clone()
    qo_bank_o = qo_bank_o.clone()
    kv_bank_k = kv_bank_k.clone()
    kv_bank_v = kv_bank_v.clone()

    group_size = num_heads // num_kv_heads
    n_layers = qo_bank_qs.shape[0]
    seeds: Dict[str, int] = {}

    for i in range(n_layers):
        for g in range(num_kv_heads):
            seed = base_seed + i * 1000 + g
            R = build_rotation(head_dim, seed).to(dtype=qo_bank_o.dtype)
            R_T = R.T.contiguous()

            # V rotation: pre-multiply the KV-head slice by R.
            v_slice = kv_bank_v[i, g * head_dim : (g + 1) * head_dim, :]
            kv_bank_v[i, g * head_dim : (g + 1) * head_dim, :] = R @ v_slice

            # O counter-rotation: for every Q-head h in this KV-group, post-multiply
            # the corresponding input-column slice by R.T.
            for h_in_group in range(group_size):
                h = g * group_size + h_in_group
                o_slice = qo_bank_o[i, :, h * head_dim : (h + 1) * head_dim]
                qo_bank_o[i, :, h * head_dim : (h + 1) * head_dim] = o_slice @ R_T

            seeds[f"layer{i}_kvgroup{g}"] = seed

    return qo_bank_qs, qo_bank_o, kv_bank_k, kv_bank_v, seeds


# ---------- extract banked tensors from a real state_dict ----------


def load_banks_from_ckpt(ckpt_path: Path, num_layers: int = 11):
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd and "qo_bank" not in sd \
            and "blocks.0.attn.c_q.weight" not in sd:
        sd = sd["state_dict"]

    if "qo_bank" in sd and "kv_bank" in sd:
        # Banked layout (what the in-memory GPT model uses).
        qo_bank = sd["qo_bank"].to(dtype=torch.float32).clone()
        kv_bank = sd["kv_bank"].to(dtype=torch.float32).clone()
        n = num_layers
        return (
            qo_bank[:n].contiguous(),
            qo_bank[n : 2 * n].contiguous(),
            kv_bank[:n].contiguous(),
            kv_bank[n : 2 * n].contiguous(),
        )

    # Unbanked layout (what _unbank_state_dict() produces). Stack the per-layer
    # keys into the banked shapes our minimal forward pass expects.
    n = num_layers
    qs = torch.stack(
        [sd[f"blocks.{i}.attn.c_q.weight"].to(dtype=torch.float32) for i in range(n)],
        dim=0,
    ).contiguous()
    os_ = torch.stack(
        [sd[f"blocks.{i}.attn.proj.weight"].to(dtype=torch.float32) for i in range(n)],
        dim=0,
    ).contiguous()
    ks = torch.stack(
        [sd[f"blocks.{i}.attn.c_k.weight"].to(dtype=torch.float32) for i in range(n)],
        dim=0,
    ).contiguous()
    vs = torch.stack(
        [sd[f"blocks.{i}.attn.c_v.weight"].to(dtype=torch.float32) for i in range(n)],
        dim=0,
    ).contiguous()
    return qs, os_, ks, vs


# ---------- the actual tests ----------


def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())


def _relative_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = float(a.detach().abs().mean().item()) + 1e-12
    return float((a - b).pow(2).mean().sqrt().item()) / denom


def run_test_baseline(qo_qs, qo_o, kv_k, kv_v, *, num_heads, num_kv_heads, head_dim):
    """No rotation => bit-exact forward pass."""
    torch.manual_seed(0)
    x = torch.randn(2, 32, qo_qs.shape[-1], dtype=torch.float32)
    y0 = minimal_attention_forward(
        x, qo_qs, qo_o, kv_k, kv_v,
        num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim,
    )
    y1 = minimal_attention_forward(
        x, qo_qs.clone(), qo_o.clone(), kv_k.clone(), kv_v.clone(),
        num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim,
    )
    diff = _max_abs_diff(y0, y1)
    ok = diff == 0.0
    print(f"[baseline]      max abs diff = {diff:.3e}   {'PASS' if ok else 'FAIL'}")
    return ok


def run_test_internal_only(qo_qs, qo_o, kv_k, kv_v, *,
                           num_heads, num_kv_heads, head_dim,
                           base_seed=42, rel_tol=1e-4):
    """R_a on V/O should preserve the forward pass to float32 relative tolerance.

    We use relative tolerance because the minimal forward pass has no RMSNorm
    between layers, so activation magnitudes grow unboundedly with trained
    weights (O(1e4) at the real checkpoint). 1e-4 relative is well above
    float32 noise and well below any algorithmic rotation bug (which would
    manifest as O(1) relative diff).
    """
    torch.manual_seed(1)
    x = torch.randn(2, 32, qo_qs.shape[-1], dtype=torch.float32)
    y_orig = minimal_attention_forward(
        x, qo_qs, qo_o, kv_k, kv_v,
        num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim,
    )

    qo_qs_r, qo_o_r, kv_k_r, kv_v_r, seeds = apply_Ra_to_banked(
        qo_qs, qo_o, kv_k, kv_v,
        base_seed=base_seed,
        num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim,
    )
    y_rot = minimal_attention_forward(
        x, qo_qs_r, qo_o_r, kv_k_r, kv_v_r,
        num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim,
    )
    max_abs = _max_abs_diff(y_orig, y_rot)
    scale = float(y_orig.abs().max().item())
    rel_max = max_abs / (scale + 1e-30)
    rel_l2 = _relative_l2(y_orig, y_rot)
    ok = rel_max < rel_tol
    status = "PASS" if ok else "FAIL"
    print(
        f"[internal_only] max_abs = {max_abs:.3e}  |y_orig|_max = {scale:.3e}  "
        f"rel_max = {rel_max:.3e}  rel_l2 = {rel_l2:.3e}  "
        f"(rel_tol={rel_tol})  {status}"
    )
    if not ok:
        print("  Diagnostic: per-layer relative diff:")
        for n_prefix in range(1, qo_qs.shape[0] + 1):
            y_o = minimal_attention_forward(
                x, qo_qs[:n_prefix], qo_o[:n_prefix], kv_k[:n_prefix], kv_v[:n_prefix],
                num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim,
            )
            y_r = minimal_attention_forward(
                x, qo_qs_r[:n_prefix], qo_o_r[:n_prefix],
                kv_k_r[:n_prefix], kv_v_r[:n_prefix],
                num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim,
            )
            a = _max_abs_diff(y_o, y_r)
            s = float(y_o.abs().max().item())
            print(f"    layers[:{n_prefix}] abs={a:.3e}  scale={s:.3e}  "
                  f"rel={a / (s + 1e-30):.3e}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=False,
                    default="runs/008-1736-reproduction/seed_42/final_model.pt",
                    help="Path to spec 008's final_model.pt")
    ap.add_argument("--num-layers", type=int, default=11)
    ap.add_argument("--num-heads", type=int, default=8)
    ap.add_argument("--num-kv-heads", type=int, default=4)
    ap.add_argument("--head-dim", type=int, default=64)
    ap.add_argument("--base-seed", type=int, default=42)
    # Relative tolerance (max_abs_diff / max_abs(y_orig)). 1e-4 is well above
    # float32 compounding noise through 11 layers of residual matmul+softmax
    # (empirically ~1e-5 on real checkpoints and ~1e-5 on synthetic). Any real
    # rotation bug shows up as ≫ 1e-2. We do not use absolute tolerance because
    # trained checkpoints produce large unbounded activation magnitudes in the
    # minimal forward (no RMSNorm between layers).
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--synthetic", action="store_true",
                    help="Use random weights instead of loading from --ckpt.")
    args = ap.parse_args()

    if args.synthetic:
        torch.manual_seed(12345)
        n = args.num_layers
        d = args.num_heads * args.head_dim
        kv = args.num_kv_heads * args.head_dim
        qo_qs = torch.randn(n, d, d, dtype=torch.float32) / math.sqrt(d)
        qo_o  = torch.randn(n, d, d, dtype=torch.float32) / math.sqrt(d)
        kv_k  = torch.randn(n, kv, d, dtype=torch.float32) / math.sqrt(d)
        kv_v  = torch.randn(n, kv, d, dtype=torch.float32) / math.sqrt(d)
        print(f"Running in SYNTHETIC mode (random weights).")
    else:
        ckpt_path = Path(args.ckpt)
        if not ckpt_path.is_file():
            print(f"Checkpoint not found: {ckpt_path}", file=sys.stderr)
            print("Tip: pass --synthetic to test on random weights, or provide --ckpt.",
                  file=sys.stderr)
            return 1
        print(f"Loading banked tensors from {ckpt_path}")
        qo_qs, qo_o, kv_k, kv_v = load_banks_from_ckpt(
            ckpt_path, num_layers=args.num_layers
        )
        print(f"  qo_bank Qs: {tuple(qo_qs.shape)} (expect (n, d, d))")
        print(f"  qo_bank O:  {tuple(qo_o.shape)}  (expect (n, d, d))")
        print(f"  kv_bank K:  {tuple(kv_k.shape)}  (expect (n, kv, d))")
        print(f"  kv_bank V:  {tuple(kv_v.shape)}  (expect (n, kv, d))")

    ok1 = run_test_baseline(
        qo_qs, qo_o, kv_k, kv_v,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
    )
    ok2 = run_test_internal_only(
        qo_qs, qo_o, kv_k, kv_v,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim, base_seed=args.base_seed, rel_tol=args.tol,
    )

    if ok1 and ok2:
        print("ALL TESTS PASS")
        return 0
    print("TESTS FAILED")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
