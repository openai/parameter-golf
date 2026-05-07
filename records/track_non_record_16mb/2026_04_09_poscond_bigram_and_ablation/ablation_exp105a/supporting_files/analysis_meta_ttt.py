#!/usr/bin/env python3
"""Weight-space analysis: exp101 (meta-TTT on) vs exp105a (meta-TTT off).

Runs five comparative analyses on the two final_model.pt files and dumps
results to JSON + prints a summary. No GPU required — pure CPU weight-space.

The two runs share:
  * Identical architecture, seed, LRs, wallclock cap, TTT knobs
  * Same ~27M-param U-Net transformer (11 layers, 512 dim, 8Q/4KV heads)
  * Bit-identical train_gpt.py (exp105a was scaffolded from exp101)

The ONLY difference is META_TTT_ENABLED (1 for exp101, 0 for exp105a). This
makes the comparison the cleanest possible ablation of meta-TTT in our
codebase, and the two checkpoints are ideal for understanding WHAT exactly
the meta-TTT training signal did to the weights.

ANALYSES
--------
1. Per-layer weight deltas (cosine, L2 distance, norm ratio).
2. Quantization sensitivity (int6 roundtrip MSE per tensor, ranked).
3. Regularizer signature: per-layer op-norm (largest SV), condition number,
   stable rank, Frobenius norm, and Lipschitz-constant product (the product
   of top singular values across all layers — correlates with loss landscape
   sharpness).
4. Functional similarity: SVD subspace overlap via principal angles — if
   two matrices span the same k-dim subspace even in a different basis,
   they're functionally equivalent after an orthogonal remapping.
5. Summary + novelty write-up ready to paste into README.

Usage
-----
    python3 analysis_meta_ttt.py
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent.parent
EXP101 = (
    REPO
    / "records"
    / "phase3"
    / "exp101_poscond-bigram-trigram_from_exp95"
    / "final_model (1).pt"
)
EXP105A = (
    REPO
    / "records"
    / "phase3"
    / "exp105a_no-metattt_from_exp101"
    / "_pod"
    / "final_model.pt"
)
OUT_JSON = Path(__file__).resolve().parent / "analysis_meta_ttt.json"


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _diff_stats(a: torch.Tensor, b: torch.Tensor) -> dict:
    """Per-tensor comparison stats: Frobenius norms, difference norm,
    relative L2, and cosine similarity (flattened)."""
    a32 = a.detach().float().reshape(-1)
    b32 = b.detach().float().reshape(-1)
    na, nb = a32.norm().item(), b32.norm().item()
    diff_norm = (b32 - a32).norm().item()
    cos = (a32 @ b32).item() / (max(na, 1e-12) * max(nb, 1e-12))
    return {
        "a_norm": na,
        "b_norm": nb,
        "diff_norm": diff_norm,
        "rel_l2": diff_norm / max(na, 1e-12),
        "cosine": cos,
    }


def _quantize_2d_mse(t32: torch.Tensor, clip_range: int) -> tuple[float, int]:
    """Per-row int6 simulation on a 2D matrix. Returns (sum_sq_err, numel)."""
    row_clip = t32.abs().amax(dim=1)
    s = (row_clip / clip_range).clamp_min(1.0 / clip_range)
    q = torch.clamp(torch.round(t32 / s[:, None]), -clip_range, clip_range)
    recon = q * s[:, None]
    sq_err = (t32 - recon).pow(2).sum().item()
    return sq_err, int(t32.numel())


def _quantize_int6_mse(t: torch.Tensor, clip_range: int = 31) -> float:
    """Symmetric per-row int6 quantization, returning mean-squared error.

    Mirrors the real pipeline (`quantize_int6_per_row` → unbanked matrices):
      * 3D BANK tensor (n, rows, cols): quantize each slot independently with
        per-row scales. This matches the unbank-then-quantize flow in
        train_gpt.py main().
      * 2D MATRIX: per-row scales.
      * 0D/1D: single global scale.
    """
    t32 = t.detach().float()
    if t32.ndim == 3:
        total_sq, total_n = 0.0, 0
        for i in range(t32.shape[0]):
            sq, n = _quantize_2d_mse(t32[i], clip_range)
            total_sq += sq
            total_n += n
        return total_sq / max(total_n, 1)
    if t32.ndim == 2:
        sq, n = _quantize_2d_mse(t32, clip_range)
        return sq / max(n, 1)
    if t32.ndim == 1 or t32.ndim == 0:
        amax = t32.abs().max().item()
        if amax == 0:
            return 0.0
        scale = amax / clip_range
        q = torch.clamp(torch.round(t32 / scale), -clip_range, clip_range)
        recon = q * scale
        return (t32 - recon).pow(2).mean().item()
    # Higher-rank: flatten trailing dims
    flat = t32.reshape(t32.shape[0], -1)
    sq, n = _quantize_2d_mse(flat, clip_range)
    return sq / max(n, 1)


def _quantize_int6_per_slot_mse(t: torch.Tensor, clip_range: int = 31) -> list[float]:
    """For 3D banks, return per-slot MSE as a list. Used to see which layer
    within a bank is more / less quantization-robust in each model."""
    t32 = t.detach().float()
    if t32.ndim != 3:
        return [_quantize_int6_mse(t32, clip_range)]
    out = []
    for i in range(t32.shape[0]):
        sq, n = _quantize_2d_mse(t32[i], clip_range)
        out.append(sq / max(n, 1))
    return out


def _svd_stats(W: torch.Tensor) -> dict:
    """Operator norm, Frobenius norm, stable rank, condition number, and
    the full singular value spectrum (for later subspace-overlap analyses).

    Skips 3D+ by reshaping to (first_dim, -1)."""
    if W.ndim >= 3:
        W = W.reshape(W.shape[0], -1)
    if W.ndim == 1 or W.numel() < 4:
        return {
            "op_norm": float(W.abs().max()),
            "fro_norm": float(W.norm()),
            "stable_rank": 1.0,
            "cond_number": 1.0,
            "top5_sv": [float(W.abs().max())],
        }
    try:
        # Using float32 for SVD stability; CPU is fine for these sizes
        S = torch.linalg.svdvals(W.float())
        op = float(S[0])
        fro = float(W.norm())
        stable_rank = (fro ** 2) / (op ** 2 + 1e-12)
        min_sv = float(S[-1])
        cond = op / max(min_sv, 1e-12)
        return {
            "op_norm": op,
            "fro_norm": fro,
            "stable_rank": stable_rank,
            "cond_number": cond,
            "top5_sv": [float(s) for s in S[:5].tolist()],
            "bottom5_sv": [float(s) for s in S[-5:].tolist()],
            "min_sv": min_sv,
        }
    except Exception as exc:
        return {"error": str(exc)}


def _principal_angles(A: torch.Tensor, B: torch.Tensor, k: int) -> list[float]:
    """Compute principal angles between the top-k left-singular-vector
    subspaces of A and B. Returns cosines of angles (1 = same subspace, 0 =
    orthogonal subspaces). Uses a standard SVD-based formulation:

        cos(principal angles) = SVD(U_A^T U_B)

    where U_A and U_B are the top-k left singular vectors of A, B.
    """
    if A.ndim >= 3:
        A = A.reshape(A.shape[0], -1)
    if B.ndim >= 3:
        B = B.reshape(B.shape[0], -1)
    if A.shape != B.shape:
        return []
    k = min(k, A.shape[0], A.shape[1])
    try:
        UA, _, _ = torch.linalg.svd(A.float(), full_matrices=False)
        UB, _, _ = torch.linalg.svd(B.float(), full_matrices=False)
        M = UA[:, :k].T @ UB[:, :k]
        sv = torch.linalg.svdvals(M)
        return [float(s) for s in sv.tolist()]
    except Exception:
        return []


def _load_checkpoints() -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    for p in (EXP101, EXP105A):
        if not p.exists():
            raise FileNotFoundError(str(p))
    print(f"Loading exp101 from: {EXP101}")
    sd101 = torch.load(str(EXP101), map_location="cpu", weights_only=True)
    print(f"Loading exp105a from: {EXP105A}")
    sd105 = torch.load(str(EXP105A), map_location="cpu", weights_only=True)
    print(
        f"  exp101:  {len(sd101)} keys, "
        f"{sum(t.numel() for t in sd101.values()):,} params"
    )
    print(
        f"  exp105a: {len(sd105)} keys, "
        f"{sum(t.numel() for t in sd105.values()):,} params"
    )
    return sd101, sd105


# ---------------------------------------------------------------------------
# Analysis 1: Per-layer weight deltas (cosine, L2 distance, norm ratio)
# ---------------------------------------------------------------------------

def analysis_weight_deltas(
    sd101: dict[str, torch.Tensor], sd105: dict[str, torch.Tensor]
) -> dict:
    common = sorted(set(sd101.keys()) & set(sd105.keys()))
    entries = []
    for k in common:
        a, b = sd101[k], sd105[k]
        if a.shape != b.shape or a.numel() < 2:
            continue
        d = _diff_stats(a, b)
        d["name"] = k
        d["numel"] = int(a.numel())
        d["shape"] = tuple(a.shape)
        entries.append(d)

    entries.sort(key=lambda e: -e["rel_l2"])
    return {
        "n_common": len(common),
        "n_compared": len(entries),
        "top10_most_different": entries[:10],
        "bottom10_most_similar": entries[-10:],
        "all_entries": entries,
    }


# ---------------------------------------------------------------------------
# Analysis 2: Quantization sensitivity (int6 roundtrip MSE per tensor)
# ---------------------------------------------------------------------------

def analysis_quant_sensitivity(
    sd101: dict[str, torch.Tensor], sd105: dict[str, torch.Tensor]
) -> dict:
    """Simulate per-row int6 quantization on both checkpoints and compare.

    For 3D BANK tensors (qo, kv, mlp_up, mlp_down) we unpack the bank into
    per-layer slots and report BOTH the bank-aggregate MSE and the per-slot
    MSE. That matches what the real pipeline does when it unbanks before
    calling quantize_int6_gptq per matrix.
    """
    quant_cats_substrings = (
        ".mlp.", ".attn.",
        "qo_bank", "kv_bank", "mlp_up_bank", "mlp_down_bank",
    )
    per_tensor = []
    per_slot_bank = {}
    total_mse_101 = 0.0
    total_mse_105 = 0.0
    total_numel = 0
    for k in sorted(sd101.keys()):
        if k not in sd105:
            continue
        if sd101[k].shape != sd105[k].shape:
            continue
        if not any(s in k for s in quant_cats_substrings):
            continue
        if sd101[k].numel() <= 65536:
            continue
        a, b = sd101[k], sd105[k]
        mse101 = _quantize_int6_mse(a)
        mse105 = _quantize_int6_mse(b)
        per_tensor.append({
            "name": k,
            "shape": tuple(a.shape),
            "numel": int(a.numel()),
            "mse_101": mse101,
            "mse_105": mse105,
            "delta_mse": mse105 - mse101,
            "ratio_101_over_105": mse101 / max(mse105, 1e-12),
        })
        total_mse_101 += mse101 * a.numel()
        total_mse_105 += mse105 * b.numel()
        total_numel += a.numel()

        # Per-slot breakdown for 3D banks
        if a.ndim == 3:
            slots_101 = _quantize_int6_per_slot_mse(a)
            slots_105 = _quantize_int6_per_slot_mse(b)
            per_slot_bank[k] = {
                "slots_101": slots_101,
                "slots_105": slots_105,
                "n_slots_101_lower": sum(
                    1 for x, y in zip(slots_101, slots_105) if x < y
                ),
                "n_slots_total": len(slots_101),
            }

    per_tensor.sort(key=lambda e: e["delta_mse"])

    avg_mse_101 = total_mse_101 / max(total_numel, 1)
    avg_mse_105 = total_mse_105 / max(total_numel, 1)
    return {
        "total_numel": int(total_numel),
        "avg_mse_101": avg_mse_101,
        "avg_mse_105": avg_mse_105,
        "ratio_101_over_105": avg_mse_101 / max(avg_mse_105, 1e-12),
        "n_tensors_101_lower": sum(
            1 for e in per_tensor if e["mse_101"] < e["mse_105"]
        ),
        "n_tensors_101_higher": sum(
            1 for e in per_tensor if e["mse_101"] > e["mse_105"]
        ),
        "n_total": len(per_tensor),
        "per_tensor": per_tensor,
        "per_slot_banks": per_slot_bank,
    }


# ---------------------------------------------------------------------------
# Analysis 3: Regularizer signature (spectral + norm properties)
# ---------------------------------------------------------------------------

def analysis_regularizer(
    sd101: dict[str, torch.Tensor], sd105: dict[str, torch.Tensor]
) -> dict:
    """Compute per-layer op-norm, condition number, stable rank, and Frobenius
    norm for every quantizable matrix in each model. Also compute the product
    of top singular values across key layers (Lipschitz proxy)."""
    keys = [
        k for k in sorted(sd101.keys())
        if k in sd105 and sd101[k].shape == sd105[k].shape
        and sd101[k].numel() >= 65536
    ]
    per_layer = []
    lipschitz_101 = 1.0
    lipschitz_105 = 1.0
    for k in keys:
        a = sd101[k]
        b = sd105[k]
        sa = _svd_stats(a)
        sb = _svd_stats(b)
        per_layer.append({
            "name": k,
            "shape": tuple(a.shape),
            "op_norm_101": sa.get("op_norm"),
            "op_norm_105": sb.get("op_norm"),
            "fro_norm_101": sa.get("fro_norm"),
            "fro_norm_105": sb.get("fro_norm"),
            "stable_rank_101": sa.get("stable_rank"),
            "stable_rank_105": sb.get("stable_rank"),
            "cond_101": sa.get("cond_number"),
            "cond_105": sb.get("cond_number"),
            "min_sv_101": sa.get("min_sv"),
            "min_sv_105": sb.get("min_sv"),
            "top5_sv_101": sa.get("top5_sv"),
            "top5_sv_105": sb.get("top5_sv"),
        })
        if sa.get("op_norm") and sb.get("op_norm"):
            lipschitz_101 *= sa["op_norm"]
            lipschitz_105 *= sb["op_norm"]

    # Aggregate stats
    def _safe_mean(xs):
        xs = [x for x in xs if x is not None and math.isfinite(x)]
        return sum(xs) / max(len(xs), 1)

    return {
        "n_layers": len(per_layer),
        "avg_op_norm_101": _safe_mean([e["op_norm_101"] for e in per_layer]),
        "avg_op_norm_105": _safe_mean([e["op_norm_105"] for e in per_layer]),
        "avg_fro_norm_101": _safe_mean([e["fro_norm_101"] for e in per_layer]),
        "avg_fro_norm_105": _safe_mean([e["fro_norm_105"] for e in per_layer]),
        "avg_stable_rank_101": _safe_mean([e["stable_rank_101"] for e in per_layer]),
        "avg_stable_rank_105": _safe_mean([e["stable_rank_105"] for e in per_layer]),
        "avg_cond_101": _safe_mean([e["cond_101"] for e in per_layer]),
        "avg_cond_105": _safe_mean([e["cond_105"] for e in per_layer]),
        # Lipschitz product grows like exp(sum log sigma); use log for stability
        "log_lipschitz_101": sum(
            math.log(e["op_norm_101"])
            for e in per_layer
            if e["op_norm_101"] and e["op_norm_101"] > 0
        ),
        "log_lipschitz_105": sum(
            math.log(e["op_norm_105"])
            for e in per_layer
            if e["op_norm_105"] and e["op_norm_105"] > 0
        ),
        "per_layer": per_layer,
    }


# ---------------------------------------------------------------------------
# Analysis 4: Functional similarity (SVD subspace overlap)
# ---------------------------------------------------------------------------

def analysis_subspace_overlap(
    sd101: dict[str, torch.Tensor], sd105: dict[str, torch.Tensor]
) -> dict:
    """For the main quantizable matrices, compute principal angles between
    the top-k left singular vector subspaces of exp101 and exp105a. Averages
    the cosines to produce a single "subspace overlap" score per matrix."""
    per_layer = []
    matrix_keys = [
        k for k in sorted(sd101.keys())
        if k in sd105 and sd101[k].shape == sd105[k].shape
        and sd101[k].numel() >= 65536
    ]
    for k in matrix_keys:
        a = sd101[k]
        b = sd105[k]
        # Choose k_subspace based on matrix dims — smaller of (32, min_dim/4)
        if a.ndim >= 3:
            min_dim = min(a.shape[0], a.reshape(a.shape[0], -1).shape[1])
        else:
            min_dim = min(a.shape)
        k_sub = min(32, max(1, min_dim // 4))
        angles = _principal_angles(a, b, k=k_sub)
        if not angles:
            continue
        avg_cos = sum(angles) / len(angles)
        # Count how many angles are > 0.9 (essentially same direction)
        near_1 = sum(1 for c in angles if c > 0.9)
        per_layer.append({
            "name": k,
            "shape": tuple(a.shape),
            "k_subspace": k_sub,
            "angles": angles,
            "avg_cosine": avg_cos,
            "n_near_aligned": near_1,
            "frac_near_aligned": near_1 / len(angles),
        })

    # Aggregate
    avg_avg_cosine = (
        sum(e["avg_cosine"] for e in per_layer) / max(len(per_layer), 1)
    )
    avg_frac_aligned = (
        sum(e["frac_near_aligned"] for e in per_layer) / max(len(per_layer), 1)
    )
    per_layer.sort(key=lambda e: -e["avg_cosine"])
    return {
        "n_layers": len(per_layer),
        "avg_avg_cosine": avg_avg_cosine,
        "avg_frac_near_aligned": avg_frac_aligned,
        "top5_most_aligned": per_layer[:5],
        "bottom5_most_divergent": per_layer[-5:],
        "per_layer": per_layer,
    }


# ---------------------------------------------------------------------------
# Analysis 5: Linear mode connectivity proxy (pure weight space)
# ---------------------------------------------------------------------------

def analysis_interp_weight_distance(
    sd101: dict[str, torch.Tensor], sd105: dict[str, torch.Tensor]
) -> dict:
    """Without running the model, we can still measure how far apart the two
    solutions are in weight space and project a naive 'midpoint' model.

    If the two runs ended in the SAME loss basin (linear mode connected),
    interpolating along a straight line should produce a model that is
    close in norm + structure to both. If they're in DIFFERENT basins,
    the midpoint will be degenerate (smaller norms, washed-out structure).

    We report:
      * total L2 distance (sum of per-tensor ||W101 - W105||_F)
      * per-tensor midpoint norm ratios (||0.5 * (A+B)||_F / ||A||_F)
      * mean cosine between corresponding layers (reused from analysis 1)

    If mean cosine ~ 1.0, the solutions are essentially the same and any
    straight-line interpolation will stay in the basin. If cosine is lower
    (say 0.5-0.8), the midpoint is in a lower-loss ridge between two basins
    and you'd need to actually eval to know whether it works.
    """
    keys = [k for k in sorted(sd101.keys()) if k in sd105 and sd101[k].shape == sd105[k].shape]
    total_l2 = 0.0
    total_norm_a = 0.0
    total_norm_b = 0.0
    total_norm_mid = 0.0
    per_layer = []
    for k in keys:
        a = sd101[k].detach().float()
        b = sd105[k].detach().float()
        mid = 0.5 * (a + b)
        na = a.norm().item()
        nb = b.norm().item()
        nm = mid.norm().item()
        diff = (a - b).norm().item()
        total_l2 += diff
        total_norm_a += na
        total_norm_b += nb
        total_norm_mid += nm
        per_layer.append({
            "name": k,
            "norm_a": na,
            "norm_b": nb,
            "norm_mid": nm,
            "mid_over_a": nm / max(na, 1e-12),
            "diff": diff,
        })
    return {
        "n_layers": len(per_layer),
        "total_l2_distance": total_l2,
        "total_norm_101": total_norm_a,
        "total_norm_105": total_norm_b,
        "total_norm_midpoint": total_norm_mid,
        "midpoint_norm_ratio": total_norm_mid / max(total_norm_a, 1e-12),
        "per_layer": per_layer[:10],  # just top few for JSON brevity
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    t0 = time.perf_counter()
    sd101, sd105 = _load_checkpoints()
    print()

    print("[1/5] Running weight-delta analysis...")
    t = time.perf_counter()
    delta_results = analysis_weight_deltas(sd101, sd105)
    print(f"  done in {time.perf_counter() - t:.1f}s")

    print("[2/5] Running quantization sensitivity analysis...")
    t = time.perf_counter()
    quant_results = analysis_quant_sensitivity(sd101, sd105)
    print(f"  done in {time.perf_counter() - t:.1f}s")

    print("[3/5] Running regularizer signature analysis (SVD spectra)...")
    t = time.perf_counter()
    reg_results = analysis_regularizer(sd101, sd105)
    print(f"  done in {time.perf_counter() - t:.1f}s")

    print("[4/5] Running SVD subspace overlap analysis (principal angles)...")
    t = time.perf_counter()
    overlap_results = analysis_subspace_overlap(sd101, sd105)
    print(f"  done in {time.perf_counter() - t:.1f}s")

    print("[5/5] Running weight-space interpolation (linear mode proxy)...")
    t = time.perf_counter()
    interp_results = analysis_interp_weight_distance(sd101, sd105)
    print(f"  done in {time.perf_counter() - t:.1f}s")

    all_results = {
        "exp101_pt": str(EXP101),
        "exp105a_pt": str(EXP105A),
        "analysis_1_weight_deltas": delta_results,
        "analysis_2_quant_sensitivity": quant_results,
        "analysis_3_regularizer_signature": reg_results,
        "analysis_4_subspace_overlap": overlap_results,
        "analysis_5_interp_distance": interp_results,
    }

    OUT_JSON.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults dumped to: {OUT_JSON}")
    print(f"Total analysis time: {time.perf_counter() - t0:.1f}s")
    print()

    # ------------------------------------------------------------------
    # Print executive summary
    # ------------------------------------------------------------------
    print("=" * 70)
    print("EXECUTIVE SUMMARY")
    print("=" * 70)

    print(f"\n[1] Weight deltas — how much did exp101 diverge from exp105a?")
    print(f"  compared {delta_results['n_compared']} tensors")
    print(f"  top 5 most divergent (high rel_l2 = different directions):")
    for e in delta_results["top10_most_different"][:5]:
        print(f"    {e['name']:<48s} rel_l2={e['rel_l2']:.3f} cos={e['cosine']:+.3f}")
    print(f"  top 5 most aligned (low rel_l2 = same direction):")
    for e in delta_results["bottom10_most_similar"][-5:]:
        print(f"    {e['name']:<48s} rel_l2={e['rel_l2']:.3f} cos={e['cosine']:+.3f}")

    print(f"\n[2] Quantization sensitivity (int6 roundtrip MSE, per-row scales)")
    print(f"  avg MSE exp101: {quant_results['avg_mse_101']:.6e}")
    print(f"  avg MSE exp105a: {quant_results['avg_mse_105']:.6e}")
    print(f"  ratio 101/105a: {quant_results['ratio_101_over_105']:.4f}  "
          f"({'exp101 BETTER' if quant_results['ratio_101_over_105'] < 1.0 else 'exp105a BETTER'})")
    print(f"  tensors where 101 quantizes better: {quant_results['n_tensors_101_lower']}/{quant_results['n_total']}")
    print(f"  tensors where 105a quantizes better: {quant_results['n_tensors_101_higher']}/{quant_results['n_total']}")
    print(f"  per-bank slot breakdown (slots where exp101 < exp105a):")
    for name, d in quant_results.get("per_slot_banks", {}).items():
        print(f"    {name:<18s} {d['n_slots_101_lower']}/{d['n_slots_total']}  "
              f"mean(101)={sum(d['slots_101'])/len(d['slots_101']):.6e}  "
              f"mean(105)={sum(d['slots_105'])/len(d['slots_105']):.6e}")

    print(f"\n[3] Regularizer signature (spectral)")
    print(f"  avg op-norm:     exp101={reg_results['avg_op_norm_101']:.3f}  "
          f"exp105a={reg_results['avg_op_norm_105']:.3f}")
    print(f"  avg Fro norm:    exp101={reg_results['avg_fro_norm_101']:.3f}  "
          f"exp105a={reg_results['avg_fro_norm_105']:.3f}")
    print(f"  avg stable rank: exp101={reg_results['avg_stable_rank_101']:.3f}  "
          f"exp105a={reg_results['avg_stable_rank_105']:.3f}")
    print(f"  avg cond num:    exp101={reg_results['avg_cond_101']:.1f}  "
          f"exp105a={reg_results['avg_cond_105']:.1f}")
    print(f"  log Lipschitz:   exp101={reg_results['log_lipschitz_101']:.3f}  "
          f"exp105a={reg_results['log_lipschitz_105']:.3f}")

    print(f"\n[4] SVD subspace overlap (principal angles)")
    print(f"  compared {overlap_results['n_layers']} matrices")
    print(f"  avg subspace cosine: {overlap_results['avg_avg_cosine']:.3f}")
    print(f"  avg frac dims aligned (>0.9): {overlap_results['avg_frac_near_aligned']:.3f}")
    print(f"  most aligned matrices:")
    for e in overlap_results["top5_most_aligned"]:
        print(f"    {e['name']:<48s} avg_cos={e['avg_cosine']:.3f}  frac_aligned={e['frac_near_aligned']:.3f}")
    print(f"  most divergent matrices:")
    for e in overlap_results["bottom5_most_divergent"]:
        print(f"    {e['name']:<48s} avg_cos={e['avg_cosine']:.3f}  frac_aligned={e['frac_near_aligned']:.3f}")

    print(f"\n[5] Weight-space interpolation proxy")
    print(f"  total L2 distance:      {interp_results['total_l2_distance']:.2f}")
    print(f"  total exp101 norm:      {interp_results['total_norm_101']:.2f}")
    print(f"  total exp105a norm:     {interp_results['total_norm_105']:.2f}")
    print(f"  midpoint norm:          {interp_results['total_norm_midpoint']:.2f}")
    print(f"  midpoint norm ratio:    {interp_results['midpoint_norm_ratio']:.3f}")
    print(f"    (if ~1.0: same basin, midpoint is viable)")
    print(f"    (if <0.8: different basins, midpoint is degenerate)")


if __name__ == "__main__":
    main()
