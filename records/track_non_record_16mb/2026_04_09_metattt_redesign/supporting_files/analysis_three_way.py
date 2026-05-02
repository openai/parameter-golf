#!/usr/bin/env python3
"""Three-way weight-space analysis: exp101 vs exp105a vs exp106.

Extends the two-way analysis (analysis_meta_ttt.py) to the full trio of
meta-TTT variants, adding error-surface geometry metrics that illuminate
WHY three different training procedures converge to functionally-equivalent
but weight-space-distinct solutions.

Models compared:
  - exp101:  FOMAML meta-TTT (same-batch inner/outer)     → legal_ttt 1.11588
  - exp105a: no meta-TTT (ablation, one flag changed)      → legal_ttt 1.11624
  - exp106:  redesigned meta-TTT (cross-chunk + Δ-loss + MetaSGD) → float-TTT 1.11469

Key questions this analysis answers:
  1. How do the three solutions relate in weight space?
  2. Do they span the same functional subspaces despite different bases?
  3. Is the loss landscape degenerate (many equivalent minima)?
  4. Why is the TTT delta invariant (~0.023 bpb) across all three?
  5. Does exp106's redesign produce a measurably different solution
     topology than exp101's same-batch FOMAML?

No GPU required — pure CPU weight-space manipulations.
Runtime: ~5–8 seconds on Apple M2.

Usage:
    python3 records/phase3/analysis_three_way.py
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from itertools import combinations

import torch

# ── Paths ──────────────────────────────────────────────────────────────────

REPO = Path(__file__).resolve().parent.parent.parent
PHASE3 = REPO / "records" / "phase3"

MODELS = {
    "exp101": PHASE3 / "exp101_poscond-bigram-trigram_from_exp95" / "_pod" / "final_model.pt",
    "exp105a": PHASE3 / "exp105a_no-metattt_from_exp101" / "_pod" / "final_model.pt",
    "exp106": PHASE3 / "exp106_metasgd-crosschunk-delta_from_exp101" / "_pod" / "final_model.pt",
}
OUT_JSON = PHASE3 / "analysis_three_way.json"

# Descriptive labels for the models (used in output)
LABELS = {
    "exp101":  "FOMAML same-batch",
    "exp105a": "no meta-TTT",
    "exp106":  "cross-chunk + Δ-loss + MetaSGD",
}

# ── Helpers ────────────────────────────────────────────────────────────────

def _diff_stats(a: torch.Tensor, b: torch.Tensor) -> dict:
    """Cosine similarity, relative L2, and norms between two tensors."""
    a32 = a.detach().float().reshape(-1)
    b32 = b.detach().float().reshape(-1)
    na, nb = a32.norm().item(), b32.norm().item()
    diff_norm = (b32 - a32).norm().item()
    cos = (a32 @ b32).item() / (max(na, 1e-12) * max(nb, 1e-12))
    return {
        "a_norm": na, "b_norm": nb,
        "diff_norm": diff_norm,
        "rel_l2": diff_norm / max(na, 1e-12),
        "cosine": cos,
    }


def _quantize_2d_mse(t32: torch.Tensor, clip_range: int) -> tuple[float, int]:
    """Per-row int6 simulation on a 2D matrix."""
    row_clip = t32.abs().amax(dim=1)
    s = (row_clip / clip_range).clamp_min(1.0 / clip_range)
    q = torch.clamp(torch.round(t32 / s[:, None]), -clip_range, clip_range)
    recon = q * s[:, None]
    return (t32 - recon).pow(2).sum().item(), int(t32.numel())


def _quantize_int6_mse(t: torch.Tensor, clip_range: int = 31) -> float:
    """Symmetric per-row int6 quantization MSE, handling 3D banks correctly."""
    t32 = t.detach().float()
    if t32.ndim == 3:
        total_sq, total_n = 0.0, 0
        for i in range(t32.shape[0]):
            sq, n = _quantize_2d_mse(t32[i], clip_range)
            total_sq += sq; total_n += n
        return total_sq / max(total_n, 1)
    if t32.ndim == 2:
        sq, n = _quantize_2d_mse(t32, clip_range)
        return sq / max(n, 1)
    if t32.ndim <= 1:
        amax = t32.abs().max().item()
        if amax == 0: return 0.0
        scale = amax / clip_range
        q = torch.clamp(torch.round(t32 / scale), -clip_range, clip_range)
        return (t32 - q * scale).pow(2).mean().item()
    flat = t32.reshape(t32.shape[0], -1)
    sq, n = _quantize_2d_mse(flat, clip_range)
    return sq / max(n, 1)


def _svd_stats(W: torch.Tensor) -> dict:
    """Spectral properties: op-norm, Fro norm, stable rank, condition number."""
    if W.ndim >= 3: W = W.reshape(W.shape[0], -1)
    if W.ndim == 1 or W.numel() < 4:
        return {"op_norm": float(W.abs().max()), "fro_norm": float(W.norm()),
                "stable_rank": 1.0, "cond_number": 1.0}
    try:
        S = torch.linalg.svdvals(W.float())
        op = float(S[0]); fro = float(W.norm())
        return {"op_norm": op, "fro_norm": fro,
                "stable_rank": fro**2 / (op**2 + 1e-12),
                "cond_number": op / max(float(S[-1]), 1e-12),
                "min_sv": float(S[-1]),
                "top5_sv": [float(s) for s in S[:5]],
                "bottom5_sv": [float(s) for s in S[-5:]]}
    except Exception as exc:
        return {"error": str(exc)}


def _principal_angles(A: torch.Tensor, B: torch.Tensor, k: int) -> list[float]:
    """Cosines of principal angles between top-k left-SV subspaces."""
    if A.ndim >= 3: A = A.reshape(A.shape[0], -1)
    if B.ndim >= 3: B = B.reshape(B.shape[0], -1)
    if A.shape != B.shape: return []
    k = min(k, A.shape[0], A.shape[1])
    try:
        UA = torch.linalg.svd(A.float(), full_matrices=False)[0]
        UB = torch.linalg.svd(B.float(), full_matrices=False)[0]
        return [float(s) for s in torch.linalg.svdvals(UA[:, :k].T @ UB[:, :k])]
    except Exception:
        return []


def _safe_mean(xs):
    xs = [x for x in xs if x is not None and math.isfinite(x)]
    return sum(xs) / max(len(xs), 1) if xs else float("nan")


# ── Load ───────────────────────────────────────────────────────────────────

def load_all() -> dict[str, dict[str, torch.Tensor]]:
    sds = {}
    for name, path in MODELS.items():
        if not path.exists():
            raise FileNotFoundError(f"{name}: {path}")
        print(f"Loading {name} from: {path}")
        sd = torch.load(str(path), map_location="cpu", weights_only=True)
        print(f"  {len(sd)} keys, {sum(t.numel() for t in sd.values()):,} params")
        sds[name] = sd
    return sds


# ── Analysis 1: Pairwise weight deltas ────────────────────────────────────

def analysis_pairwise_deltas(sds: dict) -> dict:
    """For each pair of models, compute per-tensor cosine/L2 for all shared
    keys. This reveals how much meta-TTT rotates the weight space.

    KEY FINDING: all three pairs show bank cosines of ~0.05-0.07 (near-
    orthogonal) despite sharing 97%+ of their training trajectory. This is
    a Muon effect — Newton-Schulz gradient orthogonalization amplifies any
    small perturbation (like the meta-TTT gradient) into a full basis
    rotation over 7000 steps. The scalar control parameters (attn_scale,
    mlp_scale etc.) are unaffected (cosine ~0.91) because they're trained
    with Adam, not Muon, and occupy a non-degenerate part of the landscape.
    """
    results = {}
    for (n1, sd1), (n2, sd2) in combinations(sds.items(), 2):
        pair = f"{n1}_vs_{n2}"
        common = sorted(set(sd1.keys()) & set(sd2.keys()))
        entries = []
        for k in common:
            a, b = sd1[k], sd2[k]
            if a.shape != b.shape or a.numel() < 2: continue
            d = _diff_stats(a, b)
            d["name"] = k; d["numel"] = int(a.numel())
            d["shape"] = list(a.shape)
            entries.append(d)
        entries.sort(key=lambda e: -e["rel_l2"])

        # Summary stats for banks vs scalars
        banks = [e for e in entries if "bank" in e["name"]]
        scalars = [e for e in entries if "bank" not in e["name"]]
        results[pair] = {
            "n_compared": len(entries),
            "bank_avg_cosine": _safe_mean([e["cosine"] for e in banks]),
            "bank_avg_rel_l2": _safe_mean([e["rel_l2"] for e in banks]),
            "scalar_avg_cosine": _safe_mean([e["cosine"] for e in scalars]),
            "scalar_avg_rel_l2": _safe_mean([e["rel_l2"] for e in scalars]),
            "top5_divergent": entries[:5],
            "top5_similar": entries[-5:],
        }
    return results


# ── Analysis 2: Pairwise subspace overlap ─────────────────────────────────

def analysis_pairwise_subspace(sds: dict) -> dict:
    """Principal-angle subspace overlap for each pair. This answers: are the
    models in the same functional subspace despite different element-wise
    bases?

    KEY FINDING: exp105a (no meta) and exp106 (cross-chunk meta) have the
    HIGHEST subspace overlap (0.727), while exp101 (same-batch FOMAML) has
    the LOWEST overlap with both (0.615, 0.659). Same-batch FOMAML's biased
    meta-gradient (adapt on seen data, evaluate on seen data) systematically
    rotates the functional subspace MORE than the cross-chunk variant or no
    meta-TTT at all. The cross-chunk meta-gradient is closer to noise and
    thus less disruptive to the learned subspace.

    The mlp_up_bank shows this most dramatically: 105a-vs-106 cos=0.949 but
    101-vs-105a cos=0.551. Same-batch FOMAML half-rotated the MLP input
    features; cross-chunk FOMAML preserved them.
    """
    results = {}
    for (n1, sd1), (n2, sd2) in combinations(sds.items(), 2):
        pair = f"{n1}_vs_{n2}"
        common = sorted(set(sd1.keys()) & set(sd2.keys()))
        per_layer = []
        for k in common:
            a, b = sd1[k], sd2[k]
            if a.shape != b.shape or a.numel() < 65536: continue
            min_dim = min(a.shape[0], a.reshape(a.shape[0], -1).shape[1]) if a.ndim >= 3 else min(a.shape)
            k_sub = min(32, max(1, min_dim // 4))
            angles = _principal_angles(a, b, k=k_sub)
            if not angles: continue
            avg_cos = sum(angles) / len(angles)
            near_1 = sum(1 for c in angles if c > 0.9)
            per_layer.append({
                "name": k, "k": k_sub, "avg_cosine": avg_cos,
                "frac_aligned": near_1 / len(angles),
            })
        per_layer.sort(key=lambda e: -e["avg_cosine"])
        results[pair] = {
            "n_layers": len(per_layer),
            "avg_subspace_cosine": _safe_mean([e["avg_cosine"] for e in per_layer]),
            "avg_frac_aligned": _safe_mean([e["frac_aligned"] for e in per_layer]),
            "per_layer": per_layer,
        }
    return results


# ── Analysis 3: Per-model spectral properties ─────────────────────────────

def analysis_spectral(sds: dict) -> dict:
    """Spectral properties (op-norm, condition number, stable rank) for each
    model independently. Differences here reveal how meta-TTT reshapes the
    loss landscape curvature around each solution.

    KEY FINDING: all three models have nearly identical spectral properties.
    The only meaningful difference is condition number: exp101 (meta on) has
    avg cond 5.6 vs exp105a's 6.1 (−8.2%), with exp106 at 5.9. This
    represents a tiny amount of implicit spectral regularization from the
    meta-TTT gradient noise — not enough to affect any downstream metric.
    """
    results = {}
    for name, sd in sds.items():
        keys = [k for k in sorted(sd.keys()) if sd[k].numel() >= 65536]
        per_layer = []
        for k in keys:
            stats = _svd_stats(sd[k])
            stats["name"] = k; stats["shape"] = list(sd[k].shape)
            per_layer.append(stats)
        # Lipschitz proxy
        log_lip = sum(
            math.log(e["op_norm"]) for e in per_layer
            if e.get("op_norm") and e["op_norm"] > 0
        )
        results[name] = {
            "n_layers": len(per_layer),
            "avg_op_norm": _safe_mean([e.get("op_norm") for e in per_layer]),
            "avg_fro_norm": _safe_mean([e.get("fro_norm") for e in per_layer]),
            "avg_stable_rank": _safe_mean([e.get("stable_rank") for e in per_layer]),
            "avg_cond_number": _safe_mean([e.get("cond_number") for e in per_layer]),
            "log_lipschitz": log_lip,
            "per_layer": per_layer,
        }
    return results


# ── Analysis 4: Per-model quantization sensitivity ────────────────────────

def analysis_quant(sds: dict) -> dict:
    """Int6 quantization MSE for each model. Reveals whether any meta-TTT
    variant produces more quantization-friendly weight distributions."""
    quant_cats = (".mlp.", ".attn.", "qo_bank", "kv_bank", "mlp_up_bank", "mlp_down_bank")
    results = {}
    for name, sd in sds.items():
        total_sq, total_n = 0.0, 0
        per_tensor = []
        for k in sorted(sd.keys()):
            if not any(s in k for s in quant_cats): continue
            if sd[k].numel() <= 65536: continue
            mse = _quantize_int6_mse(sd[k])
            numel = sd[k].numel()
            per_tensor.append({"name": k, "mse": mse, "numel": numel})
            total_sq += mse * numel; total_n += numel
        results[name] = {
            "avg_mse": total_sq / max(total_n, 1),
            "total_numel": total_n,
            "per_tensor": per_tensor,
        }
    return results


# ── Analysis 5: Mode connectivity (three-way) ─────────────────────────────

def analysis_mode_connectivity(sds: dict) -> dict:
    """Weight-space distance and midpoint norm ratio for all pairs.
    Also computes the CENTROID of all three models — if the centroid
    preserves norm, all three live in one broad basin.

    KEY FINDING: exp105a-exp106 midpoint ratio = 0.807 (borderline same
    basin), while exp101 pairs are at 0.786-0.793 (different basins).
    The 3-way centroid ratio is 0.704 (30% norm loss → substantial vector
    cancellation). Conclusion: the three solutions occupy distinct but
    neighboring basins. Same-batch FOMAML pushes further from the natural
    optimum than cross-chunk FOMAML does.
    """
    results = {"pairwise": {}, "centroid": {}}

    # Pairwise
    for (n1, sd1), (n2, sd2) in combinations(sds.items(), 2):
        pair = f"{n1}_vs_{n2}"
        common = [k for k in sorted(sd1.keys()) if k in sd2 and sd1[k].shape == sd2[k].shape]
        total_l2, norm_a, norm_b, norm_mid = 0.0, 0.0, 0.0, 0.0
        for k in common:
            a, b = sd1[k].float(), sd2[k].float()
            mid = 0.5 * (a + b)
            na, nb, nm = a.norm().item(), b.norm().item(), mid.norm().item()
            total_l2 += (a - b).norm().item()
            norm_a += na; norm_b += nb; norm_mid += nm
        results["pairwise"][pair] = {
            "l2_distance": total_l2,
            "norm_a": norm_a, "norm_b": norm_b,
            "norm_midpoint": norm_mid,
            "midpoint_ratio": norm_mid / max(norm_a, 1e-12),
        }

    # Three-way centroid
    names = list(sds.keys())
    common = sorted(set.intersection(*[set(sd.keys()) for sd in sds.values()]))
    common = [k for k in common if all(sds[n][k].shape == sds[names[0]][k].shape for n in names)]
    total_centroid_norm, total_avg_norm = 0.0, 0.0
    for k in common:
        tensors = [sds[n][k].float() for n in names]
        centroid = sum(tensors) / len(tensors)
        avg_norm = sum(t.norm().item() for t in tensors) / len(tensors)
        total_centroid_norm += centroid.norm().item()
        total_avg_norm += avg_norm
    results["centroid"] = {
        "n_keys": len(common),
        "centroid_norm": total_centroid_norm,
        "avg_individual_norm": total_avg_norm,
        "centroid_ratio": total_centroid_norm / max(total_avg_norm, 1e-12),
    }
    return results


# ── Analysis 6: Error surface geometry ─────────────────────────────────────

def analysis_error_surface(sds: dict) -> dict:
    """Metrics that characterize the local error surface around each solution.

    The key insight: if the TTT delta is invariant (~0.023 bpb), the LOCAL
    CURVATURE of the loss landscape (which determines how much a few SGD
    steps can improve the banks) must be similar at all three solutions.

    We measure:
    1. Per-bank gradient sensitivity proxy (Hessian trace via SV spectrum)
       — The sum of squared singular values approximates Tr(W^T W), which
       correlates with the gradient magnitude under small perturbations.
    2. Bank-specific condition numbers — high condition = sharp valley
       (hard for SGD to navigate), low condition = gentle basin.
    3. Spectral gap (σ₁ - σ₂) — measures how "peaked" the landscape is.
       A large gap means one direction dominates adaptation.
    4. Effective rank (Shannon entropy of normalized SV spectrum) — how
       many directions contribute to the learned function.

    KEY FINDING: bank-level condition numbers (1.03–1.38), effective ranks
    (22 for attn, 11 for MLP), and top-5 energy fractions (0.26/0.47) are
    IDENTICAL across all three models. This is why TTT gets the same ~0.023
    bpb from every starting point — the local curvature that SGD navigates
    during adaptation is invariant.

    The one exception is spectral gap: exp106's kv_bank (1.169) and
    mlp_up_bank (1.520) have 3-12x larger gaps than the others. The cross-
    chunk meta-gradient created a more "peaked" dominant SV, but this
    doesn't help TTT because SGD convergence depends on condition number
    (worst direction), not spectral gap (best direction).
    """
    bank_keys = ["qo_bank", "kv_bank", "mlp_up_bank", "mlp_down_bank"]
    results = {}

    for name, sd in sds.items():
        model_banks = {}
        for bk in bank_keys:
            if bk not in sd: continue
            W = sd[bk].float()
            # Reshape 3D bank → 2D for SVD
            if W.ndim == 3:
                W = W.reshape(W.shape[0], -1)
            try:
                S = torch.linalg.svdvals(W)
                op = float(S[0])
                # Effective rank via Shannon entropy of normalized spectrum
                S_norm = S / S.sum()
                S_pos = S_norm[S_norm > 1e-12]
                entropy = -(S_pos * S_pos.log()).sum().item()
                eff_rank = math.exp(entropy)

                # Hessian trace proxy: sum(σ²) = Tr(W^T W)
                hessian_trace = float((S ** 2).sum().item())

                # Spectral gap: σ₁ - σ₂
                spectral_gap = float(S[0] - S[1]) if len(S) > 1 else 0.0

                # Top-5 SV concentration: what fraction of total energy
                # is in the top 5 directions?
                top5_energy = float((S[:5] ** 2).sum().item()) / max(hessian_trace, 1e-12)

                model_banks[bk] = {
                    "shape": list(sd[bk].shape),
                    "op_norm": op,
                    "cond_number": op / max(float(S[-1]), 1e-12),
                    "stable_rank": hessian_trace / (op**2 + 1e-12),
                    "effective_rank": eff_rank,
                    "hessian_trace_proxy": hessian_trace,
                    "spectral_gap": spectral_gap,
                    "top5_energy_frac": top5_energy,
                    "min_sv": float(S[-1]),
                    "sv_spectrum_summary": {
                        "top10": [float(s) for s in S[:10]],
                        "bottom5": [float(s) for s in S[-5:]],
                        "median": float(S[len(S)//2]),
                    },
                }
            except Exception as exc:
                model_banks[bk] = {"error": str(exc)}
        results[name] = model_banks

    return results


# ── Analysis 7: exp106 MetaSGD parameter analysis ─────────────────────────

def analysis_metasgd_params(sds: dict) -> dict:
    """Analyze the 66 MetaSGD scale parameters from exp106.

    These are per-layer-per-bank learned inner-loop LR scales. If meta-TTT
    learned useful per-layer adaptation speeds, the scales should diverge
    from their 1.0 init. If not, they converge to ~1.0 (uniform = no
    per-layer differentiation learned)."""
    meta_keys = ["meta_sgd_qo", "meta_sgd_kv", "meta_sgd_up", "meta_sgd_down"]
    sd106 = sds.get("exp106", {})

    results = {}
    for mk in meta_keys:
        if mk not in sd106:
            results[mk] = {"status": "not_found"}
            continue
        t = sd106[mk].float()
        vals = t.tolist()
        results[mk] = {
            "shape": list(t.shape),
            "values": vals,
            "mean": float(t.mean()),
            "std": float(t.std()),
            "min": float(t.min()),
            "max": float(t.max()),
            "deviation_from_init": float((t - 1.0).abs().mean()),
            "all_near_one": bool((t - 1.0).abs().max().item() < 0.1),
        }

    # Total count
    total = sum(len(r.get("values", [])) for r in results.values() if "values" in r)
    all_vals = []
    for r in results.values():
        if "values" in r:
            all_vals.extend(r["values"])
    if all_vals:
        t_all = torch.tensor(all_vals)
        results["aggregate"] = {
            "total_params": total,
            "global_mean": float(t_all.mean()),
            "global_std": float(t_all.std()),
            "global_min": float(t_all.min()),
            "global_max": float(t_all.max()),
            "global_deviation_from_init": float((t_all - 1.0).abs().mean()),
            "converged_to_uniform": bool((t_all - 1.0).abs().max().item() < 0.1),
        }
    return results


# ── Analysis 8: Triangle geometry ──────────────────────────────────────────

def analysis_triangle(sds: dict) -> dict:
    """The three models form a triangle in weight space. Characterize its
    shape — is it equilateral (all equally far), isosceles (two close, one
    far), or degenerate (all at the same point)?

    This reveals the topology of the meta-TTT perturbation: does cross-chunk
    FOMAML (exp106) push further from no-meta (exp105a) than same-batch
    FOMAML (exp101) does? Or are they all equidistant?

    KEY FINDING: near-equilateral (sides 2324–2356 L2). Meta-TTT doesn't
    push you in a consistent direction — it pushes you to a random
    neighboring basin. The specific basin depends on the meta-gradient
    formulation, but all basins are equidistant. This rules out the idea
    of a "meta-optimal" region in weight space.
    """
    names = list(sds.keys())
    common = sorted(set.intersection(*[set(sd.keys()) for sd in sds.values()]))
    common = [k for k in common if all(sds[n][k].shape == sds[names[0]][k].shape for n in names)]

    # Per-bank distances
    bank_keys = ["qo_bank", "kv_bank", "mlp_up_bank", "mlp_down_bank"]
    all_keys_set = set(bank_keys)
    bank_common = [k for k in common if k in all_keys_set]
    nonbank_common = [k for k in common if k not in all_keys_set]

    def _pairwise_l2(keys):
        dists = {}
        for (n1, sd1), (n2, sd2) in combinations(sds.items(), 2):
            total = 0.0
            for k in keys:
                total += (sd1[k].float() - sd2[k].float()).norm().item()
            dists[f"{n1}_vs_{n2}"] = total
        return dists

    bank_dists = _pairwise_l2(bank_common)
    nonbank_dists = _pairwise_l2(nonbank_common)
    total_dists = _pairwise_l2(common)

    # Cosine centroid
    def _pairwise_avg_cosine(keys):
        cosines = {}
        for (n1, sd1), (n2, sd2) in combinations(sds.items(), 2):
            cos_vals = []
            for k in keys:
                a = sd1[k].float().reshape(-1)
                b = sd2[k].float().reshape(-1)
                cos_vals.append(
                    (a @ b).item() / (max(a.norm().item(), 1e-12) * max(b.norm().item(), 1e-12))
                )
            cosines[f"{n1}_vs_{n2}"] = _safe_mean(cos_vals)
        return cosines

    bank_cosines = _pairwise_avg_cosine(bank_common)

    return {
        "bank_l2_distances": bank_dists,
        "nonbank_l2_distances": nonbank_dists,
        "total_l2_distances": total_dists,
        "bank_avg_cosines": bank_cosines,
        "triangle_shape": _classify_triangle(list(total_dists.values())),
    }


def _classify_triangle(sides: list[float]) -> str:
    """Rough classification of the three-model triangle."""
    if len(sides) != 3: return "unknown"
    sides = sorted(sides)
    ratio_short = sides[0] / max(sides[2], 1e-12)
    ratio_mid = sides[1] / max(sides[2], 1e-12)
    if ratio_short > 0.85 and ratio_mid > 0.85:
        return "near-equilateral (all three equally far)"
    elif ratio_short < 0.6:
        return "elongated (two close, one far)"
    else:
        return "scalene (unequal but not extreme)"


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    t0 = time.perf_counter()
    sds = load_all()
    print()

    print("[1/8] Pairwise weight deltas...")
    t = time.perf_counter()
    r1 = analysis_pairwise_deltas(sds)
    print(f"  done in {time.perf_counter()-t:.1f}s")

    print("[2/8] Pairwise subspace overlap...")
    t = time.perf_counter()
    r2 = analysis_pairwise_subspace(sds)
    print(f"  done in {time.perf_counter()-t:.1f}s")

    print("[3/8] Per-model spectral properties...")
    t = time.perf_counter()
    r3 = analysis_spectral(sds)
    print(f"  done in {time.perf_counter()-t:.1f}s")

    print("[4/8] Per-model quantization sensitivity...")
    t = time.perf_counter()
    r4 = analysis_quant(sds)
    print(f"  done in {time.perf_counter()-t:.1f}s")

    print("[5/8] Mode connectivity (pairwise + centroid)...")
    t = time.perf_counter()
    r5 = analysis_mode_connectivity(sds)
    print(f"  done in {time.perf_counter()-t:.1f}s")

    print("[6/8] Error surface geometry (bank-level)...")
    t = time.perf_counter()
    r6 = analysis_error_surface(sds)
    print(f"  done in {time.perf_counter()-t:.1f}s")

    print("[7/8] MetaSGD parameter analysis (exp106)...")
    t = time.perf_counter()
    r7 = analysis_metasgd_params(sds)
    print(f"  done in {time.perf_counter()-t:.1f}s")

    print("[8/8] Triangle geometry...")
    t = time.perf_counter()
    r8 = analysis_triangle(sds)
    print(f"  done in {time.perf_counter()-t:.1f}s")

    all_results = {
        "models": {n: str(p) for n, p in MODELS.items()},
        "labels": LABELS,
        "analysis_1_pairwise_deltas": r1,
        "analysis_2_pairwise_subspace": r2,
        "analysis_3_spectral": r3,
        "analysis_4_quant": r4,
        "analysis_5_mode_connectivity": r5,
        "analysis_6_error_surface": r6,
        "analysis_7_metasgd": r7,
        "analysis_8_triangle": r8,
    }

    OUT_JSON.write_text(json.dumps(all_results, indent=2, default=str))
    elapsed = time.perf_counter() - t0
    print(f"\nResults dumped to: {OUT_JSON}")
    print(f"Total time: {elapsed:.1f}s")

    # ── Executive summary ──────────────────────────────────────────────
    print()
    print("=" * 72)
    print("THREE-WAY ANALYSIS EXECUTIVE SUMMARY")
    print("=" * 72)

    # 1. Pairwise deltas
    print("\n[1] PAIRWISE WEIGHT DELTAS")
    print("    (bank_cos = element-wise cosine of banked weights; low → different basis)")
    for pair, d in r1.items():
        print(f"    {pair:24s}  bank_cos={d['bank_avg_cosine']:.3f}  "
              f"bank_l2={d['bank_avg_rel_l2']:.3f}  "
              f"scalar_cos={d['scalar_avg_cosine']:.3f}")

    # 2. Subspace overlap
    print("\n[2] PAIRWISE SUBSPACE OVERLAP")
    print("    (avg_cos = principal-angle cosine; 1.0 = same subspace)")
    for pair, d in r2.items():
        print(f"    {pair:24s}  avg_cos={d['avg_subspace_cosine']:.3f}  "
              f"frac_aligned={d['avg_frac_aligned']:.3f}")

    # 3. Spectral
    print("\n[3] PER-MODEL SPECTRAL PROPERTIES")
    for name, d in r3.items():
        print(f"    {name:8s}  op_norm={d['avg_op_norm']:.1f}  "
              f"cond={d['avg_cond_number']:.1f}  "
              f"stable_rank={d['avg_stable_rank']:.1f}  "
              f"log_lip={d['log_lipschitz']:.2f}")

    # 4. Quant sensitivity
    print("\n[4] QUANTIZATION SENSITIVITY (int6 per-row MSE)")
    for name, d in r4.items():
        print(f"    {name:8s}  avg_mse={d['avg_mse']:.6e}")

    # 5. Mode connectivity
    print("\n[5] MODE CONNECTIVITY")
    for pair, d in r5["pairwise"].items():
        print(f"    {pair:24s}  l2={d['l2_distance']:.1f}  "
              f"midpoint_ratio={d['midpoint_ratio']:.3f}")
    c = r5["centroid"]
    print(f"    {'3-way centroid':24s}  centroid_ratio={c['centroid_ratio']:.3f}")

    # 6. Error surface
    print("\n[6] ERROR SURFACE GEOMETRY (bank-level)")
    for name, banks in r6.items():
        conds = [b.get("cond_number", 0) for b in banks.values() if isinstance(b, dict) and "cond_number" in b]
        eranks = [b.get("effective_rank", 0) for b in banks.values() if isinstance(b, dict) and "effective_rank" in b]
        gaps = [b.get("spectral_gap", 0) for b in banks.values() if isinstance(b, dict) and "spectral_gap" in b]
        print(f"    {name:8s}  avg_cond={_safe_mean(conds):.1f}  "
              f"avg_eff_rank={_safe_mean(eranks):.1f}  "
              f"avg_spectral_gap={_safe_mean(gaps):.2f}")

    # 7. MetaSGD
    print("\n[7] METASGD PARAMETERS (exp106 only)")
    agg = r7.get("aggregate", {})
    if agg:
        print(f"    {agg['total_params']} params  "
              f"mean={agg['global_mean']:.4f}  std={agg['global_std']:.4f}  "
              f"range=[{agg['global_min']:.4f}, {agg['global_max']:.4f}]  "
              f"converged_to_uniform={agg['converged_to_uniform']}")

    # 8. Triangle
    print("\n[8] TRIANGLE GEOMETRY")
    print(f"    shape: {r8['triangle_shape']}")
    print(f"    bank L2 distances:")
    for pair, d in r8["bank_l2_distances"].items():
        print(f"      {pair:24s}  {d:.1f}")
    print(f"    bank avg cosines:")
    for pair, d in r8["bank_avg_cosines"].items():
        print(f"      {pair:24s}  {d:.4f}")


if __name__ == "__main__":
    main()
