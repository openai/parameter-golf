"""Quantization strategy comparison — run on a saved final_model.pt checkpoint.

Tests learned codebooks, OWQ-style outlier detection, and entropy-weighted
vocab protection against baseline int6. Reports reconstruction MSE, compressed
size, and per-tensor sensitivity rankings.

Usage:
  python3 experiments/quant_analysis.py final_model.pt

Outputs a markdown table suitable for FINDINGS.md.
"""

import sys
import io
import zlib
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

try:
    import zstandard as zstd
    _HAS_ZSTD = True
except ImportError:
    _HAS_ZSTD = False


def load_checkpoint(path: str) -> dict[str, Tensor]:
    state = torch.load(path, map_location="cpu")
    # Handle both raw state_dict and wrapped formats
    if isinstance(state, dict) and "model_state_dict" in state:
        return state["model_state_dict"]
    return state


# ─── Baseline: uniform int6 per-row ──────────────────────────────────────────

def quantize_uniform(t: Tensor, bits: int = 6) -> tuple[Tensor, Tensor, float]:
    """Uniform per-row quantization. Returns (q, scale, mse)."""
    max_val = (2 ** (bits - 1)) - 1
    t32 = t.float()
    if t32.ndim < 2:
        clip = t32.abs().max().item()
        scale = torch.tensor(clip / max_val if clip > 0 else 1.0)
        q = torch.clamp(torch.round(t32 / scale), -max_val, max_val).to(torch.int8)
        deq = q.float() * scale
        mse = (t32 - deq).square().mean().item()
        return q, scale, mse
    clip_abs = t32.abs().amax(dim=1)
    scale = (clip_abs / max_val).clamp_min(1.0 / max_val)
    q = torch.clamp(torch.round(t32 / scale[:, None]), -max_val, max_val).to(torch.int8)
    deq = q.float() * scale[:, None]
    mse = (t32 - deq).square().mean().item()
    return q, scale, mse


def quantize_gptq_lite(t: Tensor, bits: int = 6) -> tuple[Tensor, Tensor, float]:
    """GPTQ-lite: per-row optimal clip search."""
    max_val = (2 ** (bits - 1)) - 1
    t32 = t.float()
    if t32.ndim < 2:
        return quantize_uniform(t, bits)
    abs_vals = t32.abs()
    best_q = None
    best_scale = None
    best_err = None
    for ratio in [0.9, 0.95, 0.99, 0.999, 0.99999]:
        clip_abs = torch.quantile(abs_vals, ratio, dim=1)
        scale = (clip_abs / max_val).clamp_min(1.0 / max_val)
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        q = torch.clamp(torch.round(clipped / scale[:, None]), -max_val, max_val).to(torch.int8)
        deq = q.float() * scale[:, None]
        err = (t32 - deq).square().mean(dim=1)
        if best_err is None:
            best_q, best_scale, best_err = q, scale, err
        else:
            better = err < best_err
            best_q[better] = q[better]
            best_scale[better] = scale[better]
            best_err[better] = err[better]
    mse = best_err.mean().item()
    return best_q, best_scale, mse


# ─── Learned codebook ────────────────────────────────────────────────────────

def quantize_codebook(t: Tensor, k: int = 256, n_iter: int = 20) -> tuple[Tensor, Tensor, float]:
    """K-means codebook quantization. Returns (indices, codebook, mse).
    Indices are uint8 (0-255), codebook is (k,) float32."""
    t32 = t.float().flatten()
    n = t32.numel()

    # Mini-batch K-means for speed
    # Init: uniform sample from data range
    vmin, vmax = t32.min().item(), t32.max().item()
    centroids = torch.linspace(vmin, vmax, k)

    for _ in range(n_iter):
        # Assign: find nearest centroid for each weight
        # Process in chunks to avoid OOM on large tensors
        chunk_size = min(n, 1_000_000)
        assignments = torch.empty(n, dtype=torch.long)
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            dists = (t32[start:end, None] - centroids[None, :]).square()
            assignments[start:end] = dists.argmin(dim=1)

        # Update centroids
        for j in range(k):
            mask = assignments == j
            if mask.any():
                centroids[j] = t32[mask].mean()

    # Final assignment
    chunk_size = min(n, 1_000_000)
    indices = torch.empty(n, dtype=torch.uint8)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        dists = (t32[start:end, None] - centroids[None, :]).square()
        indices[start:end] = dists.argmin(dim=1).to(torch.uint8)

    deq = centroids[indices.long()]
    mse = (t32 - deq).square().mean().item()
    return indices.reshape(t.shape), centroids, mse


# ─── OWQ-style outlier detection ─────────────────────────────────────────────

def detect_outliers(t: Tensor, threshold_sigma: float = 3.0) -> dict:
    """Profile weight distribution and identify outlier rows/columns."""
    t32 = t.float()
    if t32.ndim < 2:
        return {"outlier_frac": 0.0, "max_abs": t32.abs().max().item()}

    row_norms = t32.norm(dim=1)
    mean_norm = row_norms.mean().item()
    std_norm = row_norms.std().item()
    outlier_rows = (row_norms > mean_norm + threshold_sigma * std_norm).sum().item()

    col_norms = t32.norm(dim=0)
    mean_col = col_norms.mean().item()
    std_col = col_norms.std().item()
    outlier_cols = (col_norms > mean_col + threshold_sigma * std_col).sum().item()

    # Per-element outlier fraction
    abs_vals = t32.abs()
    elem_mean = abs_vals.mean().item()
    elem_std = abs_vals.std().item()
    outlier_elems = (abs_vals > elem_mean + threshold_sigma * elem_std).sum().item()
    total_elems = t32.numel()

    return {
        "outlier_rows": outlier_rows,
        "total_rows": t32.shape[0],
        "outlier_cols": outlier_cols,
        "total_cols": t32.shape[1],
        "outlier_elem_frac": outlier_elems / total_elems,
        "max_abs": abs_vals.max().item(),
        "mean_abs": elem_mean,
        "kurtosis": ((t32 - t32.mean()).pow(4).mean() / t32.var().pow(2)).item() - 3,
    }


# ─── Entropy analysis for embeddings ─────────────────────────────────────────

def embedding_row_entropy(t: Tensor) -> Tensor:
    """Compute per-row entropy of embedding weights (higher = more informative)."""
    t32 = t.float()
    # Discretize to 256 bins for entropy estimation
    vmin, vmax = t32.min(), t32.max()
    bins = 256
    binned = ((t32 - vmin) / (vmax - vmin + 1e-8) * (bins - 1)).long().clamp(0, bins - 1)
    entropies = torch.zeros(t32.shape[0])
    for i in range(t32.shape[0]):
        counts = torch.bincount(binned[i], minlength=bins).float()
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        entropies[i] = -(probs * probs.log2()).sum()
    return entropies


# ─── Compressed size estimation ──────────────────────────────────────────────

def compressed_size(data: Tensor) -> int:
    """Estimate zstd-22 compressed size of a tensor."""
    buf = io.BytesIO()
    torch.save(data, buf)
    raw = buf.getvalue()
    if _HAS_ZSTD:
        cctx = zstd.ZstdCompressor(level=22)
        return len(cctx.compress(raw))
    return len(zlib.compress(raw, 9))


# ─── Main analysis ───────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 experiments/quant_analysis.py <final_model.pt>")
        sys.exit(1)

    path = sys.argv[1]
    print(f"Loading checkpoint: {path}")
    state = load_checkpoint(path)
    print(f"Loaded {len(state)} tensors, {sum(t.numel() for t in state.values()):,} params")

    # ─── Per-tensor analysis ─────────────────────────────────────────────
    print("\n## Per-Tensor Quantization Comparison")
    print()
    print("| Tensor | Shape | Params | Int6 MSE | GPTQ-lite MSE | Codebook MSE | Outlier% | Kurtosis |")
    print("|--------|-------|--------|----------|--------------|-------------|----------|----------|")

    total_int6_mse = 0.0
    total_gptq_mse = 0.0
    total_codebook_mse = 0.0
    total_params = 0
    tensor_sensitivities = []

    for name, t in state.items():
        if not t.is_floating_point() or t.numel() < 1000:
            continue

        n = t.numel()
        total_params += n

        # Baseline int6
        _, _, int6_mse = quantize_uniform(t, bits=6)
        total_int6_mse += int6_mse * n

        # GPTQ-lite
        _, _, gptq_mse = quantize_gptq_lite(t, bits=6)
        total_gptq_mse += gptq_mse * n

        # Codebook (only for 2D tensors, skip huge ones for speed)
        if t.ndim == 2 and n < 2_000_000:
            t0 = time.time()
            _, _, cb_mse = quantize_codebook(t, k=256, n_iter=10)
            cb_time = time.time() - t0
            total_codebook_mse += cb_mse * n
        else:
            cb_mse = float('nan')
            total_codebook_mse += int6_mse * n  # fallback

        # Outlier detection
        outliers = detect_outliers(t)
        outlier_pct = outliers.get("outlier_elem_frac", 0.0) * 100
        kurtosis = outliers.get("kurtosis", 0.0)

        tensor_sensitivities.append((name, int6_mse, n))

        shape_str = "x".join(str(s) for s in t.shape)
        print(f"| {name[:40]:40s} | {shape_str:12s} | {n:>8,} | {int6_mse:.2e} | {gptq_mse:.2e} | {cb_mse:.2e} | {outlier_pct:.1f}% | {kurtosis:.1f} |")

    print()
    avg_int6 = total_int6_mse / total_params if total_params > 0 else 0
    avg_gptq = total_gptq_mse / total_params if total_params > 0 else 0
    avg_cb = total_codebook_mse / total_params if total_params > 0 else 0
    print(f"**Weighted average MSE**: Int6={avg_int6:.2e}, GPTQ-lite={avg_gptq:.2e}, Codebook={avg_cb:.2e}")
    if avg_int6 > 0:
        print(f"**GPTQ-lite improvement**: {(1 - avg_gptq/avg_int6)*100:.1f}% lower MSE")
        print(f"**Codebook improvement**: {(1 - avg_cb/avg_int6)*100:.1f}% lower MSE")

    # ─── Sensitivity ranking ─────────────────────────────────────────────
    print("\n## Tensor Sensitivity Ranking (highest MSE = most sensitive)")
    print()
    tensor_sensitivities.sort(key=lambda x: x[1], reverse=True)
    print("| Rank | Tensor | MSE | Params | Recommendation |")
    print("|------|--------|-----|--------|----------------|")
    for i, (name, mse, n) in enumerate(tensor_sensitivities[:15]):
        rec = "fp16 keep" if i < 3 else "int7" if i < 8 else "int6"
        print(f"| {i+1} | {name[:40]:40s} | {mse:.2e} | {n:>8,} | {rec} |")

    # ─── Embedding entropy analysis ──────────────────────────────────────
    emb_key = None
    for name in state:
        if "tok_emb" in name or "embed" in name.lower():
            emb_key = name
            break

    if emb_key is not None:
        print(f"\n## Embedding Row Entropy Analysis ({emb_key})")
        print()
        emb = state[emb_key]
        entropies = embedding_row_entropy(emb)
        sorted_ent, sorted_idx = entropies.sort(descending=True)
        print(f"Entropy range: {entropies.min():.2f} - {entropies.max():.2f}")
        print(f"Mean entropy: {entropies.mean():.2f}")
        print()
        # Suggest which rows to protect
        top_10_pct = int(emb.shape[0] * 0.1)
        print(f"Top 10% highest-entropy rows ({top_10_pct} rows): protect in fp16")
        print(f"Bottom 90% ({emb.shape[0] - top_10_pct} rows): safe to quantize to int6")
        fp16_cost = top_10_pct * emb.shape[1] * 2
        full_fp16_cost = emb.shape[0] * emb.shape[1] * 2
        print(f"Selective fp16 cost: {fp16_cost:,} bytes vs full fp16: {full_fp16_cost:,} bytes (saves {full_fp16_cost - fp16_cost:,} bytes)")

    # ─── Compressed size comparison ──────────────────────────────────────
    print("\n## Compressed Size Comparison (single large tensor)")
    print()
    # Pick the largest 2D tensor for size comparison
    largest = max(((n, t) for n, t in state.items() if t.ndim == 2 and t.is_floating_point()),
                  key=lambda x: x[1].numel(), default=None)
    if largest:
        name, t = largest
        print(f"Tensor: {name} ({t.shape})")

        q_int6, s_int6, _ = quantize_uniform(t, bits=6)
        size_int6 = compressed_size(q_int6) + compressed_size(s_int6)

        q_gptq, s_gptq, _ = quantize_gptq_lite(t, bits=6)
        size_gptq = compressed_size(q_gptq) + compressed_size(s_gptq)

        if t.numel() < 2_000_000:
            idx_cb, centroids, _ = quantize_codebook(t, k=256, n_iter=10)
            size_cb = compressed_size(idx_cb) + compressed_size(centroids)
        else:
            size_cb = None

        print(f"| Method | Compressed bytes |")
        print(f"|--------|-----------------|")
        print(f"| Int6 uniform | {size_int6:,} |")
        print(f"| GPTQ-lite | {size_gptq:,} |")
        if size_cb is not None:
            print(f"| Codebook K=256 | {size_cb:,} |")

    print("\nDone.")


if __name__ == "__main__":
    main()
