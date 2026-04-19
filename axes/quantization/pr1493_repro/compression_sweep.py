"""Per-tensor Pareto sweep: (MSE, post-Brotli bytes) across (bits, k, grid).

Answers the question: "Is PR-1493's (int6, k=12.85, uniform) on the Pareto
frontier for every tensor? Does NF dominate int? Is low-bit + high-k
universally better after Brotli (the hypothesis from the tensor-spike
observation)?"

No model eval. Pure weight-space analysis. RTN (no GPTQ) so we measure the
grid/bit tradeoff rather than GPTQ's error-compensation effect.

Encoding matches PR-1493's artifact format:
    quantized values stored as int8 tensor
    per-row fp16 scale
    byte_shuffle stride=2 + brotli quality=11

Output: CSV to stdout + a summary table.

Usage:
    BUNDLE_DIR=./local_bundle_seed42 \\
        /tmp/torch_env/bin/python3 compression_sweep.py > sweep_results.csv
"""
import os
import math
import csv
import sys
import io
import time
import brotli
import torch
from pathlib import Path


BUNDLE_DIR = Path(os.environ.get("BUNDLE_DIR", "local_bundle_seed42"))
BROTLI_QUALITY = int(os.environ.get("BROTLI_QUALITY", 11))

# Sweep axes
BITS = [2, 3, 4, 5, 6, 7, 8]
K_VALUES = [1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 9.0, 12.85, 20.0]


def nf_levels(bits: int) -> torch.Tensor:
    """NormalFloat reconstruction levels: conditional expectations of 2^b equal-
    probability bins under a standard normal distribution.
    """
    n = 2 ** bits
    boundaries = torch.erfinv(2 * torch.linspace(0, 1, n + 1) - 1) * math.sqrt(2)
    boundaries[0] = -10.0
    boundaries[-1] = 10.0
    phi = lambda x: torch.exp(-0.5 * x ** 2) / math.sqrt(2 * math.pi)
    # centroid of bin i = (phi(b_i) - phi(b_{i+1})) / (1/n) for equal-probability bins
    return (phi(boundaries[:-1]) - phi(boundaries[1:])) * n


def quantize_uniform(w: torch.Tensor, bits: int, k: float):
    """Per-row symmetric uniform quantization: clip at k*row_std, stored as int8.

    Returns:
        q:     int8 tensor (same shape as w), values in [-clip_range, clip_range]
        scale: fp16 per-row scale, s[i] = k * row_std[i] / clip_range
        recon: dequantized fp32 tensor (for MSE computation)
    """
    rows, cols = w.shape
    clip_range = 2 ** (bits - 1) - 1
    row_std = w.std(dim=1, keepdim=True).clamp_min(1e-10)
    scale = (k * row_std / clip_range).to(torch.float16).float()  # simulate fp16 storage
    q = torch.clamp(torch.round(w / scale), -clip_range, clip_range).to(torch.int8)
    recon = q.float() * scale
    return q, scale.to(torch.float16), recon


def quantize_nf(w: torch.Tensor, bits: int, k_scale: float = None):
    """Per-row NF quantization: snap z-scores to nearest Gaussian-quantile level.

    NF has an implicit "k" set by its outer bin edges (~±3σ for nf4). We expose
    k_scale as an optional override: if given, we scale row_std by k_scale/3
    (widens/narrows the grid the same way k does for uniform).
    """
    rows, cols = w.shape
    levels = nf_levels(bits)  # (2^b,)
    row_std = w.std(dim=1, keepdim=True).clamp_min(1e-10)
    # NF is naturally scaled to unit-variance Gaussian; the "bin edge ratio"
    # is the max level magnitude. Default scale = row_std / max_level,
    # optionally further scaled by k_scale/max_level for comparability with uniform-k.
    base_scale = row_std  # weights are dequantized as level * scale
    # Encoding: fp16 scale per row
    scale = base_scale.to(torch.float16).float()  # simulate fp16 storage
    # snap each normalized value to nearest level
    w_norm = w / scale  # should be roughly N(0,1) if weights are Gaussian
    # nearest-level lookup: |w_norm - levels| argmin over levels
    # use chunked computation to avoid O(rows*cols*2^b) memory
    q = torch.empty(rows, cols, dtype=torch.int8)
    chunk = 1024
    for i in range(0, rows, chunk):
        s = w_norm[i:i + chunk].reshape(-1, 1)  # (chunk*cols, 1)
        d = (s - levels.unsqueeze(0)).abs()     # (chunk*cols, n_levels)
        idx = d.argmin(dim=1).to(torch.int8)
        # recenter to signed int8 range so byte-shuffle + brotli sees clustering near zero
        idx_signed = idx.to(torch.int16) - (2 ** (bits - 1))
        q[i:i + chunk] = idx_signed.to(torch.int8).reshape(-1, cols)
    # recon: lookup levels[q + offset] * scale
    offset = 2 ** (bits - 1)
    recon = levels[(q.long() + offset)].reshape(w.shape) * scale
    return q, scale.to(torch.float16), recon


# byte-shuffle matches PR-1493's _byte_shuffle(data, stride=2); stride=2 on int8
# payloads effectively separates odd/even positions; brotli then compresses.
_BSHF_MAGIC = b"BSHF"


def byte_shuffle(data: bytes, stride: int = 2) -> bytes:
    if stride <= 1 or len(data) < stride:
        return data
    src = memoryview(data)
    n = len(src)
    out = bytearray(n)
    dest = 0
    for pos in range(stride):
        chunk = bytes(src[pos::stride])
        out[dest:dest + len(chunk)] = chunk
        dest += len(chunk)
    return _BSHF_MAGIC + bytes([stride]) + bytes(out)


def compressed_size(q: torch.Tensor, scale: torch.Tensor, nf_lut: torch.Tensor | None = None) -> int:
    """Simulate PR-1493's storage cost:
    - byte_shuffle(q.numpy().tobytes()) then brotli
    - scale stored as fp16 raw bytes, brotli'd
    - NF LUT stored as fp16 raw bytes, brotli'd
    Returns total compressed size in bytes.
    """
    q_bytes = q.contiguous().numpy().tobytes()
    q_shuffled = byte_shuffle(q_bytes, stride=2)
    q_compressed = brotli.compress(q_shuffled, quality=BROTLI_QUALITY)

    scale_bytes = scale.contiguous().numpy().tobytes()
    scale_compressed = brotli.compress(scale_bytes, quality=BROTLI_QUALITY)

    lut_compressed = 0
    if nf_lut is not None:
        lut_bytes = nf_lut.to(torch.float16).contiguous().numpy().tobytes()
        lut_compressed = len(brotli.compress(lut_bytes, quality=BROTLI_QUALITY))

    return len(q_compressed) + len(scale_compressed) + lut_compressed


def mse(recon: torch.Tensor, orig: torch.Tensor) -> float:
    """MSE normalized per element (comparable across tensor sizes)."""
    return (recon - orig).pow(2).mean().item()


def relative_mse(recon: torch.Tensor, orig: torch.Tensor) -> float:
    """MSE / Var(orig) — dimensionless, comparable across tensors."""
    v = orig.var().item()
    if v <= 0:
        return 0.0
    return (recon - orig).pow(2).mean().item() / v


def main():
    print(f"# compression_sweep.py — bundle={BUNDLE_DIR}, brotli_q={BROTLI_QUALITY}", file=sys.stderr)
    ema = torch.load(BUNDLE_DIR / "ema_weights.pt", map_location="cpu", weights_only=True)

    # Match PR-1493's "large" rule: 2D floating point with numel > 65536
    target_names = sorted(
        n for n, t in ema.items()
        if t.is_floating_point() and t.ndim == 2 and t.numel() > 65536
    )
    print(f"# analyzing {len(target_names)} tensors", file=sys.stderr)

    # Precompute NF levels
    nf_cache = {b: nf_levels(b) for b in BITS}

    # CSV header
    writer = csv.writer(sys.stdout)
    writer.writerow([
        "tensor", "rows", "cols", "numel", "bits", "k", "grid",
        "mse", "rel_mse", "raw_bytes", "compressed_bytes", "bits_per_weight_effective",
    ])

    t0 = time.perf_counter()
    total = 0
    for name in target_names:
        w = ema[name].float()
        rows, cols = w.shape
        numel = w.numel()

        for bits in BITS:
            # Uniform grid — sweeps k
            for k in K_VALUES:
                q, scale, recon = quantize_uniform(w, bits, k)
                m = mse(recon, w)
                rm = relative_mse(recon, w)
                comp = compressed_size(q, scale)
                eff_bpw = comp * 8 / numel
                raw_bytes = numel  # int8 storage
                writer.writerow([
                    name, rows, cols, numel, bits, k, "uniform",
                    f"{m:.6e}", f"{rm:.6e}", raw_bytes, comp, f"{eff_bpw:.4f}",
                ])
                total += 1

            # NF grid — no k (uses natural Gaussian quantiles)
            q, scale, recon = quantize_nf(w, bits)
            m = mse(recon, w)
            rm = relative_mse(recon, w)
            comp = compressed_size(q, scale, nf_lut=nf_cache[bits])
            eff_bpw = comp * 8 / numel
            raw_bytes = numel
            writer.writerow([
                name, rows, cols, numel, bits, 0.0, "nf",
                f"{m:.6e}", f"{rm:.6e}", raw_bytes, comp, f"{eff_bpw:.4f}",
            ])
            total += 1

        # Flush periodically so we can see progress
        sys.stdout.flush()
        elapsed = time.perf_counter() - t0
        print(f"# [{elapsed:>6.1f}s] {name} done  ({total} rows)", file=sys.stderr)

    elapsed = time.perf_counter() - t0
    print(f"# done: {total} rows in {elapsed:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
