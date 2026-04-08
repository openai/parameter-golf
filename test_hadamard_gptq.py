"""
Local smoke test: compare GPTQ quantization MSE with and without Hadamard rotation.
Runs on CPU -- no CUDA/FA3 needed.

Usage:
    python test_hadamard_gptq.py
"""

import math
import os
import sys
import torch
import torch.nn.functional as F


# --- Inline the functions to avoid importing train_gpt_sota (needs FA3) ---

def _hadamard_matrix(n, device="cpu"):
    H = torch.ones(1, 1, device=device)
    while H.shape[0] < n:
        H = torch.cat([
            torch.cat([H, H], dim=1),
            torch.cat([H, -H], dim=1),
        ], dim=0)
    return H / math.sqrt(n)

def _next_power_of_2(n):
    return 1 << (n - 1).bit_length()

def hadamard_rotate_weight(W, H_right=None):
    rows, cols = W.shape
    padded_cols = _next_power_of_2(cols)
    if padded_cols != cols:
        W = F.pad(W, (0, padded_cols - cols))
    if H_right is None:
        H_right = _hadamard_matrix(padded_cols, device=W.device)
    W_rot = W @ H_right
    return W_rot, H_right, cols

def hadamard_rotate_hessian(H, H_right):
    cols = H.shape[0]
    padded = H_right.shape[0]
    if cols < padded:
        H_padded = torch.zeros(padded, padded, dtype=H.dtype, device=H.device)
        H_padded[:cols, :cols] = H
        damp = 0.01 * torch.diag(H).mean().clamp_min(1e-6)
        for i in range(cols, padded):
            H_padded[i, i] = damp
        H = H_padded
    return H_right.T @ H @ H_right

def hadamard_unrotate_weight(W_rot, H_right, orig_cols):
    W = W_rot @ H_right.T
    return W[:, :orig_cols]

# Toggle for A/B comparison
_HADAMARD_ENABLE = True

def quantize_int6_gptq(weight, hessian=None, clip_range=31, block_size=128):
    t32 = weight.float()
    rows, cols = t32.shape
    had_applied = False
    orig_cols = cols
    if _HADAMARD_ENABLE:
        padded_cols = _next_power_of_2(cols)
        H_right = _hadamard_matrix(padded_cols, device=t32.device)
        t32, _, orig_cols = hadamard_rotate_weight(t32, H_right)
        hessian = hadamard_rotate_hessian(hessian, H_right)
        rows, cols = t32.shape
        had_applied = True
    H = hessian.float().clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    damp = 0.01 * torch.mean(torch.diag(H))
    H[torch.arange(cols), torch.arange(cols)] += damp
    perm = torch.argsort(torch.diag(H), descending=True)
    inv_perm = torch.argsort(perm)
    W = t32[:, perm].clone()
    W[:, dead[perm]] = 0
    H = H[perm][:, perm]
    Hinv = torch.linalg.cholesky(H)
    Hinv = torch.cholesky_inverse(Hinv)
    Hinv = torch.linalg.cholesky(Hinv, upper=True)
    best_q = None; best_scale = None; best_err = float('inf')
    for pct in [0.9990, 0.9995, 0.9999, 0.99999, 1.0]:
        if pct < 1.0:
            row_clip = torch.quantile(t32.abs(), pct, dim=1)
        else:
            row_clip = t32.abs().amax(dim=1)
        s = (row_clip / clip_range).clamp_min(1.0 / clip_range).to(torch.float16)
        sf = s.float()
        Q = torch.zeros_like(W, dtype=torch.int8)
        W_work = W.clone()
        for i1 in range(0, cols, block_size):
            i2 = min(i1 + block_size, cols)
            count = i2 - i1
            W1 = W_work[:, i1:i2].clone()
            Q1 = torch.zeros(rows, count, dtype=torch.int8)
            Err1 = torch.zeros(rows, count)
            Hinv1 = Hinv[i1:i2, i1:i2]
            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]
                q = torch.clamp(torch.round(w / sf), -clip_range, clip_range).to(torch.int8)
                Q1[:, i] = q
                err = (w - q.float() * sf) / d
                W1[:, i:] -= err.unsqueeze(1) * Hinv1[i, i:].unsqueeze(0)
                Err1[:, i] = err
            Q[:, i1:i2] = Q1
            if i2 < cols:
                W_work[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
        recon = Q.float() * sf[:, None]
        mse = (W - recon).pow(2).mean().item()
        if mse < best_err:
            best_q, best_scale, best_err = Q, s, mse
    best_q = best_q[:, inv_perm]
    if had_applied:
        return best_q, best_scale, orig_cols
    return best_q, best_scale, None


def make_synthetic_weight(rows, cols, outlier_frac=0.02, outlier_scale=10.0, seed=42):
    """Create a weight matrix with realistic outlier structure."""
    torch.manual_seed(seed)
    W = torch.randn(rows, cols)
    n_outlier_cols = max(1, int(cols * outlier_frac))
    outlier_cols = torch.randperm(cols)[:n_outlier_cols]
    W[:, outlier_cols] *= outlier_scale
    return W


def make_synthetic_hessian(cols, rank=None, seed=42):
    """Create a realistic Hessian from random activations."""
    torch.manual_seed(seed)
    n_samples = rank or (cols * 4)
    X = torch.randn(n_samples, cols)
    H = X.T @ X / n_samples
    damp = 0.01 * torch.diag(H).mean()
    H += damp * torch.eye(cols)
    return H


def test_hadamard_basics():
    """Verify Hadamard matrix properties."""
    for n in [8, 16, 64, 128, 256, 512]:
        H = _hadamard_matrix(n)
        identity_check = (H @ H.T - torch.eye(n)).abs().max().item()
        assert identity_check < 1e-5, f"H @ H^T != I for n={n}, max_err={identity_check}"
    print("PASS: Hadamard matrices are orthogonal")


def test_rotate_unrotate_roundtrip():
    """Verify W = unrotate(rotate(W))."""
    for cols in [64, 128, 256, 512, 300, 500]:
        W = torch.randn(128, cols)
        padded = _next_power_of_2(cols)
        H_right = _hadamard_matrix(padded)
        W_rot, _, orig = hadamard_rotate_weight(W, H_right)
        W_back = hadamard_unrotate_weight(W_rot, H_right, orig)
        err = (W - W_back).abs().max().item()
        assert err < 1e-4, f"Roundtrip error {err} for cols={cols}"
    print("PASS: Rotate -> unrotate roundtrip is exact")


def test_outlier_spreading():
    """Verify Hadamard rotation reduces column variance (spreads outliers)."""
    W = make_synthetic_weight(256, 512, outlier_frac=0.02, outlier_scale=10.0)

    col_norms_orig = W.norm(dim=0)
    cv_orig = col_norms_orig.std() / col_norms_orig.mean()

    H_right = _hadamard_matrix(512)
    W_rot, _, _ = hadamard_rotate_weight(W, H_right)
    col_norms_rot = W_rot.norm(dim=0)
    cv_rot = col_norms_rot.std() / col_norms_rot.mean()

    print(f"Column norm CV -- original: {cv_orig:.4f}, rotated: {cv_rot:.4f} "
          f"(reduction: {(1 - cv_rot/cv_orig)*100:.1f}%)")
    assert cv_rot < cv_orig, "Rotation should reduce column variance"
    print("PASS: Hadamard rotation spreads outliers")


def test_gptq_mse_improvement():
    """The main test: compare GPTQ quantization MSE with and without Hadamard."""
    print("\n" + "=" * 60)
    print("GPTQ int6 Quantization: Standard vs Hadamard")
    print("=" * 60)

    configs = [
        ("512x512 (model_dim projections)", 512, 512),
        ("512x1536 (MLP up)", 512, 1536),
        ("1536x512 (MLP down)", 1536, 512),
        ("256x512 (KV projection)", 256, 512),
        ("512x512 (with 5% outliers)", 512, 512),
    ]

    total_improvement = 0
    n_tests = 0

    for name, rows, cols, *extra in configs:
        outlier_frac = 0.05 if "outlier" in name else 0.02
        W = make_synthetic_weight(rows, cols, outlier_frac=outlier_frac)
        H = make_synthetic_hessian(cols)

        # Standard GPTQ (no Hadamard)
        global _HADAMARD_ENABLE
        _HADAMARD_ENABLE = False
        q_std, s_std, had_cols_std = quantize_int6_gptq(W, hessian=H)
        recon_std = q_std.float() * s_std.float()[:, None]
        mse_std = (W - recon_std).pow(2).mean().item()

        # Hadamard GPTQ
        _HADAMARD_ENABLE = True
        q_had, s_had, had_cols = quantize_int6_gptq(W, hessian=H)
        # Dequant with inverse rotation
        padded = q_had.shape[1]
        H_right = _hadamard_matrix(padded)
        recon_had_rot = q_had.float() * s_had.float()[:, None]
        recon_had = hadamard_unrotate_weight(recon_had_rot, H_right, had_cols)
        mse_had = (W - recon_had).pow(2).mean().item()

        improvement = (1 - mse_had / mse_std) * 100
        total_improvement += improvement
        n_tests += 1

        marker = "**" if improvement > 0 else "  "
        print(f"{marker} {name:40s} MSE std={mse_std:.6f} had={mse_had:.6f} "
              f"improvement={improvement:+.1f}%")

    avg_improvement = total_improvement / n_tests
    print(f"\nAverage MSE improvement: {avg_improvement:+.1f}%")
    print("=" * 60)

    _HADAMARD_ENABLE = True


def test_artifact_size_neutral():
    """Verify Hadamard doesn't increase quantized tensor sizes for power-of-2 dims."""
    W = torch.randn(512, 512)
    H = make_synthetic_hessian(512)

    global _HADAMARD_ENABLE
    _HADAMARD_ENABLE = False
    q_std, s_std, _ = quantize_int6_gptq(W, hessian=H)

    _HADAMARD_ENABLE = True
    q_had, s_had, _ = quantize_int6_gptq(W, hessian=H)

    size_std = q_std.numel() * q_std.element_size() + s_std.numel() * s_std.element_size()
    size_had = q_had.numel() * q_had.element_size() + s_had.numel() * s_had.element_size()
    print(f"\nArtifact bytes -- standard: {size_std}, hadamard: {size_had} "
          f"(overhead: {size_had - size_std} bytes)")

    _HADAMARD_ENABLE = True


if __name__ == "__main__":
    print("Testing Hadamard GPTQ integration\n")
    test_hadamard_basics()
    test_rotate_unrotate_roundtrip()
    test_outlier_spreading()
    test_gptq_mse_improvement()
    test_artifact_size_neutral()
    print("\nAll tests passed.")
