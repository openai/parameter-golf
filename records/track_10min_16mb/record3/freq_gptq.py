"""
FreqGPTQ — Frequency-aware GPTQ post-training quantization.

FreqGPTQ works by decomposing each weight matrix W into low- and high-frequency
bands (using a real-valued FFT along the column / input-feature dimension) before
running the standard GPTQ error-compensation loop.  Each band gets an independent
GPTQ pass, so quantization error in the smooth (low-frequency) components does not
contaminate the fine-grained (high-frequency) components and vice-versa.

Usage
-----
Import and call from a training script:

    from freq_gptq import collect_hessians, freq_gptq_quantize

    hessians = collect_hessians(model, val_tokens, device, seq_len=1024, num_batches=16)
    q, scale = freq_gptq_quantize(weight_tensor, hessians["blocks.0.attn.c_q.weight"])
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# Hessian collection
# ---------------------------------------------------------------------------

def collect_hessians(
    model: nn.Module,
    val_tokens: Tensor,
    device: torch.device,
    seq_len: int,
    num_batches: int,
) -> dict[str, Tensor]:
    """Collect H = X^T X for every ``nn.Linear`` layer in *model*.

    Parameters
    ----------
    model:
        The trained model (must expose ``nn.Linear`` sub-modules).
    val_tokens:
        1-D integer token tensor used as calibration data (e.g. the
        validation split).  Must contain at least ``num_batches * seq_len + 1``
        tokens.
    device:
        CUDA device on which to run calibration forward passes.
    seq_len:
        Sequence length for each calibration batch.
    num_batches:
        Number of calibration batches to accumulate.

    Returns
    -------
    dict mapping ``"<module_path>.weight"`` → damped Hessian ``Tensor``.
    """
    hessians: dict[str, Tensor] = {}
    hooks: list = []

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        key = name + ".weight"
        cols = module.weight.shape[1]
        hessians[key] = torch.zeros(cols, cols, dtype=torch.float32)

        def make_hook(k: str):
            def fn(m: nn.Module, inp: tuple, out: Tensor) -> None:
                x = inp[0].detach().float()
                if x.ndim == 3:
                    x = x.reshape(-1, x.shape[-1])
                hessians[k].add_((x.T @ x).cpu())
            return fn

        hooks.append(module.register_forward_hook(make_hook(key)))

    model.eval()
    total_seqs = max((val_tokens.numel() - 1) // seq_len, 1)
    with torch.inference_mode():
        for i in range(num_batches):
            s = (i * seq_len) % (total_seqs * seq_len)
            tok = val_tokens[s : s + seq_len + 1].to(device=device, dtype=torch.int64)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                model(tok[:-1].unsqueeze(0), tok[1:].unsqueeze(0))
    for h in hooks:
        h.remove()
    model.train()

    # Apply damping to make H positive-definite for Cholesky
    for key in hessians:
        H = hessians[key]
        H /= num_batches
        damp = 0.01 * H.diagonal().mean().clamp_min(1e-6)
        H.diagonal().add_(damp)

    return hessians


# ---------------------------------------------------------------------------
# Core GPTQ
# ---------------------------------------------------------------------------

def gptq_quantize(
    W: Tensor,
    H: Tensor,
    block_size: int = 128,
    clip_range: int = 127,
) -> tuple[Tensor, Tensor]:
    """Hessian-aware int8 quantization using GPTQ with Cholesky error compensation.

    Parameters
    ----------
    W:
        2-D weight matrix ``[out_features, in_features]``.
    H:
        Hessian ``[in_features, in_features]`` (H = X^T X, already damped).
    block_size:
        Number of columns quantised per GPTQ block.
    clip_range:
        Symmetric integer range (default 127 → int8).

    Returns
    -------
    (q, scale) where *q* is ``torch.int8`` and *scale* is per-row ``torch.float16``.
    """
    t32 = W.float()
    rows, cols = t32.shape
    H = H.float().clone()

    dead = H.diagonal() == 0
    H[dead, dead] = 1
    damp = 0.01 * H.diagonal().mean()
    H.diagonal().add_(damp)

    perm = torch.argsort(H.diagonal(), descending=True)
    inv_perm = torch.argsort(perm)
    Wp = t32[:, perm].clone()
    Wp[:, dead[perm]] = 0
    Hp = H[perm][:, perm]

    Hinv = torch.linalg.cholesky(Hp)
    Hinv = torch.cholesky_inverse(Hinv)
    Hinv = torch.linalg.cholesky(Hinv, upper=True)

    s = (t32.abs().amax(dim=1).clamp_min(1e-8) / clip_range).to(torch.float16)
    sf = s.float()
    Q = torch.zeros(rows, cols, dtype=torch.int8)
    Ww = Wp.clone()

    for i1 in range(0, cols, block_size):
        i2 = min(i1 + block_size, cols)
        cnt = i2 - i1
        W1 = Ww[:, i1:i2].clone()
        Q1 = torch.zeros(rows, cnt, dtype=torch.int8)
        E1 = torch.zeros(rows, cnt)
        Hi = Hinv[i1:i2, i1:i2]
        for i in range(cnt):
            w = W1[:, i]
            d = Hi[i, i]
            q = torch.clamp(torch.round(w / sf), -clip_range, clip_range).to(torch.int8)
            Q1[:, i] = q
            err = (w - q.float() * sf) / d
            W1[:, i:] -= err.unsqueeze(1) * Hi[i, i:].unsqueeze(0)
            E1[:, i] = err
        Q[:, i1:i2] = Q1
        if i2 < cols:
            Ww[:, i2:] -= E1 @ Hinv[i1:i2, i2:]

    return Q[:, inv_perm], s


# ---------------------------------------------------------------------------
# FreqGPTQ
# ---------------------------------------------------------------------------

def freq_gptq_quantize(
    W: Tensor,
    H: Tensor | None,
    block_size: int = 128,
    clip_range: int = 127,
) -> tuple[Tensor, Tensor]:
    """FreqGPTQ: split W into low- and high-frequency bands via rfft, then
    quantise each band independently with GPTQ before recombining.

    Parameters
    ----------
    W:
        2-D weight matrix ``[out_features, in_features]``.
    H:
        Hessian ``[in_features, in_features]``.  When *None* or W is not 2-D,
        falls back to plain per-row int8 quantization.
    block_size:
        GPTQ block size.
    clip_range:
        Symmetric integer range for the final storage quantization.

    Returns
    -------
    (q, scale) in the same format as :func:`gptq_quantize`.

    Algorithm
    ---------
    1. W_freq = rfft(W, dim=-1)                         → [out, n_freq] complex
    2. W_low  = irfft(W_freq with high half zeroed)     → smooth content
    3. W_high = irfft(W_freq with low half zeroed)      → fine-grained content
    4. Run GPTQ(W_low,  H) and GPTQ(W_high, H) independently.
    5. W_combined = dequant(W_low) + dequant(W_high)
    6. Re-quantise W_combined to int8 for storage.

    Both bands receive the same input activations x, so the full Hessian
    H = X^T X is valid for both.  The independent GPTQ passes ensure that
    error compensation in one band does not bleed into the other.
    """
    t32 = W.float()
    if H is None or t32.ndim != 2:
        # Fallback: plain per-row int8 (no Hessian available)
        return _plain_int8_quantize(t32)

    rows, cols = t32.shape

    # --- Step 1-3: frequency split -----------------------------------------
    W_freq = torch.fft.rfft(t32, dim=-1)   # [rows, n_freq] complex
    n_freq = W_freq.shape[1]
    half = n_freq // 2
    if half < 1:
        # Weight too narrow to split; fall back to standard GPTQ
        return gptq_quantize(t32, H, block_size, clip_range)

    W_f_low = W_freq.clone()
    W_f_low[:, half:] = 0
    W_f_high = W_freq.clone()
    W_f_high[:, :half] = 0

    W_low = torch.fft.irfft(W_f_low, n=cols, dim=-1)   # low-frequency content
    W_high = torch.fft.irfft(W_f_high, n=cols, dim=-1) # high-frequency content

    # --- Step 4: independent GPTQ passes -----------------------------------
    Q_low, s_low = gptq_quantize(W_low, H, block_size, clip_range)
    Q_high, s_high = gptq_quantize(W_high, H, block_size, clip_range)

    # --- Step 5-6: recombine and re-quantise --------------------------------
    W_combined = (
        Q_low.float() * s_low.float()[:, None]
        + Q_high.float() * s_high.float()[:, None]
    )
    return _plain_int8_quantize(W_combined)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _plain_int8_quantize(t32: Tensor) -> tuple[Tensor, Tensor]:
    """Simple per-row int8 quantization (fallback, no Hessian)."""
    if t32.ndim == 2:
        clip_abs = t32.abs().amax(dim=1).clamp_min(1e-8)
        scale = (clip_abs / 127.0).to(torch.float16)
        q = torch.clamp(torch.round(t32 / scale.float()[:, None]), -127, 127).to(torch.int8)
        return q.contiguous(), scale.contiguous()
    # Scalar / vector fallback
    amax = float(t32.abs().max().item()) if t32.numel() else 1.0
    scale = torch.tensor(amax / 127.0 if amax > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(t32 / scale), -127, 127).to(torch.int8)
    return q.contiguous(), scale
