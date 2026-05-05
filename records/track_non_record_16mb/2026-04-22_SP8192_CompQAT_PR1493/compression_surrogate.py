"""Compression-aware QAT — illustrative standalone implementation.

This is the mechanism added on top of PR #1493's training loop. The actual
training script (train_gpt.py in this directory) ships in the standard
lzma2+b85-obfuscated form; this file exists so reviewers can read the
mechanism without decoding the obfuscated blob.

Usage in the training loop (matching the integrated train_gpt.py):

    loss = model(x, y)                                  # standard CE loss
    if h.comp_lambda > 0. and step >= h.comp_warmup_steps:
        csl = compression_surrogate_loss(base_model, h.comp_beta)
        loss = loss + h.comp_lambda * csl
    loss.backward()

With COMP_LAMBDA=0.001, COMP_BETA=10.0, COMP_WARMUP_STEPS=200, this is the
configuration used for the 3-seed submission at val_bpb = 1.10314.

Validation: this surrogate correlates with actual zstd compression ratio at
Pearson +0.994 across six synthetic weight distributions (tight Gaussian,
wide Gaussian, uniform, bimodal, concentrated near zero, sparse). See
tests/validate_compression_surrogate.py in the research repo.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn


def compression_surrogate_loss(model: nn.Module, beta: float = 10.0) -> Tensor:
    """Differentiable Shannon entropy of a soft-int6-level histogram.

    Lower return value => weights concentrated near a smaller number of int6
    grid centers => brotli ratio on the packed byte stream improves.

    Args:
        model:  any nn.Module; all 2D linear weight matrices with both shape
                dims >= 32 contribute. Smaller matrices (scalars, embeddings'
                scale tensors) are skipped as they don't meaningfully affect
                the final artifact size.
        beta:   softmax temperature in the soft-assignment step. Larger beta
                = sharper assignment (more like hard rounding); smaller beta
                = smoother gradient. beta=10 is validated to track zstd
                ratio at Pearson +0.994.

    Returns:
        A scalar tensor (mean entropy across contributing matrices), carrying
        gradients back to model weights.
    """
    dev = next(model.parameters()).device
    levels = torch.arange(-31, 32, dtype=torch.float32, device=dev)  # 63 levels
    total = torch.zeros((), dtype=torch.float32, device=dev)
    n = 0

    for m in model.modules():
        if not isinstance(m, nn.Linear):
            continue
        w = m.weight
        if w.ndim != 2 or w.shape[0] < 32 or w.shape[1] < 32:
            continue

        wf = w.float()
        # Scale by the 99.9-th percentile of |w| so tail values don't dominate
        # the soft-assignment; outlier-robust.
        scale = wf.abs().quantile(0.999).clamp_min(1e-8)
        normalized = (wf / scale * 31.0).reshape(-1)

        # Soft assignment to each of the 63 int6 levels via Gaussian kernel
        # in the squared-distance domain. beta controls sharpness.
        soft = torch.softmax(-beta * (normalized.unsqueeze(-1) - levels) ** 2, dim=-1)

        # Pool to per-level probabilities (histogram over the flattened matrix)
        hist = soft.mean(dim=0)

        # Shannon entropy in nats. log(63) = 4.14 is the maximum.
        total = total - (hist * (hist + 1e-10).log()).sum()
        n += 1

    return total / max(n, 1)


if __name__ == "__main__":
    # Quick self-test on a small model
    import numpy as np
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Linear(128, 256, bias=False),
        nn.Linear(256, 128, bias=False),
    )
    nn.init.normal_(model[0].weight, std=0.02)
    nn.init.normal_(model[1].weight, std=0.02)

    loss = compression_surrogate_loss(model, beta=10.0)
    print(f"surrogate entropy: {loss.item():.4f}")
    print(f"max possible (ln 63):  {np.log(63):.4f}")

    loss.backward()
    grad_norm_sum = sum(p.grad.norm().item() for p in model.parameters())
    print(f"sum of grad norms: {grad_norm_sum:.6f}")
    assert loss.item() > 0 and torch.isfinite(loss).item()
    assert grad_norm_sum > 0
    print("OK")
