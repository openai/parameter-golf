"""
Entropy-Constrained Training (ECT) Module

Learned Per-Layer Compression Allocation for Rate-Distortion Optimal Neural Networks.

Key idea: Each weight matrix gets a learnable clip parameter `log_k` that controls
how aggressively it will be quantized. During training, quantization noise proportional
to k is injected via straight-through estimator. A rate penalty encourages the model
to push k higher (more compression) on insensitive layers.

The model learns its own rate-distortion curve per layer during training.

Math:
  compressed_bits_per_param ≈ b - log2(k) + 0.5*log2(πe/2)  [from SDClip theory]

  L_total = L_CE + λ * Σ_i (bits_per_param(k_i) * n_params_i)

  The gradient of the rate penalty w.r.t. log_k_i is:
    ∂R/∂log_k_i = -n_params_i / (k_i * ln(2))

  This pushes k higher (more compression) unless the CE loss pushes back.
"""

import torch
import torch.nn as nn
import math


class ECTController:
    """Manages per-layer compression allocation during training.

    Usage:
        ect = ECTController(model, target_bytes=16_000_000, code_bytes=19000)

        # In training loop:
        ect.inject_noise(model, training_frac)  # before forward
        rate_loss = ect.rate_loss()              # add to CE loss
        ect.remove_noise(model)                  # after backward
    """

    def __init__(self, model, target_bytes=16_000_000, code_bytes=19000,
                 bits=6, embed_bits=8, initial_k=12.85, embed_k=20.0,
                 lambda_rate=0.001, warmup_frac=0.3):
        """
        Args:
            model: GPT model with named_parameters
            target_bytes: total artifact size limit
            code_bytes: code size (subtracted from budget)
            bits: quantization bits for matrices
            embed_bits: quantization bits for embeddings
            initial_k: starting clip sigmas for matrices
            embed_k: fixed clip sigmas for embeddings
            lambda_rate: weight of rate penalty
            warmup_frac: fraction of training before ECT kicks in
        """
        self.target_bytes = target_bytes
        self.code_bytes = code_bytes
        self.model_budget = target_bytes - code_bytes
        self.bits = bits
        self.embed_bits = embed_bits
        self.embed_k = embed_k
        self.lambda_rate = lambda_rate
        self.warmup_frac = warmup_frac

        # Collect matrix parameters and create learnable log_k per layer
        self.param_info = []
        self.log_k_params = nn.ParameterList()

        for name, p in model.named_parameters():
            if p.ndim == 2 and p.numel() > 65536:
                if 'tok_emb' in name:
                    # Embeddings use fixed k
                    self.param_info.append({
                        'name': name, 'param': p, 'n_params': p.numel(),
                        'bits': embed_bits, 'fixed_k': embed_k, 'log_k_idx': None
                    })
                else:
                    # Matrix params get learnable k
                    log_k = nn.Parameter(torch.tensor(math.log(initial_k), dtype=torch.float32))
                    idx = len(self.log_k_params)
                    self.log_k_params.append(log_k)
                    self.param_info.append({
                        'name': name, 'param': p, 'n_params': p.numel(),
                        'bits': bits, 'fixed_k': None, 'log_k_idx': idx
                    })

        self.total_matrix_params = sum(info['n_params'] for info in self.param_info
                                        if info['fixed_k'] is None)
        self._saved_weights = {}

    def get_k(self, info):
        """Get current k value for a layer."""
        if info['fixed_k'] is not None:
            return info['fixed_k']
        return torch.exp(self.log_k_params[info['log_k_idx']]).item()

    def estimated_model_bytes(self):
        """Estimate total compressed model size with current k values."""
        total_bits = 0
        for info in self.param_info:
            if info['fixed_k'] is not None:
                k = info['fixed_k']
                b = info['bits']
            else:
                k = torch.exp(self.log_k_params[info['log_k_idx']]).item()
                b = info['bits']
            bits_per_param = b - math.log2(k) + 0.5 * math.log2(math.pi * math.e / 2)
            total_bits += bits_per_param * info['n_params']
        return total_bits / 8  # bytes

    def rate_loss(self):
        """Compute differentiable rate penalty.

        Returns scalar that, when minimized, pushes k higher on insensitive layers.
        """
        total_bits = torch.tensor(0.0, device=self.log_k_params[0].device)
        for info in self.param_info:
            if info['fixed_k'] is not None:
                continue  # fixed k, no gradient
            log_k = self.log_k_params[info['log_k_idx']]
            k = torch.exp(log_k)
            # bits_per_param = b - log2(k) + const
            # = b - log_k/ln(2) + const
            bits_pp = info['bits'] - log_k / math.log(2) + 0.5 * math.log2(math.pi * math.e / 2)
            total_bits = total_bits + bits_pp * info['n_params']
        return total_bits / 8 / 1e6  # in MB, for nicer scale

    def inject_noise(self, training_frac):
        """Inject quantization noise proportional to each layer's k.

        Uses straight-through estimator: noise in forward, gradient flows through.
        Only active after warmup period.
        """
        if training_frac < self.warmup_frac:
            return

        for info in self.param_info:
            p = info['param']
            self._saved_weights[info['name']] = p.data.clone()

            with torch.no_grad():
                if info['fixed_k'] is not None:
                    k = info['fixed_k']
                else:
                    k = torch.exp(self.log_k_params[info['log_k_idx']]).item()

                clip_range = 2 ** (info['bits'] - 1) - 1
                row_std = p.data.float().std(dim=1, keepdim=True).clamp(min=1e-10)
                scale = k * row_std / clip_range

                # Simulate quantization: round + add uniform noise
                q = torch.clamp(torch.round(p.data.float() / scale), -clip_range, clip_range)
                # Straight-through: use quantized values in forward
                p.data = (q * scale).to(p.dtype)

    def remove_noise(self):
        """Restore original weights after backward pass."""
        for info in self.param_info:
            if info['name'] in self._saved_weights:
                info['param'].data = self._saved_weights[info['name']]
        self._saved_weights.clear()

    def log_stats(self):
        """Return string with per-layer k values and estimated sizes."""
        lines = []
        for info in self.param_info:
            k = self.get_k(info)
            b = info['bits']
            bpp = b - math.log2(k) + 0.5 * math.log2(math.pi * math.e / 2)
            mb = bpp * info['n_params'] / 8 / 1e6
            lines.append(f"  {info['name']}: k={k:.2f} bpp={bpp:.2f} size={mb:.2f}MB")
        est = self.estimated_model_bytes()
        lines.append(f"  TOTAL estimated: {est/1e6:.2f}MB (budget: {self.model_budget/1e6:.2f}MB)")
        return '\n'.join(lines)


if __name__ == '__main__':
    # Quick test
    import torch.nn as nn

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok_emb = nn.Embedding(8192, 512)
            self.layers = nn.ModuleList([
                nn.Linear(512, 2048, bias=False) for _ in range(11)
            ])

    model = DummyModel()
    ect = ECTController(model, target_bytes=16_000_000)

    print("Initial state:")
    print(ect.log_stats())
    print(f"\nRate loss: {ect.rate_loss().item():.4f} MB")
    print(f"Estimated size: {ect.estimated_model_bytes()/1e6:.2f} MB")

    # Simulate pushing some k values up
    for i in range(5):
        ect.log_k_params[i].data = torch.tensor(math.log(25.0))

    print(f"\nAfter pushing 5 layers to k=25:")
    print(f"Rate loss: {ect.rate_loss().item():.4f} MB")
    print(f"Estimated size: {ect.estimated_model_bytes()/1e6:.2f} MB")
