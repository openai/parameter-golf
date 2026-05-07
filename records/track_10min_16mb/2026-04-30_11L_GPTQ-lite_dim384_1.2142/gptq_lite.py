"""
GPTQ-lite for parameter golf.

After training, this script:
1. Loads the trained model state dict
2. Runs calibration data through each linear layer
3. Computes per-row Hessian diagonal
4. Finds optimal per-row clipping thresholds
5. Quantizes with those thresholds (INT8 per-row)
6. Replaces the quantize_state_dict_int8 result with GPTQ-optimized version

Usage: integrate into train_gpt.py after training, before serialization.
"""

import torch
import torch.nn as nn
from torch import Tensor
import math
from typing import Optional


def compute_hessian_diagonal(
    layer: nn.Linear,
    inputs: list[Tensor],  # list of (batch, seq, in_features) tensors
    damping: float = 0.01,
) -> Tensor:
    """
    Compute diagonal of the Hessian H = X^T X for a linear layer.
    Returns shape (in_features,) — one value per input feature.
    """
    H = torch.zeros(layer.weight.shape[1], device=layer.weight.device, dtype=torch.float32)
    n_samples = 0
    for x in inputs:
        x_flat = x.reshape(-1, x.shape[-1]).float()  # (N, in_features)
        H += (x_flat ** 2).sum(dim=0)
        n_samples += x_flat.shape[0]
    H /= n_samples
    # Add damping for numerical stability
    H += damping * H.mean()
    return H


def gptq_lite_quantize_row(
    w_row: Tensor,       # (in_features,) float32
    h_diag: Tensor,      # (in_features,) float32 — Hessian diagonal
    bits: int = 8,
    n_clip_search: int = 20,
) -> tuple[Tensor, float]:
    """
    GPTQ-lite: find optimal per-row clipping threshold using Hessian-weighted error.
    
    For each candidate clip ratio, compute:
        error = sum_i h_diag[i] * (w[i] - quantize(w[i], clip))^2
    and pick the clip that minimizes this.
    
    Returns (quantized_row_int8, scale).
    """
    maxq = 2 ** (bits - 1) - 1  # 127 for int8
    
    w_abs = w_row.abs()
    w_max = w_abs.max().item()
    
    if w_max < 1e-8:
        return torch.zeros_like(w_row, dtype=torch.int8), 1.0
    
    best_error = float('inf')
    best_clip = w_max
    best_q = None
    best_scale = None
    
    # Search over clip ratios from 100% down to ~80% of max
    for i in range(n_clip_search):
        clip = w_max * (1.0 - 0.01 * i)  # 1.00, 0.99, ..., 0.81
        scale = clip / maxq
        if scale < 1e-8:
            continue
        
        w_clipped = w_row.clamp(-clip, clip)
        q = torch.round(w_clipped / scale).clamp(-maxq, maxq).to(torch.int8)
        w_hat = q.float() * scale
        
        # Hessian-weighted reconstruction error
        err = (h_diag * (w_row - w_hat) ** 2).sum().item()
        
        if err < best_error:
            best_error = err
            best_clip = clip
            best_q = q
            best_scale = scale
    
    return best_q, best_scale


def gptq_lite_quantize_weight(
    weight: Tensor,      # (out_features, in_features) float32
    h_diag: Tensor,      # (in_features,) float32
    bits: int = 8,
    n_clip_search: int = 20,
) -> tuple[Tensor, Tensor]:
    """
    Apply GPTQ-lite row by row.
    Returns (quantized_weight int8, scales float16) with shapes
    (out_features, in_features) and (out_features,).
    """
    out_features, in_features = weight.shape
    q_weight = torch.zeros_like(weight, dtype=torch.int8)
    scales = torch.zeros(out_features, dtype=torch.float32)
    
    for i in range(out_features):
        q_row, scale = gptq_lite_quantize_row(
            weight[i].float(), h_diag, bits=bits, n_clip_search=n_clip_search
        )
        q_weight[i] = q_row
        scales[i] = scale
    
    return q_weight, scales.to(torch.float16)


class GPTQLiteQuantizer:
    """
    Hooks into a model to collect activations for Hessian computation,
    then applies GPTQ-lite quantization to all eligible linear layers.
    """
    
    def __init__(self, model: nn.Module, bits: int = 8, n_clip_search: int = 20,
                 min_numel: int = 65536):
        self.model = model
        self.bits = bits
        self.n_clip_search = n_clip_search
        self.min_numel = min_numel
        self.hooks = []
        self.activations: dict[str, list[Tensor]] = {}
        self._name_to_module: dict[str, nn.Linear] = {}
    
    def _register_hooks(self, skip_patterns: tuple[str, ...] = ()):
        """Register forward hooks to capture input activations."""
        for name, module in self.model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if module.weight.numel() < self.min_numel:
                continue
            if any(p in name for p in skip_patterns):
                continue
            
            self.activations[name] = []
            self._name_to_module[name] = module
            
            def make_hook(n):
                def hook(mod, inp, out):
                    # inp[0] shape: (batch, seq, in_features) or (batch, in_features)
                    self.activations[n].append(inp[0].detach().cpu())
                return hook
            
            self.hooks.append(module.register_forward_hook(make_hook(name)))
    
    def _remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()
    
    @torch.no_grad()
    def collect_activations(self, calibration_inputs: list[Tensor],
                            skip_patterns: tuple[str, ...] = ()):
        """Run calibration data through model to collect activations."""
        self._register_hooks(skip_patterns)
        self.model.eval()
        self.model.eval()
        with torch.no_grad():
            for x, y in calibration_inputs:
                self.model(x, y)
        self._remove_hooks()
        print(f"Collected activations for {len(self.activations)} layers")
    
    @torch.no_grad()
    def quantize(self, state_dict: dict) -> dict:
        """
        Apply GPTQ-lite quantization, returning modified quantization objects
        compatible with the existing quantize_state_dict_int8 format.
        
        Returns dict mapping layer weight name -> (q_weight, scales).
        """
        results = {}
        device = next(self.model.parameters()).device
        
        for name, module in self._name_to_module.items():
            if name not in self.activations or not self.activations[name]:
                continue
            
            inputs = self.activations[name]
            w = module.weight.data.float()
            
            # Compute Hessian diagonal on GPU
            h_diag = compute_hessian_diagonal(
                module,
                [x.to(device) for x in inputs],
            )
            
            print(f"  GPTQ-lite quantizing {name}: {w.shape}")
            q_w, scales = gptq_lite_quantize_weight(
                w, h_diag, bits=self.bits, n_clip_search=self.n_clip_search
            )
            
            results[name + '.weight'] = (q_w, scales)
        
        return results


def apply_gptq_lite_to_state_dict(
    model: nn.Module,
    calibration_batches: list[tuple[Tensor, Tensor]],
    skip_patterns: tuple[str, ...] = ('tok_emb', 'lm_head', 'bigram'),
    bits: int = 8,
    n_clip_search: int = 20,
    min_numel: int = 65536,
    damping: float = 0.01,
) -> dict[str, tuple[Tensor, Tensor]]:
    """
    Main entry point. Returns dict of {weight_name: (q_int8, scale_fp16)}
    for all eligible linear layers.
    """
    quantizer = GPTQLiteQuantizer(
        model, bits=bits, n_clip_search=n_clip_search, min_numel=min_numel
    )
    print("Collecting calibration activations...")
    quantizer.collect_activations(calibration_batches, skip_patterns=skip_patterns)
    print("Running GPTQ-lite optimization...")
    return quantizer.quantize(model.state_dict())


if __name__ == '__main__':
    # Smoke test with a tiny model
    import torch.nn.functional as F
    
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(64, 128)
            self.fc2 = nn.Linear(128, 64)
        def forward(self, x, y=None):
            return self.fc2(F.relu(self.fc1(x)))
    
    model = TinyModel().cuda()
    calib = [(torch.randn(4, 16, 64).cuda(), None) for _ in range(8)]
    
    results = apply_gptq_lite_to_state_dict(model, calib, skip_patterns=(), min_numel=0)
    for name, (q, s) in results.items():
        print(f"{name}: q={q.shape} dtype={q.dtype}, scale={s.shape}")
    print("Smoke test passed!")
