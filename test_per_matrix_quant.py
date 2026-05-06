"""Test per-matrix bit allocation: Q,K at int4, everything else at int6."""
import torch
import numpy as np
import math
import struct
from pathlib import Path

def quantize_tensor(w, bits, k_sigma=9.0):
    """Quantize a 2D weight tensor to given bit width with SDClip."""
    levels = (1 << bits) - 1
    std = w.float().std(dim=-1, keepdim=True)
    clip_val = k_sigma * std
    w_clipped = w.float().clamp(-clip_val, clip_val)
    w_min = w_clipped.min(dim=-1, keepdim=True).values
    w_max = w_clipped.max(dim=-1, keepdim=True).values
    w_range = (w_max - w_min).clamp(min=1e-10)
    w_norm = (w_clipped - w_min) / w_range
    w_quant = (w_norm * levels).round().clamp(0, levels)
    w_deq = w_quant / levels * w_range + w_min
    mse = ((w.float() - w_deq) ** 2).mean().item()
    return w_deq, mse

print("Loading model...")
sd = torch.load('final_model.pt', map_location='cpu', weights_only=True)

# Test 1: All int6 (baseline)
print("\n=== All int6 (baseline) ===")
total_mse_6 = 0
total_params = 0
for name, w in sd.items():
    if w.dim() < 2 or w.numel() < 100:
        continue
    _, mse = quantize_tensor(w, 6)
    total_mse_6 += mse * w.numel()
    total_params += w.numel()
avg_mse_6 = total_mse_6 / total_params
print(f"  Avg MSE: {avg_mse_6:.10f}")

# Test 2: Q,K at int4, rest at int6
print("\n=== Q,K at int4, rest at int6 ===")
total_mse_mixed = 0
qk_params = 0
other_params = 0
for name, w in sd.items():
    if w.dim() < 2 or w.numel() < 100:
        continue
    is_qk = ('c_q' in name or 'c_k' in name or 'q_proj' in name or 'k_proj' in name)
    bits = 4 if is_qk else 6
    _, mse = quantize_tensor(w, bits)
    total_mse_mixed += mse * w.numel()
    if is_qk:
        qk_params += w.numel()
    else:
        other_params += w.numel()
avg_mse_mixed = total_mse_mixed / (qk_params + other_params)
print(f"  Avg MSE: {avg_mse_mixed:.10f}")
print(f"  Q,K params: {qk_params:,} at int4")
print(f"  Other params: {other_params:,} at int6")

# Test 3: Q,K at int5, rest at int6
print("\n=== Q,K at int5, rest at int6 ===")
total_mse_5 = 0
for name, w in sd.items():
    if w.dim() < 2 or w.numel() < 100:
        continue
    is_qk = ('c_q' in name or 'c_k' in name or 'q_proj' in name or 'k_proj' in name)
    bits = 5 if is_qk else 6
    _, mse = quantize_tensor(w, bits)
    total_mse_5 += mse * w.numel()
avg_mse_5 = total_mse_5 / (qk_params + other_params)
print(f"  Avg MSE: {avg_mse_5:.10f}")

# Space savings
qk_saved_4 = qk_params * 2 / 8  # 2 bits saved per param
qk_saved_5 = qk_params * 1 / 8  # 1 bit saved per param
print(f"\n=== Space Savings ===")
print(f"  Q,K int4: saves {qk_saved_4/1024:.1f} KB ({qk_saved_4/1024/1024:.2f} MB)")
print(f"  Q,K int5: saves {qk_saved_5/1024:.1f} KB ({qk_saved_5/1024/1024:.2f} MB)")

# Relative MSE increase
print(f"\n=== Quality Impact ===")
print(f"  All int6:       MSE = {avg_mse_6:.10f} (baseline)")
print(f"  Q,K int5:       MSE = {avg_mse_5:.10f} ({(avg_mse_5/avg_mse_6 - 1)*100:+.1f}%)")
print(f"  Q,K int4:       MSE = {avg_mse_mixed:.10f} ({(avg_mse_mixed/avg_mse_6 - 1)*100:+.1f}%)")
print(f"\nIf MSE increase < 5%: per-matrix allocation WORKS.")
print(f"If MSE increase > 20%: per-matrix allocation FAILS.")
