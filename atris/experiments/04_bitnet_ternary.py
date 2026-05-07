"""
Experiment 04: BitNet / Ternary Weights

HYPOTHESIS: 1.58-bit weights {-1, 0, 1} compress to near-zero under zlib
(only 3 possible values per weight). This means we can fit a DRAMATICALLY
larger model in 16MB. A model with 10× more parameters but ternary weights
could compress smaller than the current FP32→INT8 model.

KEY INSIGHT: Compression ratio. INT8 has 256 possible values per byte.
Ternary has 3 values, encodable in ~1.58 bits. Under zlib, ternary weight
matrices compress to roughly 20% of their INT8 equivalent. This means:
- Current: ~15.8MB for ~3.6M params at INT8
- Ternary: ~15.8MB could fit ~18M+ params
- 5× more parameters = dramatically more model capacity

APPROACH:
1. Implement BitLinear layer with ternary weights
2. Use absmean quantization: w_ternary = sign(w) * (|w| > threshold)
3. Activation quantization to INT8 for matmul efficiency
4. Scale factors per-row (cheap, high impact)
5. Train with STE through the quantization

MODIFICATIONS TO train_gpt.py:
- Replace CastedLinear with BitLinear
- Custom quantize_state_dict for ternary encoding (2 bits per weight, packed)
- Larger model dimensions (1024 or 1536)
- May need adjusted optimizer (ternary-aware Muon?)

VARIANTS:
- 04a: BitNet ternary, 512 dim, 9 layers (baseline arch, ternary weights)
- 04b: BitNet ternary, 1024 dim, 9 layers (2× wider)
- 04c: BitNet ternary, 768 dim, 12 layers (wider + deeper)
- 04d: Mixed: ternary attention, INT4 MLP (MLP needs more precision)

EXPECTED IMPACT: 0.02-0.05 BPB improvement (if it works)
RISK: HIGH. Ternary training is unstable. May need careful initialization,
      learning rate warmup, and gradient clipping. Quality floor unknown.

REFERENCES:
- "The Era of 1-bit LLMs" (Ma et al., 2024)
- "BitNet b1.58" (Wang et al., 2024)
"""

# BitLinear implementation sketch:
#
# class BitLinear(nn.Module):
#     def __init__(self, in_features, out_features):
#         super().__init__()
#         self.weight = nn.Parameter(torch.randn(out_features, in_features))
#         self.scale = nn.Parameter(torch.ones(out_features))
#
#     def ternary_quantize(self, w):
#         # Absmean quantization
#         gamma = w.abs().mean()
#         w_ternary = torch.sign(w) * (w.abs() > 0.5 * gamma).float()
#         # STE: forward uses ternary, backward uses real weights
#         return w + (w_ternary - w).detach()
#
#     def forward(self, x):
#         w_q = self.ternary_quantize(self.weight)
#         # Scale per output channel
#         y = F.linear(x, w_q.to(x.dtype))
#         return y * self.scale.to(x.dtype).unsqueeze(0).unsqueeze(0)
#
# Custom compression for ternary:
# - Encode {-1, 0, 1} as {0, 1, 2}
# - Pack 5 values per byte (3^5 = 243 < 256)
# - zlib on top of that
# - Theoretical: 1.58 bits/weight + zlib ≈ ~1 bit/weight effective
