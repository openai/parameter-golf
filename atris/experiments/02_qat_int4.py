"""
Experiment 02: Quantization-Aware Training (QAT)

HYPOTHESIS: The baseline loses 0.007 BPB from post-training INT8 quantization
(1.2172 → 1.2244). QAT eliminates this loss by teaching the model to be
robust to quantization during training. Going to INT4 with QAT could save
~50% of the artifact size, letting us use a much larger model.

KEY INSIGHT: The 16MB limit is on the COMPRESSED artifact. INT4 weights
compress better than INT8 under zlib. With QAT INT4, we might fit a model
that's 2× larger in parameter count.

APPROACH:
1. Implement fake quantization (STE) for INT4 during forward pass
2. Weights stay in FP32 for gradient updates, but forward sees quantized values
3. Train with this in the loop from the start
4. At export, real INT4 quantization produces near-zero quality loss

MODIFICATIONS TO train_gpt.py:
- Add FakeQuantize module with STE
- Wrap CastedLinear.forward to apply fake quant
- Modify quantize_state_dict to export INT4 instead of INT8
- Adjust INT4 packing (two values per byte)

VARIANTS:
- 02a: QAT INT8 (baseline improvement, eliminate 0.007 BPB quant loss)
- 02b: QAT INT4 (half the model bytes → room for larger model)
- 02c: QAT mixed precision (INT4 for attention, INT8 for MLP)
- 02d: QAT INT4 + wider model (use the saved bytes for MODEL_DIM=768)

EXPECTED IMPACT: 0.007-0.02 BPB improvement
RISK: INT4 training instability, need careful scale initialization.
"""

# Fake quantization with Straight-Through Estimator
#
# class FakeQuantize(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, x, bits=4):
#         qmin, qmax = -(2**(bits-1)), 2**(bits-1) - 1
#         scale = x.abs().max() / qmax
#         x_q = torch.clamp(torch.round(x / scale), qmin, qmax) * scale
#         return x_q
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         return grad_output, None  # STE: pass gradient through
#
# class QATCastedLinear(nn.Linear):
#     def __init__(self, *args, bits=4, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.bits = bits
#
#     def forward(self, x):
#         w_q = FakeQuantize.apply(self.weight, self.bits)
#         return F.linear(x, w_q.to(x.dtype), self.bias)
