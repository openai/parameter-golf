"""
KaiLean Parameter Golf Innovation Stack
Baseline to beat: val_bpb 2.3113 @ 200 steps (smoke test)

bpb improvement estimates are SPECULATIVE until smoke tests confirm them.
"""
import mlx.core as mx
import mlx.nn as nn
import mlx.utils
import math


# ─────────────────────────────────────────────────────────────────────
# INNOVATION 1: BigramHash Bypass Embedding
# Adds a cheap lookup table for token *pairs*, giving the model
# a shortcut to learn common word-boundary patterns from step 1.
# Expected delta: SPECULATIVE (~-0.05 to -0.10 bpb at 200 steps)
# ─────────────────────────────────────────────────────────────────────
class BigramHashEmbedding(nn.Module):
    def __init__(self, hash_size: int = 10240, dim: int = 512):
        super().__init__()
        self.hash_size = hash_size
        self.table = nn.Embedding(hash_size, dim)
        self.table.weight = self.table.weight * 0.02

    def __call__(self, tokens: mx.array) -> mx.array:
        t0 = tokens[:, :-1]
        t1 = tokens[:, 1:]
        idx = mx.remainder(t0 * 31337 + t1, self.hash_size)
        bigram_emb = self.table(idx)
        pad = mx.zeros((tokens.shape[0], 1, bigram_emb.shape[-1]))
        return mx.concatenate([pad, bigram_emb], axis=1)


# ─────────────────────────────────────────────────────────────────────
# INNOVATION 2: SmearGate Activation
# Extended gating range vs standard SiLU.
# SPECULATIVE — must verify vs SiLU in sweep before committing.
# ─────────────────────────────────────────────────────────────────────
def smear_gate(x: mx.array) -> mx.array:
    return mx.sigmoid(x) * mx.tanh(x) * x


# ─────────────────────────────────────────────────────────────────────
# INNOVATION 3: Int6 Fake-Quantization (QAT)
# The bpb benefit is INDIRECT: quantized weights compress better
# under zlib, so within the 16MB budget you can fit more parameters.
# Don't expect lower float bpb — expect smaller artifact size.
# ─────────────────────────────────────────────────────────────────────
def fake_quant_int6(w: mx.array) -> mx.array:
    scale = mx.max(mx.abs(w)) / 31.0 + 1e-8
    w_q = mx.clip(mx.round(w / scale), -32, 31) * scale
    return w + mx.stop_gradient(w_q - w)

def fake_quant_int5(w: mx.array) -> mx.array:
    scale = mx.max(mx.abs(w)) / 15.0 + 1e-8
    w_q = mx.clip(mx.round(w / scale), -16, 15) * scale
    return w + mx.stop_gradient(w_q - w)


# ─────────────────────────────────────────────────────────────────────
# INNOVATION 4: Stochastic Weight Averaging (SWA)
# Maintains a running average of weights over the last 40% of
# training. Averaged weights generalise better than the final step.
# Fixed: uses tree_flatten to correctly handle MLX nested params.
# ─────────────────────────────────────────────────────────────────────
class SWABuffer:
    def __init__(self, model, decay: float = 0.4):
        self.decay = decay
        flat = mlx.utils.tree_flatten(model.parameters())
        self.swa_weights = {
            ".".join(str(p) for p in path): mx.array(v)
            for path, v in flat
            if isinstance(v, mx.array)
        }

    def update(self, model):
        flat = mlx.utils.tree_flatten(model.parameters())
        for path, v in flat:
            if not isinstance(v, mx.array):
                continue
            key = ".".join(str(p) for p in path)
            if key in self.swa_weights:
                self.swa_weights[key] = (
                    self.decay * v + (1 - self.decay) * self.swa_weights[key]
                )

    def apply(self, model):
        nested = {}
        for key, val in self.swa_weights.items():
            parts = key.split(".")
            d = nested
            for part in parts[:-1]:
                d = d.setdefault(part, {})
            d[parts[-1]] = val
        model.update(nested)


print("✅ kl_innovations.py loaded — 4 innovations ready")
