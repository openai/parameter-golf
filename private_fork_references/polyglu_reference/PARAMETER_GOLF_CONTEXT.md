# Parameter Golf Challenge Context

## Challenge Overview

**OpenAI Model Craft Challenge: Parameter Golf** — train the best language model that fits in a 16MB artifact and trains in under 10 minutes on 8xH100s. Evaluated by compression on the FineWeb validation set (tokenizer-agnostic, **bits per byte / BPB**).

- **Artifact limit**: 16MB (model + code)
- **Training time**: 10 minutes on 8×H100
- **Metric**: BPB (bits per byte) on FineWeb validation set — lower is better
- **Challenge period**: March 18 – April 30, 2026

This is an `L(N)` optimization problem: optimize the lowest loss given a fixed number of parameters (N), unconstrained by data, compute budget within 10 min, steps, or architecture.

## Current Leaderboard (as of March 23, 2026)

| Score (BPB) | Author | Key Techniques |
|---|---|---|
| **1.1194** | abaybektursun | LeakyReLU(0.5)² + TTT + Parallel Muon |
| 1.1228 | signalrush | GPTQ-lite + EMA + warmdown3500 + QAT@0.15 |
| 1.1248 | jfprincz | Partial RoPE (16/64) + LN Scale + EMA + XSA4 |
| 1.1271 | jfprincz | XSA on last 4 layers + EMA replacing SWA |
| 1.1307 | unnir | Efficient Partial XSA on deepest 3 layers |
| 1.1428 | thwu1 | Int5-MLP + BigramHash(10240) |
| 1.1458 | Raahil Shah | 3x MLP + SmearGate + BigramHash |
| 1.1502 | aruniyer | 11L, 3x MLP, int6 QAT, zstd-22 |
| 1.1556 | aquariouseworkman | SmearGate + OrthoInit + Muon WD |
| 1.2244 | Baseline | 9L 512dim 1024vocab |

**Key observation**: Every single submission uses a single fixed activation. Nobody has tried adaptive activations.

## Baseline Architecture

```
Model: GPT (decoder-only transformer)
- 9 layers, 512 dim, 8 heads, 4 KV heads (GQA)
- 1024 vocab size, tied embeddings
- 2x MLP expansion with relu² activation
- RoPE positional encoding
- Logit softcap at 30.0
- Skip connections (U-Net style: encoder/decoder halves)
- Residual mixing (learned linear combination of current + initial embeddings)
```

## SOTA Architecture (Leading Submission Stack)

The current leader at 1.1194 BPB uses:

```
- 11 layers (up from 9), 512 dim, 8 heads, 4 KV heads
- 3x MLP expansion (up from 2x) with LeakyReLU(0.5)²
- Partial RoPE (16/64 dims)
- XSA (cross-sequence attention) on last 4 layers
- EMA (exponential moving average) for weight averaging
- Int6 quantization-aware training (QAT)
- GPTQ-lite quantization at export
- zstd compression
- Sliding window evaluation
- BigramHash embeddings
- SmearGate attention mechanism
- Muon optimizer with weight decay
- Test-time training (TTT) with LoRAs
- Legal score-first TTT
- LN Scale (layerwise normalization scaling)
- 2048 sequence length (training + eval)
- 786K tokens per batch
- Warmdown 3500 iterations
```

## Key Model Components to Understand

### The MLP (what we're replacing)

```python
class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        hidden = mlp_mult * dim   # e.g., 3*512=1536
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        x = torch.relu(self.fc(x))
        return self.proj(x.square())  # This is relu² (or LeakyReLU² in SOTA)
```

### The Block (where MLP lives)

```python
class Block(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())

    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x))
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x
```

### The GPT Model (skip connections)

```python
class GPT(nn.Module):
    def forward(self, input_ids, target_ids):
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips = []

        # Encoder half stores skips
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0)
            skips.append(x)
        # Decoder half reuses skips in reverse
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[self.num_encoder_layers + i](x, x0)

        x = self.final_norm(x).reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)
        logits_proj = F.linear(x, self.tok_emb.weight)  # tied embeddings
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        return F.cross_entropy(logits.float(), targets, reduction="mean")
```

## Optimizer Setup

The training uses a split optimizer strategy:

- **Muon optimizer**: For 2D matrix parameters in transformer blocks (fc, proj, attention weights)
- **Adam**: For everything else — embeddings, scalars, 1D parameters, skip weights
- **Learning rate schedule**: Warmdown (linear decay in the final N iterations based on wall clock)
- **Momentum warmup**: Muon momentum starts at 0.85-0.92, warms up to 0.95-0.99

### Critical for PolyGLU integration:
- PolyGLU routing parameters (α, β, gate_net biases, gate_net small weights) should go in the **scalar/Adam** group
- Gate_net's Linear layers (if 2D) could go in either group — but they're small, so Adam is fine
- **α and β must be exempt from weight decay** (the paper learned this the hard way)

## Quantization Pipeline

The challenge uses int8 quantization + zlib/zstd compression:

```python
# Large 2D float tensors: per-row int8 quantization with clipping
# Small tensors (< 65536 elements): kept as fp16 passthrough
# Non-float tensors: exact passthrough
# Then the whole thing is compressed with zlib level 9 or zstd
```

PolyGLU routing parameters:
- `alpha` [1536, 4] = 6,144 elements → **small enough for fp16 passthrough** (< 65,536 threshold)
- `beta` [4] = 4 elements → fp16 passthrough
- `gate_net.0.weight` [16, 512] = 8,192 elements → fp16 passthrough
- `gate_net.0.bias` [16] → fp16 passthrough
- `gate_net.2.weight` [4, 16] = 64 elements → fp16 passthrough
- `gate_net.2.bias` [4] → fp16 passthrough

**All routing parameters fall under the small-tensor threshold and will be kept as fp16**, avoiding any quantization degradation. This is favorable.

## Evaluation

```python
# BPB = bits per byte, tokenizer-agnostic compression metric
# Lower BPB = better model
# The leaderboard evaluates on the FULL FineWeb validation set
# Sliding window evaluation (stride=64) significantly helps
```

## Training Constraints

- **Wall clock**: 10 minutes maximum on 8xH100
- **Steps**: Typically ~4000-5000 steps in that time window
- **Batch**: ~524K-786K tokens per step
- **Sequence length**: 1024-2048 tokens

For tau annealing, with ~5000 steps:
- Step 0: τ = 1.0
- Step 2500: τ ≈ 0.55
- Step 4500: τ ≈ 0.19
- Step 5000+: τ = 0.1

The model has roughly the same number of annealing steps as needed for convergence.

## Compilation and Performance

The model uses `torch.compile(dynamic=False, fullgraph=True)`. Any PolyGLU modifications **must be compatible with torch.compile**. Key issues:

1. **List comprehensions with functions**: `[fn(x) for fn in self.activations]` may break fullgraph compilation. Use explicit computation instead.
2. **Gumbel-Softmax**: `F.gumbel_softmax` is supported by torch.compile.
3. **Mean pooling**: `x.mean(dim=1)` is fine.
4. **torch.stack**: Supported.

Workaround for activation application:
```python
# Instead of:
activated = torch.stack([fn(h) for fn in self.activation_fns], dim=-1)

# Use:
relu2_h = torch.relu(h).square()
tanh_h = torch.tanh(h)
silu_h = F.silu(h)
gelu_h = F.gelu(h)
activated = torch.stack([relu2_h, tanh_h, silu_h, gelu_h], dim=-1)
```
