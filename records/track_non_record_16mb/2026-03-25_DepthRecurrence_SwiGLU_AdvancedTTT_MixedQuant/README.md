# Non-Record Submission: Depth Recurrence + SwiGLU + Mixed Quantization

**Status**: Architectural proposal — untested on 8xH100 (pending compute access)

## Summary of Novel Modifications

This submission explores three architectural changes on top of the current SOTA stack (PR #549: LeakyReLU + Legal TTT + Parallel Muon). Each modification is designed to improve bits-per-byte (BPB) under the 16MB artifact constraint.

### 1. Depth Recurrence (Parameter Tying Across Depth)

**Idea**: Instead of N unique layers, use N/K unique layers looped K times, giving K*N effective depth with only N layers of parameters.

```python
# Standard: 11 unique layers, 11 effective layers
for i in range(11):
    x = block[i](x)

# Depth Recurrence (K=2): 11 unique layers, 22 effective layers
for recurrence in range(2):
    for i in range(11):
        x = block[i](x)
```

**Rationale**: With a 16MB parameter budget, depth recurrence lets us trade compute for parameters. The model gets twice the effective depth without doubling parameter count. The U-Net skip connections only activate on the final pass, giving the early passes a "refinement" role.

**Potential gain**: More effective depth improves representation quality. The parameter savings can be reinvested into wider layers (e.g., dim=576 instead of 512) or more MLP capacity.

**Risk**: As noted in the SwiGLU submission (exp010), naive layer recurrence on a single GPU halves throughput, which may offset gains in a wall-clock-limited setting. On 8xH100 with the Parallel Muon optimizer, the throughput hit may be more manageable.

### 2. SwiGLU Activation (Replacing LeakyReLU(0.5)^2)

**Idea**: Replace the LeakyReLU(0.5)^2 MLP activation with SwiGLU (Shazeer, 2020), which uses a gated linear unit with SiLU activation.

```python
# LeakyReLU^2 (current SOTA)
x = F.leaky_relu(F.linear(x, up_w), negative_slope=0.5).square()
out = F.linear(x, down_w)

# SwiGLU (this submission)
hidden = F.linear(x, up_w)  # up_w is 2x wider
gate = F.silu(hidden[..., :half])
value = hidden[..., half:]
out = F.linear(gate * value, down_w)
```

**Rationale**: SwiGLU is the standard activation in modern LLMs (LLaMA, Mistral, Gemma). It provides:
- Smooth gating that preserves gradient flow better than squaring
- Learnable gate that can adaptively suppress features
- Proven effectiveness across scales

The up projection is 2x wider to accommodate the gate/value split. The down projection operates on the gated (half-width) output, keeping the same output dimension.

**Trade-off**: SwiGLU uses more parameters in the up projection (2x wider). With depth recurrence freeing up parameter budget, this is a net win.

### 3. Mixed int5/int6 Per-Layer Quantization

**Idea**: Not all layers are equally sensitive to quantization. Use int6 for critical layers (first, last, and attention-heavy) and int5 for less sensitive middle MLP layers.

**Rationale**: The current SOTA uses uniform int6 via GPTQ-lite. Mixed quantization can save ~5-8% more space, which can be reinvested into a wider model or more layers.

## Architecture (Proposed Configuration)

| Component | Setting |
|-----------|---------|
| Unique Layers | 11 (512d, 8H, 4KV) |
| Depth Recurrence | 2 (22 effective layers) |
| MLP | SwiGLU 3x (gate+value split) |
| BigramHash | 1536 |
| XSA | Last 4 layers |
| RoPE | Partial (16/64 dims) |
| LN Scale | 1/sqrt(layer+1) |
| VE128 | Layers 9-10 |
| Weight avg | EMA(0.997) + SWA(every 50) |
| Quantization | Mixed int5/int6 + lzma |
| Optimizer | Parameter Banking + Parallel Muon |
| TTT | Legal score-first, 3 epochs SGD |

## Proposed Run Command

```bash
NUM_LAYERS=11 DEPTH_RECURRENCE=2 SWIGLU_ENABLED=1 MIXED_QUANT_ENABLED=1 \
BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Expected Impact (Hypotheses)

| Modification | Expected BPB Delta | Confidence |
|-------------|-------------------|------------|
| Depth Recurrence (K=2) | -0.002 to -0.005 | Medium (depends on throughput hit) |
| SwiGLU replacing LeakyReLU^2 | -0.001 to -0.003 | Medium-High |
| Mixed int5/int6 quant | -0.000 to -0.001 | High (frees space for wider model) |
| **Combined** | **-0.003 to -0.009** | Medium |

## Why This Is a Non-Record Submission

- **No GPU access**: These modifications have not been validated on 8xH100 hardware
- **Throughput uncertainty**: Depth recurrence doubles compute per step, which may or may not fit in the 10-minute wall-clock budget
- **SwiGLU parameter trade-off**: The 2x wider up projection increases parameters; whether depth recurrence frees enough budget is untested
- **Intended as architectural proposal**: These ideas are intended to inspire further exploration by the community

## Included Files

- `train_gpt.py` — Modified SOTA script with all three novel changes
- `submission.json` — Leaderboard metadata
- `README.md` — This file

## Credits

- **Base model + TTT + Parallel Muon**: PR #549 by @abaybektursun
- **SwiGLU activation**: Noam Shazeer (2020), "GLU Variants Improve Transformer"
- **Depth recurrence inspiration**: Universal Transformers (Dehghani et al., 2019)
- **Mixed quantization**: Inspired by GPTQ-lite (PR #374) by @signalrush
