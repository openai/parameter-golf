# AttnRes + LeakyReLU² + Legal Score-First TTT + Parallel Muon

**val_bpb: pending** | **~16 MB** | 8×H100 SXM

Built on the 2026-03-23 submission. Adds Full Attention Residuals (AttnRes) from the Kimi Team paper.

## Key Innovation: Full Attention Residuals (AttnRes)

Standard transformers pass a single global residual (`x0`, the initial embedding) to every layer.
AttnRes replaces this with **learned softmax depth-attention over all preceding layer outputs**:

```
x_ctx_l = Σ α_{i→l} · h_i
where α_{i→l} = softmax_i( w_l · RMSNorm(h_i) )
```

- One pseudo-query `w_l ∈ R^{512}` per layer, **zero-initialized** (uniform weights at start = standard residuals)
- Keys are RMSNorm-normalized preceding layer outputs (paper Table 4: removing this hurts)
- 11 layers × 512d = **5,632 parameters** (~22KB, negligible in 16MB budget)
- O(L²) = 121 attention operations — computationally trivial for 11 layers

### Implementation

`_depth_attn()` is called per block with the growing `layer_history` list:

```python
def _depth_attn(self, history: list[Tensor], w: Tensor) -> Tensor:
    if len(history) == 1:
        return history[0]
    h_stack = torch.stack(history, dim=0)               # (L, B, T, d)
    h_normed = F.rms_norm(h_stack, (h_stack.shape[-1],))# normalize keys
    scores = torch.einsum('d,lbtd->lbt', w, h_normed)   # (L, B, T)
    alpha = torch.softmax(scores, dim=0)                 # (L, B, T)
    return torch.einsum('lbt,lbtd->btd', alpha, h_stack) # (B, T, d)
```

The result replaces `x0` as the second argument to each `Block.forward()`. The existing `resid_mix`
parameter in each block then mixes `[current_x, attnres_context]` — `resid_mix[1]` is initialized
to zero so AttnRes contribution starts at zero and is learned gradually.

`attn_res_w` is updated by AdamW (scalar optimizer, same LR as `skip_weights`).
During TTT, `attn_res_w` adapts along with all block parameters (no `blocks.` prefix → not frozen).

### Why this should help

The paper (Kimi Team) reports:
- Block AttnRes (N=8) achieves **1.25× compute advantage** at 5.6 PFLOP/s-days vs standard residuals
- Consistent improvement across model sizes (1B–7B)
- Full AttnRes (all layers) is strictly better than Block AttnRes for small L

The current model already uses `resid_mix` for 2-source mixing (previous layer + initial embedding).
AttnRes upgrades this from 2 static sources to L dynamic, input-dependent sources.

## Inherited Stack (2026-03-23)

| Component | Setting |
|-----------|---------|
| Layers | 11 (512d, 8H, 4KV) |
| MLP | 3× with LeakyReLU(0.5)² |
| BigramHash | 1536 |
| XSA | Last 4 layers |
| RoPE | Partial (16/64 dims) |
| LN Scale | 1/√(layer+1) |
| VE128 | Layers 9-10 |
| Weight avg | EMA(0.997) + Tight SWA(every 50) |
| Quantization | GPTQ-lite int6 + lzma |
| Optimizer | Parameter Banking + Parallel Muon |
| **AttnRes** | **Full (11 layers), zero-init** |

## Run Command

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
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

## Credits

- **AttnRes**: Kimi Team paper "Attention Residuals" (Attention_Residuals.pdf)
- **LeakyReLU² activation**: [PR #493](https://github.com/openai/parameter-golf/pull/493) by @parinzee, [PR #518](https://github.com/openai/parameter-golf/pull/518) by @sofiabod
- **Optimizer (Parameter Banking + Parallel Muon)**: [PR #399](https://github.com/openai/parameter-golf/pull/399) by @abaybektursun
- **TTT recipe**: [PR #461](https://github.com/openai/parameter-golf/pull/461) by @Christopher-Lee-McClendon
- **Base model**: [PR #414](https://github.com/openai/parameter-golf/pull/414) by @signalrush
