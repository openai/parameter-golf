# Squeezing a GPT on One GPU in Ten Minutes

Single RTX 5090. Ten-minute runs. No cluster. Final score: **1.265 BPB**, 13.6MB artifact.

## Run-to-Run Variance

The first submission scored 1.328. A reproduction of the same run, same code, same seed, gave 1.340. The "best result" had been a lucky outlier. Run-to-run variance on this hardware is about 0.012, which means anything below a 0.015 delta is noise. Once that became clear, the methodology changed: only chase big moves, and reproduce before trusting.

## Depth-Width Tradeoff

Everyone in this competition builds 9 to 11 layers. A sweep from 9 down to 5, widening the MLP each time to hold parameters constant, found six layers optimal.

Not because depth is bad (below six the model breaks). But in a ten-minute race, fewer layers mean faster steps, and more steps mean more learning. Six layers: 4,222 steps at 142ms each. Nine layers: 3,338 steps at 180ms each. The extra 900 steps matter more than the extra depth.

The six layers split into three encoder and three decoder, connected by learned per-dimension skip weights. The decoder reaches directly back into encoder features. Remove the skips: +0.013 BPB. Load-bearing.

## Untied Embeddings

Untying input and output embeddings costs just 524K extra parameters at vocab size 1024. The win: each matrix gets one job instead of two.

When embeddings are tied, input gradients and output gradients fight. Untying stops the fight, and unlocks an embedding learning rate of 0.6, twelve times higher than anything else in this competition. Doubling the *tied* LR was catastrophic (+0.026 BPB). The output gradient dominates and destabilizes everything. But untied, the embedding has one gradient source, and 0.6 just works.

## Weight Decay and Compression

Weight decay was added to Muon just to shrink the artifact. It improved quality too.

A sweep from 0.005 to 0.050 showed monotonic improvement up to 0.040: **−0.008 BPB and 2.8MB smaller**. At 0.050 the quantization gap widened enough to cancel the gains.

Weight decay keeps weights small. Small weights have smaller dynamic range. Smaller dynamic range means INT8 bins are tighter and more precise. Weight decay is doing quantization-aware training as a side effect. The competition treats training and compression as separate problems. They're the same problem.

## Attention

The baseline shares key-value projections between pairs of heads (grouped-query attention). With only 8 heads in 512 dimensions, that's a real bottleneck. Full MHA, one KV head per query head, costs 4% more compute per step but gives each head independent attention patterns. At this scale, attention diversity matters more than the savings from sharing.

## Other Improvements

- **SwiGLU over ReLU²**: The gating mechanism is more expressive. −0.004 BPB.
- **Training at seq_len 2048**: Single biggest improvement, −0.025 BPB. RoPE extrapolation from 1024→2048 is a poor substitute for actually learning long-range patterns.
- **Logit softcap 20.0** (down from 30.0): Tighter cap, better-calibrated predictions. With 1024 tokens the distribution doesn't need extreme logits.
- **Per-dimension control knobs**: Learned residual mixing, attention/MLP scaling, per-head query gain. 18K parameters total (<0.1%), stored in fp32 through quantization.
- **Sliding window eval** with stride 256: −0.008 BPB, zero changes to training. Just measures what the model can actually do.

## What Didn't Work

Every failure taught the same lesson: at ~4,000 steps, per-step overhead is fatal.

- **Layer recurrence** (+0.051): Same weights run twice to simulate 12 layers from 6. Halves training steps. Capacity from weight reuse can't compensate for lost optimization.
- **Causal convolution** (+0.011): 17% overhead = 603 lost steps = 79M fewer tokens seen.
- **Multi-token prediction** (+0.006): 4.3% overhead, 172 lost steps. Might work at 10K+ steps.
- **EMA / SWA** (+0.24 / +0.010): Cosine LR creates a directed trajectory. Averaging along it gives a point the model already moved past.
- **Label smoothing** (+0.029): The model is underfitting, not overfitting. Regularization is poison here.
- **Parallel attention+MLP** (+0.024): The MLP needs attention-enriched features, not raw input. Sequential composition is how transformers think.

## Final Architecture

- 6 layers (3 encoder + 3 decoder with learned skip connections)
- 512 model dimension
- SwiGLU MLP with 1280 hidden
- 8 full MHA heads (8 KV heads)
- Untied embeddings, embed_lr=0.6
- Per-dimension residual mixing, attention/MLP scaling, query gain
- Muon optimizer (matrix_lr=0.045, weight_decay=0.04, momentum=0.95)
- Train seq_len=2048, eval seq_len=2048, sliding window stride=256
- INT8 + zlib quantization, fp32 control tensors
- 19.15M parameters, 13.6MB artifact

Validated on a single H100: **1.237 BPB**, within 0.013 of the 8×H100 baseline. Loss still decreasing when the clock ran out.