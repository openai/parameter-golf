# Advanced Techniques Deep Dives

## Muon Optimizer Family

- **Muon:** SGD + Nesterov momentum, post-processed via Newton-Schulz to nearest orthogonal matrix. ~35% faster training than AdamW.
- **NorMuon:** Adds per-neuron adaptive learning rates. Improves distributed scaling.
- **Muon Weight Decay:** Decoupled WD (`p.mul_(1 - wd * lr)`). First brought to competition by @notapplica in #60. Improves generalization and compressibility.

## QAT with STE (Quantization-Aware Training)

- Simulates quantization in forward pass; STE passes gradients through rounding
- **Late activation is key:** 70% (#117), 75% (#130), 85% (#76), **96% (#315)** — later = better
- STE corrupts Muon's momentum subspace; late activation minimizes damage
- #315's Late QAT (final 4% only, during low-LR warmdown): int6 gap cuts from ~0.04 to ~0.007 BPB
- Int8 QAT not worth it (20% step overhead costs ~2000 steps, per #145)
- #76 dropped QAT entirely at 12L, using WD=0.04 alone for quantization robustness

## SmearGate & Bigram Hash Embedding

- **SmearGate:** ~512-param gate blending each token's embedding with the previous token's. Injects bigram context before transformer processing.
- **BigramHash:** Hash table (2048-10240 buckets, dim=128) mapping token pairs to learned embeddings. Near-zero parameter cost.
- **OrthoInit is critical:** SmearGate without OrthoInit hurts BPB by 0.003 (#212 ablation). Every successful SmearGate submission uses OrthoInit.
- **Origin:** @unnir in PR #102/#135

## Exclusive Self-Attention (XSA)

- Removes self-value bias from attention output via orthogonal projection (arXiv:2603.09078)
- Applied to last 3-4 layers only ("Partial XSA")
- Zero parameters, ~2ms/step overhead with GQA-aware implementation
- Combined with EMA (0.997), XSA on last 4 layers → best pending: #287 at 1.1280
- Near-universal among frontier submissions
- **Origin:** @unnir in PR #265

## Test-Time Training (TTT)

- Adapts model weights during evaluation using backward-looking context
- **LoRA TTT** (@samacqua, #77): rank-8 LoRA adapters, ~0.003 BPB gain (small but only 1/10 eval budget)
- **Full-model SGD TTT** (@timowhite88, #152): 0.034 BPB but pre-eval version ruled invalid
- **Key interaction:** TTT hurts strong XSA+EMA bases (+0.016 worse in #303) but helps weak ones (-0.024 in #317)
- **Reptile meta-TTT** (#296): 0.011 BPB on SmearGate models (10x naive). Trains inner-loop directions during last 20% of wallclock. Error-guided TTT is negative.

## #315's Three New Techniques (Best Pending: 1.1250)

1. **Partial RoPE (16/64 dims):** Only 25% of head dims get position encoding. Remaining dims learn semantic similarity independent of distance. Zero params.
2. **LN Scale (1/sqrt(layer+1)):** Damps deeper layer contributions to residual stream. Layer 0: x1.0, Layer 10: x0.302. Zero params.
3. **Late QAT (final 4% only):** STE activates only when lr_scale < 0.1 during warmdown. Model converges in full precision first. Zero params.
- Together: +0.0023 BPB over #287. Individually small but compound cleanly.

## Other Notable Techniques

- **Low-Rank Q Factorization** (#215): Q matrices have extreme condition numbers (100M+). Factoring 512→192→512 saves 25% Q params, 22% faster steps.
- **Late-K Passthrough** (#99): Keep key projection weights of final 2 layers in fp16. Small errors in late-layer keys cascade into large attention changes.
- **SWA (Stochastic Weight Averaging):** Averages weights across checkpoints. Must use fp32 accumulation (bf16 causes catastrophic loss). With 84+ checkpoints, quant gap can reverse.
- **EMA (Exponential Moving Average):** decay=0.997 outperforms SWA on XSA stack; SWA outperforms EMA on #198 base. decay=0.999 hurts (too slow).
