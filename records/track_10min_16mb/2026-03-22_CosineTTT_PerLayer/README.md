# Cosine TTT Scheduling with Per-Layer Learning Rates

Mean val_bpb = 1.0970 (3 seeds, std=0.0010) | 8×H100 SXM | 600s train + 465s TTT + 187s eval

## Results

| Seed | Steps | Pre-TTT | Post-TTT | Artifact |
|------|-------|---------|----------|----------|
| 1337 | 7,101 | 1.1577 | 1.0959 | 15.4 MB |
| 42 | 6,700 | 1.1588 | 1.0971 | 15.5 MB |
| 7 | 6,987 | 1.1580 | 1.0979 | 15.8 MB |

## Background

Starting from the community stack (PRs #162, #180, #315, #398), we spent several days exploring ways to improve compression and eval-time adaptation. Many of these did not improve the result but informed the direction that eventually worked.

### Compression research (did not improve score)

We analyzed trained checkpoints to evaluate alternative quantization and compression approaches:

- **Learned codebook quantization** (K-means, K=256): 87% lower reconstruction MSE than uniform int6, but 25% larger compressed artifact under zstd-22. Codebook indices have higher byte entropy than clamped int6 values.
- **Symmetry-transport** (Procrustes alignment across layers): Layers share 91-93% rotational structure, but storing the rotation matrices costs more than storing the weights directly. Low-rank approximation of the rotation delta (rank-128) captured only 16.6% of variance.
- **Embedding low-rank factorization** (SVD): Rank-64 explains 41.9% of variance on tok_emb (1024×512). Not viable at this vocabulary size.
- **Magnitude pruning**: Non-monotonic interaction with zstd-22. 3% pruning increased artifact size by 728KB on our checkpoint.

These results indicated that int6+zstd is close to optimal for this model architecture and that compression was not the path to further improvement.

### Architectural exploration (did not improve score)

- **Progressive layer dropping**: Randomly skipping layers during training for regularization. Caused 0.06 BPB regression at step 1000 when combined with head dropout. The DDP implementation also introduced higher-order ops incompatible with torch.compile + DDPOptimizer.
- **Depth recurrence** (Huginn-style, 3 shared blocks × 3 loops): Blocks learned position-specific functions rather than general refiners. Eval at 2× trained depth produced val_bpb 4.34. Not viable below ~100M params per unique layer.
- **Neural cache** (cross-window KV caching at eval): Implemented but not validated on hardware. The original proposal (PR #318) was blocked by a torch.compile issue.

### TTT analysis (led to the finding)

Analyzing our trained checkpoint, we observed:

1. **Quantization error is uniformly distributed** — the top 1% of weights by error magnitude account for only 3.9% of total reconstruction error. This confirmed that outlier protection approaches would not be effective.
2. **Quantization damage varies 3.4× across layer types** — MLP output projections (512×1536) have systematically higher relative error than input projections (1536×512).
3. **TTT improvement exceeds quantization repair** — the TTT contribution (~0.06 BPB on our model) is roughly 2.4× larger than the quantization gap (~0.008), indicating TTT performs distribution adaptation beyond repairing quantization damage.

These observations motivated exploring the TTT schedule rather than the training architecture or compression scheme.

## TTT schedule

Two modifications to AdamW TTT (PR #442):

**Cosine lr decay** over 30 epochs instead of flat lr over 10 epochs. Quantization introduces both large-scale damage (outlier weight rounding) and distributed noise (small perturbations across all weights). A flat lr must compromise between these two regimes. Cosine decay applies full lr early to address large damage, then progressively reduces to refine without overshooting.

**Per-layer lr groups** based on the quantization damage measurements above. MLP output projections receive 3× base lr, input projections 0.5×, all other parameters 1×. This allocates more adaptation capacity to more damaged layers. The ratios are specific to our model — other architectures may show different damage profiles.

We tested 34 TTT configurations across optimizers (AdamW, Adam, SGD), learning rates (1e-4 to 2e-3), epoch counts (3 to 30), schedules (flat, cosine, warmup+cosine), per-layer groupings, freeze strategies, and loss functions (cross-entropy, focal loss γ=1-3, KL divergence from pre-quant model).

Focal loss did not improve over cross-entropy — hard tokens appear to be unpredictable rather than undertrained. KL divergence from the pre-quant model was less effective than cross-entropy — the pre-quant and post-quant models are similar enough that the KL signal is weak relative to the cross-entropy signal from the validation data.

## TTT config

```
TTT_OPTIMIZER=adamw  TTT_LR=0.0005  TTT_EPOCHS=30
TTT_COSINE=1  TTT_PERLAYER=1  TTT_FREEZE_BLOCKS=0
TTT_BATCH_SEQS=64 (per GPU, 512 total with DDP sharding)
```

Each GPU processes a contiguous 1/8 shard of the validation tokens with gradient all_reduce (ReduceOp.AVG). 30 epochs at ~15.5s/epoch = ~465s total.

## Training config

Standard community stack. 11L, 512d, 8H/4KV (GQA), 3x MLP (relu-squared), U-Net skips, SmearGate, BigramHash(2048), OrthoInit, Partial RoPE (16/64 dims), LN Scale, EMA(0.997), tied embeddings. XSA disabled. Int6 per-row + zstd-22.

## Notes

- All runs used FA2. FA3 Hopper would improve pre-TTT quality through faster training steps. The TTT schedule is independent of the attention kernel.
- The cosine + per-layer schedule adds no artifact cost and minimal code complexity over flat-lr TTT.
- See PR #212 for a non-record submission documenting 25+ additional experiments.

## Reproduction

```bash
git clone https://github.com/mrdavtan/parameter-golf.git
cd parameter-golf && git checkout next-gen
pip install flash-attn --no-cache-dir --no-build-isolation
pip install zstandard sentencepiece huggingface_hub
python3 data/cached_challenge_fineweb.py --variant sp1024
bash run_competition.sh 1337
```

Hardware: 8×H100 SXM (RunPod), PyTorch 2.9.1+cu128, Flash Attention 2

Builds on PRs #162, #180, #77, #398, #442, #417, #315, and modded-nanogpt.
