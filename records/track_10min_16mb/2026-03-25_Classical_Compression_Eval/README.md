# Classical Compression Eval-Time Augmentation

**val_bpb: 0.8128** (1-seed, seed=1337) | **15.88 MB** | 8xH100 SXM

## Results (8xH100 80GB SXM, PyTorch 2.9.1+cu128)

| Metric | Standard Sliding Window | Compressed Eval |
|--------|------------------------|-----------------|
| val_loss | 1.8941 | 1.3725 |
| val_bpb | 1.1218 | **0.8128** |
| eval_time | 97s | 383s |

| Training | Value |
|----------|-------|
| Steps | 7,135 / 9,000 (wallclock capped at 600s) |
| Step avg | 84.1ms |
| Peak memory | 21,481 MiB |
| Artifact size | 15,876,059 bytes (code: 100,231 + model int6+lzma: 15,775,828) |

## Approach

Novel eval-time augmentation that brings classical data compression techniques into the neural model evaluation pipeline. All techniques are backward-looking only (legal per competition rules), zero artifact cost.

### Base Model (PR #549 stack)

| Component | Setting |
|-----------|---------|
| Layers | 11 (512d, 8H, 4KV) |
| MLP | 3x with LeakyReLU(0.5)^2 |
| BigramHash | 1536 |
| XSA | Last 4 layers |
| RoPE | Partial (16/64 dims) |
| LN Scale | 1/sqrt(layer+1) |
| VE128 | Layers 9-10 |
| Weight avg | EMA(0.997) + Tight SWA(every 50) |
| Quantization | GPTQ-lite int6 + lzma |
| Optimizer | Parallel Muon(0.99) + Adam |

### Eval-Time Compression (Our Novel Contribution)

**Multi-Order N-gram Backoff (orders 2-7) with Entropy-Adaptive Alpha Mixing**

Inspired by classical data compression (cmix, PAQ8). During sliding-window evaluation:

1. Maintain flat `np.uint32` hash tables (4M buckets per order, 12 tables total)
2. For each scored token position, look up the highest-order n-gram match (7-gram first, backoff to 2-gram)
3. Compute neural model entropy: `H = -sum(p * log(p))`
4. Mix neural and n-gram predictions with entropy-adaptive alpha: `alpha = 0.05 + 0.55 * sigmoid(2 * (H - 4.0))`
5. Update all n-gram tables with the scored token (backward-looking only)

The key insight: FineWeb is web text with enormous local repetition (templates, navigation, common phrases). The tiny neural model cannot memorize these patterns, but n-gram statistics capture them perfectly at zero artifact cost.

**Implementation:** Vectorized numpy processing per-segment (not per-token Python loops). Uses `np.add.at` for scatter-add updates. Hash function: XOR with prime table, same approach as PR #727.

### What's Different From PR #727

Our implementation uses the same core n-gram technique but with:
- Full vectorized segment processing (not per-token)
- Separate context and full (context+target) hash tables per order
- Entropy computed on GPU, transferred in batch

### Planned Additions (not yet integrated)

These techniques from classical compression (cmix/PAQ) are implemented but not yet in the eval pipeline due to speed constraints:
- **Match Model**: Long-range exact substring matching (captures repeated paragraphs)
- **APM/SSE**: Adaptive Probability Maps for error correction
- **Logistic-Domain Mixing**: PAQ-style mixing in log-odds space

## Run Command

```bash
RUN_ID=full_run_seed1337 \
COMPRESS_EVAL=1 COMPRESS_MATCH_ENABLED=0 COMPRESS_APM_ENABLED=0 \
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 our_train_gpt.py
```

## Credits

- **Base model**: PR #549 by @abaybektursun (LeakyReLU^2 + Legal TTT + Parallel Muon)
- **N-gram eval technique**: Inspired by PR #727 by @Asukabot0 (first legal sub-1.0 BPB)
- **Classical compression research**: cmix by Byron Knoll, PAQ by Matt Mahoney
- **Built with**: Claude Code (AI agent)
