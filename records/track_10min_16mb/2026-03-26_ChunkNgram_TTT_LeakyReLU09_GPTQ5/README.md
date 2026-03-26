# Chunk-Based N-gram Backoff + Score-First TTT + LeakyReLU(0.9)^2 + GPTQ-Int5

**val_bpb: 0.29519** (mean of 3 seeds, std 0.00013)

Eval-time N-gram cache interpolation is the primary driver. An order-9 backoff N-gram model, built incrementally from already-scored validation tokens, is blended with the neural model's predictions using entropy-adaptive mixing weights. Processing 1M-token chunks with synchronized caches across all 8 GPUs ensures full cache utilization.

## Run Command

```bash
MODEL_PRESET=frontier_lean RUN_PROFILE=full_8gpu_600s_ttt \
SEED=1337 QAT_MODE=off ENABLE_COMPILE=1 \
LEAKY_RELU_SLOPE=0.9 GPTQ_CALIB_BATCHES=64 \
TTT_CHUNK_SIZE=2048 MAX_WALLCLOCK_SECONDS=525 \
torchrun --standalone --nproc_per_node=8 -m research.train
```

For the standalone `train_gpt.py` (as submitted):
```bash
SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## 3-Seed Results

| Seed | Steps | Train Time | Pre-Quant BPB | TTT BPB | N-gram BPB |
|------|-------|-----------|---------------|---------|------------|
| 1337 | 6,084 | 525s | 1.1408 | 1.1490 | **0.2953** |
| 42 | 6,094 | 525s | 1.1483 | 1.1483 | **0.2950** |
| 2024 | 6,096 | 525s | 1.1490 | 1.1490 | **0.2952** |
| **Mean** | **6,091** | **525s** | **1.1460** | **1.1488** | **0.2952** |
| **Std** | **6** | | **0.0046** | **0.0004** | **0.0001** |

Artifact size: 13.4MB (code: 181KB + model: 13.2MB). Well under 16MB.

## Key Techniques

### 1. Chunk-Based N-gram Eval Cache (primary contribution)

- **Order-9 backoff** N-gram model (orders 2 through 9) with vectorized XOR-of-products hashing
- **1M-token sequential chunks**: score all segments in a chunk, then update cache. All GPU ranks update cache with the full chunk data, so caches stay perfectly synchronized across GPUs
- **Score-first**: cache only contains tokens from previously-scored chunks
- **Entropy-adaptive alpha**: `alpha = alpha_min + (alpha_max - alpha_min) * sigmoid(scale * (H - center))`, where `center` varies by N-gram order (higher orders activate at lower entropy)
- **Per-order multipliers**: orders 2-3 suppressed (0.3x), orders 5-9 boosted (2.0x), all clipped to [0, 0.95]
- **4M buckets** per order, int32 counts, `np.bincount` for fast updates
- N-gram eval: ~287s on 8xH100

### 2. Score-First TTT (Test-Time Training)

- LoRA rank 8 on Q+V+LM head projections
- AdamW optimizer, LR=0.01, cosine schedule across chunks
- 2048-token chunks, 3 epochs per chunk, Polyak averaging (decay=0.998)
- Grouped LR: head 1.5x, Q 1.0x, V 1.0x
- Temperature 0.98, strict score-first enforcement
- Contributes ~0.015 BPB improvement over base model

### 3. Architecture

- 11 layers, 512 dim, 8 query heads, 4 KV heads (GQA)
- MLP 3.0x (1536 hidden) with LeakyReLU(0.9)^2 activation
- BigramHash(4096), SmearGate, OrthoInit
- XSA (exclusive self-attention) on last 4 layers
- Partial RoPE (16/64 dims), LN Scale (1/sqrt(layer+1))
- Value Embeddings on layers 9, 10 (dim=128)
- U-Net skip connections, logit softcap 30.0, tied embeddings
- 27,255,900 parameters

### 4. Training

- Muon optimizer (momentum 0.99, WD 0.04, NS5, banking)
- AdamW for embeddings/scalars (lr 0.035/0.025)
- EMA 0.997 with step-aware warmup
- Warmdown 3500 iters (wallclock-proportional)
- Shard ordering by perplexity (easy-to-hard curriculum)
- torch.compile(fullgraph=True)
- 525s wallclock on 8xH100 SXM (~6,091 steps at 86ms/step)

### 5. Export

- Full Hessian GPTQ with INT5 quantization (64 calibration batches, 1.0s)
- LZMA compression, export grid search (4 configs in parallel)
- GPTQ calibration fits within training budget (525s + 75s post-train < 600s)

## Ablation

| Configuration | BPB |
|---|---|
| Base model (post-export roundtrip) | 1.1600 |
| + TTT | 1.1449 |
| + N-gram (without TTT) | **0.2952** |

The N-gram cache is the dominant technique, reducing BPB by 0.87 from the base model.

## Timing Budget

| Phase | Time | Budget |
|---|---|---|
| Training (gradient steps) | 525s | 600s training |
| GPTQ calibration | 1s | 600s training |
| Quantize + serialize | 66s | 600s training |
| **Training phase total** | **592s** | **600s** |
| Roundtrip eval | 84s | 600s eval |
| TTT eval | 53s | 600s eval |
| N-gram eval | 287s | 600s eval |
| **Eval phase total** | **424s** | **600s** |

## Files

- `train_gpt.py`: single-file submission script (collapsed from modular `research/` surface)
- `submission.json`: leaderboard metadata
- `train_seed1337.log`, `train_seed42.log`, `train_seed2024.log`: training logs for all 3 seeds
- `train.log`: primary log (seed 1337) for validator compatibility
- `requirements.txt`: package list
