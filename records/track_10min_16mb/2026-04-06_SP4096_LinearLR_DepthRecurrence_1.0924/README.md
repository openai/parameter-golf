# Record: SP4096 + Linear LR Decay + Depth Recurrence

**val_bpb: 1.0924** (3-seed mean, std 0.0004) | **15.99 MB** | 8xH100 SXM, 600s | No TTT, no SLOT, no n-gram, no eval-time adaptation

**Improvement over current SOTA ([PR #1019](https://github.com/openai/parameter-golf/pull/1019), 1.1147 BPB):** -0.0223 BPB (Welch t=-68.85, df=3.84, p << 0.001)

## Results

| Seed | Steps | ms/step | Pre-quant BPB | **Sliding BPB** | Artifact |
|------|-------|---------|---------------|-----------------|----------|
| 42 | 5,410 | 109.1 | 1.0974 | **1.0927** | 15,987,206 |
| 314 | 5,409 | 109.1 | 1.0977 | **1.0927** | 15,989,311 |
| 999 | 5,408 | 109.1 | 1.0970 | **1.0919** | 15,988,159 |
| **Mean** | | | **1.0974** | **1.0924** | |

Current SOTA (PR #1019, exact 3-seed mean): **1.11473509 BPB**. This run: **1.09244346 BPB**. Delta: **-0.02229163 BPB**.

All four conditions from [Issue #1017](https://github.com/openai/parameter-golf/issues/1017) satisfied: no test-time training, no SLOT, no n-gram cache, no eval-time adaptation of any kind. Evaluation is pure sliding-window at stride=64.

---

## Key Finding: Linear LR Decay Reduces Quantization Gap by 61%

This submission uses an SP4096 transformer with depth recurrence, building on the architecture family from [PR #1296](https://github.com/openai/parameter-golf/pull/1296) and [PR #1285](https://github.com/openai/parameter-golf/pull/1285). The single critical change relative to our prior v4 stack is the learning rate schedule.

### The LR Schedule Change

Before (v4): cosine warmdown with floor at 5% of peak LR.
```python
return lr_floor + (1 - lr_floor) * 0.5 * (1 + cos(pi * progress))  # lr_floor = 0.05
```

After (this submission): linear warmdown to zero.
```python
return max((1.0 - frac) / warmdown_frac, 0.0)
```

This means the model's weights are fully settled before GPTQ quantization runs. The cosine floor at 5% left the optimizer still making non-trivial updates at quantization time, producing weights with wider magnitude distributions that quantize and compress worse.

### Measured Impact

| Metric | v4 (cosine, floor=0.05) | This (linear, floor=0.0) |
|--------|------------------------|--------------------------|
| Quantization gap (roundtrip) | 0.038 BPB | **0.014 BPB (-61%)** |
| Values pruned to fit 16MB | 1,860,936 | **340,142 (-82%)** |
| Unpruned artifact size | 16.23 MB | **16.09 MB** |

The quantization gap collapsed because: (1) weights converge to tighter distributions that need less GPTQ error compensation, (2) tighter weights compress better under Brotli, requiring less selective pruning, and (3) less pruning means fewer GPTQ compensation terms are destroyed.

---

## Architecture

| Component | Setting | Source |
|-----------|---------|--------|
| Layers | 11 (512d, 8 GQA heads, 4 KV heads) | Baseline |
| MLP | 4x (2048) with LeakyReLU(0.5)^2 | [#493](https://github.com/openai/parameter-golf/pull/493), [#1218](https://github.com/openai/parameter-golf/pull/1218) |
| Tokenizer | SentencePiece BPE 4096 | [#1218](https://github.com/openai/parameter-golf/pull/1218) |
| Depth Recurrence | Layers 4,5 repeated from step 3000 | [#1204](https://github.com/openai/parameter-golf/pull/1204) |
| Parallel Residuals | From layer 7 | [#1204](https://github.com/openai/parameter-golf/pull/1204) |
| Attention | XSA on all 11 layers | [#478](https://github.com/openai/parameter-golf/pull/478) |
| QK-Gain | 5.0 | [#1217](https://github.com/openai/parameter-golf/pull/1217) |
| RoPE | Partial (16/64 dims) | [#315](https://github.com/openai/parameter-golf/pull/315) |
| LN Scale | 1/sqrt(layer+1) | [#315](https://github.com/openai/parameter-golf/pull/315) |
| U-Net Skips | Gated encoder-decoder connections | [#289](https://github.com/openai/parameter-golf/pull/289) |
| MuonEq-R | Row-normalize before NS5 | [arXiv:2603.28254](https://arxiv.org/abs/2603.28254) |

## Training

| Parameter | Value |
|-----------|-------|
| Optimizer | Muon (matrix), AdamW (scalars/embeddings) |
| Matrix LR | 0.02 |
| Muon WD | 0.09 |
| Adam WD | 0.02 |
| Momentum | 0.99 (warmup from 0.92 over 1500 steps) |
| Warmdown | Fraction 0.667, **linear decay to LR=0.0** |
| EMA | Decay 0.997 |
| Batch | 786,432 tokens/step |
| Sequence length | 2048 |
| Grad clip | 0.3 |
| FlashAttention 3 | Hopper kernels |

## Quantization and Compression

| Component | Setting |
|-----------|---------|
| GPTQ | Full Hessian, 64 calibration batches from training data |
| Bit width | Int6 per-row for all attention + MLP matrices |
| Embeddings | Int8 per-row |
| Compression | Brotli quality=10 (empirically verified optimal for this weight distribution) |
| Pruning | Selective +/-1 values, factor=4x excess (minimal: 340K values mean, down from 1.86M) |

## Run Command

```bash
DATA_DIR=/workspace/pg_repo/data \
VOCAB_SIZE=4096 \
SEED=42 \
MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

All parameters are set as defaults in the script. No environment variable overrides needed beyond DATA_DIR, VOCAB_SIZE, and SEED.

## Data

This run uses 26 SP4096 training shards (of a possible 80) due to disk constraints during data preparation. With full training data, the pre-quantization BPB is expected to improve by approximately 0.003, projecting the final sliding-window BPB to approximately 1.089-1.092.

The validation shard is the standard full 50,000-document FineWeb validation set. BPB is computed as val_loss / log(2) * (token_count / byte_count), the standard tokenizer-agnostic formula.

## Lineage

```
Baseline (1.2244)
  + SP4096, MLP 4x (#1218)
  + Depth Recurrence L4,5, Parallel Residuals L7 (#1204)
  + MuonEq-R (arXiv:2603.28254)
  + QK-Gain 5.0 (#1217)
  + XSA all layers (#478)
  + Full Hessian GPTQ int6
  + Linear LR decay to zero (this work)
= 1.0924 BPB
```
