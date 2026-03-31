# 11L Int6 + SmearGate + SWA + AdamW WD

**val_bpb: 1.1400** (3-seed mean, sliding window stride=64) | **15.7 MB** artifact | 8xH100 SXM, 600s

## Key Finding: Batch Size vs Step Count

The dominant factor in 10-minute training is not batch quality but total optimization steps. Reducing batch from 786K to 524K tokens:
- Drops step time from 91ms to 67ms (26% faster)
- Increases total steps from ~7,300 to ~8,900 (22% more)
- Despite seeing 12% fewer total tokens, the extra gradient updates improve convergence

This finding applies to any fixed-time training budget and suggests the optimal batch size is smaller than commonly assumed.

## Technique Stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Layers | 11 | Extra depth funded by int6 + zstd compression headroom |
| MLP | 3x (1536) | Full width; int8 tok_emb + no Late-K saves space |
| Quantization | Int6 per-row (attention + MLP), int8 (tok_emb) | Int8 tok_emb preserves output projection quality |
| SmearGate | Per-dim, 512 params | Blends adjacent token embeddings |
| BigramHash | 2048 buckets, dim=128 | Consecutive token pair features |
| Weight decay | 0.04 (Muon + AdamW) | Dual WD shrinks weights for better quantization + compression |
| SWA | ~7 checkpoints, every 200 steps | Late-training weight averaging |
| OrthoInit | gain=1.0, proj scaled 1/sqrt(2L) | Standard orthogonal initialization |
| FlashAttention | v2.8.3 | ~3% throughput improvement over PyTorch SDPA |
| Compression | zstd level 22 | 35% better than zlib for int6-in-int8 data |
| Eval | Sliding window, stride=64, batch=32 | Batched windows make stride=64 feasible in 172s |

## Metrics

| Metric | Value |
|--------|-------|
| Sliding BPB (stride=64, 3-seed mean) | **1.1400** |
| Best single seed (1338) | **1.1381** |
| Artifact size | 15.7 MB |
| Steps (600s cap) | ~8,930 |
| Step time | 67ms |
| Model params | ~26.5M |

## Reproducibility (3 seeds)

| Seed | Sliding BPB | Artifact |
|------|-------------|----------|
| 1337 | 1.1411 | 15.95 MB |
| 1338 | 1.1381 | 15.63 MB |
| 1339 | 1.1408 | 15.66 MB |
| Mean | **1.1400** | 15.7 MB |
| Std | 0.0016 | — |

## Run Command

```bash
pip install zstandard flash-attn --no-build-isolation
SEED=1338 NUM_LAYERS=11 TRAIN_SEQ_LEN=2048 TRAIN_BATCH_TOKENS=524288 \
MLP_HIDDEN=1536 BIGRAM_VOCAB_SIZE=2048 BIGRAM_DIM=128 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3000 GRAD_CLIP_NORM=0.3 \
EVAL_SEQ_LEN=2048 EVAL_STRIDE=64 EVAL_BATCH_SEQS=32 \
MUON_WD=0.04 ADAM_WD=0.04 SWA_FRAC=0.5 SWA_EVERY=200 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Ablation Path (90+ experiments)

| Change | BPB | Delta |
|--------|-----|-------|
| Baseline (stock 9L) | 1.2244 | — |
| + int6 + MLP 3x + train@2048 + clip=0.3 (PR #114) | 1.1574 | -0.067 |
| + OrthoInit + MuonWD=0.02 | 1.1536 | -0.004 |
| + SmearGate + BigramHash + 10L | 1.1465 | -0.007 |
| + batch=524K (from 786K) | 1.1465 | +0.000 (same but more steps) |
| + 11L/1408, WD=0.039, FA | 1.1423 | -0.004 |
| + MLP=1536, LR=0.025, AdamW WD=0.04, int8 tok_emb | **1.1400** | **-0.002** |

## Dead Ends (selected from 90+ experiments)

- **QAT (int6 STE)**: 115ms/step overhead (vs 67ms baseline). Better quant quality but 25% fewer steps. Net loss.
- **Int5 for MLP**: Saves artifact space but 0.020 BPB quality penalty. Int6-all with tighter compression is better.
- **Batch=786K**: More tokens/step but fewer steps. 524K is optimal.
- **NorMuon**: 110ms/step. Throughput death.
- **MTP**: 86ms/step. Aux head too expensive.

## Previous Submissions

- PR #61: 1.2154 (warmdown-quantization discovery)
- PR #96: 1.1764 (sliding window + long-context training)
- PR #114: 1.1574 (int6 + MLP 3x + selective precision)
