# 12L Gradient-Guided Quant + Partial RoPE + LN Scale + EMA + XSA

**val_bpb: 1.1321** (sliding window, stride=64) | **15.6 MB** | 8xH100 SXM, 600s

## Key Innovation: Gradient-Guided Adaptive Quantization

Standard int6 quantization treats all weight tensors equally. But not all tensors are equally sensitive to quantization noise. We measure gradient magnitude during the last 10% of warmdown as a proxy for quantization sensitivity, then allocate precision accordingly:

- **Top 10% sensitivity**: int7 (63 levels) — weights still learning, need precision
- **Middle 70%**: int6 (31 levels) — standard
- **Bottom 20%**: int5 (15 levels) — converged weights, tolerate aggressive compression

This adaptive allocation saves ~1 MB vs uniform int6, funding a **12th transformer layer** while staying under 16 MB. The gradient sensitivity is measured at near-zero throughput cost (accumulated during steps that are already computing gradients).

### Why Late QAT Hurts at 12 Layers

We tested Late QAT (STE fake-quantization in the last 4% of training, per PR #315). At 11 layers with 600s budget, Late QAT reduces quant degradation by ~3x. But at 12 layers, the per-step overhead from fake quantization (~7ms) forces a lower wallclock cap (560s), costing ~770 training steps. The lost model quality exceeds the quant improvement. Practitioners should benchmark Late QAT against their step budget before adopting.

## Technique Stack

| Component | Choice | Origin |
|-----------|--------|--------|
| **Layers** | 12 | New — funded by gradient-guided compression |
| **MLP** | 2.75x (1408) | Swept 1280-1536 at 12L |
| **Gradient-Guided Quant** | int5/int6/int7 per-tensor | **Novel** |
| Partial RoPE | 16 of 64 dims | PR #315 |
| LN Scale | 1/sqrt(layer+1) | PR #315 |
| XSA | Last 4 layers | PR #287 |
| EMA | decay=0.997 | PR #287 |
| Weight decay | Muon + AdamW, both 0.04 | PR #198 |
| SmearGate + BigramHash(2048) | Per-dim gate, hash pairs | PR #162 |
| OrthoInit | gain=1.0, proj scaled 1/sqrt(2L) | Standard |
| FlashAttention | v2.8.3 | Standard |
| Batch size | 524K tokens | Independent discovery |
| Compression | zstd-22 | Standard |
| Eval | Sliding window, stride=64, batch=32 | Standard |

## Results

| Metric | Value |
|--------|-------|
| **Int6 sliding val_bpb (s64)** | **1.1321** |
| Pre-quant val_bpb | ~1.158 |
| Steps completed (600s cap) | 8,060 |
| Step time | 74ms |
| Model params | ~27.5M |
| Artifact size | 15,652,352 bytes |

## Reproducibility (3 seeds)

| Seed | Steps | Sliding s64 | Artifact |
|------|-------|-------------|----------|
| **1337** | **8,054** | **1.1321** | **15.65 MB** |
| 1338 | 8,060 | 1.1321 | 15.64 MB |
| 1339 | 8,070 | 1.1318 | 15.68 MB |

Mean: **1.1320** | Std: 0.0002

## Ablation

| Config | BPB | Delta |
|--------|-----|-------|
| 11L/1536 XSA+EMA (PR #287 style) | 1.1350 | baseline |
| + Partial RoPE + LN Scale + Grad Quant | 1.1347 | -0.0003 |
| + 12th layer (MLP 1408) | **1.1321** | **-0.0026** |
| + Late QAT (560s cap) | 1.1361 | +0.0040 (worse!) |

12L/1408 > 11L/1536 because extra depth outweighs narrower MLP. Gradient-guided quant enables this by compressing low-sensitivity tensors more aggressively.

## Run Command

```bash
pip install zstandard flash-attn --no-build-isolation
SEED=1337 NUM_LAYERS=12 MLP_HIDDEN=1408 BIGRAM_VOCAB_SIZE=2048 BIGRAM_DIM=128 \
TRAIN_SEQ_LEN=2048 TRAIN_BATCH_TOKENS=524288 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3000 GRAD_CLIP_NORM=0.3 \
EVAL_SEQ_LEN=2048 EVAL_STRIDE=64 EVAL_BATCH_SEQS=32 \
MUON_WD=0.04 ADAM_WD=0.04 \
SWA_ENABLED=0 EMA_ENABLED=1 EMA_DECAY=0.997 \
XSA_LAST_N=4 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 GRAD_QUANT=1 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```
