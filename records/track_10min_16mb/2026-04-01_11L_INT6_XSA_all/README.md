# 11L INT6 XSA-all + EMA + Value Embeddings

**val_bpb: 1.1487** (quality record, unsubmittable) | **19.03 MB** (over 16MB limit) | 8×H100 SXM

## Results

| Seed | step_avg | steps | Post-quant bpb | Post-TTT bpb | Artifact |
|------|----------|-------|----------------|--------------|----------|
| 1    | 133ms (pre-QAT) / 163ms (post-QAT) | 3758 | 1.14971 | **1.14870** | 19,030,284 |

## Architecture

| Component | Setting |
|-----------|---------|
| Layers | 11 (512d, 8H, 4KV) |
| MLP | 3× with LeakyReLU(0.5)² |
| BigramHash | 3072 buckets × 112 dim |
| XSA | All 11 layers |
| RoPE | Partial (16/64 dims) |
| LN Scale | 1/√(layer+1) |
| Weight avg | EMA(0.997) with QAT-activation reset |
| Quantization | INT6 MLP + INT6 attn + INT6 bigram + LZMA preset=9 |
| QAT trigger | Wallclock fraction (65% of budget) |
| Value Embeddings | ve_dim=128, last 2 layers |
| TTT | Legal score-first, lr=0.002, 3 epochs |

## Why Unsubmittable

INT6 for all weight categories (MLP+attn+bigram) stores values in the range −32..+31 (64 distinct values), vs INT4's 16 distinct values. LZMA compression is proportionally weaker: ~5.5× ratio vs ~6.5× for INT4. Result: 19MB vs the 16MB budget.

Fix: GPTQ (Run 2) forces quantized weights toward smaller values → more clustering → better LZMA compression. Expected to bring artifact below 16MB.

## Comparison

| Run | ms/step | Steps | ttt_bpb | Size |
|-----|---------|-------|---------|------|
| v10_banked seed 1 | 138ms | 4358 | 1.15711 | 16.47MB |
| **v11_int6_xsaall seed 1** | **133ms** | **3758** | **1.14870** | **19.03MB ❌** |

## QAT Overhead Note

QAT enabled at step ~2900 (65% of wallclock). After enabling INT6 QAT on all 3 categories simultaneously, step time jumped from 133ms → 163ms (MFU 22% → 18%). This cost ~900 steps vs expectation. Future runs: QAT on attn+bigram only (MLP clip is less expensive) or accept the overhead.

## Training File

`train_gpt_v2.py`
