## Record: 12L Gradient-Guided Quant + Partial RoPE + LN Scale + EMA + XSA4 (val_bpb: 1.1320)

**val_bpb: 1.1320** (sliding window, stride=64) | **15.7 MB** | 8xH100 SXM, 600s

### Progress from prior submissions

| | [PR #61](https://github.com/openai/parameter-golf/pull/61) | [PR #96](https://github.com/openai/parameter-golf/pull/96) | [PR #114](https://github.com/openai/parameter-golf/pull/114) | This | Delta vs #114 |
|---|---|---|---|---|---|
| **val_bpb (sliding)** | 1.2154 | 1.1764 (s512) | 1.1574 (s256) | **1.1320 (s64)** | **-0.0254** |
| Layers | 9 | 9 | 9 | **12** | +3 |
| Params | 17M | 17M | 21.8M | **27.6M** | +5.8M |
| Artifact | 15.4 MB | 15.9 MB | 15.98 MB | 15.7 MB | -0.3 MB |

### What's new

1. **Gradient-Guided Adaptive Quantization.** Standard int6 quantization treats all weight tensors equally, but not all tensors are equally sensitive to quantization noise. We accumulate per-tensor squared gradient magnitudes during the last 10% of warmdown (zero throughput cost — gradients are already computed), then rank tensors by sensitivity at quantization time:

   - **Top 10% sensitivity**: int7 (63 levels) — weights still learning, need precision
   - **Middle 70%**: int6 (31 levels) — standard
   - **Bottom 20%**: int5 (15 levels) — converged weights, tolerate aggressive compression

   This adaptive allocation saves ~1 MB vs uniform int6, funding a **12th transformer layer** while staying under 16 MB.

2. **12 layers (up from 9).** Extra depth funded by gradient-guided compression headroom. MLP narrowed to 1408 (from 1536 at 11L) — extra depth outweighs narrower width at this scale.

3. **Batch=524K.** Reducing batch size from 786K to 524K gives 22% more optimization steps (8,060 vs ~7,000) at lower per-step cost (74ms vs ~84ms). More gradient updates outweigh larger batch quality in a fixed-time budget.

4. **Partial RoPE (16 of 64 dims).** Rotary embeddings applied to only 25% of head dimensions. Remaining dims use position-free attention, improving generalization. Zero new parameters.

5. **LN Scale.** RMSNorm outputs scaled by 1/sqrt(layer_idx+1). Damps deeper layers' contributions, stabilizing training at 12 layers. Zero new parameters.

6. **XSA (Exclusive Self Attention) on last 4 layers.** Removes self-value bias from attention output via orthogonal projection. Forces attention to carry cross-token information only. Zero new parameters.

7. **EMA (decay=0.997) replacing SWA.** Exponential moving average every step instead of periodic checkpoint averaging. Smoother weight distribution, better generalization and compression.

### Negative finding: Late QAT at 12 layers

We tested Late QAT (STE int6 fake-quantization in the last 4% of training). At 12 layers the per-step overhead (~7ms) forces a lower wallclock cap, costing ~770 training steps. The lost model quality exceeds the quantization improvement: **1.1361 (with Late QAT) vs 1.1321 (without)**. Late QAT's value depends on the step budget — at high layer counts where step time is already elevated, the throughput cost dominates.

### Results

| Metric | Value |
|--------|-------|
| Pre-quant val_bpb | 1.1505 |
| Int6 roundtrip val_bpb | 1.1553 |
| **Int6 sliding val_bpb (s64)** | **1.1321** |
| Steps completed (600s cap) | 8,054 |
| Step time | 74ms |
| Model params | 27,618,913 |
| Artifact size | 15,652,352 bytes |

### Reproducibility (3 seeds)

| Seed | Steps | Sliding s64 | Artifact |
|------|-------|-------------|----------|
| **1337** | **8,054** | **1.1321** | **15,652,352** |
| 1338 | 8,060 | 1.1321 | 15,641,722 |
| 1339 | 8,070 | 1.1318 | 15,675,008 |

Mean: **1.1320** | Std: 0.0002 | Submitted: seed 1337

### Run command

```bash
pip install zstandard flash-attn --no-build-isolation
NUM_LAYERS=12 MLP_HIDDEN=1408 BIGRAM_VOCAB_SIZE=2048 BIGRAM_DIM=128 \
TRAIN_SEQ_LEN=2048 TRAIN_BATCH_TOKENS=524288 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3000 GRAD_CLIP_NORM=0.3 \
EVAL_SEQ_LEN=2048 EVAL_STRIDE=64 EVAL_BATCH_SEQS=32 \
MUON_WD=0.04 ADAM_WD=0.04 \
SWA_ENABLED=0 EMA_ENABLED=1 EMA_DECAY=0.997 \
XSA_LAST_N=4 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 GRAD_QUANT=1 \
MAX_WALLCLOCK_SECONDS=600 ITERATIONS=9000 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```
