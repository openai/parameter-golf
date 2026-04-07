# Non-record: Depth Recurrence + Int7 Mixed Quant + Parallel Hedge Mixer

**val_bpb: 1.1324** (3-seed mean, std 0.0131) | **~15.40 MB** | 8×H100 SXM, 600s

Improves on [PR #1384](https://github.com/openai/parameter-golf/pull/1384) (1.1441 bpb) by **−0.012 bpb** through mixed int7/int5 quantization enabling a wider MLP 3× model, and parallelized hedge mixer eval.

## Results (8×H100 80GB SXM, PyTorch 2.5.1)

| Seed | Steps | ms/step | Roundtrip | Sliding | **Hedge** | Artifact | Eval time |
|------|-------|---------|-----------|---------|-----------|----------|-----------|
| 1337 | 4,247 | 141.3ms | 1.2168 | 1.1832 | **1.1324** | 15.40 MB | 167s |
| 42 | 4,389 | 136.7ms | 1.2172 | 1.1840 | **1.1454** | 15.28 MB | 164s |
| 7 | 4,391 | 136.7ms | 1.2163 | 1.1828 | **1.1193** | 15.29 MB | 163s |
| **Mean** | **4,342** | **138.2ms** | **1.2168** | **1.1834** | **1.1324** | | **~164s** |

Additional seeds for variance analysis: seed 2024 → 1.1431, seed 99 → 1.1405. 5-seed mean: **1.1361** (std 0.0095).

## Changes vs PR #1384 (1.1441 bpb)

| Change | Effect | Impact |
|--------|--------|--------|
| MLP 2× → 3× (d=832→880) | +38% parameters, wider model | −0.013 sliding bpb |
| Int8 → **Int7 attn** + Int5 MLP | Fits larger model in 16MB budget | enables above |
| Earlier progressive depth (30/50 vs 40/65) | +55% full-depth training steps | −0.004 bpb |
| More SWA (every 30, start 0.6) | 43 checkpoints vs 13 | smoother average |
| Parallel hedge eval (8 GPU) | 580s → 164s eval time | fits 10 min budget |

## Key Finding: Int7 Attention is the Sweet Spot

Standard approaches use uniform quantization (all int8 or all int6). Experiments show that **attention and MLP weights have very different sensitivity to quantization**:

- **Attention weights** directly affect the neural expert in hedge mixer. Int6 (31 levels) causes hedge boost to drop from −0.052 to −0.039 — a significant quality loss.
- **MLP weights** tolerate aggressive quantization. Int5 (16 levels) compresses well with minimal quality impact.
- **Int7 (63 levels)** for attention recovers hedge boost to −0.051, nearly matching int8's −0.052.

The 2MB saved by using int5 MLP instead of int8 is reinvested into a wider model (d=880 with MLP 3× vs d=832 with MLP 2×).

| Quant config | Model | Sliding | Hedge | Hedge boost | Size | Fits? |
|-------------|-------|---------|-------|-------------|------|-------|
| Int8 attn + Int5 MLP | d=896 | 1.1760 | 1.1349 | −0.041 | 17.4 MB | ✗ |
| **Int7 attn + Int5 MLP** | **d=880** | **1.1832** | **1.1324** | **−0.051** | **15.4 MB** | **✓** |
| Int6 attn + Int5 MLP | d=896 | 1.1870 | 1.1480 | −0.039 | 15.4 MB | ✓ |

## Architecture: Depth Recurrence

Instead of 9–11 unique transformer blocks, **3 shared blocks are repeated 4 times** (12 effective layers). This trades unique parameters for effective depth, fitting 23.7M parameters into ~15.4 MB.

| Parameter | Value |
|-----------|-------|
| Layers × Repeats | 3 × 4 (12 effective) |
| Model dim | 880 |
| Heads / KV heads | 8 / 4 (head_dim=110) |
| MLP multiplier | 3× (hidden=2640) |
| Vocab size | 1024 (SP BPE) |
| Parameters | 23.7M |
| Logit softcap | 30.0 |

### Recurrence components

- **Cross-Repeat Skip**: Each block receives a weighted residual from its own output in the previous repeat — turns stateless recurrence into stateful
- **Loop Embedding**: Learned per-layer vector added before each block — depth-wise positional encoding for shared weights
- **Value Embeddings**: 2 extra embedding tables mixed into the residual stream at each effective layer with learned scales
- **XSA (Exclusive Self-Attention)**: On last 4 effective layers — prevents attention collapse in deep recurrent models
- **LeakyReLU(0.5)²**: Better gradient flow than ReLU² for deep/recurrent models

## Progressive Depth Training

Training uses increasing recurrence depth, recompiling at phase boundaries:

| Phase | Wallclock | Repeats | Effective layers | Step speed |
|-------|-----------|---------|-----------------|------------|
| 0–30% | 0–180s | 2 | 6 | ~90ms |
| 30–50% | 180–300s | 3 | 9 | ~105ms |
| 50–100% | 300–600s | 4 | 12 | ~130ms |

Schedule tuned for the MLP 3× config: earlier transitions (30/50% vs 40/65% in PR #1384) give +55% more steps at full depth. Warmdown 3000 iterations, SWA every 30 steps from LR scale < 0.6 (~43 checkpoints).

## Eval: Parallel Hedge Mixer

5-expert online ensemble with **8-GPU parallelized forward pass**:

| Expert | Description |
|--------|-------------|
| Neural | Model's own logits (log-softmax) |
| Unigram | Global token frequency with Laplace smoothing |
| Bigram | Conditional P(token \| prev_token) |
| Trigram | Hashed trigram context (65K buckets) |
| Entropy | Model's entropy as calibration signal |

**Parallelization**: Each batch of windows is split across 8 GPUs for the forward pass, logits gathered via `all_gather` to rank 0 for sequential mixer scoring. This reduces hedge eval from 580s (single GPU) to **164s**, fitting within the 10-minute eval budget.

Hedge provides **−0.051 bpb improvement** over sliding window (1.1834 → 1.1324 mean).

## Training Details

| Parameter | Value |
|-----------|-------|
| Optimizer | Muon (matrices) + Adam (scalars, embeddings) |
| Matrix / Scalar LR | 0.018 / 0.018 |
| Tied embed LR | 0.021 |
| Muon WD | 0.04 |
| Muon momentum | 0.95 (warmup 0.85→0.95 over 500 steps) |
| Grad clip | 0.3 |
| Batch tokens | 524,288 |
| Quantization | Int7 attn (63 levels) + Int5 MLP (16 levels) + zstd-22 |

## Evolution

| PR | Date | Score | What changed |
|----|------|-------|-------------|
| [#148](https://github.com/openai/parameter-golf/pull/148) | Mar 20 | 1.2196 (sliding) | Depth recurrence (3×4), cross-repeat skip, value embeddings |
| [#784](https://github.com/openai/parameter-golf/pull/784) | Mar 25 | 1.2065 (sliding) | + XSA(4), LeakyReLU², GPTQ-lite, zstd-22 |
| [#835](https://github.com/openai/parameter-golf/pull/835) | Mar 26 | 1.1980 (sliding) | + Progressive depth training (+30% steps) |
| [#1384](https://github.com/openai/parameter-golf/pull/1384) | Apr 5 | 1.1441 (hedge) | + Hedge Mixer (5-expert eval-time ensemble) |
| **This PR** | Apr 8 | **1.1324** (hedge) | + Int7 mixed quant, MLP 3×, d=880, parallel hedge |

## Lineage

- Depth recurrence architecture — original to this submission line
- XSA from [PR #198](https://github.com/openai/parameter-golf/pull/198) (unnir)
- LeakyReLU² from [PR #493](https://github.com/openai/parameter-golf/pull/493) (parinzee)
- Mixed int5/int6 quantization concept from [PR #549](https://github.com/openai/parameter-golf/pull/549) (thwu1), extended here to int7
- SWA, Muon WD from modded-nanogpt community

## Reproducing

```bash
SEED=1337 QUANT_LEVELS=63 MLP_QUANT_LEVELS=16 \
MODEL_DIM=880 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=3 \
NUM_LAYERS=3 NUM_REPEATS=4 XSA_LAST_N=4 NUM_VALUE_EMBEDS=2 \
PROG_DEPTH="0.30:2,0.50:3,1.0:4" \
WARMDOWN_ITERS=3000 SWA_START_FRAC=0.6 SWA_EVERY=30 \
VOCAB_SIZE=1024 TRAIN_SEQ_LEN=1024 TRAIN_BATCH_TOKENS=524288 \
torchrun --nproc_per_node=8 train_gpt.py
```
