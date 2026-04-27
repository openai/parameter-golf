# Legal Score-First TTT (10L, 1.1532 BPB)

A 10-layer GPT with competition-legal score-first test-time training,
mixed int5/int6 quantization, and community-standard architecture components.
Achieves **1.15321 BPB** on FineWeb validation (4×A100, legal TTT).

## What's Novel Here

**The main contribution is competition-legal full-model TTT integrated into
sliding-window evaluation.** Prior legal TTT work in this competition (PR #77)
used per-document LoRA adapters with resets. This submission replaces that
with a chunked score-first loop over the **full model weights** — no LoRA,
no adapter resets between documents — giving the model persistent memory
across the entire validation set.

Concretely, `eval_val_sliding_ttt()` divides validation into 32k-token
chunks, scores each chunk first (satisfying the "already graded" rule),
then trains the full 25.5M parameters with one AdamW step per chunk.
Cosine LR decay across chunks prevents catastrophic forgetting.
Improvement: **1.1600 → 1.1532 BPB** (−0.0068).

Everything else — BigramHash, SmearGate, XSA, U-Net skips, mixed int5/int6,
SWA, Late QAT, Muon — is adopted from community prior art and cited in
Attribution below. Depth recurrence was explored during development but is
**not active** in the final config (`unique_layers=10=num_layers`).

## Run Command

```bash
# Training (~2283s on 4×A100-40GB for this submitted run)
python train_gpt.py

# Evaluation only (loads quantized checkpoint)
python train_gpt.py --inference_only
```

## Results

| Metric | Value |
|--------|-------|
| val_loss | 1.94715268 |
| val_bpb | **1.15321496** |
| Pre-TTT val_bpb | 1.1600 |
| Training steps | 5,200 |
| TTT | Legal score-first, 1 epoch/chunk |
| Wall-clock (train) | 2,283s (4×A100) |
| Wall-clock (eval+TTT) | 458s |

### Artifact Budget

| Component | Bytes |
|-----------|-------|
| Compressed model (int5/int6 + zstd-22) | 15,913,211 |
| Code (`train_gpt.py`) | 66,874 |
| **Total** | **15,980,085** |
| Budget | 16,000,000 |
| Headroom | 19,915 |

## Architecture

- **Layers**: 10 unique `BlockCore` modules (no weight sharing in final config)
- **Dimensions**: d_model=512, 8 attention heads, 4 KV heads (GQA 2:1)
- **MLP**: 3× expansion with relu² activation
- **Embeddings**: BigramHash(10240) — hashes consecutive token pairs into 10,240
  buckets, providing cheap bigram context without a full 50257² embedding table
- **Gating**: SmearGate on MLP output — applies a sigmoid gate derived from
  down-projected hidden states
- **Attention**: XSA (cross-layer shared attention) on last 3 layers — later
  layers attend using earlier layers' KV cache
- **Residual**: U-Net skip connections between layer pairs (0↔9, 1↔8, …)
- **Normalization**: RMSNorm throughout

## Training Recipe

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | Muon (matrices) + AdamW (tied token embeddings, scalars) |
| Learning rate | 0.025 (Muon matrices) / 0.035 (tied token embeddings) / 0.025 (scalars) |
| Batch size | 786,432 tokens/step at seq_len=2048 |
| Warmup | 20 steps |
| Warmdown | last 3,000 of 5,200 steps |
| Weight decay | 0.04 |
| SWA | begin averaging once LR scale drops below 0.2; every 50 steps thereafter |
| Late QAT | threshold=0.1 (begins when warmdown fraction > 0.1) |

### Quantization

Mixed-precision per-row quantization:

- **MLP weights**: int5 (5-bit), zero-point + scale per row
- **Attention weights**: int6 (6-bit), zero-point + scale per row
- Compressed with **zstd level 22**
- GPTQ-lite applied to 75% of layers (calibrated on 4 batches)

### Test-Time Training (TTT) — Competition Legal

At evaluation time, the model uses a **score-first chunked loop** that is
compliant with competition rules (you can only train on tokens already scored):

1. Divide validation tokens into chunks of 32,768 tokens (~16 sequences)
2. For each chunk: **score** all sliding windows in that chunk, then **train**
   on those already-scored tokens with one AdamW step
3. Later chunks benefit from accumulated adaptation on earlier chunks

- **Optimizer**: AdamW (lr=0.0005, wd=0.0) — per PR #442 insight
- **Epochs per chunk**: 1
- **Freeze blocks**: 0 (all blocks unfrozen)
- **Cosine LR decay** across chunks
- **Improvement**: 1.1600 → 1.1532 BPB (0.0068 improvement from legal TTT)

## Key Techniques

### Depth Recurrence (Infrastructure Only)

The code separates `BlockCore` (weights) from `Block` (norms/scales) so that
multiple logical layers can share one core. With `unique_layers < num_layers`
this gives ALBERT-style weight tying. **In this submission
`unique_layers=10=num_layers`, so no sharing occurs.** The infrastructure
remains for future exploration under tighter budgets.

### SmearGate

A lightweight gating mechanism on MLP output. A small linear projection
produces a sigmoid gate vector that element-wise scales the MLP output before
the residual connection. Adds minimal parameters but improves gradient flow.

### Stochastic Weight Averaging (SWA)

Maintains a running average of model weights, updated every 50 steps once the
learning-rate multiplier drops below 0.2 (first triggered at step 4650 in this
run). The averaged model is used for final quantization and evaluation,
providing a flatter loss basin and better quantization robustness.

## Evolution

This submission is the result of 13 experimental iterations:

| Iter | Key Change | BPB | Notes |
|------|-----------|-----|-------|
| 1 | Baseline 12L int8 | 1.187 | Starting point from upstream |
| 2 | Depth recurrence exploration | 1.18+ | V100 smoke tests |
| 3 | Sweep: layers, dim, MLP width | ~1.18 | Found 10L sweet spot |
| 4 | int5/int6 mixed quant | ~1.17 | Major compression win |
| 5 | BigramHash, SmearGate | ~1.16 | Embedding + gating wins |
| 6 | XSA, U-Net skips | ~1.155 | Attention sharing + skip |
| 7 | Late QAT, SWA | ~1.15 | Quantization-aware training |
| 8 | GPTQ-lite | ~1.148 | Post-training calibration |
| 9 | Extended training (5200 steps) | ~1.145 | Longer schedule |
| 10 | TTT (freeze early blocks) | 1.1406 | Test-time training |
| 13 | Legal score-first TTT | **1.1532** | This submission |

## Hardware

- **Training**: 4× NVIDIA A100-40GB (SLURM cluster), 2283s training + 458s eval
- **Note**: This is a non-record-track submission. The model was not trained on
  8×H100 within the 10-minute record-track constraint, but the approach and
  techniques are fully compatible with that setting.

## Credits

This submission builds on work from many contributors to the parameter-golf competition:

- **Muon optimizer** — Baseline (`modded-nanogpt`); Newton-Schulz orthogonal preconditioning
- **BigramHash embeddings** — PR #65 (aquariouseworkman): hash consecutive token pairs for cheap bigram context
- **SmearGate** — PR #65 (aquariouseworkman): per-dim sigmoid gate blending adjacent token embeddings
- **XSA (Exclusive Self Attention)** — PR #187 (Idan3011): removes self-value bias via orthogonal projection; GQA-aware variant in PR #265 (unnir)
- **U-Net skip connections** — PR #65 (aquariouseworkman), PR #69 (TevBenji): encoder-decoder layer pairing with learned skip weights
- **Mixed int5/int6 quantization** — PR #76 (unixmadtoonslab / Will DePue): int5 for MLP, int6 for attention
- **SWA (Stochastic Weight Averaging)** — PR #69 (TevBenji): checkpoint averaging during warmdown
- **Late QAT** — PR #315 (jfprincz), working implementation in PR #374 (unnir): STE fake-quantization in final training phase
- **Sliding window evaluation** — PR #50 (mattqlf / Matthew Li): stride-64 overlapping windows
- **Legal TTT framework** — PR #77 (samacqua): first legal score-first TTT (LoRA); full-model variant is our novel contribution
- **ReLU² activation, GQA** — Baseline (`modded-nanogpt`)

Built on the [parameter-golf](https://github.com/openai/parameter-golf) starter code by Beren Millidge & Keller Jordan.
