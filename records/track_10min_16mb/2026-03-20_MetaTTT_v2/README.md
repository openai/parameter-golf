# Meta-Learned Test-Time Training: Empirical Analysis at 16MB Scale

**Non-record research submission** | val_bpb: 1.1645 (sliding window, stride=64) | Artifact: 12.7MB

## Summary

This submission investigates two strategies for improving compressed language models beyond standard training: **Reptile meta-learning** for test-time training (TTT) initialization, and **error-guided adaptation** that concentrates TTT budget on the model's weakest predictions. Both are tested on the current SOTA stack (11L, int6, 3x MLP, SmearGate, BigramHash). The main findings are:

1. **Reptile meta-learning improves SmearGate models by 0.011 BPB** — 10x better than naive TTT (+0.001), suggesting meta-learned initialization partially overcomes the SmearGate/TTT redundancy
2. **Error-guided TTT is a negative result** — concentrating adaptation on highest-loss tokens does not improve val_loss, indicating these tokens are genuinely unpredictable rather than under-adapted
3. **13 layers beat 10 layers on 8xH100** — despite 23% fewer training steps, the deeper model achieves 1.1884 vs 1.2090
4. **Detailed per-token loss distribution** on the full 62M-token validation set reveals that 4% of tokens (loss > 7.0) account for ~15% of total loss

## Motivation

Issue #140 reports that TTT adds only ~0.001 BPB on SmearGate models, compared to ~0.033 on non-SmearGate models. The standard explanation is that SmearGate and TTT both inject local bigram context, making them redundant.

We ask: **can meta-learned initialization or targeted adaptation break through this redundancy?**

## Method

### Reptile Meta-Learning (Positive Result)

After standard training (80% of wallclock), we switch to Reptile meta-learning (20% of wallclock) on the MLP layers of the last 3 transformer blocks.

Each Reptile outer step:
1. Save current MLP weights
2. Sample a training document chunk
3. Run 3 inner SGD steps (lr=0.1) on the chunk
4. Move base weights toward adapted weights: `base += 0.01 * (adapted - base)`

**Result**: 1576 Reptile meta-steps improved sliding window BPB from 1.1768 to 1.1661 (-0.0107).

**Caveat**: This improvement could partly come from additional training rather than meta-learning per se. Disentangling the two requires comparing Reptile against an equal number of standard training steps, which we leave for future work with more compute.

### Error-Guided TTT (Negative Result)

Hypothesis: Standard TTT spreads adaptation budget uniformly across all tokens. Since SmearGate already handles easy tokens well, the budget is wasted. Concentrating adaptation on the top 2% highest-loss windows should be much more effective.

Implementation:
1. **Pass 1** (20s distributed): Non-overlapping inference to identify highest-loss windows
2. **TTT phase** (8s): Rank-4 LoRA adapters on last 3 blocks' MLPs, trained for 3 epochs on only the top 2% highest-loss windows
3. **Pass 2** (227s): Standard sliding window eval with LoRA-improved model

**Result**: TTT training loss decreased (3.00 → 2.98) but global val_loss was unchanged. The LoRA adaptation on 151 high-loss windows did not transfer to improve predictions globally.

**Interpretation**: The highest-loss tokens (rare words, complex syntax, genuinely unpredictable content) cannot be fixed by local gradient updates. Their high loss reflects fundamental uncertainty in the data, not a model deficiency that adaptation can address.

### Depth Frontier (Positive Result)

We tested whether more layers help under the 8xH100/10min budget:

| Layers | Steps | val_bpb | Artifact |
|--------|-------|---------|----------|
| 10 (baseline) | 12,157 | 1.2090 | 15.4MB |
| 11 (+ SmearGate) | 7,630 | 1.1645 | 12.7MB |
| **13 (int8)** | **9,385** | **1.1884** | 19.8MB |
| 13 (int6) | 9,200 | 1.1973 | 15.1MB |

13 layers at int6 quantization improves by 0.012 over baseline while fitting in 16MB, despite 24% fewer training steps.

**Note**: On 1xH100/5min, more layers consistently hurt (10L: 1.4056, 11L: 1.4137, 13L: 1.4498) due to step-limited training. The depth frontier only works with sufficient training compute.

### Per-Token Loss Distribution

Analysis of the full 62M validation tokens on a SOTA-level model (non-overlapping eval BPB ~1.10):

| Loss threshold | Token count | % of tokens | % of total loss |
|---------------|------------|-------------|-----------------|
| > 1.0 | 36.9M | 59.6% | ~85% |
| > 3.0 | 20.3M | 32.7% | ~55% |
| > 5.0 | 7.2M | 11.6% | ~30% |
| > 7.0 | 1.7M | 2.7% | ~15% |
| > 10.0 | 135K | 0.2% | ~3% |

The loss distribution is heavy-tailed: the hardest 2.7% of tokens (loss > 7.0) contribute ~15% of total loss. This motivated our error-guided TTT approach, which ultimately showed that these high-loss tokens cannot be improved through local adaptation.

## Architecture

Built on PR #198's recipe:
- 11 transformer layers, 512 dim, 8 heads, 4 KV heads (GQA)
- 3x MLP expansion (hidden=1536), relu^2 activation
- SmearGate (~512 params) + BigramHash (2048-bucket, dim=128)
- Int6 per-row quantization + zstd-22 compression
- FP16 tied embeddings
- SWA (3 checkpoints during warmdown)
- Muon optimizer (momentum=0.99, WD=0.04) + AdamW for embeddings
- Sliding window eval (stride=64, eval_seq_len=2048)

## Reproduction

```bash
# Stage 1: Train base model (10 min, 8xH100)
torchrun --nproc_per_node=8 train_gpt.py

# Stage 2: Error-guided TTT eval (5 min, separate process)
torchrun --nproc_per_node=8 eval_error_guided_ttt.py
```

Key environment variables:
- `REPTILE_ENABLED=1`, `REPTILE_TIME_FRAC=0.2` — meta-learning phase
- `TTT_ENABLED=1`, `TTT_LR=0.001`, `TTT_RANK=4` — LoRA TTT
- `TTT_TOP_FRAC=0.02` — fraction of windows for error-guided TTT

## Theoretical Context

This work is motivated by the observation that language model compression and test-time adaptation sit at opposite ends of the same spectrum:

- **Compression** (training) encodes general knowledge about language into fixed weights
- **Adaptation** (eval) specializes the model to each specific document

TTT-E2E (Sun et al., 2025) proved that meta-learned initialization is critical — naive TTT barely helps, while meta-learned TTT matches full attention. Our Reptile experiment provides the first evidence that this principle holds for compressed models with SmearGate, though the effect size (0.011) is modest compared to the theoretical potential.

The error-guided TTT negative result suggests a deeper issue: at 16MB scale, the model's errors are concentrated on tokens that are fundamentally hard to predict (rare words, domain shifts, noise), not on tokens where better context modeling would help. This has implications for the design of future TTT strategies.

## Hardware

All experiments on RunPod 8x NVIDIA H100 80GB SXM. Total self-funded GPU cost: ~$180 across 15+ experimental runs.

## Files

- `train_gpt.py` — Training script with Reptile meta-learning
- `eval_error_guided_ttt.py` — Error-guided TTT evaluation (separate process)
- `submission.json` — Metadata
- This README

## Author

Xiaoan Liu | NYU | GitHub: @sseanliu
