# Focal Loss for Language Model Pretraining: 1.1567 int8 BPB on RTX 4000 Ada

## TL;DR

Applying **focal loss** (Lin et al. 2017) to language model pretraining gives massive, monotonic improvements. Combined with cosine LR scheduling and asymmetric encoder-decoder split, achieves **1.1567 int8 BPB at 5000 steps on a single RTX 4000 Ada** using the baseline `train_gpt.py` — within 0.037 of the 8xH100 SOTA record (1.1194 BPB).

## The Core Finding

Standard cross-entropy treats all tokens equally. But in natural language, most tokens are easy to predict (articles, common words, punctuation). Focal loss down-weights easy tokens and focuses the model on hard-to-predict tokens:

```python
# Standard cross-entropy:
loss = F.cross_entropy(logits, targets, reduction="mean")

# Focal loss (our change):
ce = F.cross_entropy(logits, targets, reduction="none")
pt = torch.exp(-ce)  # probability of correct class
focal_weight = (1 - pt) ** gamma
loss = (focal_weight * ce).mean()
```

This is a well-known technique in object detection (Lin et al., "Focal Loss for Dense Object Detection", 2017) but has not been applied to language model pretraining in this competition. The intuition transfers perfectly: just as object detection has a foreground/background class imbalance, language modeling has an easy/hard token imbalance.

## Results

All experiments run on **single RTX 4000 Ada 20GB** ($0.20/hr) using the **baseline `train_gpt.py`** code (no SOTA optimizations).

### Focal Loss Gamma Sweep (with Cosine LR)

| Gamma | 3000 steps | 5000 steps | Delta vs baseline (5000) |
|-------|-----------|-----------|--------------------------|
| 0 (cosine only) | 1.6538 | 1.5706 | -0.072 |
| 1 | 1.5643 | — | — |
| 1.5 | 1.5255 | — | — |
| 2 | 1.4849 | 1.4045 | -0.238 |
| 3 | 1.4246 | 1.3496 | -0.293 |
| 5 | 1.3339 | 1.2617 | -0.381 |
| 8 | 1.2288 | 1.1604 | -0.482 |
| **8 + asym** | — | **1.1567** | **-0.486** |
| 10 | 1.1806 | — | — |

**Baseline at 5000 steps (no cosine, no focal): 1.6422 BPB**

### Isolating Each Technique (3000 steps)

| Technique | BPB | Delta vs baseline (1.7233) |
|-----------|-----|---------------------------|
| Baseline (linear warmdown) | 1.7233 | — |
| Cosine LR only | 1.6538 | -0.070 |
| Focal γ=2 only (no cosine) | 1.5647 | -0.159 |
| Cosine + Focal γ=2 | 1.4849 | -0.238 |

Both techniques contribute independently and stack cleanly.

### Cosine LR Scaling Validation

| Steps | Baseline | Cosine LR | Delta |
|-------|----------|-----------|-------|
| 1000 | 2.0568 | 1.9334 | -0.123 |
| 2000 | 1.8330 | 1.8050 | -0.028 |
| 3000 | 1.7233 | 1.6538 | -0.070 |
| 5000 | 1.6422 | 1.5706 | -0.072 |

Cosine LR advantage is consistent and not diminishing with training length.

### Additional Findings

**What helped:**
- Asymmetric 1/10 encoder-decoder split: -0.004 to -0.009 on top of other techniques (see PR #1275)
- Higher focal gamma: monotonically better up to γ=10

**What did NOT help:**
- Higher base LR (0.08) with cosine: +0.050 worse
- Lower min_lr_frac (0.01, 0.0): worse than default 0.1
- Cosine with warm restarts: worse than plain cosine
- Label smoothing: hurt significantly
- Gradient noise: hurt significantly
- Weight decay scheduling: destroyed performance
- Gradient clipping with cosine: slightly worse

## Why This Works

Focal loss addresses the **easy token dominance** problem in language modeling:

1. **Token frequency imbalance**: Common tokens like "the", "is", "and" are easy to predict (high pt). They dominate the gradient in standard CE but contribute little information.
2. **Capacity allocation**: With focal loss, the model allocates more of its limited capacity (16MB budget) to learning hard patterns — rare words, complex syntax, domain-specific terms.
3. **Implicit curriculum**: Higher gamma creates a natural curriculum where the model progressively focuses on harder examples as easy ones are mastered.

The monotonic improvement with gamma suggests the baseline code significantly under-allocates capacity to hard tokens.

## Full Experiment Log (55+ experiments, 13 rounds)

This finding came from systematic experimentation across 3 GPUs over multiple days:

### Phase 1: M4 MacBook Exploration (27 experiments, PR #1073)
Using MLX on M4 MacBook 16GB to rapidly explore the design space:
- Deep supervision (auxiliary losses at intermediate layers): -0.05 BPB at small batch, vanishes at large batch
- LR sweep: LR 0.08 beats default 0.04 by -0.025 at 300 steps
- Gradient clipping: -0.019
- EMA/SWA: hurt at 300 steps, help at 9000 (consistent with leaderboard)
- Batch scaling analysis across 4K-524K tokens

### Phase 2: RTX 5090 + 8xH100 Validation (PR #1275)
- **Asymmetric encoder-decoder split** (num_encoder_layers=1): monotonic improvement across all configs
- RTX 5090 baseline sweep: 5/6→3/8→2/9→1/10 split, each better (-0.016 BPB total)
- SOTA code validation: -0.004 on PR #549 stack
- **8xH100 partial run**: 1.1492 pre-quant BPB at step 5666/9000 (pod crashed before final eval)

### Phase 3: RTX 4000 Ada Deep Dive (55+ experiments, this PR)
Systematic sweep on $0.20/hr GPU, testing dozens of techniques at 1000-5000 steps:

**Round 1-3**: Baseline calibration, asymmetric split validation at 500-1000 steps
- Confirmed asymmetric 1/10 helps on CUDA (-0.004 at 500 steps)
- Decoder-to-decoder skip connections: -0.005 at 500 steps, vanished at 1000

**Round 4-5**: Architecture & convergence techniques (1000 steps each)
- Decoder-skip 3-back connections: marginal, not robust
- Stochastic depth: torch.compile issues, inconclusive
- Enriched x0 (encoder feeds back to x0): no effect
- Skip connection dropout: compile issues
- Label smoothing: hurt significantly
- Gradient noise: hurt significantly
- Weight decay scheduling: destroyed performance

**Round 6**: Longer training validation (2000-3000 steps)
- **Cosine LR schedule discovered**: -0.123 at 1000 steps, -0.070 at 3000 steps
- Cosine with warm restarts: worse than plain cosine
- Gradient clipping + cosine: slightly worse

**Round 7**: Cosine LR stacking (2000-5000 steps)
- Cosine + asymmetric: -0.030 at 2000 steps (they stack!)
- Cosine + LR 0.06: -0.029 at 2000 steps
- Cosine + LR 0.08: +0.050 WORSE (too high for cosine)
- **Cosine at 5000 steps: 1.5706 vs baseline 1.6422 (-0.072, gap holds!)**

**Round 8**: Baselines and min_lr_frac sweep
- Baseline 5000 steps: 1.6422 (critical reference point)
- Cosine + asymmetric 5000 steps: 1.5619
- min_lr_frac=0.01: worse than 0.1
- min_lr_frac=0.0: even worse

**Round 9**: Novel techniques (focal loss, attention temp, lookahead)
- **FOCAL LOSS DISCOVERED**: γ=2 gives -0.169 at 3000 steps!
- Focal γ=1: -0.090
- Attention temperature annealing: crashed (torch.compile scope issue)
- Lookahead optimizer: crashed (same issue)

**Round 10**: Focal loss deep dive
- Focal γ=2 without cosine: 1.5647 (-0.159 alone, independently powerful)
- Focal γ=2 with cosine at 5000 steps: 1.4045
- Focal γ=3: 1.4246 at 3000 steps (still improving!)
- Focal γ=1.5: 1.5255

**Round 11**: Higher gamma sweep
- Focal γ=3 at 5000 steps: 1.3496
- Focal γ=4: 1.3845 at 3000 steps
- Focal γ=5: 1.3339 at 3000 steps, 1.2617 at 5000 steps
- **Still monotonically improving!**

**Round 12**: Push to the limit
- Focal γ=8 at 3000 steps: 1.2288
- **Focal γ=8 at 5000 steps: 1.1604** (approaching SOTA record!)
- Focal γ=8 + asymmetric 5000 steps: **1.1567** (our best)

**Round 13**: Ceiling test
- Focal γ=10 at 3000 steps: 1.1806 (still improving vs γ=8 at 3000)
- Focal γ=8 + asymmetric at 5000 steps: 1.1567 (confirmed)

## Experimental Setup

- **GPU**: Single RTX 4000 Ada 20GB (RunPod, $0.20/hr)
- **Code**: Baseline `train_gpt.py` with FA2 patch (no SOTA optimizations)
- **Batch**: TRAIN_BATCH_TOKENS=8192, GRAD_ACCUM_STEPS=64 (effective 524K)
- **Evaluation**: `final_int8_zlib_roundtrip_exact` (the competition metric)
- **Total experiments**: 55+ across 13 rounds, 3 GPUs (M4, RTX 5090, RTX 4000 Ada)
- **Total GPU cost**: ~$2.50 for RTX 4000 Ada experiments, ~$16 for 8xH100 run (PR #1275)

## Caveats and Open Questions

1. **Not validated on 8xH100**: These results are on a single GPU with small micro-batch. The optimal gamma may differ at the 8xH100 scale with larger batch sizes.
2. **Not tested on SOTA stack**: The SOTA code uses EMA, SWA, QAT, Muon, TTT, and other techniques. Focal loss may interact differently with these.
3. **High gamma concerns**: At γ=8, tokens predicted with 50% probability get weighted at 1/256 of normal. This aggressive down-weighting could cause underfitting on common patterns at very long training.
4. **Needs 8xH100 validation**: Requesting GPU credits to validate on the full competition setup.

## Prior Work

- **PR #1275**: Asymmetric 1/10 encoder-decoder split finding + 8xH100 partial run (1.1492 pre-quant BPB)
- **PR #1073**: 27 systematic experiments on M4 MacBook (deep supervision, LR tuning, batch scaling, architecture)

## Reproduce

### Quick test (any single GPU)

```bash
git clone https://github.com/openai/parameter-golf.git && cd parameter-golf
pip install sentencepiece huggingface-hub datasets tiktoken flash-attn

# Apply focal loss to train_gpt.py — change the loss computation in GPT.forward():
# OLD: return F.cross_entropy(logits.float(), targets, reduction="mean")
# NEW:
#   focal_gamma = float(os.environ.get("FOCAL_GAMMA", "0"))
#   if focal_gamma > 0:
#       ce = F.cross_entropy(logits.float(), targets, reduction="none")
#       pt = torch.exp(-ce)
#       focal_weight = (1 - pt) ** focal_gamma
#       return (focal_weight * ce).mean()
#   return F.cross_entropy(logits.float(), targets, reduction="mean")

# Also replace the lr_mul() function body with cosine schedule:
#   min_lr_frac = 0.1
#   progress = step / max(args.iterations, 1)
#   return min_lr_frac + 0.5 * (1.0 - min_lr_frac) * (1.0 + math.cos(math.pi * progress))

# And for asymmetric split, change in GPT.__init__():
#   self.num_encoder_layers = 1  # instead of num_layers // 2

# Download data
python data/cached_challenge_fineweb.py --variant sp1024

# Run (single GPU, ~20 min for 5000 steps on RTX 4000 Ada)
FOCAL_GAMMA=8 COSINE_LR=1 ITERATIONS=5000 python train_gpt.py
```

### 8xH100 Record Run

```bash
#!/bin/bash
# Full competition run on 8xH100 SXM
# Apply the same 3 changes to the SOTA train_gpt.py (PR #549 stack):
# 1. Focal loss in GPT.forward() (see above)
# 2. Cosine LR in lr_mul() (see above)
# 3. self.num_encoder_layers = 1 in GPT.__init__()

cd /workspace
git clone --depth 1 https://github.com/openai/parameter-golf.git && cd parameter-golf
pip install -q sentencepiece huggingface-hub datasets tiktoken
pip install -q flash-attn --no-build-isolation

python data/cached_challenge_fineweb.py --variant sp1024

# Copy SOTA script and apply changes
cp records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py train_gpt_focal.py
# Apply: (1) focal loss, (2) cosine LR, (3) asymmetric split
# See code changes above

# Run with competition settings
NUM_LAYERS=11 FOCAL_GAMMA=8 \
torchrun --standalone --nproc_per_node=8 train_gpt_focal.py
```
