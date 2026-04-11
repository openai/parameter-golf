# Systematic Hyperparameter Search (val_bpb=1.2075)

## Summary

| Metric | Value |
|--------|-------|
| Post-quant val_bpb | 1.2075 |
| Pre-quant val_bpb | 1.2008 |
| Compressed artifact | ~15.2 MB (under 16 MB) |
| Training steps | 7,390 |
| Training time | 600s (8×H100 SXM) |
| Eval time | ~15s (standard eval) |

## Approach

Methodical hyperparameter search through 33 experiments across three GPU tiers (A40 → 1×H100 → 8×H100), using fixed-seed paired comparison for reliable delta measurement.

The process:

1. **Cheap screening** — 2-min runs with SEED=1337 for paired comparison (resolves deltas as small as ±0.001 BPB)
2. **One variable at a time** — each experiment changes exactly one thing from the current best, isolating the effect
3. **Structured logging** — every experiment documented with hypothesis, result, and analysis of why it worked or didn't
4. **Progressive scaling** — start cheap (A40), validate on target hardware (8×H100) only after narrowing the search space

## Key Findings

### What works (on 8×H100/10min)

| Technique | Effect | Evidence |
|-----------|--------|----------|
| Muon optimizer (lr=0.02, momentum=0.99, warmdown=3000) | -0.005 BPB | exp_030 vs exp_029 |
| ROPE_BASE=200000 | -0.003 BPB | exp_033 vs exp_030 |
| seq_len=4096 | -0.006 BPB | exp_029 vs exp_014 (scaled) |

### What doesn't work (with evidence)

| Technique | Effect | Why |
|-----------|--------|-----|
| int6 STE (quantization-aware training) | +0.007 worse | Conflicts with Muon optimizer (exp_032) |
| 12 layers | +0.015 worse | Too slow → fewer steps → underfits (exp_016) |
| Larger batch (786K) | +0.009 worse | Fewer steps outweighs per-step quality (exp_035) |
| Smaller batch (262K) | +0.003 worse | Too noisy gradients (exp_013) |
| Higher LR at 10min (0.10 vs 0.04) | Neutral | LR insensitive with enough steps (exp_015) |

### Compute regime insights

Optimal configurations differ dramatically across compute budgets:

| Setting | Optimal layers | Optimal LR | Optimal batch |
|---------|---------------|------------|---------------|
| A40 / 2 min | 2-3 | 0.10 | 131K |
| 1×H100 / 10 min | 6-9 | 0.04 | 524K |
| 8×H100 / 10 min | 9 | 0.02 | 524K |

Hyperparameter transfer across compute scales is unreliable. The optimal LR on A40 (0.10) is 5× the optimal on 8×H100 (0.02). This means screening on cheap hardware gives directional signal but final values must be re-tuned on target hardware.

## Changes from Baseline

Only hyperparameters changed. No architectural modifications:

```python
# Optimizer
MATRIX_LR = 0.02      # was 0.04
MUON_MOMENTUM = 0.99   # was 0.95
WARMDOWN_ITERS = 3000  # was 1200

# Position encoding
ROPE_BASE = 200000     # was 10000

# Context
TRAIN_SEQ_LEN = 4096   # was 1024
```

Additionally, a compatibility fix for PyTorch 2.4 (replace `enable_gqa` with manual `repeat_interleave` for GQA).

## Experimental Cost

| Phase | GPU | Cost | Experiments |
|-------|-----|------|-------------|
| Architecture screening | 1×A40 | ~$3 | 14 |
| Technique validation | 1×H100 PCIe | ~$12 | 15 |
| Final validation | 8×H100 SXM | ~$25 | 6 |
| **Total** | | **~$40** | **35** |

## Reproducibility

```bash
RUN_ID=reproduce \
NUM_LAYERS=9 \
TRAIN_SEQ_LEN=4096 \
TRAIN_BATCH_TOKENS=524288 \
MATRIX_LR=0.02 \
MUON_MOMENTUM=0.99 \
WARMDOWN_ITERS=3000 \
ROPE_BASE=200000 \
SEED=1337 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Test Plan

- [x] Trained on 8×H100 SXM, 600s wallclock
- [x] final_int8_zlib_roundtrip val_bpb: 1.2075
- [x] Artifact under 16,000,000 bytes
- [x] train_gpt.py compiles and runs from records folder
- [x] train.log included
