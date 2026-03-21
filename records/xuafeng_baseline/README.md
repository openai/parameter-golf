# Parameter Golf Experiments — xuafeng

## Overview

Three approaches tested on top of the #1 leaderboard entry (thwu1, 1.14276 BPB):

| Approach | val_bpb | vs #1 | Status |
|----------|---------|-------|--------|
| QAT + Int5 + TrigramHash | 1.14423 | +0.0015 | Over 16MB limit |
| QAT + Int5 (no trigram) | 1.14476 | +0.0020 | Valid, but worse |
| TTT LoRA on #1 | — | — | Implemented, not tested (budget) |

None of the completed runs beat #1. The TTT approach remains the most promising untested idea.

## Files

| File | Description |
|------|-------------|
| `train_gpt_qat.py` | QAT + TrigramHash implementation on #1's architecture |
| `train_gpt_ttt.py` | TTT LoRA implementation on #1's architecture |
| `report.md` | Initial 1x H100 baseline training report (1.3274 BPB) |
| `experiment_report.md` | QAT experiment results and analysis |
| `final_report.md` | Complete summary of all runs, costs, and next steps |
| `process.md` | End-to-end RunPod + runpodctl process guide |
| `next_ideas.md` | Original ideas document (pre-experiment) |

## Method 1: QAT + Int5/Int6 (Disproven)

Added STE fake-quantization to `CastedLinear.forward()` matching the post-training quantization levels:
- MLP layers: int5 (clip_range=15)
- Attention layers: int6 (clip_range=31)
- Straight-Through Estimator: `w + (w_quantized - w).detach()`

**Result**: QAT added ~0.002 BPB penalty vs #1's post-training quantization.

**Why it failed**: Post-training quantization + SWA acts as beneficial regularization in this setup. QAT removes that benefit by making weights pre-adapted to quantization levels.

### QAT Results (3 seeds attempted, 2 completed)

| Seed | val_bpb | Artifact | Steps | Notes |
|------|---------|----------|-------|-------|
| 42 | 1.14423 | 16.2 MB | 6614 | +TrigramHash, over 16MB |
| 1337 | 1.14476 | 15.8 MB | 6649 | QAT only, valid |
| 2024 | — | — | 4500 | Pod terminated |

## Method 2: TrigramHash (Negligible)

Added a trigram hash table (4096 buckets, dim=32) alongside the existing BigramHash:
```python
hash = xor(48271 * t[i], 36313 * t[i-1], 27191 * t[i-2]) % 4095
```

**Result**: ~0.0005 BPB improvement (within noise). Pushed artifact over 16MB. Not worth the size cost.

## Method 3: TTT LoRA (Implemented, Not Tested)

Added test-time training with LoRA adapters to the evaluation phase, exploiting the unused eval budget (#1 uses 170s of 600s).

Key design:
- LoRA rank-8 on Q/V projections + LM head (all 10 layers)
- Per-document adaptation: score chunk, train LoRA, score next chunk
- Reset between documents (no leakage)
- Batched processing (64 docs at a time)
- Uses `eval_val_ttt_lora()` instead of `eval_val_sliding()`

**Expected**: -0.003 to -0.010 BPB based on TTT entry #9's results on weaker baseline.

**Status**: Could not test — RunPod pods kept terminating (insufficient balance for 8x H100 at $21.52/hr).

## Key Learnings

1. **Post-training quantization can outperform QAT** when combined with SWA and magnitude pruning
2. **N-gram hashes have diminishing returns** beyond bigram at this scale
3. **Eval time is massively underutilized** — TTT is the most promising remaining lever
4. **RunPod 8x H100 pods require ~$7+ balance** for a full 20-min session (10 min train + 10 min eval)

## Cost

Total spent: ~$22 across 5 pod sessions. Remaining: $4.04.

## Reproducibility

```bash
# QAT run (on 8x H100):
SEED=1337 RUN_ID=qat_seed1337 TRIGRAM_VOCAB_SIZE=0 \
  torchrun --standalone --nproc_per_node=8 train_gpt_qat.py

# TTT run (on 8x H100, needs ~$7+ balance):
SEED=42 RUN_ID=ttt_seed42 \
  torchrun --standalone --nproc_per_node=8 train_gpt_ttt.py
```
