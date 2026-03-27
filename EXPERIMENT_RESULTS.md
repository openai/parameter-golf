# Parameter Golf — Experiment Results

**Config:** 12L x 512dim, 8 heads, 4 KV heads, bigram 20K, MLP 3.5x
**Hardware:** RTX 4090 (single GPU), NO_COMPILE
**Baseline leaderboard #1:** 1.1428 bpb (8xH100, 10 min)

## Master Leaderboard (all experiments, ranked by quantized val_bpb)

| Rank | Experiment | Steps | Val BPB (quant) | Notes |
|---|---|---|---|---|
| 1 | **r5_control** (12L, no SWA) | 2677 | **1.3620** | Best overall |
| 2 | long_3000 (12L, no SWA) | 2617 | 1.3647 | Reproducible |
| 3 | r5_wd1500 (warmdown 1500) | 2687 | 1.3638 | SWA 12 ckpts, ~tied |
| 4 | r5_swa_default | 2676 | 1.3740 | SWA hurts quant |
| 5 | r5_swa_freq25 | 2631 | 1.3778 | More SWA = worse |
| 6 | r5_swa_early | 2677 | 1.3958 | Most SWA = worst |
| 7 | r5_wd5000 | 2684 | 1.4022 | Too early warmdown |
| 8 | 16L_long_3000 | 3000 | 1.4011 | 16L scales worse |
| 9 | combined_v3 (1000 steps) | 1000 | 1.5029 | Best at 1000 steps |
| 10 | 16L_448d | 1000 | 1.4987 | Best 1000-step arch |
| 11 | combined_v2 | 1000 | 1.5109 | 12L + bigram 20K |
| 12 | 12_layers_v2 | 1000 | 1.5111 | 12L alone |
| 13 | 16L_kv7 | 1000 | 1.5108 | MHA baseline |
| 14 | bigram_20k | 1000 | 1.5184 | Bigram 20K alone |
| 15 | bigram_dim256 | 1000 | 1.5231 | Wider bigram |
| 16 | mlp_3.5x | 1000 | 1.5243 | MLP 3.5x alone |
| 17 | control (original) | 1000 | 1.5298 | Original baseline |
| 18 | wd_002 | 1000 | 1.5313 | Lower WD hurts |
| 19 | muon_095 | 1000 | 1.5334 | Lower momentum hurts |
| 20 | 16L_kv1 | 1000 | 1.5352 | GQA worse than MHA |
| 21 | 14_layers | 938 | 1.5379 | Too slow per step |
| 22 | 20L_384d | 979 | 1.5438 | Extreme depth hurts |
| 23 | 16L_480d_kv1 | 931 | 1.5464 | Wide GQA bad |
| 24 | 18L_416d | 884 | 1.5818 | Way too slow |
| 25 | bigger_batch | 550 | 1.6879 | Halved steps |

## Round-by-Round Analysis

### Round 1 — Baseline Experiments (1000 steps)
- **Control:** val_bpb 1.5298 (10L, 512d, bigram 10K, MLP 3x)
- BigramHash is critical (~0.55 val_loss without it)
- Width over depth doesn't work (8L/640d much worse)
- Bigger batch hurts (halves steps, 2x slower per step)

### Round 2 — Architecture Sweep (1000 steps)
- **12 layers:** -0.019 bpb (biggest single gain)
- **Bigram 20K vocab:** -0.011 bpb
- **MLP 3.5x:** -0.005 bpb
- Gains are additive: combined_v3 (12L+bigram20K+MLP3.5x) = -0.027 bpb
- Lower WD and lower Muon momentum both hurt

### Round 3 — Architectural Experiments
- **SwiGLU:** worse than squared ReLU (+0.066 train_loss)
- **Trigram hash:** no benefit (+0.073 train_loss)
- **Depth recurrence 6x2:** worse (halved unique params)
- **Depth recurrence 4x3:** worse
- **16L x 448d:** best at 1000 steps (1.4987 bpb) but scales worse
- **Long run (2617 steps):** 1.3647 bpb — massive scaling confirmed

### Round 4 — 16L vs 12L Scaling
- **MHA >> GQA** for this model size
- 18L and 20L too slow per step, don't compensate
- **12L x 512d scales better than 16L x 448d** at longer runs
- 12L long run: 1.3647 bpb vs 16L long run: 1.4011 bpb

### Round 5 — SWA, Warmdown, Sliding Eval
- **SWA hurts quantization** consistently — more checkpoints = worse
- Pre-quant val_bpb identical with/without SWA (~1.359)
- SWA smooths weights in ways that quantize poorly with int5/int6
- **Warmdown 3000:** optimal (default)
- **Warmdown 1500:** slightly worse but SWA damage minimal
- **Warmdown 5000:** much worse (enters warmdown too early)
- **Sliding eval:** Could not complete full test on 4090 (too slow with retraining). Leaderboard winners use stride=64 — use it for submission (eval-only, no training cost).

## Confirmed Best Config

```
NUM_LAYERS=12
MODEL_DIM=512
NUM_HEADS=8
NUM_KV_HEADS=4
MLP_MULT=3.5
BIGRAM_VOCAB_SIZE=20480
BIGRAM_DIM=128
SWA_ENABLED=0
WARMDOWN_ITERS=3000
EVAL_STRIDE=64
WEIGHT_DECAY=0.04
MUON_MOMENTUM=0.99
TIED_EMBED_LR=0.03
MATRIX_LR=0.02
SCALAR_LR=0.02
```

## Key Insights

1. **Depth helps but has diminishing returns** — 12L is sweet spot for 512dim
2. **BigramHash is the single most important feature** — don't remove it
3. **Squared ReLU > SwiGLU** at this model scale
4. **SWA hurts quantization** — disable for single-GPU, retest on 8xH100
5. **More steps always help** — curve is still steep at 2700 steps
6. **Gains are additive** across independent improvements
7. **Train on 4090, validate decisions, deploy on 8xH100**
