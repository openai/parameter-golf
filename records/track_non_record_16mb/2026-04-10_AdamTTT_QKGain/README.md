# Non-Record Submission: Adam-TTT + QK-Gain Search

**Track:** non-record (no training logs — compute unavailable before deadline)
**Base:** PR #1493 (bigbag, 1.0810 BPB)

This submission presents a novel algorithmic idea — replacing SGD with Adam in the TTT eval loop — without validation results due to lack of compute access. Submitted to put the idea on the record and for community review.

## Key Changes

### 1. Adam-TTT (primary novel contribution)
The SOTA TTT loop uses SGD with momentum=0.9. We replace it with **Adam** (beta1=0.9, beta2=0.999).

**Why Adam is better for TTT:**
- Adam adapts per-parameter learning rates based on gradient history
- TTT is essentially fast fine-tuning — Adam is the standard choice for fine-tuning
- SGD with fixed momentum can overshoot on some parameters while undershooting on others
- Adam's normalization by second moment makes it less sensitive to LR choice

Controlled by `TTT_OPTIMIZER=adam` (default). Also supports `TTT_OPTIMIZER=adamw` (with weight decay) and `TTT_OPTIMIZER=sgd` (original behavior).

New hyperparameters:
- `TTT_OPTIMIZER` — optimizer choice (default: `adam`)
- `TTT_ADAM_BETA1` — Adam beta1 (default: `0.9`)
- `TTT_ADAM_BETA2` — Adam beta2 (default: `0.999`)
- `TTT_ADAM_EPS` — Adam epsilon (default: `1e-8`)
- `TTT_WEIGHT_DECAY` — weight decay for AdamW (default: `0.0`)
- `TTT_FRESH_OPTIMIZER` — reset optimizer state each chunk (default: `0`)

### 2. QK-Gain above 5.25
The SOTA showed monotonic improvement from QK_GAIN_INIT=4.0→4.5→5.0→5.25.
We search 5.5, 6.0, 6.5 to find the optimum.

## Architecture (unchanged from PR #1493)

11L × 512d × 8H / 4KV, MLP 4×, LeakyReLU(0.5)², Partial RoPE (16/64 dims),
layerwise LN scale, tied embeddings, logit softcap=30.0.
Depth recurrence: loops layers 3-5, activated at step ~35%.
Parallel residuals from layer 7.

## Experiment Plan

| Experiment | TTT_OPTIMIZER | QK_GAIN_INIT | Expected |
|---|---|---|---|
| 1 (Adam baseline) | adam | 5.25 | vs SOTA 1.0810 |
| 2 (higher QK-gain) | adam | 5.5 | better? |
| 3 (even higher) | adam | 6.0 | better? |
| 4 (AdamW TTT) | adamw | best above | better? |
| 5 (fresh Adam per chunk) | adam + fresh | best above | better? |

## Reproduction

```bash
pip install brotli sentencepiece
pip install flash_attn_3 --no-deps --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/
python3 data/cached_challenge_fineweb.py --variant sp8192

# Experiment 1: Adam-TTT at QK-gain 5.25
SEED=42 QK_GAIN_INIT=5.25 TTT_ENABLED=1 TTT_OPTIMIZER=adam TTT_LR=0.005 TTT_EPOCHS=3 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py

# Experiment 2: Higher QK-gain
SEED=42 QK_GAIN_INIT=5.5 TTT_ENABLED=1 TTT_OPTIMIZER=adam TTT_LR=0.005 TTT_EPOCHS=3 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Credits

- **@bigbag** — PR #1493 SOTA stack (all core techniques)
- **Lakshay Bansal** — Adam-TTT replacement, QK-gain search
