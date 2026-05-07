# JEPA V2: EMA Teacher + Attention Predictor + 11L + LeakyReLU² + SWA + Partial RoPE

**Track:** Non-record (unlimited compute, ≤16MB artifact)
**Author:** Maksimilians Aleksandrovics (@Ochakov64)
**Score:** TBD (fill after GPU run)

## Summary

Builds on the V1 JEPA submission with four improvements taken from the top leaderboard entries, stacked on top of the JEPA training objective.

## Changes from V1

### 1. 11 Layers (from 10)
More capacity without exceeding the 16MB artifact limit (~27.8M params, ~12MB int8+zlib).

### 2. LeakyReLU(0.5)² Activation
Replaces ReLU² in the MLP with `LeakyReLU(0.5)²`. Preserves negative gradient flow and prevents dead neurons. Single-line change delivering ~0.003 BPB improvement per the top submission ablations.

### 3. Student EMA Weight Averaging (SWA, decay=0.997)
A shadow copy of the student model is maintained as an EMA of its weights throughout training. The averaged weights are used for serialization, giving smoother and better-generalizing final weights. Inspired by the EMA technique in the #2 leaderboard entry.

### 4. Partial RoPE (25% of head dims)
Rotary positional embeddings applied only to the first 16 of 64 head dimensions. The remaining 48 dims attend without positional bias, learning position-invariant patterns. Zero parameter cost. Inspired by the #3 leaderboard entry.

## JEPA (inherited from V1)

- EMA teacher encoder provides stable latent targets
- Attention-based predictor (1-layer causal MHA, 4 heads) maps student → teacher space
- Loss: `0.3·cosine_jepa + 0.7·CE`, alpha anneals to 0 over final 20%
- Teacher discarded before serialization — zero byte cost

## Results

| Metric | Value |
|--------|-------|
| val_bpb (int8+zlib roundtrip) | TBD |
| Submission size | TBD MB |
| Training time | TBD |

## Hyperparameters

```
NUM_LAYERS=11
MLP_MULT=3
MODEL_DIM=512
NUM_HEADS=8
NUM_KV_HEADS=4
VOCAB_SIZE=1024
PARTIAL_ROPE_FRAC=0.25
SWA_ENABLED=1
SWA_DECAY=0.997
JEPA_ENABLED=1
JEPA_ALPHA=0.3
JEPA_TRANSITION_FRAC=0.8
JEPA_EMA_START=0.99
JEPA_EMA_END=0.999
JEPA_NORMALIZE=1
JEPA_PRED_HEADS=4
```

## Running

```bash
RUN_ID=jepa_v2 VAL_LOSS_EVERY=1000 torchrun --standalone --nproc_per_node=8 train_gpt.py
```
