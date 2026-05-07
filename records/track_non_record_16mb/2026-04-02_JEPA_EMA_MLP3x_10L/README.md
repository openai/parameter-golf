# JEPA EMA Teacher + Attention Predictor + MLP3x + 10L

**Track:** Non-record (unlimited compute, ≤16MB artifact)
**Author:** Maksimilians Aleksandrovics (@maleks)
**Score:** TBD (fill after GPU run)

## Summary

This submission implements **JEPA (Joint Embedding Predictive Architecture)** as an auxiliary training objective for the parameter-constrained language model — fulfilling the [JEPA wishlist item](https://github.com/openai/parameter-golf#requests-for-prs) in the challenge README.

## Approach

### JEPA Auxiliary Loss

During training, alongside the standard cross-entropy loss, the model minimizes a latent prediction loss:

```
loss = alpha * jepa_cosine_loss + (1 - alpha) * CE_loss
```

- **Student encoder**: the standard causal GPT forward pass (`forward_hidden`)
- **Attention predictor**: a 1-layer causal self-attention block (4 heads) followed by a linear projection, inspired by [lang-jepa](https://github.com/jerber/lang-jepa). This lets each position gather context before predicting, rather than projecting token-by-token independently. The predictor also serves as the final projection head for logits.
- **EMA teacher**: a slowly-updated copy of the student encoder, providing stable latent targets that the predictor learns to match via cosine similarity loss
- **Alpha schedule**: alpha=0.3 constant for the first 80% of training, then linearly anneals to 0 so the final 20% is pure CE

Key property: **the teacher is discarded before serialization** — it costs zero bytes in the 16MB budget. The entire JEPA cost is compute-only.

### Why CE-dominant (alpha=0.3)?

Early experiments with alpha=0.9 caused the model to under-optimize for token prediction during training. With alpha=0.3, CE always provides 70% of the gradient signal, so representations stay well-calibrated for next-token prediction throughout. The JEPA loss acts as a regularizer encouraging smoother, more stable representations.

### Architecture Changes from Baseline

| Parameter | Baseline | This submission |
|-----------|----------|----------------|
| Layers | 9 | 10 |
| MLP multiplier | 2x | 3x |
| Predictor | linear | 1-layer attention (4 heads) + linear |
| Parameters | ~17.3M | ~25.5M |
| Submission size | ~7.75 MB | ~11 MB |

The larger model fits comfortably within the 16MB artifact limit (10.78 MB int8+zlib).

## Results

| Metric | Value |
|--------|-------|
| val_bpb (pre-quant) | TBD |
| val_bpb (int8+zlib roundtrip) | TBD |
| Submission size | TBD MB |
| Training time | TBD min on 8xH100 |

## Hyperparameters

```
NUM_LAYERS=10
MLP_MULT=3
MODEL_DIM=512
NUM_HEADS=8
NUM_KV_HEADS=4
VOCAB_SIZE=1024
TIE_EMBEDDINGS=1
JEPA_ENABLED=1
JEPA_ALPHA=0.3
JEPA_TRANSITION_FRAC=0.8
JEPA_EMA_START=0.99
JEPA_EMA_END=0.999
JEPA_NORMALIZE=1
```

## Running

```bash
# Single GPU (testing)
RUN_ID=jepa_test \
torchrun --standalone --nproc_per_node=1 train_gpt.py

# 8xH100 (leaderboard)
RUN_ID=jepa_submission \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Implementation Notes

- The `JEPAPredictor` (causal attention + linear) doubles as both the JEPA prediction head and the final projection before the tied embedding lookup, keeping the architecture clean and the param count honest. The attention head count is configurable via `JEPA_PRED_HEADS` (default 4).
- The EMA update skips the predictor — only the encoder (blocks, norms, skip weights) is tracked by the teacher.
- `forward_hidden()` exposes the encoder output cleanly; `_logits_from_hidden()` applies predictor + softcap for reuse in both training and eval.
