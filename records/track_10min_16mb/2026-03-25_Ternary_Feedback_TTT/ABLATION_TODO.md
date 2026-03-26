# Ablation TODO

## Priority order (highest expected impact first)

### 1. Baseline: ternary trunk only

```bash
FEEDBACK_ENABLED=0 TTT_ENABLED=0 CAPSULE_ENABLED=0 SHARED_BLOCKS=0 \
BIGRAM_HASH_ENABLED=0 VRL_ENABLED=0 EMA_ENABLED=0 GPTQ_LITE_ENABLED=0 \
NGRAM_CACHE_ENABLED=0 SEED=42 bash run_cuda_feedback.sh
```

Repeat with SEED=1337 and SEED=7. **Measures**: raw ternary trunk BPB, artifact size.

### 2. LeakyReLU² activation (expected: -0.002 BPB)

```bash
ACTIVATION=lrelu2 LEAKY_RELU_SLOPE=0.5 FEEDBACK_ENABLED=0 \
SEED=42 bash run_cuda_feedback.sh
```

### 3. BigramHash (expected: -0.002 BPB)

```bash
BIGRAM_HASH_ENABLED=1 BIGRAM_HASH_BUCKETS=4096 BIGRAM_HASH_DIM=128 \
FEEDBACK_ENABLED=0 SEED=42 bash run_cuda_feedback.sh
```

### 4. EMA + GPTQ-lite (expected: -0.003 to -0.004 BPB combined)

```bash
EMA_ENABLED=1 EMA_DECAY=0.997 EMA_START_FRACTION=0.5 \
GPTQ_LITE_ENABLED=1 GPTQ_LITE_PERCENTILES=5 \
FEEDBACK_ENABLED=0 SEED=42 bash run_cuda_feedback.sh
```

**Warning**: EMA may interact poorly with ternary. If BPB degrades, disable EMA.

### 5. Partial RoPE (expected: -0.001 to -0.003 BPB)

```bash
PARTIAL_ROPE_DIMS=16 FEEDBACK_ENABLED=0 SEED=42 bash run_cuda_feedback.sh
```

### 6. VRL on deep layers (expected: -0.002 BPB)

```bash
VRL_ENABLED=1 VRL_START_LAYER=8 FEEDBACK_ENABLED=0 SEED=42 bash run_cuda_feedback.sh
```

### 7. LN Scale Damping (expected: small stability gain)

```bash
LN_SCALE_DAMPING=1 FEEDBACK_ENABLED=0 SEED=42 bash run_cuda_feedback.sh
```

### 8. Feedback: single pass vs off

```bash
FEEDBACK_ENABLED=1 FEEDBACK_PASSES=1 SEED=42 bash run_cuda_feedback.sh
FEEDBACK_ENABLED=0 SEED=42 bash run_cuda_feedback.sh
```

### 9. Iterative correction: 2-pass and 3-pass

```bash
# 2 correction passes (blind + 1 feedback)
FEEDBACK_ENABLED=1 FEEDBACK_PASSES=1 SEED=42 bash run_cuda_feedback.sh
# 3 passes total (blind + 2 feedback)
FEEDBACK_ENABLED=1 FEEDBACK_PASSES=2 SEED=42 bash run_cuda_feedback.sh
# Extra eval passes (train with 1, eval with 2)
FEEDBACK_ENABLED=1 FEEDBACK_PASSES=1 EVAL_FEEDBACK_PASSES=2 SEED=42 bash run_cuda_feedback.sh
```

### 10. Capsule bank (with and without recurrence)

```bash
# Capsules with single feedback pass
CAPSULE_ENABLED=1 CAPSULE_NUM=16 CAPSULE_DIM=64 FEEDBACK_ENABLED=1 FEEDBACK_PASSES=1 \
SEED=42 bash run_cuda_feedback.sh
# Capsules with 2-pass iterative correction (capsule state accumulates)
CAPSULE_ENABLED=1 FEEDBACK_ENABLED=1 FEEDBACK_PASSES=2 \
SEED=42 bash run_cuda_feedback.sh
```

### 11. Best non-TTT stack

Combine all winners from above:

```bash
ACTIVATION=lrelu2 BIGRAM_HASH_ENABLED=1 EMA_ENABLED=1 GPTQ_LITE_ENABLED=1 \
PARTIAL_ROPE_DIMS=16 VRL_ENABLED=1 LN_SCALE_DAMPING=1 \
FEEDBACK_ENABLED=1 FEEDBACK_PASSES=1 CAPSULE_ENABLED=1 \
SEED=42 bash run_cuda_feedback.sh
```

### 12. N-gram cache (expected: -0.07 to -0.16 BPB — the biggest single lever)

```bash
NGRAM_CACHE_ENABLED=1 NGRAM_MAX_ORDER=5 \
<best non-TTT config from above> \
SEED=42 bash run_cuda_feedback.sh
```

Try orders 3, 5, 7:
```bash
for ORDER in 3 5 7; do
  NGRAM_CACHE_ENABLED=1 NGRAM_MAX_ORDER=$ORDER SEED=42 bash run_cuda_feedback.sh
done
```

### 13. TTT (expected: -0.015 to -0.020 BPB on top of n-gram)

```bash
TTT_ENABLED=1 TTT_SCOPE=feedback TTT_LR=0.002 TTT_EPOCHS=1 \
<best config> SEED=42 bash run_cuda_feedback.sh
```

TTT LR sweep:
```bash
for LR in 0.001 0.002 0.005; do
  TTT_ENABLED=1 TTT_LR=$LR SEED=42 bash run_cuda_feedback.sh
done
```

### 14. Full stack (3-seed validation for submission)

```bash
for SEED in 42 1337 7; do
  <full best config> SEED=$SEED bash run_cuda_feedback.sh
done
```

Need p<0.01 significance vs current SOTA (1.1194 BPB).

## Shared block experiments (if artifact size is an issue)

```bash
SHARED_BLOCKS=3 MODEL_DIM=896 bash run_cuda_feedback.sh
SHARED_BLOCKS=2 MODEL_DIM=896 bash run_cuda_feedback.sh
```

## Optimizer tuning (based on research reports)

```bash
# Higher Muon momentum (0.99 reported as optimal)
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_STEPS=1500 bash run_cuda_feedback.sh

# Extended warmdown (40% of training)
WARMDOWN_FRACTION=0.4 bash run_cuda_feedback.sh

# Weight decay
MUON_WD=0.04 bash run_cuda_feedback.sh
```

## Quick smoke test

```bash
ITERATIONS=200 VAL_LOSS_EVERY=0 MAX_WALLCLOCK_SECONDS=60 \
FEEDBACK_ENABLED=0 SLIDING_EVAL=0 TEMP_SCALING=0 \
NPROC_PER_NODE=1 bash run_cuda_feedback.sh
```

## Notes

- Always check `budget:` line in log — total must be < 16,000,000 bytes
- Code is ~100KB; budget for model is ~15.9MB compressed
- Sliding eval stride=16 is expensive (~60-90s); for quick iteration use SLIDING_EVAL=0
- N-gram cache eval is sequential and slow (~30-60s); use only for final measurements
- If artifact is over budget: reduce model_dim, num_layers, or increase shared_blocks
- EMA may not work with ternary — monitor carefully, disable if BPB degrades
- Report both sliding and non-sliding BPB for comparisons
