# Behemoth: Macro-Sidechannel Transformer with Encoder-Decoder Distillation

## Architecture

Behemoth is a 34.7M-parameter transformer with several novel components designed to maximize BPB under the 16MB/10-minute constraint:

- **Encoder-Decoder Split** with U-Net skip connections (5 encoder + 5 decoder layers)
- **Macro Sidechannel**: Causal cross-attention at the encoder/decoder boundary. Interval summaries (every 16 tokens) are distilled via a student-teacher mechanism — the student sees only past intervals, trained to predict the current one via MSE distillation
- **Macro Pyramid**: Two-level hierarchy (interval=16, interval=64) for multi-scale context
- **Adaptive Depth Gates**: Per-token routing between residual stream and initial embeddings
- **Gated Skip Connections**: Bottleneck-gated U-Net skips replacing scalar weights
- **Orthogonal Branch Forcing (OBF)**: Cosine-similarity penalty pushing attention and MLP branches toward specialization
- **Surprisal-Weighted Loss**: sqrt-reweighted cross-entropy emphasizing difficult tokens
- **SmearGate + BigramHash**: Local context injection at the embedding layer
- **Full int4 QAT** from step 1 with straight-through estimator

## The Gradient Explosion Problem (12 Iterations)

The architecture trained cleanly on 1 GPU (val_bpb 1.3842 in 600s) but consistently produced NaN on 8xH100, dying between step 150-1300 depending on learning rate.

### Root Cause

The macro sidechannel fed un-normalized encoder output directly into QKV projection layers:

```python
# BEFORE (broken): raw residual stream into projections
q = self.macro_q(x)        # x has growing variance after 5 encoder layers
k = self.macro_k(student)  # student derived from un-normalized x
```

Standard self-attention always pre-norms before projections (`attn_norm(x)` before QKV). The macro sidechannel skipped this step. After 5 encoder layers of residual accumulation, `x`'s variance grew large enough that `q @ k.T` exceeded bfloat16 range (~65,504), producing NaN in softmax.

The same flaw existed in the adaptive depth gate: `self.depth_gate(x)` operated on raw `x`, causing sigmoid saturation and potential overflow.

### Fix (v12)

```python
# AFTER (fixed): pre-norm before projections
x_norm = F.rms_norm(x, (D,))
student_norm = F.rms_norm(student, (D,))
q = self.macro_q(x_norm)
k = self.macro_k(student_norm)
v = self.macro_v(student_norm)
context = F.rms_norm(context, (D,))  # normalize cross-attention output
```

Plus depth gate fix:
```python
x_norm_for_gate = F.rms_norm(x, (x.size(-1),))
gate_logits = self.depth_gate(x_norm_for_gate).clamp(-10.0, 10.0)
```

### Additional Bugs Fixed in v12

1. **Orphaned distillation heads**: `macro_distill_proj` and `macro_distill_pred` were not registered in any optimizer group. Gradients were computed but never applied. Fixed by adding to `collect_matrix_params()`.

2. **SkipGate bias too conservative**: Initialized at -4.0 (sigmoid=0.018), effectively blocking 98% of U-Net skip information. Changed to 0.0 (sigmoid=0.5) for 50% initial pass-through.

3. **OBF penalizing complementary branches**: `.abs()` on cosine similarity penalized both redundant (cosine=+1) and complementary (cosine=-1) branch behavior equally. Changed to `F.relu()` to only penalize redundancy.

## Results

| Run | GPUs | LR | Steps | Val BPB | Status |
|-----|------|-----|-------|---------|--------|
| v8 (1-GPU baseline) | 1 | 0.023 | 1,097 | 1.3842 | Clean (wallclock cap) |
| v1-v6 (8-GPU) | 8 | 0.018-0.023 | 150-350 | - | NaN |
| v7-v8 (8-GPU) | 8 | 0.010-0.015 | 500-1050 | ~1.40 | NaN |
| v9 (8-GPU, nonfinite guard) | 8 | 0.010 | 1,306 | 1.3818 | NaN (guard abort) |
| v10 (8-GPU, gate stabilization) | 8 | 0.010 | 1,200 | 1.3997 | NaN |
| v11 (8-GPU, soft context cap) | 8 | 0.023 | 200 | - | NaN |
| **v12 (8-GPU, pre-norm fix)** | **8** | **0.015** | **5,577** | **1.2564** | **Clean (full 600s)** |

## File Structure

```
Behemoth/
├── 8H100/
│   ├── train_gpt.py          # Base architecture (v0)
│   ├── train_gpt_1.py-11.py  # Iterations 1-11 (stability debugging)
│   ├── train_gpt_12.py       # Final stable version
│   ├── logs.txt-logs_v12.txt # Training logs for each iteration
│   └── sota_test.py          # Baseline validation
├── iter-001 through iter-010/ # 1-GPU development iterations
└── README.md                  # This file
```

## Next Steps

- Push LR toward 0.020-0.025 (pre-norm fix eliminates the stability ceiling)
- Switch from int4 to int6 quantization (+0.058 BPB from community findings)
- Add EMA weight averaging (decay=0.997)
- Enable sliding window evaluation (stride=64)
- Scale to 11 layers (leaderboard consensus)
