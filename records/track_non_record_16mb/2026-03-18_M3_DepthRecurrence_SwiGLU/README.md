# Depth Recurrence + SwiGLU — Apple M3 (8GB) Submission

## Summary

This is a non-record submission exploring **depth recurrence** (weight sharing across layers) and **SwiGLU MLPs** as parameter-efficient alternatives to the naive baseline, trained on a resource-constrained Apple M3 laptop with 8GB RAM.

### Key Ideas

1. **Depth Recurrence**: Instead of 9 unique transformer blocks, we use **4 unique blocks each reused 3 times** = 12 effective layers. This gives more effective depth for the same artifact size, inspired by Universal Transformers and ALBERT. Per-recurrence learnable gate scalars allow each pass to specialize.

2. **SwiGLU MLP**: Replaces the baseline's relu² MLP with a gated SiLU (SwiGLU) variant, which has been shown to be more parameter-efficient in the Chinchilla/LLaMA literature. We use 2/3 hidden dim (rounded to 64) to keep parameter count similar.

3. **Wider Model**: Weight sharing saves ~60% of artifact space on transformer blocks, which we reinvest into a wider model (640 dim vs 512 baseline, 10 heads vs 8).

4. **U-Net Skip Connections**: Preserved from baseline — encoder half stores skip tensors, decoder half consumes them in reverse order.

5. **Gradient Clipping**: Added `grad_clip_norm=1.0` for training stability with depth recurrence.

### Limitations

- **Trained on Apple M3 with 8GB RAM** — severely compute-limited compared to the target 8×H100 setup.
- Training batch sizes and iteration counts are much smaller than what's possible on H100s.
- Results are directional: the architecture ideas are sound but the score reflects hardware limitations, not the approach's ceiling.
- With proper H100 compute, we'd expect significantly better results from this architecture.

### Architecture Details

| Parameter | Value |
|-----------|-------|
| Unique layers | 4 |
| Recurrence factor | 3× |
| Effective layers | 12 |
| Model dim | 640 |
| Attention heads | 10 |
| KV heads | 4 |
| Head dim | 64 |
| MLP type | SwiGLU |
| Vocab size | 1024 |
| Tied embeddings | Yes |
| Logit softcap | 30.0 |

### Run Command (Mac)

```bash
RUN_ID=m3_depth_recurrence \
ITERATIONS=500 \
TRAIN_BATCH_TOKENS=8192 \
VAL_LOSS_EVERY=0 \
VAL_BATCH_SIZE=8192 \
MLX_MAX_MICROBATCH_TOKENS=4096 \
MAX_WALLCLOCK_SECONDS=0 \
python3 records/track_non_record_16mb/2026-03-18_M3_DepthRecurrence_SwiGLU/train_gpt_mlx.py
```

### Included Files

- `train_gpt_mlx.py` — training script with depth recurrence + SwiGLU
- `submission.json` — leaderboard metadata
- `train.log` — training log from Mac run
- `README.md` — this file
