# MLX Smoke Test on MacBook Air M3

**Non-record submission (preliminary local test)**
**Author:** Aleksandr Gaun ([@Rhoahndur](https://github.com/Rhoahndur))

---

## Summary

Baseline `train_gpt_mlx.py` smoke test run on a MacBook Air M3 (2024) using Apple MLX framework. This is a preliminary local development run to validate the training pipeline before scaling to 8xH100 with compute credits.

| Metric | Value |
|--------|-------|
| val_bpb | 2.3029 |
| val_loss | 3.8884 |
| Steps | 200 |
| ms/step | ~1024 |
| Artifact size | 9.64 MB (int8+zlib) |
| Hardware | MacBook Air M3, 16GB unified memory |
| Framework | MLX (Apple Silicon) |

## Approach

Unmodified baseline `train_gpt_mlx.py` with default hyperparameters:

- 9 transformer layers, 512 model dim
- 8 attention heads, 4 KV heads (GQA)
- 2x MLP expansion, tied embeddings
- Vocab size 1024, sequence length 1024
- 200 steps (wallclock-limited on consumer hardware)

## Why this score is low

The val_bpb of 2.3029 is well above the baseline (1.2244 on 8xH100 with ~5000+ steps) because:

1. **Only 200 steps** vs ~5000+ on H100 (wallclock-limited by consumer hardware)
2. **Single Apple M3 GPU** vs 8xH100 SXM (orders of magnitude less compute)
3. **No modifications** to the baseline training script

## Next steps

This submission establishes the local development workflow. Plan to:

1. Request compute credits to train on 8xH100 infrastructure
2. Experiment with architectural modifications and optimization techniques
3. Submit competitive results as a follow-up

## Reproducing

```bash
cd /path/to/parameter-golf
python train_gpt_mlx.py
```

No special environment variables needed -- uses all defaults.
