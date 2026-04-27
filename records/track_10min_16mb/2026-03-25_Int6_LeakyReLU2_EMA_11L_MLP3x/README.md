# Int6 GPTQ-lite + LeakyReLU(0.5)^2 + EMA + 11L MLP3x

## Improvements over baseline

| Change | Expected BPB Delta |
|--------|-------------------|
| 11 layers (vs 9) | -0.005 |
| MLP 3x (vs 2x) | -0.005 |
| LeakyReLU(0.5)^2 | -0.003 |
| Int6 GPTQ-lite + zstd-22 | -0.005 |
| Late QAT (STE, LR < 0.15) | -0.003 |
| EMA (decay 0.997) | -0.005 |
| Sliding window eval (stride 64) | -0.005 |

## Run Command

```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Single GPU test:
```bash
torchrun --standalone --nproc_per_node=1 train_gpt.py
```
