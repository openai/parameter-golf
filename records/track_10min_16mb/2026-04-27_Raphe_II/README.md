# Raphe_II

| seed | val_bpb | bytes |
|------|---------|-------|
| 4    | 0.85020690 | 15,995,307 |
| 42   | 0.84994524 | 15,999,255 |
| 300  | 0.85039293 | 15,999,412 |
| **mean** | **0.85018169** | — |

```
torchrun --standalone --nproc_per_node=8 train_gpt.py
```
