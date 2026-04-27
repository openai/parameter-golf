# Raphe

| seed | val_bpb (sliding) | bytes |
|------|-------------------|-------|
| 42   | 0.87280270        | 13,492,522 |
| 300  | 0.87170361        | 13,475,412 |
| 444  | 0.87168317        | 13,487,656 |
| **mean** | **0.87206316** | — |

```
torchrun --standalone --nproc_per_node=8 train_gpt_8xgpu.py
```
