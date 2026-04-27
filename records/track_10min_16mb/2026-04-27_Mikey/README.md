# Mikey

| seed | val_bpb (sliding) | bytes |
|------|-------------------|-------|
| 42   | 0.86503709        | 15,639,737 |
| 300  | 0.86698133        | 15,594,375 |
| 444  | 0.86441066        | 15,653,512 |
| **mean** | **0.86547636** | — |

```
torchrun --standalone --nproc_per_node=8 train_gpt.py
```
