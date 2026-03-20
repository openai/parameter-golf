# Seq4096 Training + 10L + Sliding Window Eval

## Result

| Seed | val_bpb | Artifact |
|------|---------|----------|
| 1337 | 1.1900 | 15,115,793 B |
| 42   | 1.1908 | 15,128,724 B |
| 7    | 1.1888 | 15,154,068 B |
| **mean** | **1.1899** | |
| **std**  | **0.0008** | |

## Key change

Train on **4096-token sequences** instead of 1024. The model sees longer
dependencies during training, improving predictions. All other settings
follow the sliding window SOTA baseline (10 layers, eval_stride=64,
FP16 embeddings, Muon + weight decay, Overtone init).

## Hyperparameters

```
num_layers     = 10
train_seq_len  = 4096
eval_stride    = 64
warmdown_iters = 3600
matrix_lr      = 0.04  (default)
muon_momentum  = 0.95  (default)
```

## Hardware

Modal 8×H100 SXM, `torchrun --standalone --nproc_per_node=8`
Training capped at 600 seconds (`MAX_WALLCLOCK_SECONDS=600`).
