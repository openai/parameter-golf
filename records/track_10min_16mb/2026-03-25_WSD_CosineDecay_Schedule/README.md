# WSD Cosine Decay Schedule

**val_bpb: TBD (8xH100)** — Preliminary 1-GPU result: 1.2824 BPB

## Key Change

Replace the default linear warmdown LR schedule with a **Warmup-Stable-Decay (WSD)** cosine schedule:

| Phase | Fraction | LR behavior |
|-------|----------|-------------|
| Warmup | 0-5% of steps | Linear 0 → peak |
| Stable | 5-80% of steps | Constant at peak LR |
| Decay | 80-100% of steps | Cosine decay → 0 |

The original schedule computes warmdown based on `warmdown_iters` and remaining wallclock time, which can cause LR to start decaying from very early in training (especially with fewer steps). WSD ensures the model trains at peak LR for the majority of the run.

## Base Techniques (inherited from SOTA)

- 10 layers, 512-dim, MLP 3x expansion
- SmearGate + BigramHash(10240)
- Mixed int5 (MLP) / int6 (attention) quantization
- SWA (start_frac=0.4, every=50 steps)
- Orthogonal init + Muon optimizer (WD=0.04)
- zstd-22 compression
- Sliding window eval (stride=64)

## Preliminary Results (1 GPU, seed=42)

| Config | val_bpb | artifact_bytes |
|--------|---------|---------------|
| 1 GPU, 600s, ~877 steps | 1.2824 | 15,767,236 |

8xH100 3-seed results pending.

## Run Command

```bash
# Single GPU
python train_gpt.py

# 8xH100 (competition setting)
torchrun --standalone --nproc_per_node=8 train_gpt.py

# With specific seed
SEED=42 torchrun --standalone --nproc_per_node=8 train_gpt.py
```
