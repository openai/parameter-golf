# Non-record: LeakyReLU(0.9)² Activation Sweep

**Local validation on RTX 5060 (1 shard, 1024 seq_len)**

## Motivation

PR #466 (LeakyReLU_LegalTTT_ParallelMuon) introduced LeakyReLU(0.5)² as a one-line activation change yielding -0.003 BPB over relu². The ablation in PR #493 confirmed this gain.

However, a community sweep over negative slopes suggests that **slope=0.9 may be significantly better than 0.5** — preserving 81% of negative pre-activation signal (0.9²) vs 25% (0.5²) while still producing non-negative outputs via squaring. This submission investigates the optimal slope for LeakyReLU² activations in the sub-16MB regime.

## Method

Starting from the PR #466 stack (11L/512d/8H/4KV, 3x MLP, XSA last 4, EMA, BigramHash, SmearGate, Partial RoPE 16/64, GPTQ-lite int6, sliding window stride=64), the only change is:

```python
# PR #466 (slope=0.5)
x = F.leaky_relu(self.fc(x), negative_slope=0.5)

# This submission (slope=0.9)
x = F.leaky_relu(self.fc(x), negative_slope=0.9)
```

## Local Results (RTX 5060, 1 shard, preliminary)

Local validation on a single RTX 5060 8GB with 1 training shard (limited data regime). These results are directional only — full 80-shard 8xH100 validation pending compute credit allocation.

| Config | val_bpb (1 shard) | Notes |
|--------|-------------------|-------|
| Baseline (9L, relu²) | 1.4395 | INT8+zlib roundtrip |
| PR #466 stack + LeakyReLU(0.9)² | TBD | INT6+zstd roundtrip |

Note: The 27M-parameter model (11L/3xMLP) underperforms the 17M baseline (9L/2xMLP) in the limited-data regime (1 shard, 8K batch tokens). This is expected — larger models require more diverse training data and larger batch sizes to realize their capacity advantage. Full validation on 8xH100 with 80 shards is needed.

## Planned Experiments (pending compute)

1. **Slope sweep**: 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95 — systematic ablation to find optimal slope
2. **3-seed validation**: Seeds 1337, 42, 2025 for statistical significance
3. **Ablation table**: Isolate LeakyReLU(0.9)² contribution vs relu² on the full PR #466 stack

## Hardware Compatibility

FA3 → PyTorch SDPA fallback for non-H100 GPUs:

```python
try:
    from flash_attn_interface import flash_attn_func as flash_attn_3_func
    _HAS_FA3 = True
except ImportError:
    _HAS_FA3 = False
```

This enables local development on consumer GPUs while maintaining H100 compatibility for final submission.

## Files

- `train_gpt.py` — Modified training script (based on PR #466 stack)
- `README.md` — This file
