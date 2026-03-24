# Score-First TTT with LoRA on 11L EMA+GPTQ-lite

**val_bpb: 1.1349** (sliding window stride=64) | 8xH100 SXM, 600s training

## Overview

This submission introduces **Score-First Test-Time Training (TTT)** infrastructure on top of signalrush's #1 architecture (PR #414, 1.1228 BPB). The key idea: inject LoRA adapters at eval time and do per-window SGD adaptation, scoring each token BEFORE adapting on it (legal per challenge rules).

This is a non-record submission demonstrating the TTT approach. The BPB does not beat SOTA due to two infrastructure limitations (see Notes below).

## Architecture (fork of signalrush PR #414)

- 11 transformer layers, 512-dim, 8 heads, 4 KV heads (GQA)
- 3x MLP expansion, relu-squared, U-Net skip connections
- XSA on last 4 layers, Partial RoPE (16/64 dims)
- SmearGate, BigramHash (2048 buckets), Value Embedding (layers 9,10)
- EMA (0.997), GPTQ-lite int6, zstd-22 compression
- Sliding window eval (stride=64, seq_len=2048)

## Key Innovation: Score-First TTT

The TTT implementation adds:

- **LoRAAdapter class**: Low-rank adapters (rank 4-8) wrapping attention Q/K/V projections
- **apply_lora_to_model()**: Injects LoRA into selected layers (all, last4, or last6)
- **eval_val_sliding_ttt()**: Modified sliding window that does SGD per window:
  1. Forward pass — score novel tokens (these scores are FINAL)
  2. Compute loss on full window (context + novel tokens)
  3. Backward + SGD update on LoRA params only
- **Score-first guarantee**: Token t is scored BEFORE any adaptation on tokens <= t

Configurable via environment variables:

- `TTT_ENABLED=1` (on/off)
- `TTT_LR=5e-4` (SGD learning rate)
- `TTT_LORA_RANK=4` (LoRA rank)
- `TTT_TARGET_LAYERS=all` (all, last4, last6)
- `TTT_TEMP=1.0` (temperature scaling)

## Changes from signalrush

- Added FlashAttention 2 fallback (FA3 > FA2 > SDPA) for non-Hopper environments
- Added Score-First TTT infrastructure (LoRA adapters, SGD adaptation per eval window)
- TTT not used in final reported eval (see Notes)

## Results (8xH100 SXM, seed 1337)

| Metric | Value |
|--------|-------|
| Training steps | 5440 (110ms/step) |
| val_loss (train end) | 1.9426 |
| val_bpb (train end) | 1.1505 |
| Post-EMA val_bpb | 1.1499 |
| Int6 roundtrip val_bpb | 1.1585 |
| **Sliding window val_bpb** | **1.1349** |
| Model size (int6+zstd) | 16,114,377 bytes |
| Code size | 77,262 bytes |
| Total | 16,191,639 bytes |

## Notes

Two limitations prevented beating SOTA:

1. **No FA3 (FlashAttention 3)**: FA3 requires a special Hopper build (`flash_attn_interface`). Without it, step time is ~110ms vs signalrush's ~48ms, yielding only 5440 steps vs ~7100. This accounts for most of the BPB gap.

2. **TTT eval too slow**: The uncompiled model (LoRA injection breaks `torch.compile(fullgraph=True)`) makes TTT eval ~10x slower than compiled sliding window. TTT eval did not complete within the 10-minute budget. Possible fixes: compile before LoRA injection, batch windows, or use larger stride.

## Run Command

```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

With TTT enabled:

```bash
TTT_ENABLED=1 TTT_LR=5e-4 TTT_LORA_RANK=4 torchrun --standalone --nproc_per_node=8 train_gpt.py
```
