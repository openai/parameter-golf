## 11L EMA+GPTQ-lite with FA2 fallback (signalrush fork)

**val_bpb: 1.1349** (sliding window stride=64) | **16.19 MB** | 8xH100 SXM, 600s

### Architecture (fork of signalrush PR #414)
- 11 transformer layers, 512-dim, 8 heads, 4 KV heads (GQA)
- 3x MLP expansion, relu-squared, U-Net skip connections
- XSA on last 4 layers, Partial RoPE (16/64 dims)
- SmearGate, BigramHash (2048 buckets), Value Embedding (layers 9,10)
- EMA (0.997), GPTQ-lite int6, zstd-22 compression
- Sliding window eval (stride=64, seq_len=2048)
- FA2/SDPA fallback when FA3 (Hopper) not available

### Changes from signalrush
- Added FlashAttention 2 fallback (FA3 > FA2 > SDPA)
- Added Score-First TTT infrastructure (LoRA adapters, SGD adaptation)
- TTT not used in final eval (too slow without torch.compile support)

### Results (8xH100 SXM, seed 1337)

| Metric | Value |
|--------|-------|
| Training steps | 5440 (110ms/step) |
| val_loss (train) | 1.9426 |
| val_bpb (train) | 1.1505 |
| Post-EMA val_bpb | 1.1499 |
| Int6 roundtrip val_bpb | 1.1585 |
| **Sliding window val_bpb** | **1.1349** |
| Model size (int6+zstd) | 16,114,377 bytes |
| Code size | 77,262 bytes |
| Total | 16,191,639 bytes |

### Notes
- Step count limited to 5440 (vs signalrush's ~7100) because FA2 is ~2x slower than FA3 on H100
- TTT infrastructure is implemented but eval is too slow without torch.compile (LoRA breaks fullgraph)
- Submission is ~192KB over the 16MB limit — needs code size reduction for official submission
