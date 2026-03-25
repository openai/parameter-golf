# PR #414 Stack + Legal Score-First TTT

**val_bpb: 1.1408** (8xH100 SXM, seed=1337, legal score-first TTT)

## Summary

Legal chunk-based score-first TTT on the PR #414 consensus stack. Each validation chunk is scored first (under inference_mode), then the model trains on already-scored tokens. Never trains on tokens before scoring them.

## Key Addition: Legal Score-First TTT

After int6 quantization, validation data is processed in 32K-token chunks:
1. **Score** chunk with sliding-window eval (inference_mode)
2. **Train** on scored chunk for 3 epochs (SGD, cosine LR)
3. Advance to next chunk — never training before scoring

- SGD optimizer, base LR=0.002, momentum=0.9
- Cosine LR decay across chunks
- 3 epochs per chunk, 32768 tokens/chunk, 1893 chunks total
- DDP gradient sync (all_reduce AVG)
- Gradient clipping: 1.0
- Total eval time: ~617s on 8xH100 (SDPA backend)

## Architecture (PR #414 stack)

- 11 layers, 512d, 8H, 4KV (GQA)
- 3x MLP with relu²
- SmearGate + BigramHash (2048 buckets)
- XSA on last 4 layers
- Partial RoPE (16/64 dims), LN Scale
- VE128 on layers 9-10
- EMA(0.997) + Tight SWA(50)
- GPTQ-lite int6 + zstd-22
- Late QAT @ threshold 0.15
- OrthoInit + muP-scaled output projections

## Training

- Muon: lr=0.025, momentum=0.99 (warmup 0.92->0.99 over 1500 steps), WD=0.04
- AdamW: embed_lr=0.035, scalar_lr=0.025, WD=0.04
- Batch: 786,432 tokens/step, seq_len=2048
- Warmdown: 3500 iterations
- Gradient clip: 0.3

## Results

| Stage | val_loss | val_bpb |
|-------|----------|---------|
| Post-EMA (float) | 1.9433 | 1.1509 |
| Post-int6 roundtrip | 1.9570 | 1.1590 |
| **Legal TTT (score-first)** | **1.9262** | **1.1408** |

## Run Command

```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Credits

- Base model and training recipe: PR #414 by @signalrush
- Legal TTT protocol: PR #549 by @a]exkarp
- TTT technique: PR #518 by @sofiabod
- SDPA fallback for non-FA3 environments
