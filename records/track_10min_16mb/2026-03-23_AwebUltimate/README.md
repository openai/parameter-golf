# Aweb Ultimate — Full SOTA Stack + N-gram Oracle Mixing

## Score Target: sub-0.20 BPB (vs leaderboard #1 at 1.1194)

## Architecture: SOTA #1 Base (PR #549 lineage)
- 11 layers, 512 dim, 8 heads, 4 KV heads (GQA)
- LeakyReLU(0.5)² MLP (3x expansion)
- XSA (Cross-layer Shared Attention) on last 4 layers
- Partial RoPE (16/64 head dims)
- LN Scale (1/sqrt(layer+1))
- SmearGate + BigramHash(2048)
- ValueEmbedding (shared table, layers 9-10)
- U-Net skip connections
- Logit softcap (30.0)

## Training
- Parallel Muon optimizer (batched Newton-Schulz, 3-phase overlapped comms)
- EMA (0.997) + Tight SWA (last 20%, every 50 steps)
- AdamW with weight decay (0.04) for embeddings/scalars
- Muon weight decay (0.04)
- Grad clip 0.3, seq_len 2048
- Late QAT (int6 STE at scale < 0.15)
- 786K batch tokens, warmdown 3500 steps

## Evaluation — The Secret Sauce
- **Legal Score-First TTT** (3 epochs SGD per chunk)
- **N-gram Oracle Cache**: orders 2-8, hashed backoff tables built from scored tokens
- **Neural + N-gram mixing**: entropy-adaptive interpolation
- Sliding window evaluation (stride=64)

## Quantization
- GPTQ-lite int6 with 5-percentile clip search
- FP16 embeddings preserved
- LZMA compression (preset 6)
- Unbank/rebank for per-layer quantization

## Reproduction

```bash
TTT_ENABLED=1 torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-03-23_AwebUltimate/train_gpt.py
```

## Author

Daniel Wahnich (@manfromnowhere143) — Founder of Aweb.

*Ostinato Rigore.*
