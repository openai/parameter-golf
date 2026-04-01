# Baseline 10L Int5-MLP on 1xH100

## Results

| Seed | val_bpb | val_loss | bytes_total |
|------|---------|----------|-------------|
| 42   | 1.2604  | 2.1281   | 15,815,650  |

## Setup

Reproduction of the #1 leaderboard submission (2026-03-20_10L_Int5MLP_MuonWD04_SWA50) on a single H100 GPU for 20 minutes of training (vs the standard 8xH100 10min).

## Architecture

- 10 transformer layers, dim=512, 8 heads (4 KV heads, GQA)
- Mixed int5/int6 quantization-aware training (MLP in int5, attention in int6)
- BigramHash embeddings: 10240 buckets, dim=128
- SmearGate token blending
- Tied input/output embeddings
- SWA (start_frac=0.4, every 50 steps)

## Training

- Muon optimizer with Newton-Schulz orthogonalization
- LR: embed=0.03, matrix=0.02, scalar=0.02
- Weight decay: 0.04, warmdown: 3000 iters
- Gradient clipping: 0.3
- Batch tokens: 786,432 (grad_accum=8 on 1 GPU)
- ~1700 steps in 20 minutes on 1xH100 SXM

## Evaluation

- Sliding window eval with stride=64
- int8+zlib roundtrip for final bpb measurement
- Serialized model size: 15,762,736 bytes (int6+zstd)

## Notes

This was trained on 1xH100 for 20 min rather than 8xH100 for 10 min. The reduced compute means fewer training steps (~1700 vs ~20000), so this result does not represent the full potential of this architecture.
