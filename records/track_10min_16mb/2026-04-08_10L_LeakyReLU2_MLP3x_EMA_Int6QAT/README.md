# 10L LeakyReLU² MLP3x + EMA + Int6 QAT

**Track:** `track_10min_16mb`  
**Date:** 2026-04-08  
**Status:** Implementation complete, pending GPU run

## Architecture

| Component | Value |
|-----------|-------|
| Layers | 10 |
| Model dim | 512 |
| Heads / KV heads | 8 / 4 (GQA) |
| MLP hidden | 1536 (3x) |
| Activation | LeakyReLU(0.5)² |
| Vocab | 1024 (SentencePiece BPE, tied) |
| Positional | RoPE (base=10000) |
| Softcap | 30.0 |
| Skip connections | U-Net (5 encoder → 5 decoder) |
| Estimated params | ~24.1M |

## Training Techniques

- **Optimizer:** Muon (matrices) + Adam (embed/scalars), decoupled WD=0.04
- **EMA:** Polyak averaging, decay=0.997, start at step 100
- **QAT:** Int6 per-row fake quantization (STE), enabled at 80% of training
- **Warmdown:** 3500 steps (LR → 0, wallclock-aware)
- **Warmup:** 20 steps (compile priming + state reset)
- **Batch:** 524K tokens, seq_len=1024

## Serialization

- **Quantization:** Int6 per-row for large matrices, int8→fp16 for small tensors
- **Compression:** LZMA preset=9 (stdlib, no extra deps)
- **Estimated artifact:** ~14.2 MB (well under 16 MB limit)

## Evaluation

- Standard non-overlapping eval during training (every 1000 steps)
- **Sliding window eval** (stride=256) on EMA weights post int6+LZMA roundtrip for final BPB

## Key Differences from Baseline

1. **10 layers** (vs 9) – more capacity within budget
2. **LeakyReLU(0.5)²** (vs ReLU²) – non-zero gradients for negative inputs
3. **MLP 3x** (vs 2x) – wider MLP for better expressiveness
4. **EMA** – smoothed weights reduce noise in final model
5. **Int6 QAT** – quantization-aware training for better int6 fidelity
6. **LZMA-9** (vs zlib) – better compression ratio
7. **Sliding window eval** – lower BPB from better context utilization
8. **Muon weight decay** – decoupled regularization

## How to Run

```bash
# Smoke test (1 GPU, ~2 min)
bash run_smoke_1gpu.sh

# Full leaderboard run (8xH100, 10 min)
bash run_leaderboard_8xh100.sh
```

## Results

| Seed | val_bpb | Notes |
|------|---------|-------|
| 1337 | TBD | Pending run |
