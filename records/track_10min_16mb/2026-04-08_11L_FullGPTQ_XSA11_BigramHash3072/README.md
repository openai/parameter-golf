# 11L AR Self-Gen GPTQ + XSA-all + BigramHash 3072×112

## Architecture
- **11 transformer layers**, 512d, 8 heads / 4 KV heads (GQA)
- **MLP**: 3× expansion (1536 hidden), LeakyReLU(0.5)² activation
- **BigramHash 3072×112**: Bigram frequency hashing for additional token-pair context
- **XSA (Cross-Sequence Attention)**: All 11 layers — removes self-value projection component to improve generalization
- **SmearGate**: Learned position mixing gate for smoothing embeddings
- **Partial RoPE**: 16 of 64 head dims — leaves most of head for learned position
- **LN Scale**: 1/√(layer+1) — deeper layers see smaller-norm inputs
- **Value Embedding VE128**: Reinjects token identity into attention values at layers 9,10
- **U-Net skip connections**: Encoder (5L) → Decoder (6L) with learned skip weights
- **Logit softcap**: 30.0
- **Tied embeddings**: Token embedding = LM head weight (transposed)

## Training
- **Parameter Banking**: 4 contiguous 3D banks (qo, kv, mlp_up, mlp_down) for batched Muon
- **Parallel Muon**: Async reduce-scatter → local Newton-Schulz-5 → all-gather (overlaps comm with compute)
- **AdamW** for embeddings and scalar/control params
- **EMA(0.997)** + Tight SWA (every 50 steps when lr_scale < 0.2)
- **Late QAT**: STE fake-quantization activates when LR scale < 0.15
- **Warmdown**: 4000 iterations (wallclock-adaptive)
- **Grad clip**: 0.3
- **Batch**: 786,432 tokens/step, seq_len=2048

## Quantization (Post-Training)
- **Full Hessian GPTQ**: Cholesky + column reordering + block error compensation
- **AR Self-Generated Calibration**: Model generates its own calibration data (64 seqs × 2048 tokens, temp=0.8)
- **Percentile Search**: 5 clip percentiles tried per weight matrix, best MSE wins
- **Selective ±1 Pruning**: Binary search to zero out quantized ±1 values with smallest reconstruction error
- **LZMA preset=9** compression

## Evaluation
- **Sliding window** stride=64 for final BPB scoring
- **Flash Attention 3** on Hopper if available, SDPA fallback otherwise

## How to run

### Smoke test (1 GPU)
```bash
bash run_smoke_1gpu.sh
```

### Leaderboard run (8×H100 SXM)
```bash
bash run_leaderboard_8xh100.sh
```
