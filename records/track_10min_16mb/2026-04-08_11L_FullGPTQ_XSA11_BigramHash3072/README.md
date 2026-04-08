# Non-record: 11L FullGPTQ + XSA-all + BigramHash 3072×112

**Track**: 10min_16mb | **Author**: AVINASH0052 | **Date**: 2026-04-08

## Results

| Seed | val_bpb | val_loss | artifact_bytes |
|------|---------|----------|----------------|
| 1337 | **1.11564047** | 1.88370722 | 15,832,508 |

- Post-EMA (before GPTQ): val_bpb 1.1350
- After int6 GPTQ + sliding-window exact eval (stride=64): **val_bpb 1.11564047**
- Steps: 6891 | Avg step time: 87.08ms | FA3: True
- Hardware: 8×H100 80GB SXM

## Architecture

| Component | Detail |
|-----------|--------|
| Layers / dim | 11L, 512d |
| Attention heads | 8H query / 4KV (GQA) |
| MLP | 3× expansion (1536 hidden), LeakyReLU(0.5)² |
| XSA | All 11 layers — drops self-value projection |
| Hash Embedding | BigramHash 3072×112 |
| Pos Encoding | Partial RoPE (16 of 64 head dims) |
| Skip Connections | U-Net style: layers 0↔10, 1↔9, 2↔8 |
| Value Embed | VE128 re-injection at layers 9, 10 |
| LN Scaling | 1/√(L+1) per layer — deeper layers see smaller-norm inputs |
| SmearGate | Learned position mixing gate on embedding |
| Logit softcap | 30.0 |
| Tied embeddings | Token embedding = LM head (transposed) |
| Total params | ~27M |

## Training

| Setting | Value |
|---------|-------|
| Optimizer | Parallel Muon (8-GPU) + AdamW for embeddings |
| Parameter Banking | 4 contiguous 3D banks (qo, kv, mlp_up, mlp_down) |
| Batch | 786,432 tokens/step, seq_len=2048 |
| EMA | α=0.997, tight SWA every 50 steps when lr_scale < 0.2 |
| Late QAT | STE fake-quant activates when lr_scale < 0.15 (step 6299) |
| Warmdown | 4000 iters (wallclock-adaptive) |
| Grad clip | 0.3 |
| Max wallclock | 600s (10 min) |

## Post-Training Quantization (GPTQ)

| Step | Detail |
|------|--------|
| Calibration | AR self-generated: 64 seqs × 2048 tokens, temp=0.8 |
| Hessian | Collected across all 68 quantizable layers |
| Method | Full Hessian GPTQ int6: Cholesky + column reordering + block error compensation |
| Clip search | 5 percentiles tried per weight matrix, best MSE wins |
| Pruning | Selective ±1 pruning (model fit in budget — no pruning applied) |
| Compression | LZMA preset=9 |
| Serialized model | 15,750,244 bytes (int6 + LZMA) |
| Code | 82,264 bytes |
| **Total artifact** | **15,832,508 bytes** |

## How to Run

### Leaderboard run (8×H100 SXM)
```bash
pip install flash_attn_3 --no-deps --find-links \
    https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/
cd records/track_10min_16mb/2026-04-08_11L_FullGPTQ_XSA11_BigramHash3072
SEED=1337 bash run_leaderboard_8xh100.sh
```

### Smoke test (1 GPU, ~5 min)
```bash
bash run_smoke_1gpu.sh
```

## PR

[openai/parameter-golf#1473](https://github.com/openai/parameter-golf/pull/1473)
