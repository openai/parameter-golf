# Improved Tier1 Baseline

**val_bpb = 1.0970** (quantized, sliding window) | **~16.07 MB** | 8xH100 SXM

## Results

| Seed | Pre-Quant BPB (SW) | Quantized BPB (SW) | Steps | Train Time | Artifact |
|------|--------------------|--------------------|-------|------------|----------|
| 42   | 1.0857             | **1.0970**         | 4575  | 588.9s     | 16,072,361 |
| 999  | 1.0851             | —                  | 4614  | 588.1s     | 16,019,390 |

All eval uses sliding window (stride=64, seq_len=2048). No TTT.
Note: Seed 999 quantized eval did not complete (log truncated).

## Key Techniques

1. **SP8192 Tokenizer** — 8192-vocab SentencePiece BPE
2. **3-Layer Depth Recurrence** (layers 3–5, activated at frac=0.35) — encoder [0,1,2,3,4,5,3,4] decoder [5,3,4,5,6,7,8,9,10]
3. **Parallel Residuals** (layers 7+) — GPT-J style, attention and MLP read from same input
4. **Skip Gates** — sigmoid-gated U-Net skip connections
5. **QK-Gain 5.25** — learnable per-head query scaling
6. **GPTQ SDClip** — int6 for attention/MLP matrices (k=12.85), int8 for embeddings (k=20.0)
7. **Brotli Compression** — model serialization
8. **Sliding Window Eval** — stride=64, seq_len=2048

## Architecture

11L × 512d × 8H / 4KV, MLP 4×, fused MLP, RoPE (32 dims), layerwise LN scale, tied embeddings, logit softcap=20.0. Depth recurrence: 2 loops of layers 3–5. Parallel residuals from layer 7. 35.9M parameters.

## Training

- **Optimizer**: MuonEq-R (row-normalized, Newton-Schulz 5 steps) + AdamW for embeddings/scalars
- **Steps**: ~4575–4614 / 20000 (stopped early by 588s wallclock cap)
- **LR**: matrix=0.022, scalar=0.022, embed=0.6, tied_embed=0.03, head=0.008
- **EMA**: decay=0.9965
- **WD**: muon=0.095, adam=0.02, embed=0.085
- **Warmdown**: 85%
- **Batch**: 786,432 tokens, grad_accum=1
- **Grad Clip**: 0.3
- **Hardware**: 8xH100 SXM
- **Throughput**: ~6.1–7.8M tok/s (slower after loop activation at step ~2045)
- **Peak Memory**: 39,046 MiB allocated per GPU

## Quantization

Full-Hessian GPTQ with SDClip (64 calibration batches, hessian_clip_lambda=0.3):
- int6: attention (c_q, c_k, c_v, proj) + MLP (fc, proj) — clip=12.85σ
- int8: token embeddings — clip=20.0σ
- Passthrough (float16): q_gain, attn_scale, mlp_scale, resid_mix, skip_gates, skip_weights
- Brotli compression

## Attribution

- **SP8192 + GPTQ Embeddings + SDClip + MuonEq-R + Depth Recurrence** — @clarkkev (PR #1394)
- **3-Layer Depth Recurrence** — @dexhunter (PR #1331, #1437)
- **Parallel Residuals** — @Robby955 (PR #1412), @msisovic (PR #1204)
- **Hyperparameter Tuning** (WD=0.095, MLR=0.022, EMA=0.9965) — @X-Abhishek-X (PR #1445)
- **Full-Hessian GPTQ + XSA** — @abaybektursun (PR #1019)
- **LeakyReLU² + Score-First TTT precedent** — @abaybektursun (PR #549)

## Notes

- No TTT — eval is sliding window only
- Both seeds exceed 16,000,000 byte artifact limit (code size optimization needed)
- Seed 42 code: 78,269 bytes; Seed 999 code: 23,801 bytes (different code snapshots)
- Baseline run for technique validation, not a leaderboard submission
