# Record: 11L XSA-all + Full GPTQ (Budget-Legal) + Parallel Muon + Selective Pruning

**val_bpb: 1.1178** (3-seed mean, std 0.0001) | **15.95 MB** max artifact | 8xH100 SXM, ~596s total compute

## Update (2026-03-26)

This PR was updated to fix a GPTQ budget violation identified in [issue #677](https://github.com/openai/parameter-golf/issues/677). The previous version trained for the full 600s, then ran GPTQ calibration for ~46s on top, exceeding the 600s artifact-production budget. The fix reserves 14s from the training budget for GPTQ calibration (`gptq_reserve_ms = 14000.0`), ensuring total compute (training ~586s + GPTQ ~10s = ~596s) stays within the 600s limit. All results below use the fixed code with fresh 3-seed runs.

## Results (3 seeds, 8xH100 SXM)

| Seed | Steps | ms/step | Sliding BPB (s64) | val_loss | Artifact | Train Time | GPTQ Time | Total |
|------|-------|---------|--------------------|----------|----------|------------|-----------|-------|
| 1337 | 6,674 | ~88 | **1.1177** | 1.8871 | 15,929,433 bytes | 586,128ms | 9,786ms | 595,915ms |
| 42 | 6,732 | ~87 | 1.1179 | 1.8875 | 15,949,353 bytes | 586,050ms | 9,792ms | 595,842ms |
| 7 | 6,731 | ~87 | 1.1179 | 1.8875 | 15,946,145 bytes | 586,066ms | 9,823ms | 595,889ms |

**Mean: 1.1178 | Std: 0.0001**

## Key Techniques

### XSA on All 11 Layers
Standard practice applies Exclusive Self-Attention to only the last 4 layers. Applying to all 11 forces cross-position information mixing from layer 0, improving representation quality. Zero new parameters — just a config change. -0.0016 BPB vs XSA-last-4 in ablation.

### Full Hessian GPTQ (Budget-Legal)
- 64-batch GPU Hessian calibration from training data
- Column-wise int6 quantization with Cholesky error compensation, block size 128, percdamp 0.01
- QAT STE aligned to export quantizer using row-maximum (amax) clipping with [-32, 31] range
- **Budget reservation:** `gptq_reserve_ms = 14000.0` — training stops ~14s early so GPTQ calibration fits within 600s
- Log verification: `gptq:budget_check train:586128ms + gptq:9786ms = 595915ms (budget:600000ms)`

### Parallel Muon Optimizer with Parameter Banking
- Weight matrices stored in contiguous parameter banks (qo_bank, kv_bank, mlp_up_bank, mlp_down_bank)
- 3-phase overlapped optimizer step: async reduce-scatter -> batched Newton-Schulz orthogonalization -> async all-gather
- Eliminates DDP double-communication overhead, achieving ~87ms/step (~6,700 steps in 586s)

### Selective Magnitude Pruning
Post-GPTQ, sort quantized values at +/-1 by reconstruction error (scale^2), zero least-impactful first until artifact fits target. Binary search for exact target size.

### LZMA Compression
LZMA preset 6 replacing zstd-22. Better compression ratio on int6 quantized weights.

## Architecture

- 11 transformer layers, dim=512, 8 heads, 4 KV heads (GQA)
- 3x MLP expansion (hidden=1536) with **LeakyReLU(0.5)^2** activation
- **XSA on all 11 layers** (Exclusive Self-Attention)
- Partial RoPE (16/64 dims) + NTK-aware scaling
- LN Scale Factor 1/sqrt(layer_idx+1)
- U-Net skip connections (5 encoder, 6 decoder)
- SmearGate temporal gating
- BigramHash (2048 buckets, 128-dim)
- Shared Value Embedding (dim=128, layers 9-10)
- FlashAttention 3 (Hopper native kernels)
- Orthogonal init, logit softcap 30, tied embeddings

## Training

- Parallel Muon optimizer (matrices): lr=0.025, momentum=0.99, WD=0.04, 5 Newton-Schulz steps
- AdamW (embeddings): lr=0.035, (scalars): lr=0.025, WD=0.04
- Gradient clip: 0.3
- Batch: 786,432 tokens/step, seq_len=2048
- Warmdown: 3,500 iters (wallclock-based)
- EMA (decay=0.997) + Tight SWA (every 50 steps, scale<0.2)
- Late QAT: STE int6 fake-quantization when LR scale<0.15

## Quantization & Compression

- Full GPTQ with 64-batch GPU Hessian calibration, block_size=128, percdamp=0.01
- Int6 per-row with amax clipping, range [-32, 31]
- Selective magnitude pruning (target 15.9MB)
- Small tensors + tok_emb.weight in fp16
- LZMA preset 6 compression

## Compliance

- [x] 3 seeds, all total compute <= 600s on 8xH100 SXM (verified: max 595,915ms)
- [x] GPTQ calibration WITHIN training budget (14s reserved, verified via `gptq:budget_check`)
- [x] All artifacts <= 16,000,000 bytes (max: 15,949,353)
- [x] No TTT on validation data
- [x] No training data accessed during evaluation
- [x] No network calls during evaluation
- [x] Sliding window eval stride=64, consistent across seeds (std=0.0001)

## Run Command

```bash
SEED=1337 TARGET_MB=15.9 torchrun --standalone --nproc_per_node=8 train_gpt.py
```
