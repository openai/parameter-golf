# Improved Tier1 Baseline

**val_bpb = 1.0751** (sliding window, quantized) | **~16.07 MB** | 4xA100 80GB PCIe

## Results

| Seed | val_bpb (quantized) | val_bpb (sliding window) | Artifact Size |
|------|---------------------|--------------------------|---------------|
| 1337 | 1.0915              | **1.0751**               | 16,074,000    |

Pre-quantization post-EMA val_bpb: **1.0799**

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

11L × 512d × 8H / 4KV, MLP 4×, fused MLP, RoPE (32 dims), layerwise LN scale, tied embeddings, logit softcap=20.0. Depth recurrence: 2 loops of layers 3–5. Parallel residuals from layer 7.

## Training

- **Optimizer**: MuonEq-R (row-normalized, Newton-Schulz 5 steps) + AdamW for embeddings/scalars
- **Steps**: 5392 / 20000 (stopped early by wallclock cap at ~3588s)
- **LR**: matrix=0.022, scalar=0.022, embed=0.6, tied_embed=0.03, head=0.008
- **EMA**: decay=0.9965
- **WD**: muon=0.095, adam=0.02, embed=0.085
- **Warmdown**: 85%
- **Batch**: 786432 tokens, grad_accum=2
- **Hardware**: 4xA100 80GB PCIe (development run, not leaderboard config)

## Quantization

Full-Hessian GPTQ with SDClip:
- int6: attention (c_q, c_k, c_v, proj) + MLP (fc, proj) — clip=12.85σ
- int8: token embeddings — clip=20.0σ
- Passthrough (float16): q_gain, attn_scale, mlp_scale, resid_mix, skip_gates, skip_weights
- Brotli compression

## Notes

- This is a development run on 4xA100 (not the 8xH100 leaderboard config)
- Training log will be added from a proper 8xH100 run in the future
- Total artifact size (16,074,000 bytes) exceeds the 16MB limit — needs optimization
- Based on `train_gpt_improved.py` with run ID `improved_tier1_2baseline`
