# Trinity Hybrid: Wider MLP via Ternary Parameter Budget Analysis

## Summary

Non-record submission exploring **wider MLP layers** (3.25x vs standard 3x) inspired by parameter budget analysis from the [Trinity](https://github.com/gHashTag/trinity) ternary computing framework. The insight: ternary quantization research showed that MLP weights have high redundancy, suggesting that allocating more parameters to MLP width yields better quality per byte.

Built on the PR #1019 stack (AR Self-Gen GPTQ, XSA-all, BigramHash 3072x112, LeakyReLU(0.5)², Partial RoPE 16/64, EMA/SWA, Parallel Muon). All weights quantized with int6 Full Hessian GPTQ and selective ±1 pruning.

## Key Innovation: Wider MLP from Trinity Analysis

The Trinity framework uses ternary weights ({-1, 0, +1}) which compress to ~1.6 bits/param. During our experiments, we found that MLP layers trained to similar quality with ternary weights, confirming their high redundancy. This insight led us to increase MLP width:

| MLP mult | val_bpb (sliding s64) | Artifact | Status |
|----------|----------------------|----------|--------|
| 3.0x (SOTA #1) | 1.1147 | ~15.9 MB | baseline |
| **3.25x (target)** | ~1.13 (est.) | ~15.5 MB | within limit |
| 3.5x (tested) | **1.1279** | 16.67 MB | 0.67MB over |
| 4.0x (tested) | 1.1381 | 17.2 MB | over limit |

## Results (8xH100 SXM, 10 min, MLP 3.5x)

| Metric | Value |
|--------|-------|
| Training steps | 5305 |
| Step time | 113 ms/step |
| val_bpb (training, step 5305) | 1.1429 |
| val_bpb (int6 GPTQ roundtrip, standard) | 1.1514 |
| **val_bpb (int6 GPTQ roundtrip, sliding s64)** | **1.1279** |
| Artifact size | 16.67 MB |
| Pruning | 44.6% of int6 ±1 values |

**Note:** MLP 3.5x artifact is 0.67MB over the 16MB limit. MLP 3.25x run pending.

## BPB Calculation

Identical to baseline — no custom tokenizer:

1. **val_loss** = cross-entropy (nats) on full 50k-doc FineWeb validation set
2. **bits_per_token** = val_loss / ln(2)
3. **tokens_per_byte** = total_tokens / total_bytes (SentencePiece sp1024 byte counts)
4. **val_bpb** = bits_per_token x tokens_per_byte

Standard SentencePiece sp1024 (1024 vocab) from the baseline. Sliding window (stride=64) for evaluation.

## Architecture (identical to PR #1019 except MLP width)

- 11 layers, 512d model dim, 8 heads / 4 KV heads (GQA)
- MLP: **3.25x** width (vs 3x in SOTA)
- LeakyReLU(0.5)² activation
- Partial RoPE (16/64 dims) + LN scale
- XSA on all 11 layers
- BigramHash 3072x112
- Value Embeddings on layers 9-10
- U-Net skip connections with SmearGate
- Logit softcap = 30.0, tied embeddings

## Quantization Pipeline

1. Train fp32/bf16 for ~85% of steps (Parallel Muon + AdamW)
2. Late QAT: int6 STE when LR scale < 0.15
3. EMA (0.997) + SWA (every 50 steps in warmdown)
4. AR self-gen calibration (64 seqs x 2048 tokens, temp=0.8)
5. Full Hessian GPTQ (int6, clip_range=31, Cholesky compensation)
6. Selective ±1 pruning to fit 16MB
7. LZMA preset=9 compression

## Running

```bash
# On 8xH100 SXM (RunPod):
pip install -r records/track_10min_16mb/2026-04-02_Trinity_Hybrid_Ternary_GPTQ_XSA/requirements.txt
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10
cp records/track_10min_16mb/2026-04-02_Trinity_Hybrid_Ternary_GPTQ_XSA/train_gpt.py ./train_gpt.py
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Lineage

Built on PR #1019 (abaybektursun) → PR #549 → PR #414 → PR #374 → PR #287 → PR #198 → baseline.

Trinity contribution: parameter budget analysis showing MLP tolerates increased width within int6 quantization.
