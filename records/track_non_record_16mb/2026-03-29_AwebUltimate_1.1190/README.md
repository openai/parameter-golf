# Aweb Ultimate — 1.1190 BPB (10min, 8×H100)

## Results

| Metric | Score |
|--------|-------|
| **val_bpb (TTT)** | **1.1190** |
| val_bpb (sliding window) | 1.1217 |
| val_bpb (chunked roundtrip) | 1.1450 |
| Submission size | 15,948,863 bytes |
| Steps | 7,166 in 600s on 8×H100 SXM |

## Summary

Independent reproduction and slight improvement of the PR #549 SOTA stack (1.1194 BPB). Achieves 1.1190 BPB under the standard 10-minute 8×H100 constraint with a 15.95MB artifact.

## Techniques

| Category | Details |
|----------|---------|
| Architecture | 11L, 512d, 8 heads, 4 KV heads (GQA), tied embeddings |
| Activation | LeakyReLU(0.5)² |
| Cross-layer attention | XSA on last 4 layers |
| Positional encoding | Partial RoPE (16/64 head dims) |
| Normalization | LN Scale (1/√(layer+1)) |
| Weight averaging | EMA (0.997) + SWA |
| Optimizer | Parallel Muon (batched NS5, 3-phase overlapped comms) + AdamW |
| Quantization | GPTQ-lite int6 (MLP+attn) + int8 (rest) + LZMA |
| Input enrichment | SmearGate + BigramHash(2048) + ValueEmbedding(128, layers 9-10) |
| Skip connections | U-Net encoder-decoder with learned skip weights |
| Late QAT | Int6 STE at LR scale < 0.15 |
| Evaluation | Sliding window (stride=64) + Legal Score-First TTT (3 epochs SGD) |

## Reproduction

```bash
TTT_ENABLED=1 torchrun --standalone --nproc_per_node=8 \
  records/track_non_record_16mb/2026-03-29_AwebUltimate_1.1190/train_gpt.py
```

## Author

Daniel Wahnich (@manfromnowhere143)
