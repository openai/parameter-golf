# 10L Seq2048 WarmdownQuant + TTT LoRA

## Approach

Pure-model approach focused on training efficiency and quantization-aware optimization.

### Training
- 10-layer transformer (512 dim, 8 heads, 4 KV heads, tied embeddings)
- Sequence length 2048 for richer context
- Muon optimizer: momentum 0.98, matrix LR 0.03, scalar LR 0.03
- Reduced batch size (393K tokens) for more optimizer updates in 10 minutes
- Always-decaying warmdown (WD=15000) constrains weight magnitudes, reducing int8 quantization error
- Gradient clipping 1.0 for stability with aggressive schedule
- Overtone spectral embedding init + phase-transition residual mixing

### Quantization
- Int8 per-row quantization + zlib compression
- FP16 tied embeddings (avoids int8 error compounding through input and output)
- Warmdown-tightened weights minimize quantization penalty

### Evaluation
- Test-time training with batched LoRA adapters (rank 8) on Q, V projections and LM head
- Per-document isolation with chunk-wise Adam optimization (chunk_size=256)

## Results

Seed 1337 on 8xH100 SXM:

| Metric | Value |
|--------|-------|
| val_bpb (pre-quant, final step) | 1.1787 |
| int8+zlib artifact size | 15.56 MB |
| Steps completed | 13,282 |
| Step avg | 45.18 ms |

### Val BPB progression
| Step | val_bpb |
|------|---------|
| 1000 | 1.3883 |
| 5000 | 1.2598 |
| 8000 | 1.2333 |
| 10000 | 1.2130 |
| 12000 | 1.1917 |
| 13282 | 1.1787 |

## Run log ([`run_seed1337.log`](run_seed1337.log))

Training + int8+zlib export for seed 1337: ends at step 13282 with **val_bpb 1.1787**, then peak memory and submission byte counts. TTT LoRA (above) is separate eval — not in this capture.

Val was still trending down at the 10-minute cap. More 8×H100 wall (or faster hardware) plus a full TTT pass is the path to a **~1.15 BPB** stretch; same stack, not logged end-to-end here yet.
