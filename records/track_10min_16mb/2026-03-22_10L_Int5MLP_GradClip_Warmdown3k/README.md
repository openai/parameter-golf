# 10L Int5-MLP + Mixed Quantization + GradClip + Warmdown3k

**val_bpb: 1.20262** (mean of 3 seeds, post int5/int6+zlib quantization roundtrip)

## Run Command
```bash
RUN_ID=submission SEED=42 VAL_LOSS_EVERY=0 SWA_ENABLED=0 torchrun --nproc_per_node=8 train_gpt.py
```

## 3-Seed Results

| Seed | val_bpb | artifact_bytes | valid |
|------|---------|----------------|-------|
| 1337 | 1.20375402 | 15,727,520 | yes |
| 42   | 1.20134663 | 15,731,966 | yes |
| 123  | 1.20275459 | 15,719,392 | yes |
| **Mean** | **1.20262** | | |

## Key Techniques

### Mixed Int5/Int6 Quantization
- **Int5 [-16,15]** for MLP weights — funds the 10th layer within 16MB
- **Int6 [-32,31]** for attention weights  
- **FP16** for tied embeddings (zero quantization error)

### Architecture
- 10 layers, 512 dim, 8 heads, 4 KV heads (GQA)
- MLP 3x expansion (hidden=1536), relu² activation
- U-Net skip connections, tied embeddings

### Training
- WARMDOWN_ITERS: 3000 (baseline 1200)
- MLP_MULT: 3 (baseline 2)
- NUM_LAYERS: 10 (baseline 9)
- GRAD_CLIP_NORM: 0.3 (baseline 0.0)
- SWA disabled (wallclock-capped runs)