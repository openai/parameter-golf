# Depth-Recurrent 3×3 GPT — Non-Record Submission

## Architecture

- **Depth recurrence**: 3 unique Transformer blocks × 3 recurrent loops = 9 effective layers
- **Dimensions**: dim=768, 8 heads, 4 KV heads (GQA), MLP mult=2
- **Per-loop conditioning**: AdaLN/FiLM scale+shift, sigmoid cycle gates, mini U-Net skip connections
- **Parameters**: 13.2M total, compressed to **9.35 MB** (int8 + zlib, payload ratio 3.86×)

## Results

| Metric | Value |
|---|---|
| val_bpb | 1.6729 |
| val_loss | 2.8247 |
| Compressed size | 9.35 MB |
| Training | 2000 steps on Apple M3 Max (MLX) |
| H100-equivalent | ~150 steps (severely undertrained) |

## Key Experiments

### Test-Time Training (TTT)
- Attention-only adaptation (~5.3M params) during eval
- Best result: **-0.73% BPB** at lr=0.1 on chunk-2+ tokens (200 documents)
- Zero storage cost: adaptation is transient, per-document reset

### Sparsity Trade-off
- 20% magnitude pruning: **+0.06% BPB** (essentially free)
- Knee at 30%: +0.30% BPB
- Enables dimension scaling within 16 MB budget

### Curriculum (tested, abandoned)
- 60% 1-loop → 40% 3-loop: **-4.8% BPB** vs fixed 3-loop
- 1-loop creates incompatible local minimum for 3-loop recovery

### Bigram LUT (tested, ruled out)
- Model already far exceeds bigram predictions (2.85 bits/token vs 6.07 bits/token bigram)
- 2 MB storage better spent on more parameters

## H100 Experiments Planned

1. Full 10-min training run with depth recurrence
2. dim=768→960 scaling (~14 MB with 20% sparsity)
3. Reptile meta-learning to amplify TTT gains
4. TTT eval timing verification (K=3-5 steps within eval budget)

## Files

- `train_gpt_mlx_v2.py` — Training script with depth-recurrent architecture
- `submission.json` — Metrics
- `train.log` — Training log (2000 steps)
