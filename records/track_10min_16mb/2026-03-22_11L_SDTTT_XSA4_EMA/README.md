# 11L XSA4 + EMA + Self-Distillation TTT

## Summary

**Mean val_bpb = 1.1287** (3-seed verified, sliding window stride=64)

Uses @dannywillowliu-uchi's PR #379 architecture with Self-Distillation TTT at eval time. Third progressive submission from our team — showing continued improvement through systematic experimentation informed by 231K-chunk expert knowledge base analysis.

| Seed | val_bpb (sliding) | val_bpb (standard) | Steps | ms/step | Artifact |
|------|-------------------|-------------------|-------|---------|----------|
| **1337** | **1.1280** | 1.1530 | 6,084 | 99.5 | 15.7MB |
| 42 | 1.1287 | 1.1527 | 6,029 | 99.5 | 15.7MB |
| 7 | 1.1294 | 1.1530 | 6,084 | 99.5 | 15.7MB |
| **Mean** | **1.1287** | | | | |

Std: 0.0007 | All artifacts under 16MB

## Submission progression

| PR | BPB | Technique added | Days in |
|----|-----|-----------------|---------|
| #273 | 1.1575 | Baseline SOTA, 10L | Day 1 |
| #385 | 1.1488 | WD=0.04, SWA=0.4, 11L | Day 2 |
| **This** | **1.1287** | XSA4 + EMA + SDTTT | Day 4 |

0.029 BPB improvement across 3 submissions in 4 days.

## Architecture (from PR #379)

- 11 layers, 512 dim, 8 heads, 4 KV heads (GQA), 3× MLP
- XSA (Exclusive Self-Attention) on last 4 layers
- EMA (decay=0.997) replacing SWA
- Int6 QAT with STE, zstd compression
- FP16 tied embedding passthrough
- SmearGate, U-Net skip connections

## Self-Distillation TTT

KL-divergence adaptation on validation data at eval time:
- Temperature: 2.0
- SGD lr: 0.001, momentum: 0.9
- 2 epochs, first 4 blocks frozen
- Preserves XSA attention patterns

## Note on step speed

Running at 99ms/step on stock PyTorch SDPA (no Flash Attention 3, no custom Triton kernels). The SOTA achieves ~55ms/step with FA3 + fused kernels. With equivalent infrastructure, this technique stack would yield significantly more training steps and better BPB.

## Total experiment context

40+ experiments, ~$150 RunPod spend, 4 days. Systematic analysis using a 231K-chunk searchable knowledge base spanning 50+ AI/ML experts.

## Run command

```bash
NCCL_NVLS_ENABLE=0 SEED=7 \
MAX_WALLCLOCK_SECONDS=600 \
SDTTT_ENABLED=1 SDTTT_EPOCHS=2 SDTTT_LR=0.001 \
SDTTT_TEMPERATURE=2.0 SDTTT_FREEZE_BLOCKS=4 \
GPTQ_ENABLED=0 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Submission checklist

- [x] 3-seed verification (mean=1.1287, std=0.0007)
- [x] All artifacts < 16MB (15.7MB)
- [x] Wallclock < 600s on 8×H100
- [x] Train logs included (3 seeds)
- [x] Reproducible train_gpt.py included
- [x] README with detailed explanation
