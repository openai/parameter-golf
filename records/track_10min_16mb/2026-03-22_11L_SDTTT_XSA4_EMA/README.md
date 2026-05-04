# 11L XSA4 + EMA (PR379 base) - SDTTT disabled for legality

## Update (April 13, 2026)

@MatoTeziTanka correctly flagged this submission on 2026-04-12 via The Agora community compliance tracker. The original reported number (1.1287 3-seed mean) was computed with Self-Distillation TTT (SDTTT) running 2 epochs of SGD over all val_tokens before the sliding-window eval. That violates score-first discipline per Issue #402 and Issue #677.

Updating this submission to report the **pre-SDTTT post-swa numbers** from the same 3-seed runs. These are legal: they come from a clean `eval_val` run on the EMA-averaged weights, before any SDTTT adaptation touched val_tokens. The train logs for all 3 seeds still contain the clean DIAGNOSTIC post_swa lines that we're now using as the reported number.

The `train_gpt.py` in this folder has `sdttt_enabled` defaulting to `"0"` (line 140), so running it as-is produces legal results. The illegal numbers came from runs where `SDTTT_ENABLED=1` was set as an env var override.

Thanks to @MatoTeziTanka for the careful review and the explicit "pre-SDTTT score of 1.1448 would be clean" pointer that made this fix straightforward.

## Legal Summary

**Mean val_bpb = 1.1455** (3-seed mean, clean post-EMA eval, no SDTTT)

| Seed | val_bpb (post_swa, legal) | val_bpb (sliding + SDTTT, illegal) | Steps | ms/step | Artifact |
|------|---------------------------|-------------------------------------|-------|---------|----------|
| 7 | **1.1461** | 1.1294 | 6,084 | 99.5 | 15.7MB |
| 42 | **1.1457** | 1.1287 | 6,029 | 99.5 | 15.7MB |
| 1337 | **1.1448** | 1.1280 | 6,084 | 99.5 | 15.7MB |
| **Mean** | **1.1455** | 1.1287 | | | |

The post_swa column is the DIAGNOSTIC val_bpb computed after EMA weight averaging but before any eval-time adaptation. That is the legal predecessor of the submitted sliding_window number, which was contaminated by SDTTT. All artifacts still under 16MB.

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
