# Farnsworth-Adapted: 11L MLP3x + INT6 + SmearGate + BigramHash + TTT + FA2 + WD Tuning

**Score:** val_bpb = 1.1381 (3-seed mean, sliding window stride=64, post-TTT)

## Summary

Adapts the FarnsworthEngine architecture (PR #254) with FlashAttention 2 (instead of FA3 Hopper) and weight decay optimization for artifact size control. Key finding: **cuDNN SDP is 40% faster per attention op than Flash SDP on H100 but produces worse model quality** (1.1455 vs 1.1418 BPB). Weight decay directly controls compressed artifact size: WD=0.042 targets the optimal ~15.5MB.

## Architecture

| Component | Details |
|-----------|---------|
| **Layers** | 11 transformer layers, 512 dim, 8 heads, 4 KV heads (GQA) |
| **MLP** | 3x expansion (hidden=1536), ReLU^2 activation |
| **Quantization** | Int6 mixed precision (MLP+attention), Int8 (embeddings) |
| **Compression** | zstd-22, artifact ~15.50 MB |
| **SmearGate** | Learned sigmoid token blending gate (~512 params) |
| **BigramHash** | 2048-bucket hash embedding for token-pair features (dim 128) |
| **Skip Connections** | U-Net style encoder-decoder with learned skip weights |
| **Initialization** | Orthogonal + muP output scaling |
| **Optimizer** | Muon (WD=0.042, momentum=0.99, warmup 1500 steps, warmdown 3000) |
| **SWA** | 7 checkpoint average during warmdown (every 200 steps) |
| **Attention** | FlashAttention 2.8.3 (torch SDPA flash backend) |
| **Position** | NTK-RoPE (base=10000) |
| **Sequence** | Train@2048, eval@2048 |
| **TTT** | Full-weight SGD adaptation on val data (lr=0.002, momentum=0.9, 3 epochs) |
| **Eval** | Sliding window stride=64 with TTT-adapted weights |

## Results

| Seed | Steps | Step Avg | Pre-quant BPB | Post-TTT Sliding BPB | Artifact |
|------|-------|----------|--------------|----------------------|----------|
| 1337 | 9,000 | 65.7ms | 1.1546 | **1.1374** | 15.50 MB |
| 42 | 9,000 | 65.7ms | 1.1530 | **1.1372** | 15.66 MB |
| 7 | 9,000 | 65.9ms | 1.1560 | **1.1397** | 15.62 MB |
| **Mean** | | | **1.1545** | **1.1381** | **15.59 MB** |

## Attribution

### [SOTA-ADOPT] From FarnsworthEngine (PR #254)
- 11L MLP3x INT6 architecture
- SmearGate + BigramHash
- U-Net skip connections with learned weights
- Orthogonal init with muP scaling
- TTT (Test-Time Training) with sliding window eval
- Muon optimizer with momentum warmup
- SWA during warmdown

### [SOTA-ADOPT] From PR #236 (saml212)
- TRAIN_BATCH_TOKENS=524288 (smaller batch = more gradient updates in fixed time)
- Weight decay as artifact size control

### [ORIGINAL] Findings
1. **cuDNN SDP vs Flash SDP benchmark on H100:** cuDNN is 40% faster per attention op (0.134ms vs 0.221ms) but produces worse BPB (1.1455 vs 1.1418). We verified this is a quality issue, not a speed tradeoff — cuDNN gets MORE training steps but still underperforms. This suggests cuDNN uses different internal accumulation precision that hurts final model quality.

2. **Weight decay sweep for artifact size targeting:** Systematic sweep from WD=0.040 to WD=0.050 revealed that WD=0.042 optimally targets 15.5MB (within the 16MB budget) while minimizing BPB:
   - WD=0.040: 16.3MB (invalid), 1.1414 BPB
   - WD=0.041: 15.6MB, 1.1378 BPB
   - WD=0.042: 15.5MB, **1.1374 BPB** (optimal)
   - WD=0.045: 15.6MB (with QAT), 1.1466 BPB
   - WD=0.050: 15.0MB, 1.1418 BPB

3. **QAT hurts at this scale:** Enabling INT6 quantization-aware training (STE) during forward pass reduces the quant gap (0.005 vs 0.009 BPB) but increases training loss enough to negate the benefit (1.1466 vs 1.1374 overall).

4. **INT4 quantization is a dead end for this architecture:** All-INT4 (clip=7) achieves excellent pre-quant BPB (1.1521) by fitting 33.5M params instead of 26.8M, but the 0.06 BPB quantization gap makes it strictly worse than INT6 with fewer params.

5. **FA2 on H100 is competitive:** Without the FA3 Hopper-native kernels, FA2.8.3 achieves ~66ms/step (vs Farnsworth's reported 81ms with FA3). The speed advantage doesn't fully translate to BPB (1.1374 vs 1.1303), suggesting FA3 may have different numerical properties that help model quality.

## Reproduction

```bash
SEED=1337 NUM_LAYERS=11 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 \
MLP_MULT=3.0 BIGRAM_VOCAB_SIZE=2048 BIGRAM_DIM=128 \
MUON_WD=0.042 ADAM_WD=0.042 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3000 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_MOMENTUM=0.9 \
TRAIN_BATCH_TOKENS=524288 VAL_LOSS_EVERY=0 TRAIN_LOG_EVERY=500 \
SWA_ENABLED=1 SWA_EVERY=200 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Timing Budget

| Phase | Time |
|-------|------|
| Training | 591s |
| TTT adaptation | 46s |
| Sliding window eval | 80s |
| **Total** | **~717s** |

## What We'd Try Next

1. Build FA3 Hopper kernels from source — could close remaining 0.007 BPP gap to SOTA
2. Minify code (~69KB to ~40KB) to free ~29KB for additional model weights
3. Explore larger models (d=528 or d=544) if code minification provides space
4. Test TTT with more epochs (5) or higher LR (0.003)
5. Try MUON_MOMENTUM=0.995 or different warmdown schedules
