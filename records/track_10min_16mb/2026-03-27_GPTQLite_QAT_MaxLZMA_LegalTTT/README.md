# Ultimate: GatedAttention + ValueResidual + Full QAT + lzma-9 + BigramHash(2048)

**Target val_bpb: < 1.1144** (to beat competition leaderboard #1 by 0.005)
**Base: 2026-03-23_LeakyReLU_LegalTTT_ParallelMuon → 1.1194 BPB**

## Results (8×H100 80GB SXM) — TBD after run

| Seed | step_avg | steps | Pre-TTT bpb | **Post-TTT bpb** | TTT gain | TTT time | Artifact |
|------|----------|-------|-------------|-----------------|----------|----------|----------|
| 1337 | —        | —     | —           | **—**            | —        | —        | —        |
| 42   | —        | —     | —           | **—**            | —        | —        | —        |
| 2025 | —        | —     | —           | **—**            | —        | —        | —        |
| **Mean** | —   | —     | —           | **—**            | —        | —        | —        |

## Changes vs 2026-03-23 Base (1.1194 BPB)

| Change | Why | Expected Δ BPB |
|--------|-----|----------------|
| **GatedAttention=True** | Per-head sigmoid gate (PR #841, nearly no-op init: weight=0, bias=4.0). Already implemented, just enabling default. | -0.002 to -0.005 |
| **ValueResidual=True** | Mixes layer-0 value v0 into all subsequent layers (PR #841, init: λ=[0.5,0.5]). Already implemented, enabling default. | included above |
| **QAT_ENABLED=True from step 1** | Model trains with int6 fake-quant noise for all ~7000 steps vs only the last 175 (5%). Full quantization adaptation. | -0.001 to -0.003 |
| **LATE_QAT_THRESHOLD=0.05** | CastedLinear QAT (for non-bank params) only activates in final 5% of warmdown — minimal noise during critical fine-tuning phase. | included above |
| **lzma preset=6 → 9** | CLAUDE.md: lzma-9 is 8-15% smaller than lzma-6 on int6 weights. Frees ~1.3MB artifact budget. | ~0 (enables BigramHash upgrade) |
| **BigramHash 1536 → 2048** | 2026-03-23 downgraded from 2048→1536 to fit 16MB with lzma-6. lzma-9 savings enable going back to 2048. Ablation: 2048→3072 was -0.0009; expect similar gain here. | -0.001 to -0.002 |
| **Total expected gain** | | **-0.004 to -0.010** → ~1.109 to 1.115 BPB |

## Preserved from 2026-03-23 (unchanged)

| Feature | Setting |
|---------|---------|
| **Architecture** | 11L, 512d, 8H, 4KV, 3× MLP |
| **Activation** | **LeakyReLU(0.5)²** — hardcoded, always on |
| **Optimizer** | **Parallel Muon + Parameter Banking** — unchanged |
| **EMA** | decay=0.997, applied at end of training |
| **SWA** | every 50 steps when LR scale < 0.2 |
| **Sliding window eval** | stride=64, seq_len=2048 |
| **Legal TTT** | score-first, 3 epochs, freeze=0, lr=0.002, SGD+momentum(0.9) |
| **BigramHash** | vocab=**2048**, dim=128 (restored from 1536) |
| **XSA** | Last 4 layers |
| **Partial RoPE** | 16/64 dims, NTK scaling |
| **LN Scale** | 1/√(layer+1) |
| **VE** | dim=128, layers 9,10 |
| **Quantization** | GPTQ-lite int6, per-row clip search, **lzma-9** |
| **Training** | TRAIN_SEQ_LEN=2048, EVAL_STRIDE=64, WARMDOWN_ITERS=3500 |

## Training Architecture

PR #414 + PR #399 + PR #841 stack:

| Component | Setting |
|-----------|---------|
| Layers | 11 (512d, 8H, 4KV) |
| MLP | 3× with **LeakyReLU(0.5)²** |
| **GatedAttention** | Per-head sigmoid gate (NEW, weight=0, bias=4.0 init) |
| **ValueResidual** | Layer-0 value injection λ=[0.5,0.5] (NEW) |
| BigramHash | **2048** vocab, 128 dim (restored from 1536) |
| XSA | Last 4 layers |
| RoPE | Partial (16/64 dims), NTK dynamic scaling |
| LN Scale | 1/√(layer+1) |
| VE128 | Layers 9-10 |
| Weight avg | EMA(0.997) + Tight SWA(every 50) |
| **QAT** | **Full from step 1** (int6 fake-quant on CastedLinear) |
| Quantization | GPTQ-lite int6 + **lzma-9** |
| Optimizer | Parameter Banking + Parallel Muon |

## Run Command

```bash
# All changed defaults are now baked in; no env var overrides needed for the 4 improvements.
# Explicit env vars shown for transparency / to allow override:
NUM_LAYERS=11 XSA_LAST_N=4 \
EMA_ENABLED=1 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
GATED_ATTENTION=1 VALUE_RESIDUAL=1 \
QAT_ENABLED=1 LATE_QAT_THRESHOLD=0.05 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee train_seed1337.log
```

## Timing Budget (estimated)

| Phase | Time |
|-------|------|
| Training | 600s |
| EMA apply + int6 roundtrip eval | ~30s |
| Sliding window eval (2048, stride=64) | ~74s |
| lzma-9 compression (slower than lzma-6) | ~45s |
| Legal TTT (3ep, all blocks, 2048 ctx) | ~410s |
| **Total eval** | **~560s (< 10 min ✓)** |

## Credits

- **GatedAttention + ValueResidual**: PR #841 by @someone114514
- **LeakyReLU² activation**: PR #493 by @parinzee, PR #518 by @sofiabod
- **Optimizer (Parameter Banking + Parallel Muon)**: PR #399 by @abaybektursun
- **TTT recipe**: PR #461 by @Christopher-Lee-McClendon
- **Base model**: PR #414 by @signalrush
