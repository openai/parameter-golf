# Extended Compute Scaling Analysis (50K Steps, ~12 Hours)

**val_bpb: 1.0858** (50K steps, 3-seed mean, std 0.0005) | **~14.30 MB** | 4×A100 MIG (Unlimited Compute Track)

## Summary

This submission is a **non-record submission**. It extends the [20K scaling run](https://github.com/openai/parameter-golf/pull/1407) to 50K steps (~12 hours), using the same architecture and code from ([PR #549](https://github.com/openai/parameter-golf/pull/549) by @abaybektursun). Training runs on 4×A100 MIG instances (approximately 10× slower per step than 8×H100 SXM).

Key findings:
- **50K steps achieves 1.0858 BPB post-TTT** (3-seed mean, std 0.0005) — improvement of **−0.0102 BPB** over the 20K run (1.0960)
- **Artifact shrinks further to ~14.30MB** as warmdown smooths weight entropy more aggressively at longer horizons
- **TTT gains scale with compute**: TTT provides −0.0089 BPB at 50K steps vs −0.0058 at 20K steps (+53% more TTT benefit)
- **Artifact size plateau at 10K–30K steps** (~17.2MB) confirms the non-monotonic pattern; warmdown recovery is steeper and more complete at 50K than at 20K

## Results

### 50K steps, ~12 hours (4×A100 MIG, 3-seed comparison)

| Seed | step_avg | steps | Pre-TTT bpb | **Post-TTT bpb** | TTT gain | Artifact |
|------|----------|-------|-------------|-----------------|----------|----------|
| 1337 | 829.0ms | 50,000 | 1.0942 | **1.0853** | -0.0089 | 14,330,478 |
| 42 | 830.9ms | 50,000 | 1.0945 | **1.0857** | -0.0088 | 14,206,210 |
| 2024 | 828.7ms | 50,000 | 1.0953 | **1.0865** | -0.0088 | 14,353,974 |
| **Mean** | **829.5ms** | **50,000** | **1.0947** | **1.0858 (std 0.0005)** | **-0.0088** | **14,296,887** |

### Comparison: 20K vs 50K

| Metric | 20K steps (~6h) | 50K steps (~12h) | Δ |
|--------|----------------|-----------------|---|
| Post-TTT BPB | 1.0960 | 1.0858 | **−0.0102** |
| Artifact size | ~15.05 MB | ~14.30 MB | −0.75 MB |
| TTT gain | −0.0058 | −0.0088 | −0.0030 (53% more) |

## Scaling Analysis: BPB & Artifact Size vs Training Steps

Artifact size is computed as int6+LZMA compressed model + code bytes. Every 2,500 steps.

### Seed 2024 (2,500-step intervals)

| Steps | Pre-TTT val_bpb | artifact_bytes | Under 16MB? |
|------:|--------:|---------------:|:-----------:|
| 0 | 4.1037 | 4,577,366 | Yes |
| 2,500 | 1.2552 | 13,069,870 | Yes |
| 5,000 | 1.2282 | 13,892,598 | Yes |
| 7,500 | 1.2185 | 15,496,510 | Yes |
| 10,000 | 1.2027 | 17,215,490 | **No** |
| 12,500 | 1.1996 | 17,232,630 | **No** |
| 15,000 | 1.1945 | 17,237,242 | **No** |
| 17,500 | 1.1923 | 17,219,546 | **No** |
| 20,000 | 1.1912 | 17,213,066 | **No** |
| 22,500 | 1.1903 | 17,206,306 | **No** |
| 25,000 | 1.1904 | 17,201,238 | **No** |
| 27,500 | 1.1883 | 17,193,574 | **No** |
| 30,000 | 1.1865 | 17,190,822 | **No** |
| 32,500 | 1.1813 | 17,052,022 | **No** |
| 35,000 | 1.1742 | 16,786,866 | **No** |
| 37,500 | 1.1677 | 16,553,402 | **No** |
| 40,000 | 1.1592 | 16,067,974 | **No** |
| 42,500 | 1.1493 | 15,598,778 | Yes |
| 45,000 | 1.1350 | 15,025,010 | Yes |
| 47,500 | 1.1178 | 14,743,726 | Yes |
| 50,000 | 1.0953* | 14,353,974 | Yes |

*Step 50K artifact reflects the final model with full warmdown applied.

**Note:** The artifact peaks at ~17.24MB between steps 10K–30K — a much wider over-budget window than at 20K training. Warmdown fully recovers compressibility, landing at ~14.30MB, ~1.7MB below the 16MB limit. Early stopping without a dedicated warmdown phase remains non-viable for this architecture.

### BPB vs Steps (ASCII plot)

Power-law decay with two distinct phases: rapid early learning (0–10K), plateau (10K–30K), then warmdown-driven final drop.

```
BPB
4.10 |*
     |
     |
2.50 |
     |
1.26 | *
1.23 |   *
1.22 |     *
1.20 |       *  * * * * * * * *
1.18 |                           *
1.16 |                             *
1.14 |                               *
1.12 |                                 *
1.10 |                                   *
     +---------+--------+---------+-------> steps (K)
     0        10       20        30       50

     |<early>|<--- plateau --->|<warmdown>|
      (rapid)  (BPB ≈1.19–1.20)  (sharp drop)
```

### Artifact Size vs Steps (ASCII plot)

Non-monotonic: grows to peak at ~10K–30K steps (~17.2MB), then shrinks rapidly during warmdown.

```
MB
17.2 |       * * * * * * * * * * * *
16.0 |--------------------------------------  16MB limit
15.5 |
14.9 |                                * *
14.4 |                                    *
13.9 |    *
13.1 | *
 4.6 |*
     +---------+--------+---------+-------> steps (K)
     0        10       20        30       50

     |<fits>|<-------over budget------->|<fits>|
```

Intermediate checkpoints between steps ~10K–42K exceed the 16MB budget. Only the final model (with warmdown complete) fits at ~14.30MB.

### Observations

1. **BPB vs steps: plateau then sharp warmdown drop.** Unlike the 20K run (where warmdown is visible but brief), the 50K warmdown spans 19,500 steps and drives a dramatic BPB drop from 1.1904 (step 25K) to 1.0953 (step 50K) — a drop of 0.095 BPB in the final half of training.
2. **Artifact size plateau is wider at 50K.** The over-budget window spans steps 10K–42K (32K steps vs ~10K steps at 20K scale). Warmdown at larger scale is both longer and more effective — final artifact is 14.30MB vs 15.05MB at 20K.
3. **TTT scales with compute.** At 50K steps, TTT provides −0.0088 BPB (up from −0.0058 at 20K steps). A stronger base model enables more adaptation at test time.
4. **Diminishing returns on BPB.** Doubling compute from 20K→50K (steps) yields −0.0102 BPB improvement. Power-law decay holds but with diminishing marginal returns as expected.

## Architecture

Identical to [PR #549](https://github.com/openai/parameter-golf/pull/549) (LeakyReLU² + Legal TTT + Parallel Muon):

| Component | Setting |
|-----------|---------|
| Layers | 11 (512d, 8H, 4KV) |
| MLP | 3× with LeakyReLU(0.5)² |
| BigramHash | 1536 |
| XSA | Last 4 layers |
| RoPE | Partial (16/64 dims) |
| LN Scale | 1/√(layer+1) |
| VE128 | Layers 9-10 |
| Weight avg | EMA(0.997) + Tight SWA(every 50) |
| Quantization | GPTQ-lite int6 + lzma |
| Optimizer | Parameter Banking + Parallel Muon |

### Hyperparameter scaling for extended training

LR schedule parameters scaled proportionally to maintain the same warmup/warmdown ratios:

| Parameter | 20K steps | This work (50K steps) |
|-----------|-----------|-----------|
| ITERATIONS | 20,000 | 50,000 |
| WARMDOWN_ITERS | 7,800 (39.0%) | 19,500 (39.0%) |
| MUON_MOMENTUM_WARMUP_STEPS | 3,340 (16.7%) | 8,350 (16.7%) |
| VAL_LOSS_EVERY | 2,000 | 2,500 |
| MAX_WALLCLOCK_SECONDS | 0 (unlimited) | 0 (unlimited) |

## Run Command

```bash
RUN_ID=train_step50k_seed1337 \
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=8350 WARMDOWN_ITERS=19500 \
ITERATIONS=50000 MAX_WALLCLOCK_SECONDS=0 EVAL_STRIDE=64 \
VAL_LOSS_EVERY=2500 \
SEED=1337 \
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 train_gpt.py
```

## Hardware

- 4×NVIDIA A100 MIG instances
- Seeds 1337 & 2024: ~829ms/step | Seed 42: ~831ms/step (vs 83ms/step on 8×H100 SXM — approximately 10× slower)
- grad_accum_steps=2 (to match 8-GPU effective batch size of 786,432 tokens)
- ~11.5h training + ~34min TTT eval = ~12 hours total per seed

## Credits

This submission uses the full architecture and code from the record-track PR #549 with no ML changes — only extended compute and proportionally scaled LR schedules.

- **Base submission [PR #549](https://github.com/openai/parameter-golf/pull/549)**: @abaybektursun — LeakyReLU² + Legal TTT + Parallel Muon
- **LeakyReLU² activation**: [PR #493](https://github.com/openai/parameter-golf/pull/493) by @parinzee, [PR #518](https://github.com/openai/parameter-golf/pull/518) by @sofiabod
- **Optimizer (Parameter Banking + Parallel Muon)**: [PR #399](https://github.com/openai/parameter-golf/pull/399) by @abaybektursun
- **TTT recipe**: [PR #461](https://github.com/openai/parameter-golf/pull/461) by @Christopher-Lee-McClendon
- **Base model**: [PR #414](https://github.com/openai/parameter-golf/pull/414) by @signalrush
