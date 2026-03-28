# Extended Compute Scaling Analysis (20K–50K Steps)

**val_bpb: 1.0853** (50K steps, seed 1337) | **~14.35 MB** | 4×A100 MIG (Unlimited Compute Track)

## Summary

This submission is a **non-record submission**. It studies how the current record-track SOTA ([PR #549](https://github.com/openai/parameter-golf/pull/549) by @abaybektursun) scales under extended compute, removing the 10-minute wall-clock constraint. The same architecture and code are trained for 20K–50K steps (5.5-11.5 hours training) on 4×A100 MIG instances (approximately 10× slower per step than 8×H100 SXM).

Key findings:
- **50K steps achieves 1.0853 BPB post-TTT**, a -0.034 improvement over the record-track SOTA (1.1194)
- **Artifact size balloons mid-training** (peaking at 17.2MB around step 10K–15K) but **recovers to 14.35MB** after warmdown completes — warmdown smooths weight entropy and restores compressibility
- **Diminishing returns** set in beyond ~30K steps, with the BPB curve flattening significantly
- **TTT gains scale with base model quality**: TTT provides -0.014 BPB on the 50K model vs -0.0025 on the 7K SOTA

## Results

### Best run: 50K steps and 11.5 hours (4×A100 MIG, seed 1337)

| Phase | val_loss | val_bpb | Artifact |
|-------|----------|---------|----------|
| Pre-TTT (EMA) | 1.8469 | 1.0939 | 14,348,646 |
| Int6 roundtrip | 1.8963 | 1.1231 | 14,348,646 |
| Sliding window (s=64) | 1.8566 | 1.0996 | 14,348,646 |
| **Legal TTT** | **1.8325** | **1.0853** | **14,348,646** |

### 20K steps and 5.5 hours (4×A100 MIG, 2-seed comparison)

| Seed | step_avg | steps | Pre-TTT bpb | **Post-TTT bpb** | TTT gain | Artifact |
|------|----------|-------|-------------|-----------------|----------|----------|
| 1337 | 828.7ms | 20,000 | 1.1018 | **1.0957** | -0.0061 | 15,077,933 |
| 42 | 828.8ms | 20,000 | 1.1020 | **1.0962** | -0.0058 | 15,137,145 |
| **Mean** | **828.8ms** | **20,000** | **1.1019** | **1.0960 (std 0.0004)** | **-0.0060** | **15,107,539**|

### Comparison with record-track SOTA

| Config | Steps | Hardware | Pre-TTT bpb | Post-TTT bpb | Delta vs SOTA |
|--------|-------|----------|-------------|--------------|---------------|
| SOTA (PR #549) | ~7,185 | 8×H100 SXM | 1.1218 | 1.1194 | — |
| This work (20K) | 20,000, 5.5 hours | 4×A100 MIG | 1.1019 | 1.0960 | **-0.0234** |
| This work (50K) | 50,000, 11.5 hours | 4×A100 MIG | 1.0939 | **1.0853** | **-0.0341** |

## Scaling Analysis: BPB & Artifact Size vs Training Steps

Data from the 50K run with validation every 2,500 steps. Artifact size is computed as int6+LZMA compressed model + code bytes.

| Steps | Pre-TTT val_bpb | artifact_bytes | Under 16MB? |
|------:|--------:|---------------:|:-----------:|
| 0 | 4.1046 | 4,577,902 | Yes |
| 2,500 | 1.2554 | 13,062,362 | Yes |
| 5,000 | 1.2290 | 14,137,046 | Yes |
| 7,500 | 1.2201 | 15,465,582 | Yes |
| 10,000 | 1.2022 | 17,185,494 | **No** |
| 12,500 | 1.1987 | 17,195,458 | **No** |
| 15,000 | 1.1936 | 17,238,818 | **No** |
| 17,500 | 1.1918 | 17,178,826 | **No** |
| 20,000 | 1.1905 | 17,182,262 | **No** |
| 22,500 | 1.1900 | 17,168,362 | **No** |
| 25,000 | 1.1898 | 17,167,526 | **No** |
| 27,500 | 1.1877 | 17,158,910 | **No** |
| 30,000 | 1.1861 | 17,148,538 | **No** |
| 32,500 | 1.1804 | 17,016,594 | **No** |
| 35,000 | 1.1740 | 16,753,850 | **No** |
| 37,500 | 1.1669 | 16,422,374 | **No** |
| 40,000 | 1.1582 | 16,034,950 | **No** |
| 42,500 | 1.1487 | 15,698,694 | Yes |
| 45,000 | 1.1340 | 15,144,098 | Yes |
| 47,500 | 1.1167 | 14,726,650 | Yes |
| 50,000 | 1.0942 | 14,330,478 | Yes |

**Note:** Intermediate checkpoints do not benefit from warmdown. The final model (step 50K) has full warmdown applied, resulting in 14.35MB — well under the 16MB limit. The artifact size peaks mid-training when weights are high-entropy, then drops as warmdown smooths them.

### BPB vs Steps (ASCII plot)

Power-law decay with three distinct phases: rapid early learning, mid-training plateau, warmdown-driven final drop.

```
BPB
4.10 |*
     |
     |
     |
2.50 |
     |
1.26 | *
1.23 |  *
1.22 |   *
1.20 |    *
1.19 |     * * * * *
1.18 |             * *
1.17 |               *
1.16 |                *
1.15 |                 *
1.13 |                  *
1.12 |                   *
1.09 |                    *
     +----+----+----+----+----+-> steps (K)
     0   10   20   30   40   50

     |<early >|<--- plateau --->|<warmdown>|
      (rapid)                    (sharp drop)
```

The plateau between steps 20K–30K shows diminishing returns from additional gradient updates alone. The sharp drop after step 30K is driven by warmdown reducing the learning rate toward zero, smoothing weights and simultaneously improving both BPB and compressibility.

### Artifact Size vs Steps (ASCII plot)

Non-monotonic: grows rapidly to a peak at ~15K steps, plateaus above 16MB, then shrinks back below budget during warmdown.

```
MB
17.2 |         * * * * * * * * * * * *
16.8 |                               *
16.4 |                                *
16.0 |------------------------------------*--------  16MB limit
15.7 |                                    *
15.1 |                                     *
14.7 |     *                                *
14.1 |  *                                    *
13.1 | *
 4.6 |*
     +----+----+----+----+----+-> steps (K)
     0   10   20   30   40   50

     |<-fits->|<--- OVER 16MB ------>|<fits->|
```

Intermediate checkpoints between steps ~10K–42K exceed the 16MB budget and cannot be submitted as-is. Only the final model (with warmdown complete) fits. This means **early stopping is not viable** for this architecture without a separate warmdown phase.

### Observations

1. **BPB vs steps follows a power-law decay** with diminishing returns. The biggest gains are in the first 7,500 steps. Beyond 25K–30K steps, warmdown begins and BPB drops sharply again.
2. **Artifact size is non-monotonic**: it grows rapidly from 4.6MB (init) to 17.2MB (step 10K–15K peak), plateaus during mid-training, then shrinks back to 14.3MB during warmdown. This means intermediate checkpoints without warmdown may exceed 16MB even if the final model fits.
3. **TTT gain scales with compute**: at 7K steps (SOTA), TTT provides -0.0025 BPB. At 20K steps, -0.006 BPB. At 50K steps, -0.014 BPB. The better-trained base model benefits more from test-time adaptation.

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

| Parameter | SOTA (9K steps) | 20K steps | 50K steps |
|-----------|----------------|-----------|-----------|
| ITERATIONS | 9,000 | 20,000 | 50,000 |
| WARMDOWN_ITERS | 3,500 (38.9%) | 7,800 (39.0%) | 19,500 (39.0%) |
| MUON_MOMENTUM_WARMUP_STEPS | 1,500 (16.7%) | 3,340 (16.7%) | 8,350 (16.7%) |
| MAX_WALLCLOCK_SECONDS | 600 | 0 (unlimited) | 0 (unlimited) |

## Run Commands

### 20K steps
```bash
RUN_ID=run_recordsota_nonrecord_seed1337 \
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=3340 WARMDOWN_ITERS=7800 \
ITERATIONS=20000 MAX_WALLCLOCK_SECONDS=0 EVAL_STRIDE=64 \
SEED=1337 \
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 train_gpt.py
```

### 50K steps
```bash
RUN_ID=scaling_50k_seed1337 \
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
- 829ms/step (vs 83ms/step on 8×H100 SXM — exactly 10× slower)
- grad_accum_steps=2 (to match 8-GPU effective batch size of 786,432 tokens)
- 20K run: ~4.6h training + ~34min TTT eval
- 50K run: ~11.5h training + ~34min TTT eval

## Credits

This submission uses the full architecture and code from the record-track SOTA with no ML changes — only extended compute and proportionally scaled LR schedules.

- **Base submission [PR #549](https://github.com/openai/parameter-golf/pull/549)**: @abaybektursun — LeakyReLU² + Legal TTT + Parallel Muon
- **LeakyReLU² activation**: [PR #493](https://github.com/openai/parameter-golf/pull/493) by @parinzee, [PR #518](https://github.com/openai/parameter-golf/pull/518) by @sofiabod
- **Optimizer (Parameter Banking + Parallel Muon)**: [PR #399](https://github.com/openai/parameter-golf/pull/399) by @abaybektursun
- **TTT recipe**: [PR #461](https://github.com/openai/parameter-golf/pull/461) by @Christopher-Lee-McClendon
- **Base model**: [PR #414](https://github.com/openai/parameter-golf/pull/414) by @signalrush
