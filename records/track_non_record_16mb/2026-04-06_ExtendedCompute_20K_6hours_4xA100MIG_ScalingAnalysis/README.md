# Extended Compute Scaling Analysis (20K Steps, ~6 Hours)

**val_bpb: 1.0960** (20K steps, 3-seed mean, std 0.0003) | **~15.05 MB** | 4×A100 MIG (Unlimited Compute Track)

## Summary

This submission is a **non-record submission**. It studies how ([PR #549](https://github.com/openai/parameter-golf/pull/549) by @abaybektursun) scales under extended compute, removing the 10-minute wall-clock constraint. The same architecture and code are trained for 20K steps (~6 hours) on 4×A100 MIG instances (approximately 10× slower per step than 8×H100 SXM).

Key findings:
- **20K steps achieves 1.0960 BPB post-TTT** (3-seed mean)
- **Artifact size balloons mid-training** (peaking at ~17.2MB around step 10K–15K) but **recovers to ~15.05MB** after warmdown completes — warmdown smooths weight entropy and restores compressibility
- **TTT gains scale with base model quality**: TTT provides -0.006 BPB on the 20K model 

## Results

### 20K steps, ~6 hours (4×A100 MIG, 3-seed comparison)

| Seed | step_avg | steps | Pre-TTT bpb | **Post-TTT bpb** | TTT gain | Artifact |
|------|----------|-------|-------------|-----------------|----------|----------|
| 1337 | 828.7ms | 20,000 | 1.1018 | **1.0957** | -0.0061 | 15,077,933 |
| 42 | 828.8ms | 20,000 | 1.1020 | **1.0962** | -0.0058 | 15,137,145 |
| 2024 | 839.8ms | 20,000 | 1.1017 | **1.0962** | -0.0055 | 14,942,394 |
| **Mean** | **832.4ms** | **20,000** | **1.1018** | **1.0960 (std 0.0003)** | **-0.0058** | **15,052,491** |

## Scaling Analysis: BPB & Artifact Size vs Training Steps

Artifact size is computed as int6+LZMA compressed model + code bytes. Every 2,000 steps.

### Seed 2024 (2,000-step intervals)

| Steps | Pre-TTT val_bpb | artifact_bytes | Under 16MB? |
|------:|--------:|---------------:|:-----------:|
| 0 | 4.1037 | 4,577,366 | Yes |
| 2,000 | 1.2651 | 13,800,942 | Yes |
| 4,000 | 1.2286 | 16,959,534 | **No** |
| 6,000 | 1.2122 | 17,243,366 | **No** |
| 8,000 | 1.2046 | 17,248,774 | **No** |
| 10,000 | 1.2007 | 17,246,738 | **No** |
| 12,000 | 1.1994 | 17,231,058 | **No** |
| 14,000 | 1.1835 | 16,929,622 | **No** |
| 16,000 | 1.1672 | 16,321,958 | **No** |
| 18,000 | 1.1429 | 15,534,274 | Yes |
| 20,000 | 1.1110* | 14,942,394 | Yes |

*Step 20K artifact reflects the final model with full warmdown applied. Intermediate checkpoints without warmdown exceed 16MB.

**Note:** Intermediate checkpoints do not benefit from warmdown. The final model has full warmdown applied, resulting in ~15.05MB — well under the 16MB limit. The artifact size peaks mid-training when weights are high-entropy, then drops as warmdown smooths them.

### BPB vs Steps (ASCII plot)

Power-law decay with two distinct phases: rapid early learning, then warmdown-driven final drop.

```
BPB
4.10 |*
     |
     |
2.50 |
     |
1.26 | *
1.23 |    *
1.22 |       *
1.20 |          *
1.19 |            * *
1.18 |                *
1.10 |                  *
     +---------+--------+-> steps (K)
     0        10       20

     |<early >|<warmdown>|
      (rapid)  (sharp drop)
```

### Artifact Size vs Steps (ASCII plot)

Non-monotonic: grows rapidly to a peak at ~15K steps, then shrinks back below budget during warmdown.

```
MB
17.2 |    * * * * * *
16.0 |--------------------  16MB limit
15.1 |                  *
14.1 | 
13.1 | *
 4.6 |*
     +---------+--------+-> steps (K)
     0        10        20

     |<-fits->|<over>|<fits>|
```

Intermediate checkpoints between steps ~10K–17.5K exceed the 16MB budget and cannot be submitted as-is. Only the final model (with warmdown complete) fits. This means **early stopping is not viable** for this architecture without a separate warmdown phase.

### Observations

1. **BPB vs steps follows a power-law decay** with diminishing returns. The biggest gains are in the first 7,500 steps, with warmdown driving the final sharp drop.
2. **Artifact size is non-monotonic**: it grows rapidly from 4.6MB (init) to ~17.2MB (step 10K–15K peak), then shrinks back to ~15.1MB during warmdown. Intermediate checkpoints without warmdown exceed 16MB.
3. **TTT gain scales with compute**: At 20K steps, -0.006 BPB. The base model benefits from test-time adaptation. 

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

| Parameter | This work (20K steps) |
|-----------|-----------|
| ITERATIONS | 20,000 |
| WARMDOWN_ITERS | 7,800 (39.0%) |
| MUON_MOMENTUM_WARMUP_STEPS | 3,340 (16.7%) |
| MAX_WALLCLOCK_SECONDS | 0 (unlimited) |

## Run Command

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
VAL_LOSS_EVERY=2000 \
SEED=1337 \
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 train_gpt.py
```

## Hardware

- 4×NVIDIA A100 MIG instances
- Seeds 1337 & 42: ~829ms/step | Seed 2024: ~840ms/step (vs 83ms/step on 8×H100 SXM — approximately 10× slower)
- grad_accum_steps=2 (to match 8-GPU effective batch size of 786,432 tokens)
- ~4.6–4.7h training + ~34min TTT eval = ~6 hours total per seed

## Credits

This submission uses the full architecture and code from the record-track PR #549 with no ML changes — only extended compute and proportionally scaled LR schedules.

- **Base submission [PR #549](https://github.com/openai/parameter-golf/pull/549)**: @abaybektursun — LeakyReLU² + Legal TTT + Parallel Muon
- **LeakyReLU² activation**: [PR #493](https://github.com/openai/parameter-golf/pull/493) by @parinzee, [PR #518](https://github.com/openai/parameter-golf/pull/518) by @sofiabod
- **Optimizer (Parameter Banking + Parallel Muon)**: [PR #399](https://github.com/openai/parameter-golf/pull/399) by @abaybektursun
- **TTT recipe**: [PR #461](https://github.com/openai/parameter-golf/pull/461) by @Christopher-Lee-McClendon
- **Base model**: [PR #414](https://github.com/openai/parameter-golf/pull/414) by @signalrush
