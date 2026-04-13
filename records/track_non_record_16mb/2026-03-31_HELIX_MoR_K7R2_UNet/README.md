# HELIX MoR K7R2 U-Net: SOTA-base recurrence study

This folder contains a non-record 16MB, 3-seed run of the HELIX MoR K7R2 U-Net variant on 8xH100. The intent is to preserve the proven SOTA stack and test recurrence-style parameter sharing as the primary architecture variable.

## Architecture Breakdown

This model keeps most of the established winning recipe and introduces MoR weight sharing:

1. **Proven SOTA base (kept intentionally)**
   - GQA (`NUM_HEADS=8`, `NUM_KV_HEADS=4`)
   - Partial RoPE (`ROPE_DIMS=16`) + selective XSA in late blocks
   - LeakyReLU(0.5)^2 MLP with `MLP_MULT=3`
   - Smear-style and bigram-aware token representation path
   - EMA/SWA style stabilization and sliding-window eval

2. **MoR recurrence with U-Net structure**
   - `NUM_UNIQUE_BLOCKS=7`, `NUM_ITERATIONS=2` -> 14 virtual layers
   - Shared core weights across iterations, with encoder-decoder style skip reinjection
   - Goal: preserve depth-like compute behavior while reducing unique parameter footprint

3. **Per-virtual-layer control tensors**
   - Layer-specific scales/mixes to avoid identical transforms at each virtual position
   - Routed to scalar optimizer path (not Muon) so shared matrix banks remain stable

4. **Competition-aware export path**
   - int6 per-row quantization + lzma compression
   - end-to-end training/export flow designed around final submission artifact behavior

## Why this design makes sense

- **Strong baseline + controlled novelty:** Keep high-performing defaults, change one major axis (recurrence) for interpretable A/B outcomes.
- **Parameter budget efficiency:** Sharing unique blocks lets compute depth increase without linear artifact growth.
- **Practical submission focus:** The architecture is evaluated in post-quantized form, not only in train-time precision.

## Run Configuration

```bash
SEED=1337,1338,1339 \
RUN_ID=helix_mor_k7r2_unet_seed${SEED} \
NUM_UNIQUE_BLOCKS=7 \
NUM_ITERATIONS=2 \
MODEL_DIM=512 \
NUM_HEADS=8 \
NUM_KV_HEADS=4 \
MLP_MULT=3 \
TRAIN_BATCH_TOKENS=786432 \
TRAIN_SEQ_LEN=2048 \
ITERATIONS=20000 \
WARMUP_STEPS=20 \
MAX_WALLCLOCK_SECONDS=600 \
MATRIX_LR=0.025 \
SCALAR_LR=0.025 \
TIED_EMBED_LR=0.035 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Results Summary

All seeds were wallclock-limited before full training completion. The best seed is used as the canonical metadata entry in `submission.json`.

| Seed | Step at stop | Pre-quant BPB | Post-EMA BPB | Final int6 sliding BPB | Legal TTT BPB | Bytes total |
|---|---:|---:|---:|---:|---:|---:|
| 1337 | 1393/20000 | 1.2824 | 1.3070 | 1.3663 | 1.3105 | 7,274,404 |
| 1338 | 1266/20000 | 1.3047 | 1.3328 | 1.4558 | 1.3529 | 6,599,512 |
| 1339 | 1320/20000 | 1.2962 | 1.3237 | 1.4339 | 1.3550 | 6,788,072 |

Additional best-seed (1337) details:

| Metric | Value |
|---|---|
| Model params | 17,563,708 |
| Train time | 600,548 ms |
| Step average | 431.12 ms |
| Serialized int6+lzma model | 7,182,636 bytes |
| Code size | 91,768 bytes |
| Total submission size | 7,274,404 bytes |

## What can be improved next

The architecture shows upside, but this run under-delivered because it did not execute enough steps inside the 10-minute budget. The bottleneck was usage/runtime throughput behavior (including environment-level friction in Runpod setups), not just model quality.

Priority work for the next owner:

1. **Make 600s count**
   - Improve steady-state step time and remove runtime inefficiencies so many more optimization steps fit under the same wallclock.

2. **Recurrence cost control**
   - Keep K7R2 concept, but trim expensive recurrence-path overhead where BPB-per-ms is poor.
   - Rebalance shared vs per-virtual control compute to improve training velocity.

3. **Finish intended optimization schedule**
   - Current runs stop far before completion (`~1.3k / 20k` steps).
   - The next breakthrough should come from fitting the full intended training progression into a strict sub-10-minute execution path.

Given the architecture quality and early-trajectory behavior, confidence remains high that a fully optimized 10-minute execution can push this line of work much closer to, or into, SOTA territory.

## Included files

- `train_gpt.py` (exact training script used)
- `run_3seeds.sh` (3-seed launcher)
- `runs/seed_1337|1338|1339/*` (artifacts and logs)
- `submission.json` (canonical metadata from best seed)
