# Medusa: Unstable S2 — DeltaNet Crawler, Legal Resubmission

**val_bpb: 0.8822** (3-seed mean) | **~9.9MB** | 8xH100 SXM

Legal resubmission of PR #1028 (Medusa: Unstable, mean 0.9984 BPB).

**Legality fix:** PR #1028 was flagged because `gptq_calibrate_loop_aware()` reads 256 batches from training data after the 600s wallclock cap fires. Fix: `GPTQ_RESERVE_MS=30000` stops the training loop 30s early (~570s) so GPTQ calibration (~12s) completes within the budget. The log prints elapsed time at GPTQ start for reviewer verification:
```
stopping_early: wallclock_cap train_time:570052ms step:4642/20000
gptq:loop-aware calibrated 41 layers in 11.4s
```

All hyperparameters are identical to PR #1028 / Medusa_IV.

## Results

| Seed | BPB (sliding window) | Post-EMA BPB | Int6 Roundtrip | Steps |
|------|--------------------:|-------------:|---------------:|------:|
| 300  | 1.0251 | 0.6484 | 0.8987 | 4628 |
| 444  | 0.8469 | 0.4330 | 0.7159 | 4616 |
| 4    | **0.7744** | 0.4339 | 0.6271 | 4642 |
| **Mean** | **0.8822** | | | |
| **Std dev** | **~0.105** | | | |

3-seed mean improved from 0.9984 (PR #1028) to 0.8822 with the timing fix.

## Architecture

- **Topology**: 4 flat layers + 1 crawler layer × 4 loops (Frugendorff compression)
- **INST_DIM**: 32 (flow instructions)
- **DeltaNet**: 4 heads, canonical `chunk_delta_rule` from `fla.ops.delta_rule`
- **Quantization**: int6+zstd + CRAWLER_QUANT_INT8=1, loop-aware 2-phase GPTQ (41 layers)
- **Dims**: XSA_LAST_N=11, BIGRAM_VOCAB_SIZE=2048, ROPE_DIMS=16
- **Schedule**: WARMDOWN_ITERS=2000, SWA_EVERY=50, EMA_START_STEP=4400, EMA_DECAY=0.99
- **GPTQ_RESERVE_MS**: 30000 (training stops at ~570s; GPTQ runs within budget)

## Legality

1. No n-gram eval — sliding window only
2. No val data used during training
3. GPTQ calibration reads training data and runs **inside** the 600s wallclock budget (verified via `gptq:loop-aware calibrated 41 layers in ~11.5s` at ~570s elapsed)
4. Score-first protocol not applicable (no n-gram cache)

## Known Issues

High cross-seed variance (std dev ~0.105) is caused by DeltaNet heads. Two root causes identified:
1. **State dtype bug**: `chunk_delta_rule` returns Float32 `new_state` in BF16 training — causes recompile_limit warnings during eval (does not affect final score, only eval speed). Fix exists in follow-on work.
2. **Quantization unravel**: DeltaNet weight errors compound through 4 crawler loops.

Stabilization is active research.

## Reproduce

```bash
SEED=300 bash experiments/Medusa_Legal_unstable/run.sh
SEED=444 bash experiments/Medusa_Legal_unstable/run.sh
SEED=4 bash experiments/Medusa_Legal_unstable/run.sh
```

8xH100 SXM, 600s training per seed.

## Credits

- **Gated DeltaNet (GDN) — primary catalyst**: @shalyhinpavel (PR #875) — 1.0226 BPB pure neural
- **Canonical DeltaNet kernel**: `fla.ops.delta_rule` (flash-linear-attention)
- **Loop-aware GPTQ + Frugendorff crawler architecture**: @newjordan (PR #990, PR #1028)
