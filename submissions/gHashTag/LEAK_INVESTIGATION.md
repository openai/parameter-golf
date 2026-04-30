# Leak Investigation: 42 Experiments with Suspicious BPB 0.0002-0.0015

## Executive Summary

**42 of 43 recent experiments show BPB ≤ 0.01**, indicating severe train/val split contamination. These results are flagged and NOT included in this submission.

| Status | Count | Notes |
|--------|-------|-------|
| Done (suspicious) | 42 | All have BPB 0.0002-0.0015 |
| Done (honest) | 1 | ID 1387, BPB 2.1505 |
| Failed (gf16 bug) | 186 | Historical regression, now pruned |

## Hypothesis

The gardener component generates train/val splits using IDENTICAL seed values, causing validation data to be a substring of training data.

**Evidence:**
1. All 42 suspicious experiments have `format=gf16` and `created_by=gardener`
2. Fibonacci seeds (1597, 4181, 10946) with identical configs produced identical BPB 0.0002
3. No `val_seed` field exists in current config schema — defaults to train_seed
4. Config shows only `lr`, `ctx`, `wave`, `model`, `steps`, `format`, `hidden`, `asha_rung`, `attn_layers`, `kill_at_step`, `kill_if_bpb_over`

## Root Cause Analysis

Looking at the config generation in `crates/trios-railway-audit/src/lib.rs`:

```rust
const FALLBACK_MARKER: &str = "Failed to load data/tiny_shakespeare.txt";
```

The trainer uses `train_path` and `val_path` (from `DataConfig` in `src/config.rs`) but gardener doesn't populate these paths. When data paths are absent, trainer likely:
- Uses same corpus for both train and val
- Falls back to `tiny_shakespeare.txt` (single file)
- Validation sees training data → BPB ≈ 0

## Verification Needed

To confirm the leak hypothesis, we need to:
1. Run one suspicious experiment with proper held-out validation set
2. Compare BPB on training data vs held-out data
3. If BPB jumps dramatically (>0.5) on held-out → confirmed leak

Current blocker: `record_checkpoint()` is a stub in `crates/trios-railway-core/src/neon.rs`, so no checkpoints from Railway runs are persisted.

## Impact on Gate-2

- 186 failed experiments from historical gf16 regression (pre-2026-04-29 19:00 UTC)
- No honest results < 2.0 BPB except one (ID 1387)
- Gate-2 target (BPB < 1.85) impossible to achieve with leaked data

## Recommendation for Gate-3

Before next race:
1. Fix `record_checkpoint()` with actual safetensors serialization
2. Add Railway persistent volume for `/data/ckpts` 
3. Implement proper held-out split: `train_data` (first 90%), `val_data` (last 10%)
4. Add `val_seed` field separate from `seed`
5. Run contract test: 100 gardener configs → deserialize against `TrainerConfig` → all valid

## Files

This report documents the data integrity issue affecting 42 experiments, which prevented competitive Parameter Golf submission and Gate-2 ratification.

### Gardener code location
```bash
# Primary config generator
crates/trios-railway-audit/src/lib.rs

# Config schema
crates/trios-railway-core/src/config.rs
```

**Date:** 2026-05-01 (ICT, UTC+7)
