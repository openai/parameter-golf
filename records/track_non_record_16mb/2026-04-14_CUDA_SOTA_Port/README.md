# CUDA Port: Depth Recurrence + Parallel Residuals + Tuned Hyperparameters

**Author:** @seekerPrice
**Date:** 2026-04-14
**Track:** non_record_16mb (H100 validation pending)
**Status:** Code ready, awaiting compute credits to validate

## TL;DR

CUDA port of the techniques validated locally in PR #1612:
- **Depth recurrence** (env var `RECUR_LAYERS`, e.g., "3,4,5")
- **Parallel residuals** (env var `PARALLEL_RESIDUAL=1`)
- **Tuned hyperparameters** (via env vars: `MATRIX_LR=0.02 MUON_MOMENTUM=0.95 QK_GAIN_INIT=4.0`)

All features are **opt-in via env vars**. Default behavior = upstream baseline.

## Relation to PR #1612

PR #1612 proved in MLX:
| Config | val_bpb |
|--------|---------|
| SOTA defaults (Muon 0.99, QK 5.25) | 1.5596 |
| **Tuned (Muon 0.95, QK 4.0)** | **1.5096** |
| | **−0.0500** |

This CUDA port lets us validate the same recipe on H100 at 2.4B tokens.

## Changes vs upstream `train_gpt.py`

1. **Hyperparameters class** — added `recur_layers`, `recur_start_step`, `parallel_residual`, `parallel_start_layer` env vars
2. **Block.forward** — accepts `parallel: bool` parameter; when True, attn and MLP read from same pre-residual input
3. **GPT.__init__** — builds virtual-to-physical layer mapping for depth recurrence
4. **GPT.set_recurrence_active** — toggles recurrence on/off (typically called after warmup)
5. **GPT.forward** — iterates virtual layers, maps to physical via `self.v2p`
6. **Training loop** — activates recurrence at `RECUR_START_STEP` (or at init if 0)

All changes are **additive** — setting no env vars → identical behavior to upstream.

## Example run commands

### Baseline (no new features)
```bash
torchrun --nproc_per_node=8 train_gpt.py
```

### Match SOTA architecture (recurrence + parallel)
```bash
RECUR_LAYERS=3,4,5 \
RECUR_START_STEP=1500 \
PARALLEL_RESIDUAL=1 \
PARALLEL_START_LAYER=7 \
MATRIX_LR=0.02 \
MUON_MOMENTUM=0.95 \
QK_GAIN_INIT=4.0 \
torchrun --nproc_per_node=8 train_gpt.py
```

## Test plan
- [x] Code compiled and structure validated against MLX reference
- [x] Backwards compatible (no env vars = upstream behavior)
- [ ] H100 1-seed smoke test (pending Quick-start credits)
- [ ] Verify hyperparameter transfer: MLX −0.05 BPB → H100 ?
- [ ] 3-seed validation after Development grant

## Next steps
1. Obtain Quick-start H100 credits (8 hrs)
2. Run sanity: reproduce upstream baseline (~1.08 BPB)
3. Run tuned: our recipe (expected ~1.07 BPB or better)
4. Submit Development grant application with H100 evidence

## Why non-record?
This is code-ready but not yet validated on H100. Submitting as non-record for community visibility and to solicit feedback on the port before H100 run.

## Attribution
- Depth recurrence: @dexhunter (PR #1331, #1437)
- Parallel residuals: @Robby955 (PR #1412), @msisovic (PR #1204)
- Tuned hyperparameters: @seekerPrice (PR #1612, via 3-AI brainstorm)
