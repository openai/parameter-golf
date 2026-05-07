# PC1 Golf Experiment Status
**Last Updated:** 2026-03-21 15:17 EDT

## Current State
✅ **IDLE** — No active training processes

## Recent Runs (Last 48h)
All completed on **Fri Mar 20**:
- **Ablation series** (4 runs): Testing weight decay, ROPE, batch size, SWA window
  - Best: `abl_01_wd042` → **1.4183 val_bpb** @ step 1251
  - Runner: `abl_02_rope50k` → **1.4185 val_bpb** @ step 1248
- **Fast series** (2 runs): Baseline vs full stack
  - `fast_01_baseline` → **1.4255 val_bpb** @ step 1229
  - `fast_02_fullstack` → **1.4475 val_bpb** @ step 1131 (worse)

## Comparison to PR #259
**Current PR #259 target:** ~1.42 val_bpb (need to verify exact)
**Best PC1 result:** 1.4183 (abl_01_wd042)

→ **Marginal improvement** — not enough to update PR unless this is a new hyperparameter we haven't submitted yet.

## Next Steps
- Verify exact PR #259 baseline
- Check if `abl_01_wd042` config differs meaningfully from submitted version
- No `next_experiments.sh` found — PC1 awaiting new batch
