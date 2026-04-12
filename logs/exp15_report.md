# exp15: Adaptive Depth — Experiment Report

## Idea
Decoder pass runs up to `MAX_PASSES` times (default 3). Each pass reapplies
encoder skip connections with its own learned weights. Tokens whose block-output
L2 delta falls below `ACT_THRESHOLD` (default 0.05) are frozen via masking.

## Architecture (based on exp13)
- 6 weight-tied layers × 2 passes (encoder + decoder), model_dim=672
- Pass 0 (decoder): unconditional, identical to exp13
- Pass 1+ (decoder): adaptive with convergence masking
- `num_skip_weights = num_layers × max_passes` (6→12 for max_passes=2, 6→18 for max_passes=3)
- Pass 1+ skip_weights zero-initialized (start as no-op, learn gradually)

## Results

### OOM: Cannot fit max_passes ≥ 2 in 80 GiB H100

| Config | Peak Memory | Status |
|--------|-------------|--------|
| exp13 (baseline, 1 decoder pass) | ~42 GiB | OK |
| exp15 MAX_PASSES=3, batch=786432 | >80 GiB | **OOM at compile** |
| exp15 MAX_PASSES=2, batch=786432 | >80 GiB | **OOM at compile** |
| exp15 MAX_PASSES=2, batch=524288 | ~70 GiB | **OOM at SWA checkpoint (~step 50)** |

Root cause: each additional decoder pass adds `num_layers` block activations
to the backward graph. With 2 passes, total block activations go from 12
(6 enc + 6 dec) to 18 (6 enc + 12 dec), ~1.5× memory. Combined with
torch.compile's inductor scheduling (which may materialize extra buffers),
this exceeds the 80 GiB limit even with reduced batch size.

### Learning dynamics (before crash)

When the run survived (steps 1–50 with batch=524288):
- Step 0 val_bpb: 4.1085 (matches exp13's 4.1073)
- Initial train_loss spike at step 2 (~9.6) is **normal** — exp13 shows the
  same Muon-induced spike (10.05 at step 2)
- Loss was recovering: 9.6 → 7.97 by step 10
- act_stats: avg_passes=2.00, active_ratio=0.000 (all tokens converge in
  adaptive pass due to zero-init — expected, would change as training progresses)

### Bug found and fixed: zero-init masking trap

Initial implementation applied convergence masking to ALL decoder passes
including pass 0. With zero-init output projections, block deltas were tiny
from the start → all tokens immediately masked → decoder blocks produced zero
gradient → train_loss diverged to ~10 and never recovered.

**Fix**: Pass 0 runs unconditionally (same as exp13). Only pass 1+ is adaptive.

### Bug found and fixed: skip accumulation corruption

Even with the pass-0 fix, skip_weights for pass 1+ were initialized to 1.0.
When adaptive masking froze a token (mask=0), the skip connection had already
been added to x. This caused encoder skip contributions to accumulate without
corresponding block transformations, corrupting the representation.

**Fix**: Zero-initialize skip_weights for pass 1+.

## Conclusions

1. **Memory is the primary blocker.** The adaptive decoder pass concept works
   architecturally but cannot fit in 80 GiB with the current model size.

2. **Possible paths forward:**
   - Gradient checkpointing for adaptive passes (recompute instead of store)
   - Smaller model_dim (e.g., 512 instead of 672) to free memory budget
   - Run on larger-memory GPUs (A100 80GB might have better fragmentation)
   - torch.compile memory optimization (e.g., activation offloading)

3. **The design requires pass 0 to be unconditional** — convergence masking
   from layer 0 is incompatible with zero-init training.

4. **Skip weights for adaptive passes must be zero-initialized** to avoid
   corrupting frozen tokens with skip accumulation.

## Files
- `train_exp15.py` — full implementation
- `logs/exp15_v3c.txt` — last run log (crashed at step ~50)
