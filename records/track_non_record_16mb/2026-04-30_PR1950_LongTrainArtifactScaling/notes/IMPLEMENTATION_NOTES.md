# Implementation Notes

## Checkpoint Export Strategy

The checkpoint logic is inserted into the main training `while True` loop in
`train_model()`, after `step += 1` and `approx_training_time_ms` computation.

### Flow per milestone

```
1. Check if approx_training_time_ms/1000/60 >= target_minute
2. Pause training timer:
   - torch.cuda.synchronize()
   - Accumulate elapsed time into training_time_ms
3. Clone current non-EMA model state_dict (GPU tensors)
4. Apply EMA weights to base_model
5. Redirect h.model_path and h.quantized_model_path to per-checkpoint subdirectory
6. Call serialize(h, base_model, code_text):
   - collect_hessians() runs forward passes through model (~3.5s)
   - gptq_mixed_quantize() quantizes weights (~2s)
   - Per-group lrzip compression (~120s)
   - Writes .ptz file
7. Restore original artifact paths on h
8. [full mode only] Run eval_val() for diagnostic BPB
9. Restore original non-EMA weights to base_model
10. Write checkpoint metadata JSON
11. Resume training timer:
    - torch.cuda.synchronize()
    - Reset t0 = time.perf_counter()
12. Break inner for-loop (one checkpoint per training step)
```

### Why this is safe

- **Optimizer state** is never touched. The optimizer references `base_model.parameters()`,
  which are the same tensor objects. We only change their `.data` via `load_state_dict`,
  then restore them. The optimizer's momentum buffers remain valid because they reference
  parameter tensors by identity, and `load_state_dict` with `strict=True` fills existing
  tensor storage in-place.

- **EMA state** is a separate dict of float32 clones. It accumulates from
  `base_model.state_dict()` each step. Since we restore non-EMA weights before the next
  step's EMA update, the EMA continues to track the training trajectory correctly.

- **`looping_active`** is a boolean on `base_model`. We do not modify it during export.
  `serialize()` calls `collect_hessians()` which runs forward passes — these use whatever
  `looping_active` state exists. Since the EMA model has the same architecture, this is
  fine. The flag is restored automatically when we restore weights (it's not in state_dict).

- **Training time accounting** excludes export time. We accumulate into `training_time_ms`
  before export, then reset `t0` after. The wallclock cap check uses `approx_training_time_ms`
  which is computed from `training_time_ms + elapsed_since_t0`, so export time is invisible
  to the training schedule.

- **Distributed sync**: We call `dist.barrier()` after creating the checkpoint directory
  so all ranks see it before serialize writes files. The serialize function itself handles
  rank 0 file writes internally.

## Key Assumptions

1. `serialize()` only reads model weights (via `state_dict()` and forward passes).
   It does NOT modify model state, optimizer state, or the training data loader.

2. `load_state_dict(strict=True)` copies data into existing tensor storage. This
   preserves parameter tensor identity, keeping optimizer param_groups valid.

3. The `_longtrain_code_text` variable is read once at the start of `train_model()`
   to avoid re-reading the file at each checkpoint.

4. Only one checkpoint can be exported per training step iteration (the `break`
   at the end of the for-loop). If training jumps past multiple milestones in one
   step (unlikely at typical step rates), remaining milestones are caught on the
   next step.

## Overhead Analysis

Each checkpoint export costs ~130s wall time (dominated by lrzip compression):
- collect_hessians: ~3.5s
- GPTQ quantization: ~2s  
- Per-group lrzip: ~120s
- State dict clone/restore: ~0.5s
- JSON write: negligible

With 5 milestones, total overhead ≈ 650s. This is NOT counted against training time.

Total wall time for 60-min training + 5 checkpoints + final export + TTT eval:
≈ 3600 + 650 + 130 + 300 = 4680s ≈ 78 min

## Interpreting Results

The primary output is the set of `checkpoint_*min.json` files. Feed them to
`scripts/analyze_scaling.py` to get:

- `scaling_results.csv` — raw data table
- `scaling_summary.json` — structured summary with recommendation
- `scaling_summary.md` — human-readable report

The recommendation thresholds are:
- **STRONG_POSITIVE**: ≥300 KB artifact shrink + BPB improvement → test larger model
- **MODERATE_POSITIVE**: 50-300 KB shrink → report scaling benefit
- **QUALITY_ONLY**: BPB improves but artifact doesn't shrink
- **NEGATIVE**: no clear benefit

If the result is STRONG_POSITIVE, run `scripts/make_larger_variant_plan.py` to
generate candidate configurations for a larger model that fits within 16 MB.
