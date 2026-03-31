# Parallel Muon (Batched Newton-Schulz)

## Score: val_bpb = TBD

## Hypothesis

Grouping weight matrices by shape and running batched Newton-Schulz orthogonalization reduces per-step overhead. Community achieves 83ms/step with parameter banking. On 1×GPU, this is a minor speedup; on 8×H100, it's critical (more steps in 10 minutes = lower BPB).

## Changes from exp06

- Added `zeropower_via_newtonschulz5_batched()` for 3D tensor [batch, rows, cols]
- Muon optimizer now groups params by shape, batches same-shape matrices for NS5
- Momentum applied first to all grads, then batched NS5, then all-reduce
- Correctness: should produce identical val_bpb to exp06 on 1×GPU (same math, different execution order)

## Architecture

Inherits from exp06 (all features) + Parallel Muon optimizer.

## Expected Impact

Same val_bpb as exp06 on 1×GPU. Faster ms/step → more training steps within wallclock on multi-GPU.

## Results

TBD — awaiting A100 run.
