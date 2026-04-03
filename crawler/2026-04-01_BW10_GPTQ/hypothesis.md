# BW10_GPTQ — Hypothesis

**Parent:** BW8_Tap (BW5 + TAP_DIM=32 shared — stacking signals, not yet full run)
**One variable:** `LOOP_AWARE_GPTQ=1` (post-training Hessian calibration for int6 quantization)

## Background

LOOP_AWARE_GPTQ was tested in the CL2 lineage (Bandit_ClownCar ancestor):
- Delta vs naive int6: **−0.0062 BPB** at gate scale (strong signal)
- ClownCar full run 1.1874 BPB included GPTQ

When `COMPILE_FULLGRAPH=1` was added in BW5, GPTQ was stripped — PyTorch's fullgraph
compile is incompatible with forward hooks used by GPTQ calibration.

## Solution

GPTQ runs **post-training** on `base_model` (the uncompiled reference model that
already exists in the training script). The compiled model handles training; the
uncompiled `base_model` handles calibration. No changes to the training graph.

Flow:
1. Training completes on `compiled_model` (fullgraph, no hooks)
2. GPTQ calibration runs on `base_model` (uncompiled) — hooks work fine
3. Hessians used for `mixed_quantize_int6_gptq` instead of naive quantization

## Loop-aware calibration (2 phases)

Standard GPTQ calibrates flat and crawler layers independently. But after flat
layers are quantized, the crawler receives drifted activations vs what it saw in
training — causing fixed-point unraveling in the Hessian estimate.

Loop-aware fix:
- Phase 1: standard Hessian collection for ALL layers
- Phase 2: patch flat_blocks with GPTQ weights, re-collect crawler Hessians
  → crawler now sees the real post-quantized-flat activations
- Merge: flat keeps Phase 1; crawler gets Phase 2

This was the version that showed −0.0062 BPB delta in CL2.

## Expected size impact

GPTQ typically produces slightly smaller compressed artifacts than naive int6
because the quantized weights are closer to the original (less residual noise
for zstd to encode). ClownCar showed size reduction vs naive.

## Gate target (4×GPU SDPA, 2000 steps)

Control = BW8 baseline (SKIP_GPTQ=1, naive int6)
Test = BW8 + LOOP_AWARE_GPTQ=1, SKIP_GPTQ=0

Pass: GPTQ arm beats control int6_sw_bpb. Step overhead <5ms (GPTQ is post-training).
Fail: regression or >85ms step (indicates GPTQ running inside training loop — bug).

Historical reference: CL2 GPTQ delta −0.0062 BPB (4×GPU proxy environment).
Proxy inflation expected — 0.0003+ at full run would be meaningful.
