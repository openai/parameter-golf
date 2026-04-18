# Spec 002 — SWA + EMA blend screen — Summary

All 6 configs run on 1×H100 NA-1, single process (no DDP), using cached Hessian from `/workspace/runs/002-swa-plus-ema-1h-c0/hessians.pt` (232 MB, 67 keys, from C0's EMA-only weights).

Quant-only screen: `SLIDING_WINDOW_ENABLED=0` for C1–C5 (dropped after measuring C0's sliding eval at ~12 min on 1×H100, making the full 6×sliding run budget-prohibitive).

## Results

| cfg | description              | val_bpb_quantized | Δ vs C0   | artifact bytes | quant sec |
|-----|--------------------------|-------------------|-----------|----------------|-----------|
| C0  | EMA-only (control)       | **1.10518**       | (base)    | 15,979,642     | 169.9     |
| C1  | SWA all 4 post-recurrence| 1.14694           | **+0.042**| 15,977,443     | 135.1     |
| C2  | SWA late 3 (skip 1500)   | 1.12273           | +0.018    | 15,979,447     | 132.9     |
| C3  | 0.5 SWA + 0.5 EMA        | 1.12251           | +0.017    | 15,977,186     | 130.4     |
| C4  | 0.25 SWA + 0.75 EMA      | 1.11108           | +0.006    | 15,978,389     | 126.7     |
| C5  | 0.75 SWA + 0.25 EMA      | 1.13532           | +0.030    | 15,976,948     | 130.5     |

C0 also has `val_bpb_sliding = 1.08869` (from its full-pipeline run, `elapsed_sliding_sec = 717.5`). Other configs have `sliding_bpb = nan` (skipped).

## Verdict

**No signal. Clean kill.** Signal gate required `Δ_quant ≤ −0.0003` for at least one non-C0 config. **All 5 non-C0 configs are Δ > 0.** Pure EMA (C0) is the best config; any SWA fraction hurts.

Pattern is cleanly linear with SWA fraction:

| EMA fraction | SWA fraction | bpb     | Δ vs C0  |
|--------------|--------------|---------|----------|
| 100%         | 0%           | 1.10518 | 0.000    |
| 75%          | 25% (C4)     | 1.11108 | +0.006   |
| 50%          | 50% (C3)     | 1.12251 | +0.017   |
| 25%          | 75% (C5)     | 1.13532 | +0.030   |
| 0%           | 100% (C1)    | 1.14694 | +0.042   |

C2 (SWA of late 3 only, no EMA) at +0.018 is almost identical to C3 (half-EMA-blend SWA of all 4) — suggesting dropping step 1500 is roughly equivalent to adding 50% EMA weight. Either way, EMA wins.

## Validity gate

C0 on 1×H100 reproduced spec 001's λ=0 quantized bpb **bitwise-exactly** (1.1051789806396541 both runs). Confirms our sweep pipeline is deterministic on the same checkpoint + seed + calibration. Absolute number is ~+0.0009 off spec 000's 8×H100 1.10430 baseline; this is the expected 1-GPU-vs-8-GPU Hessian calibration offset, documented in spec 001's summary.

## Why SWA hurts here

Plausible interpretations (research to choose):

1. **EMA is already the right weight average.** SOTA's EMA decay (0.9965) over ~3849 steps gives an effective averaging window of ~285 steps. That's a much richer moving average than 4-snapshot uniform SWA. Adding 4-point SWA just injects noise from less-trained earlier checkpoints.
2. **The checkpoints we averaged are not equilibrated.** Ideal SWA averages snapshots from the flat low-LR plateau; spec-000's training ends with ~70% of the budget in *warmdown*. Snapshots at 1500 (LR still high, recurrence just activating), 2275, 3412 (mid-warmdown), and 3849 (near-zero LR) are from very different regions of the loss landscape.
3. **Quant penalty amplifies averaging noise.** Averaged weights might have lower-magnitude variance, which interacts with GPTQ's row-std-based clip to scale clip tighter, increasing quant error.

Without more experiments (e.g. SWA late-in-warmdown only, or SWA over a narrow post-convergence window), we can't disentangle these. Spec says to kill on first no-signal — that's the conclusion.

## Artifacts

On NA-1 volume at `/workspace/runs/002-swa-plus-ema-1h-c0/`:
- `hessians.pt` (232 MB) — reusable for any Hessian-based experiment on spec-000's post-EMA weights.
- `quantized_C0.ptz` through `quantized_C5.ptz` (~16 MB each, ~96 MB total) — retained if any follow-up wants to run sliding/TTT on a specific config.

In repo at `runs/002-swa-plus-ema/`:
- `config_C0.json` through `config_C5.json` — per-config results.
- `summary.md` (this file), `notes.md` (execution narrative), `sweep.out` (stdout/stderr).

## Handback

Research: evaluation + experiments.md row + promote/iterate/kill decision is yours. My read: clean kill — pattern is monotonic and large-magnitude, no regime where SWA helps over pure EMA on this checkpoint.
