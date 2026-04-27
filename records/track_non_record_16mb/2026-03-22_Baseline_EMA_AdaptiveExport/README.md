# Non-Record Submission: Baseline + EMA + Adaptive Export (2×H100)

## Result

This run reaches **`1.17251579 val_bpb`** on the final post-quant sliding-window roundtrip evaluation.

Key numbers:

- Final post-quant sliding-window: `val_loss=1.97973856`, `val_bpb=1.17251579`
- Pre-export validation at stop: `val_loss=2.0093`, `val_bpb=1.1900`
- Step at stop: `3807`
- Training wallclock: `1200.112s`
- Final eval wallclock: `613.676s`
- Total artifact size: `16,399,881 bytes`
- Budget miss: `449,881 bytes`

This is therefore **non-record only**: the run produced a strong final score shape, but it does **not** satisfy the 16MB artifact cap yet.

## What Changed

This submission starts from the strong `Int6 MLP3x + SmearGate + BigramHash + Muon` baseline and changes only two things:

1. **Late-stage EMA**
   - `EMA_ENABLED=1`
   - `EMA_BETA=0.9998`
   - `EMA_START_FRAC=0.8`

2. **Adaptive export-time pruning search**
   - `PRUNE_CANDIDATES=0.00,0.01,0.02,0.03,0.04,0.05`
   - `TARGET_ARTIFACT_BYTES=15950000`
   - select the smallest pruning ratio that meets the target, or the smallest artifact if none do

The backbone, optimizer family, mixed quantization setup, and final evaluation path were otherwise kept close to the strong merged baseline line.

## Why This Is Worth Looking At

The point of this run is not "new backbone, new SOTA." The point is to test a narrower hypothesis:

> under a hard artifact budget, some of the remaining optimization headroom may come from **weight smoothing before export** and **budget-aware export selection**, not only from changing the training architecture.

Even though the run is not leaderboard-valid yet, it is still informative for three reasons:

1. The final score shape is already competitive for a non-record run under limited compute.
2. The failure mode is narrow: the bottleneck is **artifact size**, not a collapse in post-quant quality.
3. The next steps are unusually clear:
   - extend pruning search beyond `5%`
   - move from global pruning to module-aware budget allocation
   - compare against a minimal top-record export-only fork

That makes this a useful potential-PR result rather than just another failed oversized run.

## Submission Checklist

- Training completes under wallclock cap: `yes`
- Final post-quant roundtrip evaluation runs successfully: `yes`
- Sliding-window final eval runs successfully: `yes`
- Code is self-contained in `train_gpt.py`: `yes`
- Artifact is under the 16MB target: `no`
- Multi-seed verification: `no`

## Compute Limitation

This run was produced under constrained compute:

- `2×H100`, not `8×H100`
- single seed only
- no remaining compute budget for additional tuning passes after the first full validation run
- substantial runtime spent in final sliding-window evaluation

So this should be read as a validated directional result, not as a fully swept or fully scaled submission.

## Included Files

- `train_gpt.py` — code snapshot for this non-record run
- `log.txt` — exact training + export + final evaluation log
- `submission.json` — run metadata
