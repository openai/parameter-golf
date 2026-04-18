# Execution notes — spec 000

## Outcome
Run **completed**, final post-TTT val_bpb **1.08621766**. **Outside accept window [1.079, 1.083] by +0.0032.**

## Timeline (UTC)
- `20:14:01` pod `t7k5v85j3fwpdh` created (8×H100 NA-1, $23.92/hr)
- `20:14:01 → 20:22:56` initial preflight + accidental stop after SSH heredoc issue (~$3.60 wasted on provisioning churn)
- `20:23:xx` pod resumed, brotli reinstalled, setsid launch
- `20:24:00` training start (estimated from log times)
- `20:33:xx` `stopping_early: wallclock_cap train_time: 588040ms step: 3849/20000`
- `20:47:30` TTT final val_bpb landed, pod stopped immediately

## Throughput deficit (primary finding)
Our pod ran at **~85% of SOTA-submission pod's step rate** in the same 588s training window.

- Ours: 3849 steps in 588s → **~6.5 steps/sec**
- SOTA: 4550 steps in 588s → **~7.74 steps/sec**
- Missing ~700 steps ≈ 0.005 bpb deficit, which tracked cleanly through every eval stage:

| Stage | Ours | SOTA | Δ |
|---|---|---|---|
| Training-end val (step 3849 vs 4550) | 1.0938 | 1.0886 | +0.0052 |
| Post-EMA, pre-quant | 1.09289 | 1.08735 | +0.0055 |
| Quantized | 1.10430 | 1.09970 | +0.0046 |
| Quantized + sliding window | 1.08774 | 1.08286 | +0.0049 |
| Quantized + TTT (final) | **1.08622** | **1.08079** | **+0.0054** |

The gap is **consistent** across stages — strong signal that the miss is *purely* the training-steps deficit, not a code or pipeline bug. Quant / sliding / TTT deltas within the run match SOTA's expected contributions almost exactly:

| Within-run contribution | Ours | SOTA |
|---|---|---|
| EMA effect | −0.0009 | −0.0013 |
| Quant penalty | +0.0114 | +0.0124 |
| Sliding window gain | −0.0166 | −0.0168 |
| TTT gain | −0.0015 | −0.0021 |

TTT is doing work, just slightly less than SOTA (−0.0015 vs −0.0021). The under-trained base model likely has slightly less "signal" for TTT to amplify.

## Root cause of throughput gap
Leading hypothesis: **hardware heterogeneity in Runpod's H100 SXM pool**. Same nominal hardware (8×H100 80GB HBM3) but likely different NVLink/NVSwitch topology or host CPU/memory bandwidth than the SOTA submission's pod. Our instantaneous tok/s (4.3M early, 5.1M stable) was persistently below SOTA's (7.5-7.7M early, 6.1-6.6M late). Not a code issue — code commit matches.

Other candidates ruled out: identical commit (01e6fcf), identical env (+ BIGRAM_VOCAB_SIZE=0, QK_GAIN_INIT=5.25, TTT_ENABLED=1), torch 2.9.1+cu128 current, data locally mounted on fast network volume.

## Cost
- Training/eval pod time: ~24 min @ $23.92/hr ≈ **$9.50**
- Earlier churn (provisioning + unintended stop): ~9 min @ $23.92/hr ≈ **$3.60**
- **Total**: **~$13.10** (vs $3.50-6 in original spec estimate)

## Artifacts produced
- `train.log` — full stdout/stderr from torchrun (size ~4KB, not gzipped)
- `launch.out` — wrapper script output (empty; all log lines went via `tee` to `train.log`)
- `final.json` — schema-compliant with val_bpb, hardware, timing, launch env, checkpoint list
- `checkpoints.md` — pointer file to 9 phase-boundary checkpoints on volume (2.7 GB total)
- **No `loss_curve.csv`** — was in EXECUTION.md default schema but low-priority given we have the structured info; can be derived from `train.log` if research needs it

## Key takeaways for research
1. **Spec 000 did NOT hit the accept criterion** on this hardware. The miss (+0.0032 bpb outside window) is **purely a throughput artifact**, not a code/config bug. The ~0.005 per-stage tracking gap is the cleanest possible evidence that our code is faithful to SOTA.
2. **Next spec should either** (a) shop for a faster pod via secure-cloud or tok/s calibration preflight, (b) accept the deficit and compare Δ vs our 1.08622 baseline rather than 1.0810, or (c) both.
3. **Checkpoints are usable as hotstart seeds** for downstream experiments — ckpt_final_pre_ema and ckpt_final_post_ema at step 3849 give us two starting points; phase-boundary ckpts (warmdown_start, pre_recurrence, final_post_ema) let downstream specs start from an already-training-validated mid-point.
4. **Pod churn is real** — initial SSH heredoc issue cost ~$3.60 in provisioning + abandoned-pod cost. Next session should use tmux/setsid from the start, avoid long heredocs over SSH.
