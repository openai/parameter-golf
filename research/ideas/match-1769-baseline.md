# Idea: Match #1769 baseline (clip=12 + TTT on 008 model)

## Problem

We have never cleanly measured our own pipeline against #1769's result. Our current
situation:

- **#1769** (dexhunter, clean #1736 + MLP_CLIP_SIGMAS=12.0): post-TTT **1.06453** (5-seed)
- **Spec 026** (cross-layer carry + clip=12 + TTT): post-TTT **1.06582**
- **Spec 021e** (recur-alpha frozen + clip=12 + TTT): post-TTT **1.06622**
- **Spec 009 baseline** (008 model + clip=**10** + TTT): post-TTT **1.06728**

We have never run: 008 model + clip=**12** + TTT.

Spec 008 reproduced #1736 exactly (pre-GPTQ float bpb **1.06922**, seed 42, 4828 steps)
with `MLP_CLIP_SIGMAS=12.0` already set in the env. But spec 008 never ran TTT — the
training wallclock filled the 600s budget, GPTQ ran, log ended. Spec 009 then ran GPTQ+TTT
on the 008 pre_gptq.pt but accidentally used clip=10 (SpinQuant script default), producing
the 1.0673 number.

The 008 pre_gptq.pt is preserved at `/runpod/runs/008-1736-reproduction/seed_42/pre_gptq.pt`
and is the cleanest float model we have — same code, same seed, no experimental changes.

## What to run

Re-run spec 009's SpinQuant eval script in baseline mode (no rotation) but with
`MLP_CLIP_SIGMAS=12.0` set explicitly. Commit `6456188` has the eval script.

```bash
git checkout 6456188

MLP_CLIP_SIGMAS=12.0 \
PHASED_TTT_ENABLED=1 PHASED_TTT_PREFIX_DOCS=2000 PHASED_TTT_NUM_PHASES=3 \
GPTQ_CALIBRATION_BATCHES=16 GPTQ_RESERVE_SECONDS=4 \
ARTIFACT_DIR=/runpod/runs/028-match-1769-baseline \
torchrun --standalone --nproc_per_node=8 spinquant_eval.py \
  --mode baseline \
  --ckpt /runpod/runs/008-1736-reproduction/seed_42/pre_gptq.pt \
  > /runpod/runs/028-match-1769-baseline/run.log 2>&1
```

No training. GPTQ (~60s) + TTT (~400s) only. Cost ~$3, ~8 minutes on 8×H100 JP.

## What it tells us

| result | interpretation | action |
|---|---|---|
| post-TTT ≈ 1.064 | Our pipeline matches #1769. Cross-layer carry hurts GPTQ by +0.002. | Close cross-layer carry arc. Stack 027 levers on clean baseline. |
| post-TTT ≈ 1.066–1.067 | Our GPTQ code produces worse results than #1769 regardless of model. | Diff our GPTQ code against #1769 line by line. |

## Why this matters

Every spec from 026 onward is being evaluated relative to a baseline of ~1.066. If our
clean baseline is actually ~1.064 (matching #1769), then:

1. Spec 026's 1.06582 is only −0.0004 vs our true clean baseline — cross-layer carry is
   nearly neutral, not the +0.004 improvement 025b's training signal suggested.
2. Spec 027 (depth curriculum + LoRA warm-start-A) targeting ~1.062 is competing against
   a bar that may already be at 1.064 — we need net −0.002, not −0.004.
3. The accept criteria for all open specs need to be updated.

## Cost

~$3 (eval-only, no training). Should be the first thing run before any further 8×H100 spend.
