# PR 1204 — Record: Parallel Residuals + Mini Depth Recurrence

**Author:** Marko Sisovic (msisovic)
**Branch date:** 2026-04-01
**Claimed BPB:** 1.1063 (3-seed mean, std 0.0017) | 1.8679 nats
**Artifact size:** ~15.94 MB (mean: 15,936,223 bytes)
**Seeds:** 1337, 42, 2024
**Hardware:** 8×H100 80GB SXM, 600s, no TTT

## Files retrieved
- `records__track_10min_16mb__2026-03-31_ParallelResiduals_MiniDepthRecurrence__README.md`
- `records__track_10min_16mb__2026-03-31_ParallelResiduals_MiniDepthRecurrence__train_gpt.py`
- `records__track_10min_16mb__2026-03-31_ParallelResiduals_MiniDepthRecurrence__submission.json`

## Environment variables (from reproduction command)
`SEED=$SEED POST_GPTQ_EVAL_ONLY=0 BIGRAM_DIM=112 MIXED_QUANT=1 N_INT6_LAYERS=32 NUM_LAYERS=11 RECUR_LAYERS=4,5 RECUR_START_STEP=3000 REPEAT_UNTIE_MLP=full REPEAT_UNTIE_MLP_LAYERS=4,5 DISABLE_LAYER0_ATTN=1 PARALLEL_RESIDUAL=1 PARALLEL_START_LAYER=7`

## Claimed changes (from README, verbatim)
"I started this submission from PR #1179, which gave me the base training stack I wanted to iterate on here. On top of that, I ported over the mixed-quantization and autoregressive GPTQ path from PR #1105.

## Parallel residuals
I took this idea from my modded-nanogpt record in KellerJordan/modded-nanogpt PR #230 and adapted it to this codebase. Starting from layer 7, attention and MLP read from different residual lanes, and each sublayer learns how strongly to write back into both lanes. The learned routing is quite asymmetric: MLP barely writes back into attention's residual stream, especially in the deeper partitioned layers.

## Mini Depth Recurrence
Instead of recurring the whole stack, I only repeated a couple of middle layers. Repeating one layer helped, repeating two consecutive layers helped more, and repeating three was already losing to the step-time penalty. I also swept the position of the repeated pair and found a clear sweet spot at layers 4,5, right around the U-Net hinge point. Delayed recurrence (RECUR_START_STEP=3000) beat always-on. Best to untie the repeated MLPs while leaving the rest of the recurrent block shared."
