# PR 1420 — Triple Loop + Fused Kernels + Parallel Residuals + N-gram Tilt

**Author:** Abay Bektursun (abaybektursun)
**Claimed BPB:** 1.08014 (submission.json) / 1.08309 (5-seed mean tilt, README table)
**Artifact size:** ~15,974,414 bytes mean
**Seeds:** 1, 42, 1234, 1337, 2025

## Files retrieved
- `records__track_10min_16mb__2026-04-06_TripleLoop_FusedKernels_Ngram__README.md`
- `records__track_10min_16mb__2026-04-06_TripleLoop_FusedKernels_Ngram__train_gpt.py`
- `records__track_10min_16mb__2026-04-06_TripleLoop_FusedKernels_Ngram__submission.json`

## Environment variables (from run command)
Not listed in a run_*.sh; seed-only invocation implied.

## Claimed changes (from README, verbatim)
> Changes:
> * One extra loop pass through layers 4-5. PR #1394 passes through layers 4-5 three times total (NUM_LOOPS=2, 15 virtual layers). I add a fourth pass (NUM_LOOPS=3), giving 17 virtual layers. Encoder [0,1,2,3,4,5,4,5], decoder [4,5,4,5,6,7,8,9,10]. Quadruple looping was worse.
> * Activate looping earlier (0.35 instead of 0.50). Swept {0.30, 0.35, 0.40, 0.50} on seed 1234. 0.35 won.
> * Fused MLP kernels (Triton TMA forward + CUTLASS EVT backward). Forward fuses leaky_relu(fc(x), 0.5).square() into a single Triton TMA kernel. Backward fuses (grad_out @ proj.weight) * act_grad into a CUTLASS 3.x Epilogue Visitor Tree. +127 training steps in 600s.
> * Parallel residuals for layers 7-10. GPT-J style: attention and MLP both read from the same pre-residual input, outputs summed in parallel. +68 training steps.
> * Eval-time n-gram tilt (causality-fixed). Original had causality violation caught by @Gusanidas in the within-word and word-start hint channels. Fix: only token_hint channel (orders 8-16) remains, provides -0.00014 BPB. Original -0.0029 delta came from (removed) target-dependent gating.
> * Double-buffered async data prefetch (background thread + pinned memory + separate CUDA stream).
> * PyTorch 2.9.1 instead of 2.11 (2.11 regressions documented with upstream PRs).
