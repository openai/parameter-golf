# Experiment 0042_kill_no_bigram

Parent: 0038_mamba2_kill_selectivity

## Question
**BigramHash interaction test (part 1 of 2)**: was BigramHash filling selectivity's "recall" niche, making selectivity redundant in 0038/0039?

The 0038/0039 finding (LTI Mamba-2 beats full Mamba-2 by 0.014 BPB) has the standing mechanism story "selectivity is anti-load-bearing at our regime". An alternative hypothesis: selectivity is a recall mechanism, BigramHash is also a recall mechanism, so they're redundant — and BigramHash is the better-tuned one at 200 steps. Removing BigramHash should let selectivity "show its work".

This experiment runs the kill (LTI) version *without* BigramHash. Companion 0043 will run the full (selective) version *without* BigramHash. The cross-comparison decomposes the mechanism.

## Hypothesis [CONJECTURE]
val_bpb in [2.040, 2.090]. Single-seed.

Reference points:
- 0024 (S4D-Lin sandwich + BG): 4-seed mean 2.0839
- 0017 (S4D-Lin sandwich, NO BG): single-seed 2.0945 — so removing BG from S4D-Lin family hurts ~0.011.
- 0038/0039 (kill Mamba-2 + BG): 2-seed mean 2.02723.

If BG's incremental value is similar (-0.010 to -0.015) for kill Mamba-2: predicted **val ≈ 2.04**.

Outcome decomposition (when paired with 0043):
- 0042 ≈ 0043 ≈ 2.04 (both no-BG land near full+BG): selectivity matters when BG is removed → BG was making selectivity redundant.
- 0042 < 0043 (kill still wins without BG): selectivity is genuinely anti-load-bearing regardless of BG. The 0038/0039 story is the right story.
- 0042 > 0043 (full wins without BG): clear evidence BG was making selectivity redundant.

## Change
**env.sh ONLY**: `BIGRAM_VOCAB_SIZE=0` (was 4096). All other 0038 settings unchanged: `MAMBA2_KILL_SELECTIVITY=1`, ATTN_LAYER_POSITIONS=2, MAMBA2_LAYER_POSITIONS=0,1.

## Disconfirming
- val < 2.025: kill is *better* without BG → confusing; would need re-investigate.
- val > 2.060: BG was doing >0.03 BPB of recall work; kill+no-BG falls past S4D-Lin no-attn. Suggests removing BG matters more than expected for the LTI family.
- val ∈ [2.030, 2.050]: expected; informative.

## Notes from execution
Direct env-var-only fork; no code changes. Ready to launch.
