# Experiment 0038_mamba2_kill_selectivity

Parent: 0035_mamba2_2of3 (current SSM-best, val 2.0399 single-seed)

## Question
**The decisive mechanism ablation for the writeup.** 0035 (Mamba-2 at 2 of 3 positions) lands at val 2.0399 — Δ vs transformer-best 2.0869 = -0.047 BPB. Three competing explanations:

- **(A) Selectivity is load-bearing**: input-dependent (dt, B, C) makes the SSM dynamics adapt per-token, recovering recall-like behavior. The headline mechanism claim.
- **(B) Parameter capacity**: Mamba-2 block has more params than S4D-Lin block (1.65 MB vs 0.66 MB int8 raw). The win is just "bigger block."
- **(C) Auxiliary structure**: conv1d, gating via z, in_proj producing multiple heads — these are absent in S4D-Lin and could be doing the work without selectivity.

This experiment kills selectivity while keeping (B) and (C) constant: replace input-dependent (dt, B, C) in the Mamba-2 forward with learned per-head/per-state CONSTANTS. The block becomes LTI but retains the same in_proj/conv1d/out_proj/A_log/dt_bias/D_skip/gate-via-z/SSD-chunkwise scan. If the win is from selectivity, this should regress to S4D-Lin-family floor (~2.16-2.22). If the win is from auxiliary structure or parameters, val should stay ≈ 2.04.

## Hypothesis [CONJECTURE]
val_bpb in [2.04, 2.20]. Outcome decomposition:

- **val ≈ 2.04 (win is parameters/structure)**: ~25% likely. Headline weakens to "Mamba-2's BLOCK is the lever, not selectivity per se. Could swap selective scan for FFT-conv with same auxiliary structure."
- **val ∈ [2.10, 2.16] (selectivity matters substantially)**: ~50% likely. Headline stands: "Selectivity recovers ~0.07-0.12 BPB at our regime."
- **val > 2.16 (selectivity is the dominant mechanism)**: ~25% likely. Strongest possible writeup outcome — selectivity is THE mechanism.

## Change
**In-place edit on the experiment-folder train_gpt.py** (no subagent, ~15 lines):
- Mamba2Block.__init__: read MAMBA2_KILL_SELECTIVITY env var; if =1, construct `_B_const`, `_C_const` as `nn.Parameter` of shape (d_state,) initialized like the random-projection init for B/C.
- Mamba2Block.forward: when killed, override `dt = zeros_like(dt)` (so softplus(0+dt_bias) is the constant per-head dt), `B_in = _B_const.expand(b, l, d_state)`, `C_in = _C_const.expand(b, l, d_state)`. Everything else identical.
- env.sh: `MAMBA2_KILL_SELECTIVITY=1`. Inherit all 0035 settings (Mamba-2 at positions 0,1; ATTN at 2; BigramHash; schedule).

When MAMBA2_KILL_SELECTIVITY=0 (default), behavior is byte-identical to vanilla Mamba-2.

## Disconfirming
- val < 2.05 (or ≈ 2.04): selectivity is NOT load-bearing — the win is parameters/auxiliary structure. Big writeup pivot needed.
- val ≥ 2.16: selectivity IS the mechanism — the writeup story holds.
- val ∈ [2.05, 2.16]: partial. Quantify what fraction of the 0.047 BPB win comes from selectivity vs other sources. Param-matched transformer (next experiment) helps disambiguate.

## Notes from execution
Direct in-place edit by main agent (~15 lines). _B_const, _C_const are 1D so auto-fp32 via ndim<2 rule. The unused B/C/dt outputs of in_proj are still produced and discarded — slight param waste (~6.5% of in_proj) but keeps in_proj shape identical to full Mamba-2 for cleanest ablation framing.
