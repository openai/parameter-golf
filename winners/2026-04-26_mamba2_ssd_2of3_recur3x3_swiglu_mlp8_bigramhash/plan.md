# Experiment 0035_mamba2_2of3

Parent: 0032_mamba2_ssd_smoke (current best)

## Question
Does the Mamba-2/SSD win COMPOUND when used at MORE positions? Current best (0032/0034) has Mamba-2 at 1 of 3 unique blocks. This experiment uses Mamba-2 at 2 of 3, with 1 attention at end (position 2). Pattern per K=3 group: MAMBA2-MAMBA2-ATTN, looped ×3 = 6 Mamba-2 + 3 attn effective layers (vs current best's 6 attn + 3 Mamba-2).

If the +0.024 BPB win was about replacing the LTI block, we should expect ~+0.024 BPB more from the second Mamba-2. If it was about having ONE selective block among many attention layers, we should see saturation or even regression.

## Hypothesis [CONJECTURE]
val_bpb in [2.030, 2.090]. Single-seed exploration per directive.

Three scenarios:
- **Compound win (val < 2.045, ~30%)**: more Mamba-2 helps further. Saturation curve says "selectivity > attention" at our regime; could push past 2.04.
- **Saturate (val 2.045-2.075, ~50%)**: 2 mamba is roughly tied with 1 mamba; the attention layers were doing recall work that's hard to substitute.
- **Loss (val > 2.075, ~20%)**: too few attention layers (3) hurts recall; Mamba-2 doesn't fully compensate. Mirror of 0006/0008 pattern (no-attn loss).

Cap math: replacing 1 attn (788 KB int8) with 1 Mamba-2 (1.65 MB int8 + 5KB fp32 per block) at K=3 unique = +860KB int8 = +480KB compressed. Predicted artifact ~13.25 MB. Cap-safe.

Step time: lose 1 attn (~0.18s/layer × 3 layers = 0.54s) gain 1 mamba2 (~0.45s/layer × 3 layers = 1.35s) → +0.81s/step → ~6.4 s/step total. ~22 min/exp.

## Change
**env.sh ONLY — no code changes**:
- ATTN_LAYER_POSITIONS=2 (was 0,2)
- MAMBA2_LAYER_POSITIONS=0,1 (was 1)
- All other 0032 settings unchanged.

## Disconfirming
- val < 2.045: compound win — 2 Mamba-2 helps. Try 3-of-3 (pure Mamba-2) next.
- val > 2.075: loss — selectivity has diminishing returns; sandwich-attn topology was load-bearing.
- val ∈ [2.045, 2.075]: saturate; current 1-of-3 sandwich was at the optimum ratio.

## Notes from execution
Direct in-place env.sh edit by main agent (no subagent — uses 0032's existing Mamba-2 code).
