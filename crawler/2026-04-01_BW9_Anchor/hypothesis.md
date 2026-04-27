# BW9_Anchor — Hypothesis

**Parent:** BW8_Tap (BW5 + TAP_DIM=32 shared — working baseline, pending full run)
**One variable:** `ANCHOR_DIM=32` (delta anchor, loop-to-loop causal write state)

## Mechanism

Each crawler loop commits a small learned vector (model_dim→32) representing what
it "wrote" to the residual. The next loop reads this anchor (32→model_dim) before
processing, conditioning on what was already committed rather than re-discovering it.

- Loop 0 reads zeros (warm start)
- anchor_write[loop]: model_dim→32, zero-init
- anchor_read[loop]: 32→model_dim, zero-init
- ~196K new params total (6 linears, all zero-init)

Battery differentiates reading (9,1,1 RoPE scales).
TAP differentiates encoder reference (shared anchor).
Anchor differentiates writing (per-loop committed state).
Three complementary loop coordination axes.

## Gate evidence (BW7 MegaGate, vs BW5 baseline)

| Arm | int6_sw_bpb | delta |
|-----|-------------|-------|
| CTRL-00 (BW5) | 1.28912666 | — |
| ANC-05 (dim=32) | 1.28578393 | −0.00334 |
| ANC-06 (dim=64) | 1.28749998 | −0.00163 |

dim=32 >> dim=64. Low-dimensional anchor is better.
This gate was vs BW5 baseline (no tap). Now testing on BW8 (tap baked in).

## Gate target (8×GPU FA3, 2-arm)

BW9 control = BW8 config (TAP_DIM=32 shared).
BW9 test = BW8 + ANCHOR_DIM=32.
Pass: test arm beats control, step_avg ~74ms.
