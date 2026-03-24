# H2: Cadence x Architecture Interaction

## Question
Does optimal cadence change when recursive depth increases from 4x2 to 6x2?

## Prediction
A deeper recursive system (3 crawler blocks x 2 loops = 6 effective recursive depth)
will prefer a LOWER cadence (more C-steps) than the 4x2 system because:

- More crawler layers = more refinement capacity per C-step firing.
  Each double-fire passes through 3 blocks instead of 2, so there's more
  "work" the consensus can do per C-step. The investment pays off more.
- BUT: more layers also means more gradient interference on C-steps.
  If this dominates, the 6x2 might prefer HIGHER cadence (fewer C-steps)
  to avoid gradient conflict.
- The direction of the shift tells us something fundamental:
  - If 6x2 wants lower cadence → recursion benefits from more firing
  - If 6x2 wants higher cadence → recursion creates gradient pressure that needs relief
  - If same optimal → cadence is a universal constant, not architecture-dependent

## Architecture (held constant within this front)
```
NUM_FLAT_LAYERS=3  NUM_CRAWLER_LAYERS=3  CRAWLER_LOOPS=2
MODEL_DIM=640  NUM_HEADS=10  NUM_KV_HEADS=5  MLP_MULT=4
XSA_LAST_N=3  VE_LAYERS=0,1,2
```

Note: XSA_LAST_N=3 and VE_LAYERS=0,1,2 expanded to cover all 3 crawler blocks.
This is one logical change from RC-0: architecture shape (flat/crawler split).

## Arms

| Arm | DIAG_FIXED_CADENCE | C-step ratio | Parent |
|-----|-------------------|--------------|--------|
| cad1 | 1 | 100% (all C) | 3f3cx2 base |
| cad2 | 2 | 50% (C/N) | 3f3cx2 base |
| cad3 | 3 | 33% (C/N/N) | 3f3cx2 base |
| cad4 | 4 | 25% (C/N/N/N) | 3f3cx2 base |

## Scale
0.25 (150s wallclock, 625 warmdown, TTT/distill OFF)

## Cross-Comparison with H1
After both fronts complete, compare:
1. Does the BPB-optimal cadence shift between 4x2 and 6x2?
2. Does the `delib_scale` trajectory differ? (More layers = steeper delib growth?)
3. Does the quant gap respond differently to cadence in deeper recursion?
4. Plot: cadence vs BPB for both architectures on same axes.

## Verdict
_To be filled after runs complete._

| Arm | Steps | fast_val_bpb | sliding_bpb | post_ema_bpb | quant_gap | delib_scale_final | Verdict |
|-----|-------|-------------|-------------|-------------|-----------|-------------------|---------|
| cad1 | | | | | | | |
| cad2 | | | | | | | |
| cad3 | | | | | | | |
| cad4 | | | | | | | |

## Cross-Front Conclusion
_To be filled after both H1 and H2 complete._

- Optimal cadence for 4x2: ___
- Optimal cadence for 6x2: ___
- Shift direction: ___
- Interpretation: ___
