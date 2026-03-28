## Distillation Teacher: COMPLETE
- Steps: 50000 @ 401ms (5.6 hours)
- val_bpb (float): 1.0986
- val_bpb (int6, sliding s64): **1.0246**
- Artifact: 52.25 MB (teacher, no size constraint)
- Checkpoint: workspace/ema_e15ddc5e-29bd-4b08-95b4-352d324ac4dd.pt
- Notes: Strong teacher. Ready for distillation experiments.

## Experiment DIST-2: Distillation alpha=0.5, temp=2.0
- Config: DISTILL=1 DISTILL_ALPHA=0.5 DISTILL_TEMP=2.0 + best student config
- Steps: 789 @ 761ms (only 789 steps due to 2x overhead from teacher forward!)
- val_bpb (float): 1.3399 (at step 789)
- val_bpb (int6, sliding): 1.5640
- Delta vs no-distill: +0.41 (MUCH WORSE)
- Status: DROP
- Notes: Distillation doubles step time (760ms vs 148ms), giving only 789 steps vs 4051 normally. The student can't train enough in 600s with the teacher overhead. Distillation needs >2x sample efficiency to be worthwhile, which it doesn't achieve at this early stage.

**CONCLUSION: Distillation is not viable under the 600s training constraint** on 4xH200. The teacher forward pass overhead is too expensive. Would need either:
- Cached teacher logits (precompute and save to disk)
- Or much longer training budget (unlimited compute track)

## SELFDIST-2: Cached self-distillation alpha=0.5 temp=2.0
- Steps: 1231 @ 488ms
- val_bpb: 3.88 (TERRIBLE, not learning)
- Status: DROP
- Notes: Two problems: (1) step overhead still 3x from extra forward_logits call (488ms vs 148ms), (2) KL div loss magnitude (~14000) drowns hard label signal. The student can't learn from either source. Need to either remove the extra forward pass or reduce alpha dramatically.

**KEY INSIGHT:** The extra `forward_logits()` call for student logits is the main bottleneck, not the teacher. Need to modify model.forward() to return both loss AND logits, or compute distillation loss inside the model's forward pass.

## SELFDIST-3: Hard distillation (teacher top-1), alpha=0.5
- Steps: 3969 @ 151ms (near-normal speed!)
- val_bpb (sliding): 1.559 (WORSE than no-distill 1.15)
- Status: DROP
- Notes: Step time is good (151ms, only 3ms overhead) but alpha=0.5 is too much teacher influence. The self-teacher's top-1 predictions are often wrong, hurting learning.

## SELFDIST-5a: Hard distillation alpha=0.1
- Steps: 3947 @ 152ms
- val_bpb (sliding): 1.2423 (still worse than no-distill 1.15)
- Status: DROP

## Self-Distillation Summary:
| Alpha | Sliding BPB | vs Baseline |
|-------|------------|-------------|
| 0.0 (baseline) | 1.1521 | — |
| 0.1 | 1.2423 | +0.09 (worse) |
| 0.5 | 1.5590 | +0.41 (worse) |

**CONCLUSION: Self-distillation from same-size teacher hurts at all alpha values.**
--
## SELFDIST-6: Big teacher (105M) hard distillation alpha=0.1
- Steps: 3972 @ 151ms
- val_bpb (sliding): 1.2417 (same as self-teacher, worse than baseline 1.15)
- Status: DROP

## Distillation Summary (all hard distillation, cached top-1):
| Teacher | Alpha | Sliding BPB | vs Baseline |
|---------|-------|------------|-------------|
| None | 0.0 | 1.1521 | — |
| Self (27M) | 0.1 | 1.2423 | +0.09 |
| Self (27M) | 0.5 | 1.5590 | +0.41 |
| Big (105M) | 0.1 | 1.2417 | +0.09 |

--
## Sanity Check: Modified script with all features OFF
- Steps: 4051 @ 148ms
- val_bpb (sliding): 1.1522 (matches exp 26's 1.1521 exactly!)
- Artifact: 16.12 MB (over limit, same as exp 22 which was also 16.12)
- Status: PASS, modifications don't leak

## SELFDIST-4: Soft KL div, big teacher, alpha=0.5, T=2.0
- Steps: 3751 @ 160ms (~12ms overhead)
- val_bpb (sliding): **1.1558** (+0.0036 vs baseline 1.1522)
- Status: CLOSE but still worse
- Notes: Soft KL is MUCH better than hard top-1 (1.156 vs 1.242). Only +0.004 vs baseline. The overhead is minimal (160ms vs 148ms). Try lower alpha to see if we can break even.

## Updated Distillation Table:
| Teacher | Alpha | Loss Type | Sliding BPB | Delta |
|---------|-------|-----------|-------------|-------|
| none | 0.0 | hard labels | 1.1522 | — |
| self (1.15) | 0.5 | hard top-1 | 1.559 | +0.407 |
| self (1.15) | 0.1 | hard top-1 | 1.242 | +0.090 |
| big (1.10) | 0.1 | hard top-1 | 1.242 | +0.090 |
--
## SELFDIST-5b: Soft KL, big teacher, alpha=0.3
- Steps: 3765 @ 159ms
- val_bpb (sliding): 1.1555 (+0.0033 vs baseline)
- Notes: Slightly better than alpha=0.5. Still +0.003 worse than baseline.

| Alpha | Sliding BPB | Delta vs baseline |
|-------|------------|-------------------|
| 0.0 (baseline) | 1.1522 | — |
| 0.3 | 1.1555 | +0.003 |
| 0.5 | 1.1558 | +0.004 |

## SELFDIST-5c: Soft KL, big teacher, alpha=0.1
- Steps: 3773 @ 159ms
- val_bpb (sliding): 1.1553 (+0.0031 vs baseline)

## Final Soft KL Alpha Sweep:
| Alpha | Sliding BPB | Delta |
|-------|------------|-------|
| 0.0 | 1.1522 | — |
| 0.1 | 1.1553 | +0.003 |
| 0.3 | 1.1555 | +0.003 |
| 0.5 | 1.1558 | +0.004 |

**CONCLUSION: Soft KL distillation is uniformly ~0.003 worse than baseline at all alpha values.** The distillation adds ~11ms overhead (159ms vs 148ms = ~7%) which costs ~280 steps. That step penalty roughly equals the distillation benefit, netting out to zero or slightly negative.
