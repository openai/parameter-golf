# Log Summary

The staged seed logs do not contain a completed `quantized_ttt_phased` result.
All three runs were interrupted during TTT compile/eval, so the last completed
validation metric is `quantized_sliding_window val_bpb`.

## Final Metrics

| Seed | Train stop step | Last scheduled val step | Final completed val_bpb |    Artifact size |
| ---- | --------------: | ----------------------: | ----------------------: | ---------------: |
| 42   |            8597 |                    8597 |              1.08934733 | 15,999,684 bytes |
| 314  |            8631 |                    8631 |              1.09035192 | 15,997,730 bytes |
| 999  |            8620 |                    8620 |              1.09285937 | 15,998,747 bytes |

## Mean

-   `quantized_sliding_window val_bpb` mean: `1.09085287`

## Source Logs

-   `logs/seed_42.log`
-   `logs/seed_314.log`
-   `logs/seed_999.log`

---

# Feature Uniqueness Analysis

Analyzed all 2,015 PRs (open, closed, merged) on openai/parameter-golf as of 2026-05-01.

## Unique (I think no one else tried it..)

1. **Multi-Trajectory SWA** — Each GPU rank follows independent trajectory during warmdown (grad sync off), then SWA averages combined across ranks.
2. **Scale Tuning Post-GPTQ** — Freeze quantized int weights, fine-tune only per-row scales via CE loss backprop (Adam, 20 steps).
3. **Two-Pass GPTQ** — Run GPTQ, dequantize, re-collect Hessians on quantized model, run GPTQ again.

## Partially Unique (maybe our variant is novel? But concept was explored)

4. **Selective 2:4 Sparsity (training-time)** — Mid-training one-shot 2:4 pruning on MLP weights. PR #1537 tried post-training 2:4 (negative). PR #1818 tried as compression codec (catastrophic).

## Not Unique (tried by others)

5. **Mixture of Softmax** — PRs #266, #584, #908, #1227, #1608, #1995. All neutral-to-harmful.
6. **Hourglass Downsampling** — PRs #133, #831, #1275, #1573, #2004. PR #831 called it "catastrophic."
7. **Loop Gate** — PRs #155, #1208, #1691, #1996. Well-explored by 4-5 teams.
8. **Gated MLP / SwiGLU** — 20+ PRs, 2 merged. Most widely tried feature in competition.
9. **Knowledge Distillation** — PRs #578, #687, #896, #1029, #1034, #1083, #1185, #1697. All negative.
10. **Hard Token Mining / Focal Loss** — PRs #687, #877, #1233, #1325, #1360, #1380, #1402, #1510, #1702. All negative.
11. **Byte-Weighted CE** — PRs #108, #1033, #1359, #1519. None merged.
12. **Momentum Cooldown** — PRs #534, #1337. Neither merged.

## Shared (in other merged submissions)

13. **Fused Softcapped CE (Triton)** — PR #1787
14. **Batch Size Schedule** — Ternary PR #1184
15. **Auxiliary CE / Deep Supervision** — Ternary PR #1184
16. **Phased LoRA TTT + Global SGD** — PRs #1530, #1610, #1626
17. **LQER Asymmetric Quantization** — PR #1851
18. **Value Residual Mixing** — msisovic (2026-03-31), SOTA (2026-04-09)
19. **Warmup State Reset** — msisovic (2026-03-31)
