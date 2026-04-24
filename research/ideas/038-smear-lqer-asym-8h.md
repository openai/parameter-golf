# 038 idea — add SmearGate and LQER-asym directly on top of `037`

The current sparse-family promotion line is already:

- sparse gate
- full-float frozen updated `alpha/beta`
- `MIN_LR`
- fused CE
- phased TTT

The two remaining high-signal `#1797` adds are orthogonal enough to try
together:

1. **SmearGate**
   A tiny causal residual smear over the previous token. This is a training /
   forward-path lever.

2. **LQER-asym**
   A post-GPTQ low-rank residual repair for the highest-error quantized
   matrices. This is a quantization-path lever.

Because one mainly helps the float model path and the other mainly helps the
quantized model path, the stack case is stronger than for most speculative
bundles. The downside is attribution: if it wins, we still will not know how
much came from each one without later ablations.

That tradeoff is acceptable here because:

- `037` is already a strong direct promotion line
- `#1797` suggests these levers coexist cleanly on a similar sparse-gate base
- we care more about frontier movement than tidy single-lever attribution

Plan:

- `038A` = `037` + `SMEAR_GATE_ENABLED=1` + LQER-asym on `8×H`
- use public `#1787/#1797` seed family (`42/0/1234`) for cleaner comparison
