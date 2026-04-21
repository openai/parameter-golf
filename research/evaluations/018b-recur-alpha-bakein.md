# Evaluation — Spec 018b (Recur-Alpha bake-in into block forward)

**Run dir:** `runs/018b-recur-alpha-bakein/run-d-bakein/`
**Commit:** `4c06275` on `exp/recur-alpha-bakein`
**Pod:** `g5s1rqfhia58uk` — 2×H100 SXM, US-NE-1, NA volume `hvpdph5i3g`
**Eval date:** 2026-04-21

## Hypothesis recap

Baking the α-blend into `Block.forward` as a 4-term linear combination eliminates `x_new` materialization, potentially fusing the entire residual sum into one kernel:

```python
return (
    eff_mix_0 * x + eff_mix_1 * x0
    + eff_attn_s * attn_out + eff_mlp_s * mlp_out
)
```

## Result

| Run | Commit | Config | Avg last 5 tok/s | vs baseline | vs current blend | vs lerp |
|-----|--------|--------|-----------------|-------------|-----------------|---------|
| A (016b) | 154c9b8 | no recur-alpha | 3,333K | — | — | — |
| B (016b) | 4dd2d63 | current 4-op blend | 3,234K | −2.9% | — | — |
| C (018) | 97d9854 | torch.lerp | 3,252K | −2.4% | +0.6% | — |
| **D (018b)** | **4c06275** | **bake-in** | **3,174K** | **−4.8%** | **−1.9% worse** | **−2.4% worse** |

Steps at stabilization: ~step 100. Readings at 350/375/400/425/450: 3,176K / 3,177K / 3,171K / 3,173K / 3,175K — fully stable, definitively worse.

## Decision criterion outcome

Per spec's decision table:
- K ≥ 1.05×L: bake-in materially better ✗
- 1.01×L ≤ K < 1.05×L: modest improvement ✗
- K ≈ L (within 1%): compile auto-fused lerp ✗
- **K < L: bake-in broke something ← actual** ✓

## Interpretation

The bake-in refactor made things **worse**, not better. Most likely cause: the refactor introduced a more complex expression inside `Block.forward` that **broke existing kernel fusions** torch.compile had already found for the baseline block. The 4-term sum `eff_mix_0*x + eff_mix_1*x0 + eff_attn_s*attn + eff_mlp_s*mlp` uses per-element scaled tensors derived from `self.recur_alpha` (a learned parameter), which adds dynamic data dependency that prevents the compiler from fusing as aggressively.

In other words: the manual optimization outsmarted torch.compile's existing fusions and produced a net regression. The compiler had already found a better schedule for the original block + external blend.

## Decision — SHELVE bake-in

**Do not apply.** The bake-in refactor is strictly worse than both the current blend and torch.lerp at proxy scale. There is no reason to expect this to improve at full model scale — the regression likely worsens proportionally.

**torch.lerp (spec 018) is the correct optimization.** Apply it and move on.

## Cost

| item | cost |
|---|---|
| Run D: ~6 min compile + 2 min training | ~$0.80 |
| **018b total** | **~$0.80** |

Combined 018 + 018b on same pod: **~$1.60 total** (plus pod boot ~$0.30).

## Cross-references

- Spec: `research/specs/018b-recur-alpha-bakein.md`
- lerp result: `research/evaluations/018-recur-alpha-lerp.md`
- Control data: `research/evaluations/016b-recur-alpha-throughput.md`
