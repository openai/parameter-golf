# QK_GAIN_SLOT_Gate — Hypothesis

**Date:** 2026-03-31
**Branch:** TEST_LAB
**Baseline:** Rascal II — 1.10986874 BPB, seed=444

---

## What We're Testing

Two independent signals, one experiment:

| Signal | Variable | Type | Claimed delta | Source |
|--------|----------|------|---------------|--------|
| QK_GAIN_INIT=4.0 | `QK_GAIN_INIT` env var | Training-side | ~-0.006 BPB | External: 45 runs, 3 codebases |
| SLOT | `SLOT_ENABLED=1` | Eval-side | ~-0.021 BPB | arXiv:2505.12392v2 |

---

## Mechanism

### QK_GAIN_INIT=4.0

`q_gain` is a per-head scalar learnable parameter, initialized to `QK_GAIN_INIT` (default 1.5). It multiplies the query after RMS-norm at line 1072 of train_gpt.py:

```python
q = q * self.q_gain.to(dtype=q.dtype)[None, None, :, None]
```

**Hypothesis:** Starting at 4.0 (vs 1.5) gives the attention mechanism sharper initial focus, driving better early gradient signal through the q direction. The parameter is free to train away from init — so this is an initialisation effect, not a constraint. Expected to have decaying influence as training progresses.

**This is a single env var change vs baseline. Zero code diff.**

### SLOT (Sample-specific LM Optimisation at Test-time)

At eval time, for each sliding window batch:
1. Compute frozen hidden states: `hidden = model.forward_hidden(x)` — no gradient
2. Initialise per-batch delta: `delta = zeros(1, 1, dim)`, requires_grad=True
3. Optimise delta for 8 steps of AdamW against the language modelling loss on this batch
4. Score with the optimised delta: `logits = model.compute_logits_from_hidden(hidden, delta.detach())`

Model weights are **never modified**. Only the additive delta adapts per batch.
Training trajectory is **identical to baseline** — SLOT only affects the final eval pass.

**Legality:** Score-first, self-supervised. The optimisation uses the next-token prediction loss (no external labels). Legal per competition rules.

**This is an eval-side change only. Training code is unchanged.**

---

## Test Design

4 cases × 1200 steps, seed=444, single GPU:

| Case | QK_GAIN_INIT | SLOT | Measures |
|------|-------------|------|----------|
| baseline | 1.5 (default) | off | control |
| qk_gain4 | 4.0 | off | QK training delta |
| slot_only | 1.5 | on | SLOT eval delta |
| qk_gain4_slot | 4.0 | on | interaction |

**Cross-correlation check:** If signals are independent, `combo_delta ≈ qk_delta + slot_delta`. Interaction residual > 0.002 BPB = signals interfere, test arm-by-arm.

**Key parameters (all hardcoded in run_ablation.py BASE_ENV):**
- `COPRIME_MAX_LOADED_SHARDS=1` — CRITICAL, matches SOTA run condition
- `LOADER_MODE=coprime`, `COPRIME_SHARDS_PER_BATCH=1`
- `SLOT_STEPS=8`, `SLOT_LR=0.005`, `SLOT_MAX_WINDOWS=512` (~1M tokens, fast proxy)
- `SKIP_FINAL_EVAL=0` — runs full sliding window eval to measure SLOT effect
- `POST_EMA_DIAGNOSTIC=1` — measures QK_GAIN effect on post-EMA weights

---

## Proxy Caveat

This is 1200 steps (~18% of a full run). Proxy deltas inflate 5–15× vs full run. **Never promote from proxy alone.** These results answer: "is there a directional signal?"

If the proxy shows signal → run the full 8×H100 gate (2000 steps) on the winning arm(s) before spending $15.

---

## Go / No-Go Criteria

After the cross-correlation ablation:

| Result | Decision |
|--------|----------|
| qk_gain4 `post_ema_bpb` improves by ≥ 0.001 | QK signal real → include in full gate |
| slot_only `sliding_bpb` improves by ≥ 0.003 | SLOT signal real → include in full gate |
| Interaction residual < 0.002 | Both signals compatible → combine in full gate |
| step_avg > 200ms | Broken pod — abort before running any cases |

If neither signal validates → investigate before spending the $15. Do not run the race on unvalidated hypotheses.

---

## Next Step After Validation

If both signals validate (additive, no interaction):
- Build a single race script: Rascal II base + `QK_GAIN_INIT=4.0` baked + SLOT in eval
- Run the 8×H100 full run (600s, seed=444)
- If it beats 1.10986874 on seed 444 → confirm on seed 300 → submit

Only one race. Cost: ~$3–4.
