# Non-record: SLOT Violates Causal Dependence — Empirical Analysis

## Summary

SLOT (Sample-specific LM Optimization at Test-time) optimizes a delta vector using target tokens, then scores those same targets with the optimized delta. This means the prediction at position `t` depends on tokens beyond `x_1..x_{t-1}` — a causal dependence violation.

This analysis provides an empirical proof and requests an organizer ruling on SLOT's legality.

**Affected submissions:** All SLOT variants, including:
- PR #1084, #1128 (original SLOT, @AnubhavBharadwaaj)
- PR #1172 (@dexhunter), #1176 (@bigbag)
- PR #1209 (our own submission — we are flagging ourselves)
- PR #1229 (per-sample delta + logit bias, 0.9300 BPB)

## The violation

### How SLOT works

1. Compute hidden states `H` from inputs (frozen model, `torch.no_grad()`)
2. Optimize a delta vector `δ` by minimizing NLL on the **target tokens** in the scored window
3. Score the **same target tokens** using `H + δ`

### Why this violates causal dependence

At position `t`, the prediction is `P(x_{t+1} | H_t + δ)`. The hidden state `H_t` depends only on `x_1..x_t` (causal attention). But `δ` is optimized using targets at positions `t+1, t+2, ..., t+k` — so `δ` carries information from future tokens into the prediction at position `t`.

**Formal statement:** `δ = argmin_δ Σ_{t ∈ scored} -log P_δ(x_{t+1} | H_t + δ)`. Since `δ` depends on `{x_{t+1} : t ∈ scored}`, the prediction `P_δ(x_{t+1} | H_t + δ)` at position `t` depends on tokens beyond the strict prefix — including `x_{t+1}` itself.

**Compression argument** (credit: @NoesisGenesis, PR #1172): To decode the first token in a SLOT batch, a decoder would need `δ`. But `δ` was computed from the entire batch's targets. The decoder cannot reconstruct `δ` from the prefix alone, because the later tokens that determined it have not yet been decoded. A score that requires side information unavailable to a causal decoder does not measure compression.

### The 96.9% counterargument and why it doesn't resolve the question

@AnubhavBharadwaaj (original SLOT author) correctly notes that in stride=64 sliding window evaluation, 1984/2048 tokens per window are already-scored context. So 96.9% of the gradient signal comes from known tokens, and only 3.1% from the 64 scored positions.

This is a meaningful quantitative point — the degree of information leakage from future tokens is small in the shared-delta variant. But it doesn't eliminate the violation: the prediction at position `t` still depends on targets beyond the prefix, even if the dependence is diluted. "A little bit of future information" is still future information.

The question for the organizers is whether this level of information leakage is acceptable under the competition's evaluation rules.

## Empirical proof

### Test design

We use a minimal causal LM (2-layer transformer, dim=128, random weights) to isolate the SLOT procedure from any specific model. The violation is structural — it exists in the procedure itself, regardless of model architecture or weights.

**Test A — Future-token sensitivity:**
Flip one target token `x_{t+k}`, re-run SLOT, check if NLL at position `t` changes.
If it does, the prediction at `t` depends on `x_{t+k}` — violating causal dependence.

**Test B — Self-prediction advantage:**
Score the same token `x_{t+1}` under two conditions: (1) SLOT optimizes toward `x_{t+1}`, (2) SLOT optimizes toward a different token. If NLL differs, the answer is leaking through delta.

**Test C — Systematic cross-position leakage:**
Flip each of 16 individual targets; for each flip, check all 15 other scored positions. Reports the fraction of position pairs that show information leakage.

### Results

```
Without SLOT (baseline):  predictions are perfectly causal.
                          Changing targets has ZERO effect on NLL.

With SLOT:                predictions depend on FUTURE targets.

Metric                                            Shared Per-sample
───────────────────────────────────────────── ────────── ──────────
Max NLL change from future token flip           0.255651   0.774387
Self-prediction advantage                        +0.2382    +0.7255
Cross-position violations                            240        240
Cross-position checks                                240        240
Violation rate                                    100.0%     100.0%
```

**100% of scored position pairs show cross-position information leakage.** The per-sample delta + logit bias variant (PR #1229) amplifies the violation by ~3x.

### Reproducing

```bash
# No GPU required. Works on CPU/MPS. ~30 seconds.
python prove_slot_causal_violation.py
```

## Variants and severity

| SLOT variant | Optimized params | BPB gain | Violation severity |
|---|---|---|---|
| Shared delta `[1,1,D]` (PRs #1084, #1128, #1209) | 512 | ~0.010 | Low — diluted by 96.9% context gradient |
| Per-sample delta `[B,1,D]` + logit bias `[B,1,V]` (PR #1229) | 1536/sample | ~0.189 | High — 3x amplified, 24 params per scored position |

The per-sample logit bias provides 1024 free parameters per sample that directly shift the output distribution, trained on only 64 scored positions. This high parameter-to-data ratio enables significant memorization of evaluation targets.

## What a "context-only SLOT" fix would look like

@AnubhavBharadwaaj proposed a trivially legal variant: optimize `δ` only on context positions (the 1984 already-scored tokens), not on the 64 scored positions. This would eliminate the causal violation entirely while reportedly losing only ~0.0002 BPB (source: PR #1172 comments).

This suggests that the real signal in SLOT comes from adapting to the local text distribution (legal), with only a small component from target leakage (illegal). A context-only mask would preserve the legal part.

## Prior discussion

This analysis builds on arguments already made by community members:

- **@NoesisGenesis** (PR #1172): information-theoretic argument that SLOT violates Condition 1 of Issue #1017, plus the compression/decodability argument
- **@AnubhavBharadwaaj** (PR #1172): the 96.9% context gradient counterargument and context-only SLOT proposal
- **@msisovic** (PR #1176): "This SLOT implementation, like the ones before it, violates causality"
- **@Eppie, @abaybektursun** (PR #886, Issue #677): established the precedent of empirical distribution audits for n-gram caches

Note: Issue #1017's four conditions are a community proposal by @NoesisGenesis, not official organizer rules. However, @valerio-oai referenced them approvingly in Issue #677.

## Request for ruling

@0hq @valerio-oai — SLOT has been debated across PRs #1084, #1128, #1172, #1176, and #1209 without an official ruling. This analysis provides empirical evidence that SLOT violates causal dependence. Could you weigh in on whether SLOT (in any variant) is acceptable under the competition's evaluation rules?

We are flagging our own submission (PR #1209) alongside all others that use SLOT. If SLOT is ruled illegal, we accept the consequences for our own score.
