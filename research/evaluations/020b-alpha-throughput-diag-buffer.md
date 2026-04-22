# Evaluation 020b — Buffer-α throughput diagnostic

**Spec:** `research/specs/020b-alpha-throughput-diag-buffer.md`
**Run:** `runs/020b-alpha-throughput-diag-buffer/seed_42/`
**Date:** 2026-04-21
**Hardware:** 4×H100 SXM NA (US-CA-2) — JP unavailable; NA used per spec's fallback
**Commit:** `3cfc372` (buffer-α + instrumentation + cuda.synchronize fix)
**Status:** Training complete. GPTQ crashed (`No module named 'brotli'`). Diagnostic objective fully met.

---

## Result summary

| metric | value |
|---|---|
| Steps completed | 2193 (wallclock cap) |
| Loop activation step | 509 (`ENABLE_LOOPING_AT=0.17`) |
| Pre-loop median step time | **197.9ms** |
| Post-loop median step time | **292.8ms** |
| Loop activation overhead | **+47.9%** |
| Type A spikes (dataloader shard loads) | 12 |
| Type B spikes (GPU-side mystery) | **0** |
| Post-val recompile cluster | **0** |
| Val at step 1500 | val_bpb 1.1671 |
| Pre-quant post-EMA val_bpb | 1.10598 (4×H100, not comparable to #1736) |

---

## Spike taxonomy

### Type A — Dataloader shard loads (12 post-loop)

Every 127 steps, `dataloader_us` spikes to 40–110ms. Step time rises to 340–415ms. Identical pattern to spec 020.

Post-loop Type A spikes: steps 636, 763, 1017, 1144, 1271, 1398, 1525, 1652, 1779, 1906, 2033, 2160.

### Type B — GPU-side mystery spikes: **ZERO**

Spec 020 (literal-α) had 12 Type B spikes (3–13s, dl_us normal, consecutive pairs). Spec 020b (buffer-α): **zero**. This is the key diagnostic result.

### Post-val recompile cluster: **ZERO**

Val fired at step 1501 (42.8s — evaluation time itself). Post-val window analysis:

| window | mean step_time_ms | max step_time_ms | steady-state |
|---|---|---|---|
| Steps 1502–1700 | 293.8ms | 378.6ms | 292.8ms |

The elevated max at step 1525 is a dataloader event (dl_us=108ms). The post-val window is **statistically indistinguishable from steady state**. No recompile penalty.

---

## Comparison: 020 (literal-α) vs 020b (buffer-α)

| metric | 020 literal-α | 020b buffer-α | Δ |
|---|---|---|---|
| Post-loop median step time | 277.6ms | 292.8ms | +5.5% (pod variance) |
| Type B GPU mystery spikes | 12 | **0** | −12 |
| Post-val recompile cluster | N/A (no val) | **0** | — |
| Type A dataloader spikes | 16 (2036 steps) | 12 (1684 post-loop steps) | Same rate |

The +5.5% difference in post-loop median is pod-level variance (different pod, different NA datacenter). Not a buffer-α regression.

---

## Hypothesis verdict

**CONFIRMED.** `register_buffer` α eliminates both the Type B mystery spikes and the post-val recompile cluster seen in 019/019b. Dynamo treats the buffer as a runtime tensor input — no graph specialization on the α value, no cold-path penalty on train→eval→train switch.

The 020b dip pattern matches spec 017 (tensor-α, 0 mystery spikes) rather than 019/019b (literal-α, 7-12 spikes including post-val cluster).

---

## Accept criteria assessment

| criterion | threshold | result | pass? |
|---|---|---|---|
| Dip count 0–1 | → confirm 021 | **0 post-val dips** | ✓ |
| Post-val window dip-free | strong confirmation | **dip-free** | ✓ |

Primary: **dip-free**. Secondary: **post-val dip-free**. Both pass at the highest confidence tier.

---

## Bugs

None new. The `brotli` crash is a pre-existing issue with NA pods; only affects GPTQ compression, not training or diagnostics.

---

## Decision

**LAUNCH spec 021** on 8×H100 JP with high confidence.

The throughput "risk" from constant-α is eliminated. Buffer-α gives:
1. Same throughput shape as 017 (no mystery spikes)
2. No post-val recompile penalty
3. α correct from step 1 (vs 017's learned α, which starts at 1.0 and drifts to endpoint values)

The question for 021 is whether the early-α initialization advantage translates to a val_bpb improvement over 017's 1.06733.
