# Leak Investigation: 216 experiments with BPB < 0.1 — **ROOT CAUSE CONFIRMED**

> **UPDATE 2026-04-30 18:40 UTC:** Root cause definitively confirmed after the
> `--ctx` regression was fixed in [gHashTag/trios-railway@69c3467](https://github.com/gHashTag/trios-railway/commit/69c3467f)
> and 6 fresh FIX-VERIFY runs completed with BPB ∈ [0.0006, 0.0015] despite
> the pipeline now working correctly. The only remaining explanation is
> the train/val corpus split — and inspection of
> [`Dockerfile:35-36`](https://github.com/gHashTag/trios-trainer-igla/blob/main/Dockerfile#L35-L36)
> showed `head -c 100000 tiny_shakespeare.txt > tiny_shakespeare_val.txt`,
> i.e. **val is a strict prefix of train**. See `gHashTag/trios-trainer-igla#60`
> and the fix PR `#61` (merged 2026-04-30 18:42 UTC).
>
> Fix: Dockerfile now uses `train = head -c $((SIZE-100000))`,
> `val = tail -c 100000` (byte-disjoint), plus a runtime
> `assert_train_val_disjoint()` in `src/train_loop.rs` that panics at
> startup if the first 1024 val tokens appear as a substring of train.

---

## Original investigation (pre-confirmation) — now updated for V2 evidence

**Date:** 2026-04-30 · **Auditor:** perplexity-computer-grandmaster (R5-honest lane)
**Anchor:** `phi^2 + phi^-2 = 3`
**Flag applied:** `last_error = 'SCARABAEUS-LEAK-CANDIDATE: bpb<0.1'`
(via [trios-railway#105](https://github.com/gHashTag/trios-railway/pull/105) ledger-daemon Job 3, Khepri-3 leak gate)

## Executive summary

**216 of 313 `done` experiments** returned BPB ∈ [0.0002, 0.0147]
at step = 4000. The per-byte cross-entropy of a character-level LM on
English Shakespeare cannot physically be below ≈ 1.0 even for perfect
overfitting of the training stream, because the bytes themselves carry
≥ 1 bit/byte of irreducible structure.

**CONFIRMED:** hypothesis H1 (train/val path identity) is correct.
The Dockerfile built the val file as the first 100 000 bytes of train.
After ∼ 4 000 steps the model memorises that 100 KB prefix and reports
BPB ≈ 0 on val. Fixed in
[trios-trainer-igla#61](https://github.com/gHashTag/trios-trainer-igla/pull/61)
(merged 18:42 UTC). All 216 rows are tainted and flagged in Neon with
`last_error = 'SCARABAEUS-LEAK-CONFIRMED-V2: dockerfile-100k-prefix-overlap'`.
An audit row
(`gardener_runs.action='leak_confirmed_v2'`) documents the close.

## V2 evidence — 6 fresh FIX-VERIFY runs confirm the leak

After the unrelated `--ctx` fix landed in
[trios-railway@69c3467](https://github.com/gHashTag/trios-railway/commit/69c3467f)
and all 6 Railway workers were redeployed with the current image, 6
fresh experiments ran to completion with the following verbatim results:

| canon_name | account | dur (s) | final_step | final_bpb |
|---|---|---:|---:|---:|
| `IGLA-FIX-VERIFY-h512-LR002-rng42-step4000-acc0`   | acc0 | 171 | 4000 | 0.0009 |
| `IGLA-FIX-VERIFY-h512-LR002-rng43-step4000-acc1`   | acc1 |  86 | 4000 | 0.0007 |
| `IGLA-FIX-VERIFY-h512-LR002-rng44-step4000-acc2`   | acc2 | 160 | 4000 | 0.0010 |
| `IGLA-FIX-VERIFY-h512-LR002-rng1597-step4000-acc3` | acc3 | 103 | 4000 | 0.0010 |
| `IGLA-FIX-VERIFY-h512-LR002-rng2584-step4000-acc4` | acc4 | 103 | 4000 | 0.0006 |
| `IGLA-FIX-VERIFY-h512-LR002-rng4181-step4000-acc5` | acc5 | 104 | 4000 | 0.0015 |

These 6 rows, plus the 210 historical `BLITZ-T10H` rows, total 216
tainted observations. All carry the same
`SCARABAEUS-LEAK-CONFIRMED-V2` marker.

## Distribution of suspect rows

From `experiment_queue` snapshot 2026-04-30 18:10 UTC:

| Wave | Lane | Format | Rows | Avg BPB | Min BPB | Max BPB |
|------|------|--------|-----:|--------:|--------:|--------:|
| BLITZ-T10H | GATE2-RECOVERY-T-10H | gf16   | 56 | 0.0013 | 0.0006 | 0.0047 |
| WAVE3      | —                     | —       | 31 | 0.0013 | 0.0003 | 0.0031 |
| WAVE2-LONG | —                     | —       | 19 | 0.0013 | 0.0003 | 0.0045 |
| (null)     | —                     | —       | 15 | 0.0035 | 0.0002 | 0.0130 |
| RECOVERY-PLAN-A | —                | fp32    | 14 | 0.0016 | 0.0013 | 0.0022 |
| (null) (gardener) | —              | —       | 14 | 0.0059 | 0.0016 | 0.0146 |
| HACK-DIVERSE | —                   | fp32    | 13 | 0.0024 | 0.0007 | 0.0045 |
| (null) (gardener) | —              | gf16    | 12 | 0.0049 | 0.0023 | 0.0079 |
| SPRINT-2026-04-30-REDEFINE | —     | gf16    | 12 | 0.0013 | 0.0003 | 0.0046 |
| MEGA-ASHA-R2 | —                   | fp32    | 10 | 0.0032 | 0.0003 | 0.0195 |
| HACK-LONG  | —                     | —       |  6 | 0.0007 | 0.0003 | 0.0015 |
| RECOVERY-PLAN-A-LR-SWEEP | —       | fp32    |  6 | 0.0038 | 0.0029 | 0.0047 |
| WAVE2-ULTRA | —                    | —       |  5 | 0.0002 | 0.0002 | 0.0003 |
| HACK-DIVERSE | —                   | gf16    |  3 | 0.0033 | 0.0021 | 0.0045 |

All 210 rows share:
- `status = 'done'`
- `final_step = 4000`
- `final_bpb < 0.1`
- Either `created_by = 'gardener'` (82 rows) or `created_by = 'human'`
  (128 rows, but the human inserts all descend from gardener-authored
  seeds via copy-paste).

## Hypotheses (ranked by evidence)

### H1 — Train/val path identity (strongest)

In [`trios-trainer-igla/src/train_loop.rs`](https://github.com/gHashTag/trios-trainer-igla/blob/main/src/train_loop.rs#L60):

```rust
fn load_data(path: &str) -> Vec<usize> {
    if std::path::Path::new(path).exists() {
        let raw = std::fs::read(path).unwrap_or_else(|e| panic!(...));
        return raw.into_iter().map(|b| (b as usize) % VOCAB).collect();
    }
    eprintln!("Data file '{}' not found, using synthetic fallback", path);
    let fallback = b"The quick brown fox jumps over the lazy dog. ".repeat(2500);
    fallback.into_iter().map(|b| (b as usize) % VOCAB).collect()
}
```

When both `TRIOS_TRAIN_PATH` and `TRIOS_VAL_PATH` resolve to a missing
file (i.e. `/work/data/tiny_shakespeare.txt` is absent on a given
Railway image), **both** train and val streams collapse to the same
45-byte "quick brown fox" string repeated 2,500 times. The model
overfits the 45-byte cycle within ~400 steps, then evaluates val on the
same bytes — BPB approaches zero exactly as observed.

### H2 — W-6 numerical-instability underflow (secondary)

The evaluator in
[`train_loop.rs::evaluate`](https://github.com/gHashTag/trios-trainer-igla/blob/main/src/train_loop.rs#L492)
computes per-token log-softmax in f32. Beyond step ≈ 6,000 the logits
saturate, softmax outputs round to one-hot in f32, and `log(1.0 - eps)`
rounds to 0. This is the W-6 instability already documented in
[l7_ledger #19](https://github.com/gHashTag/trios/blob/main/docs/L7_LEDGER.md).

H2 explains the min value at step > 6000, but not rows with
`final_step = 4000`. The 210 flagged rows all have step = 4000, so W-6
is NOT the dominant driver. H1 explains the mass.

### H3 — val_seed missing in config schema (contributing)

`config_json` produced by `gardener` carries `seed` but not
`val_seed`. The seed-agent does not pass a val-seed override to the
trainer. When `train_loop.rs` seeds its RNG from `args.seed`, the val
iterator draws from the same RNG state as the train iterator — even if
the corpus bytes were distinct, token-window starts collide.

## Corroborating signal: the 6 `gate2_eligible` rows

Six rows in the `gate2_eligible` view have BPB 1.75–1.82 at step=1000,
with corrective action `W-6_step_cap_applied_per_l7_ledger_19` and
`eligible_bpb` stamped by the `acc-sot` source-of-truth account:

| seed | canon_name | bpb_at_1000 |
|-----:|------------|------------:|
| 42   | IGLA-TRAIN_V2-FP32-E0055-H1536-rng42   | 1.78 |
| 43   | IGLA-TRAIN_V2-FP32-E0059-H2048-rng43   | 1.75 |
| 44   | IGLA-TRAIN_V2-FP32-E0060-H2048-rng44   | 1.75 |
| 1597 | IGLA-TRAIN_V2-FP32-E0050-rng1597       | 1.78 |
| 2584 | IGLA-TRAIN_V2-FP32-E0051-rng2584       | 1.82 |
| 4181 | IGLA-TRAIN_V2-FP32-E0052-rng4181       | 1.80 |

These were ratified at step = 1000 (before W-6 kicks in) and they do
**not** collapse to BPB < 0.1. If H1 were universal, these six would
also be poisoned. They aren't, which proves the corpus **does** exist
on some Railway images (`TRIOS_TRAIN_PATH` resolves successfully
there), and the leak is **per-image** not universal.

Remaining open question: **which images ship `tiny_shakespeare.txt` and
which don't?** Our prune query revealed the Dockerfile for
`ghcr.io/ghashtag/trios-trainer-igla:latest` does `curl` the corpus at
build time. Older images on the Railway services may predate that
line.

## Recommendations

For Gate-3 (post-deadline):

1. **Harden `load_data`** — panic instead of synthetic-fallback. Make
   missing corpus a fatal configuration error, not a silent substitute.
   Draft PR in [trios-trainer-igla#60](https://github.com/gHashTag/trios-trainer-igla/issues/60) to follow.
2. **Add `val_seed` to config schema** — default `val_seed = seed ^ 0xDEADBEEF`,
   enforced by the `trios-railway-audit` contract test
   ([trios-railway#102](https://github.com/gHashTag/trios-railway/issues/102)).
3. **Held-out eval gate** — Khepri-4 daemon re-evaluates every `done`
   row against a byte-level 90/10 split never seen at train time. BPB
   delta > 0.5 flips `status='leak_suspected'`.
4. **Golden-floor sanity check** — reject any `done` row with
   `final_bpb < 1.0` at insert time in `bin/seed-agent/src/claim.rs`.
   No bytestream can be compressed below 1 b/B without overfitting.

## Decision for this submission

- The 210 leak-flagged rows are **excluded** from any claim of Gate-2
  pass.
- The 6 `gate2_eligible` rows are **reported but not claimed** — we
  state them honestly (BPB 1.75–1.82 at step=1000 on `acc-sot`) and
  leave the ratification decision to the reviewer who can run the
  held-out eval.
- The single row we report as our "honest best" is ID 1387
  (`IGLA-MEGAASHA-h1024-LR00300-AL2-step12000-acc4-rng4181-t28860`,
  BPB 2.1505, step 12000 — well above any suspect bound).

phi² + phi⁻² = 3 · R5-honest · NEVER STOP.
