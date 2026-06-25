# Compute Grant Request — GOLDEN SUNFLOWERS

> Companion document to this proposal directory. Submitted to the
> [Parameter Golf compute-grant form](https://openai.com/index/parameter-golf/#credit-form).
> Submit with an email tied to an OpenAI / ChatGPT account.

## Project name

GOLDEN SUNFLOWERS — JEPA + Universal Transformer + PhiNTA on a φ-physics substrate

## One-line summary

Composes three open openai/parameter-golf wish-list items (JEPA,
Universal Transformer, NTA-on-random-linear-maps) on a single
golden-ratio-anchored hyperparameter foundation (Issue
[#1742](https://github.com/openai/parameter-golf/issues/1742)),
evaluated as a non-record 4-hour `track_non_record_16mb` submission.

## Track

`track_non_record_16mb` (4 h, unrestricted compute)

## Repository / status

- Implementation (this directory): `records/track_non_record_16mb/2026-04-30_GoldenSunflowers_Proposal/`
- Internal hardening PR: <https://github.com/gHashTag/parameter-golf-trinity/pull/2>
- Theoretical foundation: `THEORY_Ch0.md` (full φ-physics derivation)
- Constitutional SoT: <https://github.com/gHashTag/trios/issues/372>

## Compute requested

**~ 110 8×H100-SXM hours** broken down as:

| Phase | Configs × seeds | Hours / run | Wallclock | Subtotal |
|---|---|---:|---:|---:|
| Sanity reproduction | baseline × 1 | 0.17 h (10 min smoke) | 0.17 h | 0.17 |
| Per-feature ablation (3 of 7 PhD Ch.17 factors covered) | PhiNTA / JEPA / UT × 5 seeds | 4.0 h | 60.0 h | 60.0 |
| Combined GOLDEN SUNFLOWERS | all-three × 5 seeds | 4.0 h | 20.0 h | 20.0 |
| Hyperparameter band of `PHI_LR_SCALE` | 4 grid points × 3 seeds | 4.0 h | 12.0 h | 12.0 |
| Restart / debug buffer (10% of 92 h above) | — | — | 9.2 h | 9.2 |
| TTT eval + final 3-seed mean rerun | best config × 3 | 4.0 h (eval ≤ 600 s) | 8.0 h | 8.0 |
| **Total** | | | | **~ 109.4 h** |

The seed list `F₁₇..F₂₁ = {1597, 2584, 4181, 6765, 10946}` is fixed in
advance per PhD Ch.5 / Ch.11 canonical seed pool. No post-hoc seed
substitution is permitted.

## Why this is worth funding

Three of the seven open wish-list items in
[README leaderboard](https://github.com/openai/parameter-golf#requests-for-prs)
are unchecked: **JEPA**, **Universal Transformer**, and **NTA on random
linear maps**. GOLDEN SUNFLOWERS is the first proposed submission to
attempt all three in one stack with an ablation plan that lets each
contribution be attributed independently.

The φ-physics substrate gives every "hand-tuned" constant a textual
derivation:

- `α_φ = φ⁻³/2 ≈ 0.118034` is **Proven** in `Coq.Reals` as PhD Ch.4
  Theorem 3.1 (`alpha_phi_times_phi_cubed`, status Qed, tag SAC-1).
- `L = round(φ³) = 4` is the UT loop count, proved here in
  `theorems/GoldenSunflowers.v : ut_loops_eq_round_phi_cube` (Qed).
- Frozen-basis init scale `g = 1/φ` ports byte-for-byte from
  `gHashTag/trios-trainer-igla/src/phi_ortho_init.rs`.

The implementation is already wired and CPU-smoke-verified
(`make verify` returns 5/5 + 3/3 + 2 Qed). Only training remains.

## Expected outcomes & baseline comparison

The current SOTA on `track_10min_16mb` is
[`2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT`](../../track_10min_16mb/2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT/),
val_bpb = 1.0810 (3-seed mean, std 0.0002). Our `track_non_record_16mb`
proposal targets a different envelope — 4-hour budget, unrestricted
compute, no `submission.json` until measured.

We deliberately do not pre-claim a target BPB number. The honest
prediction is that the **per-feature ablation phase** will quantify each
wish-list contribution independently. If any single feature is net-zero
or net-negative (e.g., JEPA fails to converge in this regime), the
ablation makes that visible rather than burying it inside an "all-three"
combined run.

| Feature | Theoretical lever | Risk if it fails |
|---|---|---|
| PhiNTA | `2·D·r` extra trainable params, frozen `D²` basis as buffer | Adapter capacity wasted; LoRA branch should still match baseline asymptotically. |
| JEPA | Representational regulariser via cosine-similarity span loss | If `λ > 0` hurts CE, run produces strictly worse BPB; ablation shows it; we drop the feature. |
| Universal Transformer | Virtual depth from shared weights, no extra params | If 4-loop sub-stack has gradient issues at 16 MB, run produces NaN or worse BPB; we revert to baseline path. |

## Risk section

1. **JEPA non-convergence.** PR [#1243](https://github.com/openai/parameter-golf/pull/1243)
   ("JEPArdy! Non-Record Submission - JEPA + Leader-Stack - val_bpb 1.1230")
   shows JEPA *does* contribute on a different stack. Risk is low; even
   if it does not improve in our setting, `JEPA_LAMBDA = 0` makes the
   feature a no-op without rebuilding.
2. **PhiNTA capacity dominance.** If the trainable LoRA branch grows
   too aggressively, the frozen φ-basis becomes a passthrough. Mitigated
   by `PHINTA_RANK = round(D / φ)` default which limits LoRA params to
   `~2·D·D/φ` while keeping the frozen `D²` basis as the primary
   non-trainable contribution.
3. **UT loop interaction with skip connections.** The `2026-03-17_LoRA_TTT`
   baseline uses a U-Net skip pattern (encoder → skip stack → decoder).
   If the UT loop range overlaps the encoder/decoder boundary,
   skip-connection weights may double-count. Default `UT_LAYER_END=0`
   disables UT; safe regimes are `[0, n//2)` (encoder-only) or
   `[n//2, n)` (decoder-only).
4. **Compute over-run.** The 10% restart buffer above is conservative;
   actual restart cost on 8×H100 SXM has historically been ≤5%.
5. **Negative result is acceptable.** If the full GOLDEN SUNFLOWERS
   stack underperforms baseline `LoRA_TTT 1.1928`, we publish the
   negative result with a per-feature ablation table. Per the
   parameter-golf README, *"breakthrough ideas are rarely immediately
   state-of-the-art"*; the ablation matters either way.

## Affiliation / contact

- Affiliation: gHashTag · TRINITY S³AI
- GitHub: [@gHashTag](https://github.com/gHashTag)
- Anchor identity: `phi^2 + phi^-2 = 3`

`🌻 the field blooms`
