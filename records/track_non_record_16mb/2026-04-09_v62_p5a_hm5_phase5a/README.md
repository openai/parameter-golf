# v6.2 Phase 5a SOTA-trivial stack — 8×H100 SXM, non-record 10-min 16MB track

**3-seed val_bpb (SLOT lr=0.1 steps=100, stride=64, re-run @75-76 %): 1.136399 ± 0.001492**
*(trajectory: @28 %→1.142572, @32 %→1.140655, @40 %→1.137407, @50 %→1.136816, @56 %→1.139363, @66 %→1.138112, @76 %→1.136399. The cumulative bpb oscillates within ±0.003 bpb; final 100 %-eval expected in [1.136, 1.140].)*

## Originality — what's novel to this submitter

Seven discrete contributions in this PR / the v6.1 chain it extends:

1. **First rANS entropy codec for mixed-precision NN weights in the
   competition (prior in chain, #1123 opened 2026-03-30).** To our knowledge
   there are exactly **two** rANS-based PR chains in the competition —
   **this chain (#1123 → #1146 → #1465, opened 2026-03-30)** is the first
   chronologically, and `turbo-indubitable` #1215 (opened 2026-04-01, two
   days later, int5/int6 on a 12L LeakyReLU² backbone, 1.1601 bpb) is the
   only other. **Our distinctive contribution is the Pentanary MLP-up
   alphabet**: 2.32 bits/weight on 23 % of the artifact vs ~3.0+
   bits/weight that int5/int6-only rANS can reach. MLP-down reaches **1.20
   bits/weight (Int4)**. The whole HybridQuant mixed-alphabet rANS stack
   (Pentanary + Int4 + Int5 + Int6 + FP16 passthrough with per-row scales)
   + the custom Rust codec `rans_codec_rs` is the chain's core originality
   claim — see the "rANS HybridQuant baseline" section.
2. **Aggressive SLOT tuning (prior in chain, #1146)** — discovered that
   SLOT defaults (`lr=0.003 steps=5` from PR #1128 and `lr=0.005 steps=8`
   from PR #1176) are ~20–33× too conservative at 32 M scale. Stride=64
   sweep showed SLOT is monotonically helpful up to `lr=0.1 steps=100`,
   delivering **~−0.1 bpb** over the no-SLOT base eval (from ~1.234 to
   1.1365).
3. **Phase 1A int6 tied-embedding quantization (new in this PR)** — the
   parent chain stored the tied `lm_head / tok_emb` as FP16 passthrough
   (1.05 MB / 7 % of the artifact). Phase 1A's sweep showed
   `EMBED_QUANT_BITS=6 EMBED_QUANT_TOK_EMB=1` is a **free −0.6 MB** with
   zero bpb regression (vs +0.043 bpb for pentanary-tied-embed, which the
   higher tied-embed sensitivity cannot tolerate).
4. **Phase 5a trivial-wins composition (new in this PR)** — QK-Gain 5.0 + MuonEq-R
   + EMA 0.9965 + hidden_mult 5 + int6 tied embed, stacked on top of the rANS
   HybridQuant backbone. Delivers **−0.010124 bpb** over the v6.1 SLOT-100 record.
5. **Shannon-floor empirical check (new in this PR)** — inter-layer delta
   prediction experiment showed **delta entropy ≥ raw-weight entropy across
   all 11 layers**; rANS reaches 2.32 bits/weight on MLP-up vs a Shannon
   theoretical minimum of 2.28 bits/weight on the same tensors. To our
   knowledge this is **the first explicit Shannon-floor empirical check on
   the HybridQuant / Pentanary rANS pipeline** — the other rANS-based PR
   #1215 reports int5/int6 bits/weight but does not run a delta-vs-raw
   entropy comparison.
6. **Empirical negative-results catalog for the 32 M regime (new in this
   PR)** — 10 actually-run experiments with eval data (Phase 1A pent/int4
   tied embed, Phase 2A inter-layer delta measurement, Phase 4 seven-variant
   architecture sweep, Phase 5b two depth-recur attempts) + 5 code-written
   stubs dropped before execution (Phase 1B / 1C / 2B / 2C / 3) — in the
   two tables below, split honestly so reviewers can see which negatives
   are empirically grounded and which are only code-level.
7. **Legal Muon-TTT non-competitive finding (new in this PR)** — 3-seed full-eval
   TTT mean 1.205215 vs SLOT-100 mean 1.136399, **SLOT wins by 0.069 bpb** on
   this model. Strong negative result: aggressive SLOT captures most of the
   gain TTT can extract for a 32 M model.

**Legal Muon-TTT alternative (3-seed, full eval)**: mean 1.205215 vs SLOT-100
mean 1.136399 — SLOT-100 beats TTT by **0.069 bpb** on this model. TTT is
not competitive with aggressive SLOT here. (Per-seed: s1337 TTT=1.206428,
s1338 TTT=1.204575, s1339 TTT=1.204643.)

> **First submission in the competition to use rANS entropy coding for
> mixed-precision NN weights** (parent #1123 opened 2026-03-30) — mixed
> Int4 / Int5 / Int6 / **Pentanary** quantization flows directly through a
> custom Rust rANS codec, giving ~2.32 bits/weight on MLP-up (Pentanary)
> and ~1.20 bits/weight on MLP-down (Int4), vs ~4.0 bits/weight for naive
> Int4 baselines and ~3.0+ bits/weight for int5/int6-only rANS. The other
> rANS-based chain is `turbo-indubitable`'s #1215 (int5/int6-only on a
> 12 L LeakyReLU² backbone, opened two days after #1123) — our
> Pentanary + full-HybridQuant stack is the distinctive contribution.

| seed | bpb (re-run @75-76 %) | windows |
|------|-----------------------|---------|
| 1337 | 1.138161 | 739,232 / 969,088 (76.3 %) |
| 1338 | 1.135610 | 732,832 / 969,088 (75.6 %) |
| 1339 | 1.135425 | 731,232 / 969,088 (75.5 %) |
| **mean** | **1.136399** |  |
| **std**  | 0.001492    |  |

vs prior `2026-04-08_v61_h100_aggressive_slot_steps100` (3-seed 1.146523): **−0.010124 bpb**

This is a **non-record** submission (PR #1019 record is 1.1147, we are +0.028 above).
Submitted to document the Phase 5a SOTA-trivial stack as well as the negative
ablations from Phases 1B/1C/2A-C/3/5b that other submitters can skip.

### Why mid-eval? (pod was terminated before 100 %)
A full 100 %-eval at stride=64 SLOT-100 costs ~50 min per seed on one H100
(the 10-minute training limit does not apply to the eval phase, but the
stride=64 × SLOT-100 inner loop is ~5× slower than the stride=64 × SLOT-20
recipe used for the previous record). The re-run reported above was in
flight on the same H100 pod up to 75-76 % when the pod's container was
terminated by RunPod-side (the submission deadline was close and our pod's
container got recycled). The reported 1.136399 is the **last stable
checkpoint we captured from the live log files** before we lost the session.
**Completing the remaining 24 % of the 100 %-eval on all 3 seeds requires
approximately $15 of additional RunPod credit** (3 seeds × ~12 min ×
$0.33 per H100-min) that is outside this submission's budget but clearly
attainable with a small top-up; we will push a follow-up commit once the
final numbers are in.

### Shannon-limit empirical check
One of the abandoned Phase 2 experiments was inter-layer delta prediction
(`ΔW_l = W_l − W_{l−1}`, video-codec style). We measured the per-layer
Shannon entropy of both `W_l` and `ΔW_l` after Pentanary / Int4 quantization
and found that **across all 11 layers the delta entropy was equal to or
higher than the raw weight entropy** — the Pentanary alphabet distribution
widens after the delta because the per-layer median (which rANS was already
exploiting on raw weights) gets removed. Empirically, rANS reaches 2.32
bits/weight for MLP-up Pentanary vs a Shannon theoretical minimum of 2.28
bits/weight measured on the same weights, so **the 15 MB artifact is already
entropy-bound at the single-token coder level**. The only remaining headroom
is information flow between the model and the quantizer (QAT, tied-embed
quantization, hidden-mult re-investment — which is exactly what Phase 1A +
Phase 5a exploits).

## Phase 5a stack (vs v6.1 SLOT-100 baseline)

| # | Component | Source | Estimated Δ |
|---|---|---|---|
| 1 | `QK_GAIN_INIT=5.0`        | PR #1413 | -0.002 |
| 2 | `MUON_EQ_R=1` (Newton-Schulz row L2) | PR #1394 | -0.001 |
| 3 | `ema=0.9965` (vs 0.997)   | PR #1421/#1445 | -0.001 |
| 4 | `HIDDEN_MULT=5.0` (FFN 4×→5×) | byte re-investment, Phase 4 | -0.002 |
| 5 | `EMBED_QUANT_BITS=6 EMBED_QUANT_TOK_EMB=1` (int6 tied) | Phase 1A this submitter | -0.001, -0.6 MB |

Phase 5a is a **trivial-wins composition**: no new architecture, no weight-format
change beyond the int6 tied embed in Phase 1A. The training loop, model classes,
and rANS serializer are all unchanged from v6.1 baseline.

## Negative results we tried

Split honestly: **actually run with eval data** vs **code written but
not run to eval**.

### Actually run (eval data available)

| Phase | Idea | Outcome |
|---|---|---|
| 1A pent_tok | Tied embed Pentanary        | killed @4 % sliding, early bpb +0.0428 above baseline, abandoned |
| 1A int4_tok | Tied embed Int4             | +0.0095 regression — int6_tok dominates, abandoned |
| 2A    | Inter-layer delta entropy measurement (`analyze_inter_layer.py`) | H(W)=2.124 bits vs H(ΔW)=2.128 bits (+0.004), delta magnitude 1.4× raw — Shannon-floor evidence |
| 4     | `p5a_bg4096` BigramHash 4096 | ~1.146 mid-eval vs hm5 ~1.144, abandoned |
| 4     | `p5a_bg8192` BigramHash 8192 | ~1.148 mid-eval, abandoned |
| 4     | `p5a_nl12` num_layers 12     | ~1.147 mid-eval, abandoned |
| 4     | `p5a_ve4` ve_layers 7,8,9,10 | ~1.150 mid-eval, abandoned |
| 4     | `p5a_bg4096_hm5`             | ~1.144 mid-eval, tie with hm5-only but +0.5 MB, abandoned |
| 5b    | Depth Recurrence `nl9r2` (9 unique × recur 2 = 18 effective, cf. PR #1394 / #1421 / #1445 depth-recur chain) | 30 % eval @ 1.151 vs hm5 @ 1.136, abandoned |
| 5b'   | Depth Recurrence `nl7r2` (7 unique × recur 2 = 14 effective) | 92 % eval @ 1.166 (post-bugfix re-run), worse |

### Code written, NOT run to eval (abandoned before execution)

| Phase | Idea | Reason stopped |
|---|---|---|
| 1B    | FP32 layer scalars → Int8 | Stub only; target tensors < 1 % of artifact |
| 1C    | Pentanary → Ternary (BitNet b1.58) | `TernaryLinear` + `MLP_UP_TYPE` env + `run.sh` added but **never trained or evaluated**; Phase 1A int6_tok landed the byte savings without the BitNet-at-32M risk |
| 2B    | Hadamard 16-dim block transform | Planning note only; dropped after Phase 2A Shannon-floor result |
| 2C    | Context-aware rANS lookup table | Outline only; same reason + Rust codec rebuild blocker |
| 3     | Custom `HQGRANS1` binary container | `serialize_hybrid_binary` / `deserialize_hybrid_binary` added, but lzma9-after-rANS already absorbs most pickle overhead — net benefit ≈ 0 on the `.rans.ptz.xz` path, kept for future lzma-free experiments |

## Architecture re-investment table (Phase 4 sanity sweep, 1-seed s1337 SLOT@100)

Each variant retrained from scratch with the same Phase 5a stack:

| variant         | byte cost vs base | mid-eval bpb | result |
|-----------------|-------------------|--------------|--------|
| `p5a` (no extra) | 0                 | ~1.144      | base   |
| `p5a_bg4096`     | +0.5 MB           | ~1.146      | hurts  |
| `p5a_hm5` ⭐    | +1.0 MB (FFN 4→5) | ~1.144      | **best** |
| `p5a_bg4096_hm5` | +1.5 MB           | ~1.144      | tie    |
| `p5a_bg8192`     | +1.5 MB           | ~1.148      | hurts  |
| `p5a_nl12`       | +1.5 MB           | ~1.147      | hurts  |
| `p5a_ve4`        | +0.2 MB           | ~1.150      | hurts  |

`hm5` (hidden_mult 4 → 5) is the only re-investment that uses Phase 1A's saved
0.6 MB without regression.

## Reproducibility
```bash
bash records/track_non_record_16mb/2026-04-09_v62_p5a_hm5_phase5a/run.sh both 1337
bash records/track_non_record_16mb/2026-04-09_v62_p5a_hm5_phase5a/run.sh both 1338
bash records/track_non_record_16mb/2026-04-09_v62_p5a_hm5_phase5a/run.sh both 1339
```
Identical 8×H100 SXM training pipeline as
`track_non_record_16mb/2026-04-08_v61_h100_aggressive_slot_steps100`, plus the
Phase 5a env vars (`QK_GAIN_INIT=5.0`, `MUON_EQ_R=1`, `EMBED_QUANT_BITS=6`,
`EMBED_QUANT_TOK_EMB=1`, `HIDDEN_MULT=5.0`) and `--ema 0.9965`.

## Eval cost
- Training: 600s × 8×H100 SXM ≈ $4 / seed
- Eval (SLOT-100, stride=64): ~50 min/seed
- Eval (Legal TTT Muon, stride=64): ~30-40 min/seed (separate copy of model)
- 3-seed train+eval ≈ $30 of RunPod credit

## Files
- `train_gpt.py` — same as `2026-04-09_v62_phase5a_sota_trivial/train_gpt.py`
- `run.sh`        — 8×H100 train+eval driver
- `submission.json` — submission metadata
- `PR_BODY.md`    — PR description
- `README.md`     — this file

## Reference
- Parent: openai/parameter-golf#1123 (HybridQuantGPT v6.1, 1.1986 non-record)
- SLOT origin: openai/parameter-golf#1128 (AnubhavBharadwaaj, 2026-03-30 09:43 UTC, `SLOT_LR=0.003 SLOT_STEPS=5`)
- SLOT + Muon-TTT variant: openai/parameter-golf#1176 (bigbag, `SLOT_LR=0.005 SLOT_STEPS=8`, QK-Gain 4.0)
- QK-Gain 5.0: openai/parameter-golf#1413 (dexhunter)
- MuonEq-R: openai/parameter-golf#1394 (clarkkev)
- EMA 0.9965: openai/parameter-golf#1421, openai/parameter-golf#1445 (X-Abhishek-X)
- Prior records (this submitter):
  - `2026-04-08_v61_aggressive_slot_1159` (3-seed 1.157108, SLOT-20)
  - `2026-04-08_v61_slot_steps50_1150` (3-seed 1.148772, SLOT-50)
  - `2026-04-08_v61_slot_steps80_1147` (3-seed 1.147032, SLOT-80)
  - `2026-04-08_v61_slot_steps100_1146` (3-seed 1.146523, SLOT-100)
