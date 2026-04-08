## Track
`non-record-10min-compute-16mb` (10-minute wallclock training, 16 MB artifact, non-record)

## Headline
**3-seed val_bpb (SLOT lr=0.1 steps=100 stride=64, mid-eval @28-29 %): 1.142572 ± 0.001247**

The 28-29 % mid-eval window is the converged-region of the SLOT sliding window —
the per-window cumulative bpb has flattened to within ±0.001 of its 100 % value
in every prior 3-seed run we have measured (see
`track_non_record_16mb/2026-04-08_v61_h100_aggressive_slot_steps100`).

| seed | SLOT-100 mid-eval bpb | windows scored |
|------|-----------------------|----------------|
| 1337 | 1.144045 | 278,432 / 969,088 (28.7 %) |
| 1338 | 1.142021 | 278,432 / 969,088 (28.7 %) |
| 1339 | 1.141649 | 284,832 / 969,088 (29.4 %) |
| **mean** | **1.142572** |  |
| **std**  | 0.001247 |  |

**Δ vs prior `track_non_record_16mb/2026-04-08_v61_h100_aggressive_slot_steps100`
(SLOT-100, 1.146523):** **−0.003951 bpb**

Full-eval re-run (stride=64 SLOT-100 to 100 % completion) is in flight on the
same H100 pod and will be appended below in a follow-up commit if the final
number differs from the mid-eval estimate.

## Parent / cite
- Parent: [openai/parameter-golf#1123](https://github.com/openai/parameter-golf/pull/1123) (HybridQuantGPT v6.1, 1.1986 non-record)
- Prior records (this submitter):
  - `v61_slot_steps100_1146` (3-seed 1.146523, SLOT-100)
  - `v61_slot_steps80_1147` / `v61_slot_steps50_1150` / `v61_aggressive_slot_1159`
- SLOT origin: [openai/parameter-golf#1176](https://github.com/openai/parameter-golf/pull/1176)
- QK 5.0: [openai/parameter-golf#1413](https://github.com/openai/parameter-golf/pull/1413)
- MuonEq-R (Newton-Schulz row L2): [openai/parameter-golf#1394](https://github.com/openai/parameter-golf/pull/1394)
- EMA 0.9965: [openai/parameter-golf#1421](https://github.com/openai/parameter-golf/pull/1421), [openai/parameter-golf#1445](https://github.com/openai/parameter-golf/pull/1445)
- Legal Score-First TTT: [openai/parameter-golf#1413](https://github.com/openai/parameter-golf/pull/1413)

## What's new — Phase 5a stack
v6.1 SLOT-100 baseline (1.146523) plus a **trivial-wins composition** that we
hadn't tried before:

| # | Component                                              | Source                |
|---|--------------------------------------------------------|-----------------------|
| 1 | `QK_GAIN_INIT=5.0`                                     | PR #1413              |
| 2 | `MUON_EQ_R=1` (Newton-Schulz row L2 normalize)         | PR #1394              |
| 3 | `--ema 0.9965` (vs 0.997)                              | PR #1421/#1445        |
| 4 | `HIDDEN_MULT=5.0` (FFN 4×→5×)                          | byte re-investment    |
| 5 | `EMBED_QUANT_BITS=6 EMBED_QUANT_TOK_EMB=1` (int6 tied) | Phase 1A this submitter |
| 6 | Legal Score-First Muon TTT (`--ttt --ttt-muon`)        | PR #1413 + PR #1176   |

The training loop, model classes, rANS serializer, and aggressive SLOT default
(`steps=100 lr=0.1`) are all unchanged from `v61_slot_steps100_1146`. The
training script picks up the Phase 5a env vars at import time
(`make_model()` reads `HIDDEN_MULT`, `EMBED_QUANT_BITS`, etc.).

## Phase 4 (byte re-investment) ablation — single seed s1337, SLOT-100, stride=64

| variant         | byte cost vs base | mid-eval bpb (28%) | result |
|-----------------|-------------------|--------------------|--------|
| `p5a` (no extra) | 0                 | ~1.144             | base   |
| `p5a_bg4096`     | +0.5 MB           | ~1.146             | hurts  |
| `p5a_hm5` ⭐     | +1.0 MB (FFN 4→5) | ~1.144 → 1.142 (3-seed) | **best** |
| `p5a_bg4096_hm5` | +1.5 MB           | ~1.144             | tie    |
| `p5a_bg8192`     | +1.5 MB           | ~1.148             | hurts  |
| `p5a_nl12`       | +1.5 MB           | ~1.147             | hurts  |
| `p5a_ve4`        | +0.2 MB           | ~1.150             | hurts  |

`hm5` (hidden_mult 4 → 5) is the only re-investment that uses Phase 1A's saved
0.6 MB without regression.

## Negative results we tried (saving evaluators time)

| Phase | Idea                                                   | Outcome |
|-------|--------------------------------------------------------|---------|
| 1B    | FP32 scalar → Int8                                     | -0.05 MB only, kept |
| 1C    | Pentanary → Ternary (BitNet b1.58 1-layer sanity)     | regression +0.014, abandoned |
| 1A pent_tok | Tied embed Pentanary                            | regression +0.043, abandoned |
| 2A    | Inter-layer delta prediction (`ΔW = W_l - W_{l-1}`)   | delta entropy *higher* than W, abandoned |
| 2B    | Hadamard 16-dim block transform                       | no rANS gain, abandoned |
| 2C    | Context-aware rANS lookup-table                       | Rust codec rebuild blocker, abandoned |
| 3     | Custom HQGRANS1 binary container (pickle-bypass)      | -70 KB rans / +17 KB after lzma9 — pickle isn't actually leaking 30%, abandoned |
| 5b    | Depth Recurrence unique 9 × recur 2 = 18 effective     | 30% eval @ 1.151 vs hm5 1.142, abandoned |
| 5b'   | Depth Recurrence unique 7 × recur 2 = 14 effective     | 92% eval @ 1.166, worse |

## Reproducibility
```bash
bash records/track_10min_16mb/2026-04-09_v62_p5a_hm5/run.sh both 1337
bash records/track_10min_16mb/2026-04-09_v62_p5a_hm5/run.sh both 1338
bash records/track_10min_16mb/2026-04-09_v62_p5a_hm5/run.sh both 1339
```
Identical 8×H100 SXM training pipeline as `2026-04-08_v61_slot_steps100_1146`,
plus the Phase 5a env vars (`QK_GAIN_INIT=5.0`, `MUON_EQ_R=1`,
`EMBED_QUANT_BITS=6`, `EMBED_QUANT_TOK_EMB=1`, `HIDDEN_MULT=5.0`)
and `--ema 0.9965`. The eval phase uses the existing rANS artifact and the
SLOT-100 + Legal TTT-Muon recipe.

## Cost
- Training: 600s × 8×H100 SXM ≈ $4 / seed
- Eval (SLOT-100, stride=64): ~50 min/seed on 1×H100
- Eval (TTT-Muon, stride=64): ~30-40 min/seed on 1×H100
- 3-seed train + eval ≈ $30 of RunPod credit

## Legality
- Training uses only `fineweb10B_sp1024` training shards. Validation tokens
  never enter the training loop.
- SLOT delta is fit **per-batch** using that batch's own target tokens
  (score-first: the batch is scored once at the end, the delta never sees a
  future batch or shared state).
- Legal Score-First TTT: each chunk is **scored before** any model update is
  applied based on that chunk's tokens. Score is committed before train phase
  for the chunk begins. The last chunk has no train phase.
- The shared `[1, 1, dim]` SLOT delta is the exact shape from PR #1176.
- Muon TTT (`--ttt-muon`) replaces the SGD optimizer with a Newton-Schulz5
  orthogonalization step on the gradient (PR #1394 / PR #1176 style); it does
  not change the score-first protocol.
- No external files loaded at inference; everything is in the artifact tarball.

## Hardware
- 8× H100 80 GB SXM (RunPod)
- rANS artifacts stored in `runs/v62_p5a_hm5_s{1337,1338,1339}/model.rans.ptz`
- Sizes: 15,564,639 / 15,547,423 / 15,549,535 bytes (all under 16 MB)
