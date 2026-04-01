# Neural Science Board

Track: Rascal lineage · Goal: beat leaderboard #1 · Score: sliding-window BPB
Champion: **1.10986874 BPB** (seed 444) · **15.44MB** · `neural/2026-03-30_Rascal_II/`

Legend: → PROMOTED · ✓ PASS · ✗ FAIL · ⏳ PENDING · — n/a

---

## Competitive Landscape (updated 2026-03-31)

| Status | PR | Score (seed 444) | Author | Key Techniques | Notes |
|--------|-----|-----------------|--------|---------------|-------|
| MERGED #1 | #1019 | 1.1147 | abaybektursun | AR Self-Gen GPTQ + XSA-all + BigramHash 3072×112 | Official leaderboard top |
| **OUR OPEN PR** | **#1120** | **1.10987** | **Frosty40** | **Rascal II — XSA-all + Muon + Bigram2048 + SKIP_GPTQ** | **Pending merge. Beats all below.** |
| Open — beats us | #1089 | **1.1091** | mikeapedia | Turbo-Muon + EngramLite + ParamBanking + ASQU | ⚠️ 0.00077 BPB ahead of us |
| Open — we beat | #1179 | 1.1105 | dexhunter | Split-LR + BigramHash 2816×160 + Brotli | Clean |
| Open — we beat | #1135 | 1.1116 | barneywohl | Fused Triton MLP + Full GPTQ + Coprime + BH2816 | Clean |
| Open — we beat | #1169 | 1.1126 | Bortlesboat | Turbo-Muon + EngramLite + ParamBanking + GPTQ Reserve | Clean |
| Open — we beat | #1060 | 1.1122 | dexhunter | Coprime-stride loader + Full Hessian GPTQ + XSA-all | Clean |
| SLOT — no ruling | #1176 | 1.0914 | bigbag | SLOT + QK-Gain + Muon-TTT | Open. No organizer ruling. Community member flagged — not official. |
| SLOT — no ruling | #1172 | 1.1015 | dexhunter | SLOT + Split-LR + Full GPTQ | Open. No organizer ruling. Organizer requested but not received. |
| CONTESTED | #1185 | 0.9641 | — | N-gram backoff cache | Under dispute — likely invalid probability distributions |

**Summary**: We hold the best legal score in the open PR queue. PR #1089 at 1.1091 is the only clean
competitor ahead of us, by 0.00077 BPB — within 1-sigma seed variance.

**SLOT status**: No official organizer (0hq/valerio-oai/xuandong-openai) has ruled on SLOT in any PR.
The "causality violation" comment on #1176 came from community member msisovic (author_association: NONE).
All SLOT PRs remain open. Organizer ruling formally requested but not received as of 2026-03-31.

---

## What Rascal II Has (already in stack — no need to add)

| Feature | Our Config | Notes |
|---------|-----------|-------|
| LeakyReLU(0.5)² | ✅ Yes, custom Triton kernel | Lines 151-206 in vault file |
| LN_SCALE=1/√(layer+1) | ✅ Default=1 | Matches PR #1019 |
| XSA on all 11 layers | ✅ XSA_LAST_N=11 | Matches leaders |
| Full Hessian GPTQ code | ✅ Exists (lines 552-643) | **DISABLED** — SKIP_GPTQ=1 |
| Coprime loader | ✅ Exists | COPRIME_MAX_LOADED_SHARDS=**1** (CRITICAL — do NOT change) |
| Multiple LR groups | ✅ HEAD_LR, MATRIX_LR, EMBED_LR | Leaders have similar |
| WARMDOWN_ITERS | ✅ 3500 | Leaders use 4000 — gap exists |

---

## What We Are Missing vs Competition Leaders

| Feature | Our State | Leader State | Est. BPB Delta | Risk |
|---------|-----------|-------------|---------------|------|
| Full Hessian GPTQ | SKIP_GPTQ=1 | Enabled | **−0.003 to −0.009** | Medium — costs ~328 training steps |
| AR self-gen GPTQ calibration | Training data | Self-generated seqs | ~−0.001 to −0.003 | Low once GPTQ is on |
| BigramHash vocab | 2048 | 3072 | ~−0.001 to −0.002 | Low — size est. +~31KB |
| Warmdown iters | 3500 | 4000 | ~−0.0005 | Very low |
| Brotli compression | zstd-22 | Brotli-11 | Frees artifact budget | Medium — new dependency |
| Code minification | 118,521 bytes | ~28-30KB | Frees ~88KB for weights | Medium — must still run |

Budget: 15,554,053 / 16,000,000 = **445,947 bytes headroom**.
Code: 118,521 bytes. Model: 15,435,532 bytes.

---

## Thread: Rascal Architecture — XSA + Parallel Muon + Bigram

Core lineage. Rascal II is the current best legal open submission.

| Date | Leg | Change vs Parent | Gate | Full Run BPB (seed 444) | Size | Verdict | Key Finding |
|------|-----|-----------------|------|-------------------------|------|---------|-------------|
| 2026-03-30 | **Rascal_II** (CHAMPION) | 11L XSA-all + Parallel Muon + Coprime (SHARDS=1) + Bigram2048×128 + RoPE16 + Late QAT + SWA | confirmed | **1.10986874** | **15.44MB** | → PROMOTED | 3-seed mean 1.1099. 26.99M params. SKIP_GPTQ=1 naive int6 + zstd-22. 6593 steps @ ~91ms. |

Seed detail:
| Seed | BPB | Size |
|------|-----|------|
| 42   | 1.11018163 | 15,540,001 B |
| 300  | 1.10979099 | 15,542,719 B |
| 444  | 1.10986874 | 15,554,053 B |
| mean | **1.1099**  | 15,554,053 B (max) |

DO NOT CHANGE without explicit hypothesis:
- BIGRAM_DIM=128, XSA_LAST_N=11, ROPE_DIMS=16
- COPRIME_MAX_LOADED_SHARDS=**1** (changing to 4 caused LC4-class failure previously)
- COMPILE_FULLGRAPH=1

---

## Thread: Quantization — GPTQ

Biggest single gap vs competition. quant_gap = +0.0217 BPB (int6 - float32) — confirmed in sweep.
GPTQ code is already in the vault script (lines 552–643). We run SKIP_GPTQ=1 because original
Rascal I was too large with GPTQ. Rascal II is 15.44MB — with GPTQ enabled, quantization quality
improves, potentially offsetting the ~328 lost training steps from the 30s reserve window.

Current calibration (when GPTQ enabled): 256 samples from training data, 2048 token context.
PR #1019 uses AR self-generated data (64 seq × 2048 tok, temp=0.8) — better for deployment
distribution; does NOT touch val data (legal).

**BUG (2026-03-31)**: `gptq:calibrated 2 layers in 1.9s` → `gptq_quantize: 0 GPTQ layers`.
Only 2 of ~many layers are hooked during calibration. Quantizer key lookup matches 0 calibrated layers.
Likely cause: `torch.compile` changes module internals so hooks don't attach to the right places.
`gptq_full` (full training with SKIP_GPTQ=0) is the next test.

| Date | Leg | Change vs Parent | Gate | Full Run BPB | Size | Verdict | Key Finding |
|------|-----|-----------------|------|-------------|------|---------|-------------|
| 2026-03-31 | gptq (post-train) | SKIP_GPTQ=0, SKIP_TRAIN=1 (reuse baseline ckpt) | ✗ | — | — | ✗ BUG | Only 2 layers hooked, 0 quantized. torch.compile hook mismatch. Model unchanged = 0 delta. |
| — | Rascal_III_GPTQ | SKIP_GPTQ=0 (full training + GPTQ calib) | — | — | — | ⏳ PENDING | Costs ~30s → ~328 fewer steps. GPTQ_RESERVE_MS=30000. Single variable. |
| — | Rascal_III_ARcal | AR self-gen calibration (replace training-data) | — | — | — | NOT STARTED | Requires ~20 lines new code. Do AFTER GPTQ gate passes. |

---

## Thread: Architecture Capacity — Bigram Hash

Competition moved from BigramHash 2048 → 3072 (PR #1019 uses 3072×112, we use 2048×128).
More buckets = better coverage of the 2-gram space = less hash collision.
Size impact of 3072 (keep DIM=128): +1024 buckets × 128 dim = +131K params × 0.75 bytes/param × ~0.5 zstd ≈ +~50KB. Well inside 445KB headroom.

| Date | Leg | Change vs Parent | Gate | Full Run BPB | Size | Verdict | Key Finding |
|------|-----|-----------------|------|-------------|------|---------|-------------|
| 2026-03-31 | bigram_3072 (sweep) | BIGRAM_VOCAB_SIZE=2048→3072 | proxy: 0.0000 | — | 14.30MB | ✗ DEAD | Zero measured signal at proxy scale. Size increases +0.78MB. Do not run 8×GPU. |
| 2026-03-31 | bigram_4096 (sweep) | BIGRAM_VOCAB_SIZE=2048→4096 | proxy: +0.0006 | — | 14.42MB | ✗ DEAD | Hurts. Size risk (14.42MB). Dead permanently. |

---

## Thread: Training Schedule

| Date | Leg | Change vs Parent | Gate | Full Run BPB | Size | Verdict | Key Finding |
|------|-----|-----------------|------|-------------|------|---------|-------------|
| 2026-03-31 | warmdown_4k (sweep) | WARMDOWN_ITERS=3500→4000 | proxy: +0.0034 | — | 13.79MB | ✗ DEAD | HURTS significantly. Root cause: time-based schedule → longer warmdown → QAT fires EARLIER (step 2297 vs 2376). Dead permanently. Do not retry without step-based schedule. |
| 2026-03-31 | qat_early (sweep) | LATE_QAT_THRESHOLD=0.15→0.25 | proxy: +0.0004 | — | 14.23MB | ✗ DEAD | Hurts. QAT at step 2021 (355 earlier). No gain from earlier QAT at proxy scale. |
| 2026-03-31 | qat_late (sweep) | LATE_QAT_THRESHOLD=0.15→0.05 | proxy: +0.0004 | — | 14.01MB | ✗ DEAD | Hurts. QAT at step 2721 (345 later). Symmetric with qat_early — threshold doesn't matter. |
| 2026-03-31 | swa_dense (sweep) | SWA_EVERY=50→10 | proxy: +0.0010 | — | 13.60MB | ✗ DEAD | Hurts. More snapshots = worse averaging. Current SWA_EVERY=50 is correct. |
| 2026-03-31 | rope_32 (sweep) | ROPE_DIMS=16→32 | proxy: -0.0004 | — | 13.56MB | ✗ BORDERLINE | Below noise floor (~0.001 needed). Do not run 8×GPU. |

---

## Thread: SLOT (Sample-specific LM Optimisation at Test-time)

**Proxy signal: −0.0085 BPB (1200 steps, 1-GPU, SLOT_MAX_WINDOWS=512, seed=444)**
Proxy inflates 5-15×. Real signal estimate: −0.0006 to −0.0017 BPB at full run.

### What our SLOT does (from code audit, lines 1903-1923 of experiment train_gpt.py)

For each sliding window batch [ws .. ws+seq_len]:
1. Compute frozen hidden states from base model (no gradient, model unchanged)
2. Initialize per-batch delta = zeros(1,1,dim), requires_grad=True
3. 8 steps AdamW: optimize delta via `cross_entropy(logits(hidden+delta), y_batch)`
4. Score: `cross_entropy(logits(hidden+delta.detach()), y_batch)` (same y_batch)
5. Only the new stride-64 positions are counted in BPB

delta is a single broadcast vector (1×1×dim) — it shifts ALL positions by the same direction.
Model weights are never modified. Training trajectory is identical to baseline.

### Legality Analysis — Current Implementation

The competition rule (README): **"you are only allowed to test-time train on validation set tokens you've already evaluated your model on"**

| Window | Positions in y_batch used for delta opt | Positions already scored | Positions NEW (not yet scored) |
|--------|----------------------------------------|--------------------------|-------------------------------|
| First window (ws=0) | tokens[1..2047] | 0 (none) | 2047 (all) |
| Subsequent windows | tokens[ws+1..ws+2047] | seq_len−stride = 1984 (96.9%) | stride = 64 (3.1%) |

**Issue**: delta is optimized using `y_batch` which includes the 64 new-stride targets, then those same new targets are scored under the optimized delta. This is **not** strictly "score-first" — the optimization sees the targets before scoring them.

**Magnitude**: 3.1% of gradient comes from not-yet-scored tokens. delta is a single shared vector so it cannot memorize per-position — it finds a direction that helps on average across the batch. But the rule doesn't have a "3.1% is fine" exception.

**SLOT PRs that have this same structure**: #1084, #1105, #1128, #1150, #1172, #1176 — all open, all awaiting organizer ruling.

### The Legal Fix — Context-Only SLOT

Unambiguously compliant: optimize delta only on positions already scored, score only the new positions.

```
For window at ws with stride=64:
  context_y = y_batch[:, :seq_len-stride]   # already scored — legal to train on
  new_y      = y_batch[:, seq_len-stride:]   # not yet scored — score-first then stop

  optimize delta on context_y (8 steps AdamW)
  score only new_y under optimized delta
```

First window (no context): optimize on prefix of window 0 (arbitrary split, e.g. first 90%), score last 10%. Or skip SLOT on window 0.

Requires ~30 lines of code change in the eval function. Worth testing if current SLOT proves illegal.

### Status & Strategy

| Question | Answer |
|----------|--------|
| Official organizer ruling on SLOT? | **None.** All SLOT PRs open as of 2026-03-31. |
| Who said "causality violation"? | msisovic — community member (author_association: NONE), not organizer |
| Does our impl strictly satisfy "already evaluated"? | **No** — 3.1% of gradient from new tokens, first window 100% new |
| Is context-only SLOT strictly legal? | **Yes** — score first, then adapt on those scores |
| Should we submit with current SLOT? | **NO** — wait for organizer ruling first |

**Path forward**:
1. Watch #1172 / #1176 for official organizer comment from @0hq / @valerio-oai
2. If organizer blesses standard SLOT → current implementation is usable
3. If organizer rules against → pivot to context-only SLOT and retest
4. Do NOT include SLOT in any submission until ruling arrives

| Date | Leg | Change | Signal | Verdict |
|------|-----|--------|--------|---------|
| 2026-03-31 | QK_Gain_SLOT experiment | baseline vs slot_only (1200 steps, 1GPU) | ✓ −0.0085 proxy sw_bpb | ⚠️ ILLEGAL — causality violation |
| 2026-03-31 | QK_Gain_SLOT_Legal | Context-only SLOT (optimize on scored prefix only) | ✓ −0.0057 proxy sw_bpb | ✓ GATE PASSED — awaiting 8×GPU run |

---

## Thread: Artifact Compression

Low-risk infrastructure wins. Brotli-11 vs zstd-22; code minification.
Code minification potential: 118KB → ~28KB = ~90KB freed for model weights.

| Date | Leg | Change vs Parent | Gate | Full Run BPB | Size | Verdict | Key Finding |
|------|-----|-----------------|------|-------------|------|---------|-------------|
| — | Rascal_Brotli | Brotli-11 instead of zstd-22 | — | — | — | NOT STARTED | New python dep (brotli). Run AFTER architecture wins are locked in. |
| — | Rascal_Minified | Minify train_gpt.py (~90KB freed) | — | — | — | NOT STARTED | Infrastructure change. Minified code must be tested locally first. |

---

## Recommended Hypothesis Order (updated 2026-03-31 post-sweep)

Arch+Sched sweep verdict: **all 9 cases dead or borderline.** No 8×GPU runs from sweep.
GPTQ is the only open win. Legal SLOT gate passed — queued for 8×GPU.

| Priority | Leg Name | Change | Expected Gain | Risk | Est. Cost | Status |
|----------|---------|--------|--------------|------|-----------|--------|
| **1** | **Rascal_III_GPTQ** | SKIP_GPTQ=0, full training + GPTQ calib | −0.003 to −0.009 BPB | Low (code exists) | 1 env var | ⏳ BUG TO FIX FIRST |
| **2** | **QK_Gain_SLOT_Legal full run** | Context-only SLOT on 8×GPU | −0.0004 to −0.0011 BPB est. | Medium (ruling risk) | eval-only | ⏳ READY |
| **3** | **Rascal_III_ARcal** | AR self-gen GPTQ calib (after GPTQ passes) | −0.001 to −0.003 more | Low | ~20 lines code | NOT STARTED |
| 4 | Rascal_Brotli | zstd → Brotli-11 | Frees budget | Medium (new dep) | New dep | NOT STARTED |
| 5 | Rascal_Minified | Code minification | Frees ~90KB | Medium (infra) | Infra work | NOT STARTED |
| ✗ | ~~Bigram3072~~ | BIGRAM_VOCAB_SIZE=3072 | 0.0000 at proxy | — | — | DEAD (2026-03-31) |
| ✗ | ~~Warmdown4k~~ | WARMDOWN_ITERS=4000 | +0.0034 (hurts) | — | — | DEAD PERMANENTLY (2026-03-31) |
| ✗ | ~~rope_32~~ | ROPE_DIMS=16→32 | −0.0004 (noise) | — | — | DEAD (2026-03-31) |

Gate target for all new legs: beat **1.10986874** BPB on seed 444 → confirm on seed 300.

---

## All-Time Reference

| Leg | BPB (seed 444) | Size | Mean BPB | Status |
|-----|----------------|------|----------|--------|
| (pre-Rascal history — junkyard) | — | — | — | — |
| **Rascal_II** | **1.10986874** | **15.44MB** | **1.1099** | **CHAMPION (open PR #1120)** |

| 2026-03-31 | **QK_Gain_SLOT_Legal** | context-only SLOT (eval-only) | ✓ gate −0.0057 proxy | — | — | — | ⏳ GATE PASSED, 8×GPU PENDING | |
