# v13 Plan — Strip-down + Capacity Recovery

**Date:** 2026-04-28
**Branch:** shikhar
**Predecessor:** v12 bankless (sliding 1.0955, gap 0.0047, size 16.54 MB — over cap)
**Goal:** Land a submittable run (≤16 MB) at sliding ≤1.090 BPB, ideally closer to 1.083.

---

## Two coupled goals (and the tension between them)

1. **Bring pre-quant down from 1.0908 → ~1.083-1.085** (close 0.006-0.008 BPB)
2. **Get under the 16 MB submission cap** (drop ≥0.6 MB to leave wrapper headroom)

Partially aligned: removing extras frees both params **and** step time.
Partially fighting: some extras may be load-bearing for pre-quant, and we don't know which without ablation.

---

## What v12 carries (vs PR #1394 / #1493 base)

PR #1394 base: 11 layers, model_dim=512, SP8192 vocab, recurrence on 4-5, parallel residuals optional, MuonEq-R, SDClip k=12.85. **No** SWA, **no** bigram, **no** VE, **no** SmearGate, **no** PKO, **no** XSA, **no** TTT.

v12's extras on top of that base (each is a strip candidate):

| Feature | v12 setting | Estimated size | Step_avg cost | Pre-quant value (best guess) |
|---|---|---|---|---|
| **ValueEmbedding** (`ve_shared`) | enabled, vocab×128 + 128→kv_dim_ve proj, layers 9,10 | **~1.0–1.1 MB** (8192×128 ≈ 1M params @ INT8) | small (only 2 layers reinject) | Medium (~0.005–0.015 BPB) |
| **BigramHashEmbedding** | vocab=3072, dim=112, +proj 112→512 | **~0.40 MB** (344k embed + 57k proj) | small | Low–medium (~0.003–0.010 BPB) |
| **SmearGate** | per-block gate | **~0.05 MB** (per-layer model_dim params) | small | Low (~0.001–0.003 BPB) |
| **SWA** (window=256, layers 0-5) | on | **0 MB** (mask only) | negative cost (faster than full) | Disputable. v10 stripped SWA + raised WD with banks: gap unchanged, pre-quant unclear. Probably small effect. |
| **XSA_LAST_N=11** (XSA on all layers) | on | **0 MB** (gate params only) | non-trivial — extra attention path | Unknown. Our novel contribution — never ablated cleanly post-bankless. |
| **PKO** | on, full-attn layers | 0 MB | tiny | Low (~0.001–0.003 BPB) |
| **MTP heads** (mtp_num_heads) | check default | **lm_head × N**, big if >0 | training-only (drop at eval) | Training aid only |
| **TTT** | disabled (`TTT_ENABLED=0`) | 0 MB | 0 | **Free pre-quant gain** if we turn it on (PR #1493 uses this) |

**Important uncertainty:** size estimates extrapolated from param count × INT8/INT6 byte budgets without seeing the actual prune log breakdown by component. Pull `keep_float_tensor` log lines from v12 to confirm. The VE estimate dominates and is the riskiest number — could be 0.7 MB or 1.3 MB depending on `_ve_target_dim` (= `num_kv_heads × head_dim` = 4 × (512/num_heads)).

---

## What we're missing vs PR #1493 (1.0810 SOTA)

| Feature | v12 has | PR #1493 | Cost to add |
|---|---|---|---|
| **3-layer recurrence (L3-5 → 17 virtual)** | No (L4-5, 13 virtual) | Yes | 0 MB, +30% FLOPs in layers 3-5 → step_avg goes up |
| **Parallel residuals from L7+** | `PARALLEL_START_LAYER=-1` (off) | On | 0 MB (just resid_mix pattern) — **free if helpful** |
| **Legal score-first TTT (SGD lr=0.005, 3 epochs)** | Disabled | On | 0 MB. Eval-only. Improves post-quant directly. **Should be on already.** |
| **Hyperparams: WD=0.095, MLR=0.022, EMA=0.9965** | WD=0.085 likely; check MLR/EMA | These specifically tuned | 0 MB |
| **Round-robin Muon** (PR #1394 style) | Yes (we ported in v12) | Yes | Already done |
| **MuonEq-R** | Yes (`row_normalize=True`) | Yes | Already done |

**TTT is the most surprising omission.** The flag exists in v12 (`TTT_ENABLED`, lines 164-170) but defaults off. It's a pure eval-time technique that PR #1493 uses to push their post-quant score. Toggling it on costs nothing at training time and should land as a free-or-near-free post-quant gain. **Critical caveat:** TTT runs after quantization on the dequantized model — its benefit shows up in `final_int6_sliding_window`, not in `post_ema val_bpb`. We need to verify it actually fires correctly with the bankless model.

---

## Size math, in detail

v12 numbers from log:
- `unpruned=15.77 MB`, `target=15.9 MB` → `prune: already fits, no pruning needed`
- Final `.ptz` blob: **16.4 MB** (per log) / 16.54 MB (per commit message)

The 0.6-0.8 MB delta between unpruned weights and the .ptz file is the **brotli wrapper overhead + tokenizer + metadata + indexing**. That's roughly fixed — we can't compress it further without code changes.

So the *correct* tightening of `target_mb` is **~15.2-15.3** (give 0.7-0.8 MB to wrapper). At 15.77 MB unpruned, that means **0.5-0.6 MB has to actually come out** — either via pruning or by deleting a feature.

**Pruning cost:** PR #1394 / our prior runs suggest pruning ~0.5 MB at INT6 costs ~0.005-0.010 BPB. Not free.

**Strip cost:** Removing a feature shrinks unpruned size **and** can speed up training **and** can change pre-quant — three coupled effects. Strip-vs-prune is not symmetric.

### Two strip candidates that solve size by themselves

1. **Strip ValueEmbedding** (~1 MB out): goes from 15.77 → ~14.8 MB unpruned, comfortably under cap with no pruning needed. Risk: pre-quant gets worse by 0.005-0.015 BPB.
2. **Strip BigramHashEmbedding** (~0.4 MB out): borderline. 15.77 → 15.4 MB unpruned. Just barely fits with target=15.2. Pre-quant cost ~0.003-0.010 BPB.

**Both** strips: 15.77 → ~14.4 MB. Massive headroom. But two simultaneous strips means we can't attribute pre-quant changes cleanly.

---

## Staged run plan

Resist one mega-run. Three small runs in sequence give clean attribution. Each is ~10 minutes wallclock on 8×H100.

### v13a — Minimum-risk submittable run

**Hypothesis:** TTT + parallel residuals are free wins; bigram is the cleanest size cut.

```bash
RUN_ID=v13a_ttt_parallel_strip_bigram \
  TTT_ENABLED=1 TTT_LR=0.005 TTT_EPOCHS=3 \
  PARALLEL_START_LAYER=7 \
  BIGRAM_VOCAB_SIZE=0 \
  TARGET_MB=15.3 \
  # everything else as v12
```

Expected outcome:
- Size: ~15.4 MB unpruned → fits cleanly
- Pre-quant: roughly flat or slightly worse (±0.005)
- Sliding (post-quant + TTT): **0.005-0.015 BPB better** than v12 → land 1.080-1.090 territory

**Why this first:** TTT is the cheapest improvement and is already plumbed. Parallel residuals are free architecturally. Bigram is the smaller of the two embedding-class strips, so if it costs us pre-quant we still have VE to fall back on. Worst case we learn TTT is broken with bankless.

### v13b — If v13a still over budget or pre-quant worse

**Hypothesis:** VE is the better strip than bigram (more size, similar pre-quant cost).

```bash
RUN_ID=v13b_strip_ve \
  TTT_ENABLED=1 PARALLEL_START_LAYER=7 \
  VE_LAYERS="" \  # disable VE
  TARGET_MB=15.3 \
  # bigram kept
```

### v13c — Capacity recovery via 3-layer recurrence

**Only if** v13a/b show pre-quant flat or improving. Add the SOTA's recurrence span:

```bash
RUN_ID=v13c_3layer_recur \
  RECUR_LAYERS="3,4,5" \
  # winning config from v13a or v13b
```

Risk: 3-layer recurrence makes layer 3-5 weights see 3× compute → potentially reopens the quant gap that v12 closed. Watch the gap field carefully. If it moves from 0.005 → 0.015+, abort and return to 2-layer.

---

## Explicitly NOT doing in v13 — and why

- **Not stripping SWA.** v10 stripped SWA with banks and it didn't help, but post-bankless it could be different. Saving for v14 if v13 stalls.
- **Not stripping XSA.** It's our novel contribution and may be load-bearing. Needs controlled ablation later, not bundled with size cuts.
- **Not stripping SmearGate.** Tiny size payoff (~0.05 MB), pre-quant value unknown but cheap to keep.
- **Not raising WD to 0.095.** PR #1493's WD=0.095 is tuned for *their* stack with no banks, no bigram, no VE — applying it blindly to ours is cargo-culting until we strip those features first.
- **Not retuning MLR/EMA.** Same reason — they're correlated with WD and architecture.
- **Not restoring paired-head Muon yet.** v12 already shows the bankless path works; reintroducing paired-head NS5 to recover step time is a v14 optimization, not a quality move. If TTT gives us 0.01+ BPB, the speed deficit matters less.

---

## Critical caveats before running anything

1. **TTT might not be wired correctly with bankless.** The `TTT_ENABLED` flag exists but the SGD step in `train_v12_bankless.py` line ~2158 may have been written for bank weights. **Read the TTT code path before running v13a.** If broken, v13a's expected 0.01-0.015 BPB gain evaporates.

2. **The size of the .ptz wrapper isn't tightly known.** Assuming 0.7 MB based on 15.77 → 16.4 MB. If wrapper varies with content (it might — brotli compresses tokenizer and metadata differently), tightening `TARGET_MB` to 15.3 might still miss. Measure the unpruned-vs-ptz delta on v13a's actual output.

3. **TTT score-first is "legal" only if implemented correctly.** PR #1493 specifies "score before update" — i.e., score the chunk first, then adapt. Doing it the other way is a leakage bug and would invalidate the submission. Verify the order in the code.

4. **The 0.116 BPB v12 jump may not be reproducible cleanly.** v12 uses a separate file (`train_v12_bankless.py`) that may differ from `train_gpt_swa.py` in subtle ways beyond the bank rewrite. Before declaring victory, check that the v12 codepath uses the same data, same seed handling, same eval routine. **Single-seed result with no replicate.** Per the v7 doc's reproducibility data, expect ±0.005 BPB seed variance. v13a should ideally be 2 seeds.

5. **Size estimates are extrapolations**, not ground truth. Pull the per-tensor breakdown from v12's quantized state dict before trusting which strip frees how much.

---

## Pre-flight checklist (before pushing v13a)

- [ ] Read TTT code path in `train_v12_bankless.py` (line ~1196 and ~2158) — confirm it calls SGD on per-layer CastedLinear weights, not bank weights.
- [ ] Confirm TTT runs **score-then-update** order (chunk loss computed before SGD step on that chunk).
- [ ] Pull per-tensor size breakdown from v12's `final_model.int6.ptz` artifact — confirm VE ≈ 1 MB, bigram ≈ 0.4 MB.
- [ ] Confirm `BIGRAM_VOCAB_SIZE=0` cleanly disables bigram (no NaN, no shape errors). Read `GPT.__init__` line 924 to verify `if bigram_vocab_size > 0` gate.
- [ ] Confirm `PARALLEL_START_LAYER=7` is wired to per-block `_parallel` flag in v12 code.
- [ ] Sanity check: v13a target_mb=15.3 with bigram off → unpruned should be ~15.3-15.4 MB. If under 15.3, no pruning fires (good). If over, pruning fires (acceptable but track BPB cost).

---

## Bottom line

The strip+add plan can simultaneously fix size and lower pre-quant — but the cleanest version is staged, not one big run.

- **v13a:** TTT + parallel residuals on, bigram off, target_mb=15.3. Expected sliding 1.080-1.090, submittable.
- **v13b** (if v13a fails on size or pre-quant): swap bigram-strip for VE-strip.
- **v13c** (only if v13a/b succeed and pre-quant headroom remains): 3-layer recurrence for SOTA-level capacity, watching the quant gap closely.

If v13a lands at ≤1.090 sliding under 16 MB, **submit it immediately** — that beats our 1.1117 baseline by ~0.02 BPB and locks in a real improvement before the deadline. Don't gamble on v13c if v13a is already a clear improvement.
