# v13 Plan — Strip-down + Capacity Recovery

**Date:** 2026-04-28 (updated)
**Branch:** shikhar
**Predecessor:** v12 bankless (sliding 1.0955, gap 0.0047, size 16.54 MB — over cap)
**Goal:** Land a submittable run (≤16 MB) at sliding ≤1.090 BPB, ideally closer to 1.083.
**Competition SOTA:** PR #1493 at 1.0810 BPB.

---

## Master Results Table

All runs use 8×H100, 600s wallclock cap, correct tokenizer (370,952 bytes), TARGET_MB=15.2.

| Experiment | Config Change | post_ema | int6 RT | sliding BPB | quant gap | size (bytes) | pruned | steps | verdict |
|------------|--------------|----------|---------|-------------|-----------|-------------|--------|-------|---------|
| **v13a** (baseline) | VE off | 1.0895 | 1.1055 | **1.0955** | 0.0060 | 15,844,725 | 0 | 5108 | **BEST** |
| v13b | bigram off, VE on | 1.0895 | 1.1173 | 1.1075 | 0.0180 | 15,938,354 | 917K (2.9%) | 5067 | BAD — pruning kills quant |
| v13c | TTT enabled | 1.0897 | 1.1067 | — | — | — | — | — | DEAD — +0.645 divergence |
| v13d | parallel L7+ | 1.0907 | 1.1065 | 1.0969 | 0.0062 | 15,844,085 | 0 | ~4700 | -0.0014 worse |
| v13e | recur L3-5 | 1.0902 | 1.1057 | 1.0958 | 0.0056 | 15,884,099 | 0 | 4731 | neutral (+0.0003) |
| v13f | WD=0.095, MLR=0.022 | 1.0897 | 1.1091 | 1.0985 | 0.0088 | 15,571,296 | 0 | 5044 | BAD — quant gap +0.0030 |
| v13g | both VE+bigram off | 1.0918 | 1.1053 | 1.0959 | 0.0041 | 15,542,372 | 0 | 5131 | neutral, 302KB freed |
| v13h | 12 layers, strip both | 1.0886 | 1.1430 | 1.1331 | 0.0445 | 15,990,770 | 2.35M (6.8%) | 4612 | CATASTROPHIC — pruning |
| v13i | dim=528, strip both | CRASH | — | — | — | — | — | — | flash_attn head_dim%8 |

---

## Experiment Results — Detailed

### v13a — Strip VE (baseline, no TTT)
**Config:** VE_ENABLED=0, TARGET_MB=15.2, TTT_ENABLED=0, everything else same as v12.

| Metric | v12 | v13a | Delta |
|--------|-----|------|-------|
| post_ema val_bpb | 1.0908 | **1.0895** | -0.0013 (better) |
| val_loss | 2.8177 | 2.8143 | -0.0034 |
| final_int6_roundtrip_exact | 1.1050 | **1.1055** | +0.0005 |
| sliding BPB | 1.0955 | **1.0955** | 0.0000 |
| size (bytes) | 16,537,005 | **15,844,725** | -692,280 |
| submittable? | NO (over cap) | **YES** | |
| training steps | ~4836 | 5108 | +272 (fewer params = faster steps) |

**Finding: VE strip is essentially free.** Val_loss identical, post-EMA actually 0.0013 better due to 272 extra training steps from faster step time. Size comfortably under 16MB cap with no pruning. This is our submittable baseline.

### v13b — Strip Bigram (VE kept)
**Config:** BIGRAM_VOCAB_SIZE=0, VE_ENABLED=1, VE_DIM=128, VE_LAYERS="9,10".

| Metric | v13a (baseline) | v13b | Delta |
|--------|----------------|------|-------|
| post_ema val_bpb | 1.0895 | 1.0895 | 0.0000 |
| int6 roundtrip | 1.1055 | **1.1173** | **+0.0118** |
| sliding BPB | 1.0955 | **1.1075** | **+0.0120** |
| size | 15,844,725 | 15,938,354 | +93,629 |
| pruned weights | 0 | 917,238 (2.9%) | |

**Finding: Strip bigram is TERRIBLE.** Pre-quant identical to v13a, but VE forces the artifact to 15.47 MB unpruned, requiring 2.9% magnitude pruning to fit TARGET_MB=15.2. This pruning catastrophically widens the quant gap from 0.0060 to 0.0180. VE is too expensive size-wise and not load-bearing quality-wise — it's strictly dominated by the v13a config. **Never keep VE at the expense of pruning.**

### v13c — TTT on Bankless (first test)
**Config:** Same as v13a + TTT_ENABLED=1, TTT_LR=0.005, TTT_EPOCHS=3, TTT_CHUNK_TOKENS=32768, TTT_MOMENTUM=0.9.

Training phase identical to v13a (TTT only affects eval):
- post_ema val_bpb: 1.0897 (matches v13a within noise)
- final_int6_roundtrip_exact: 1.1067

**TTT eval result:**

| Metric | Without TTT (INT6) | With TTT | Delta |
|--------|-------------------|----------|-------|
| BPB | 1.1067 | **1.7519** | **+0.6452 (catastrophic)** |
| Eval time | ~9s | **571s (~9.5min)** | 63x slower |

**Finding: TTT is DEAD for our stack.** This is the worst TTT failure across all versions:
- v4 (bank-based): +0.089 BPB
- v5 (bank-based): +0.046 BPB
- v13c (bankless): **+0.645 BPB** — complete divergence

The SGD adaptation catastrophically diverges on the bankless architecture. TTT also nearly exceeds the 10-minute eval cap by itself (571s). **Do not attempt TTT again without fundamental redesign.**

### v13d — Parallel Residuals
**Config:** Same as v13a + PARALLEL_START_LAYER=7 (layers 7-10 run attn+MLP in parallel).

| Metric | v13a (baseline) | v13d | Delta |
|--------|----------------|------|-------|
| post_ema val_bpb | 1.0895 | 1.0907 | +0.0012 |
| int6 roundtrip | 1.1055 | 1.1065 | +0.0010 |
| sliding BPB | 1.0955 | **1.0969** | **+0.0014** |
| size | 15,844,725 | 15,844,085 | -640 |

**Finding: Parallel residuals hurt slightly.** Zero parameter cost but -0.0014 BPB. The sequential attn→MLP dependency matters for our architecture — MLP benefits from seeing attention-enriched representations. PR #1493 may get away with parallel because their TTT compensates; we can't.

### v13e — 3-Layer Recurrence (L3-5)
**Config:** Same as v13a but RECUR_LAYERS="3,4,5" (was "4,5").
Encoder: [0,1,2,3,4,5,3,4,5] → decoder: [3,4,5,6,7,8,9,10]. One extra recurred layer.

| Metric | v13a (baseline) | v13e | Delta |
|--------|----------------|------|-------|
| post_ema val_bpb | 1.0895 | 1.0902 | +0.0007 |
| int6 roundtrip | 1.1055 | 1.1057 | +0.0002 |
| sliding BPB | 1.0955 | **1.0958** | **+0.0003** |
| size | 15,844,725 | 15,884,099 | +39,374 |

**Finding: 3-layer recurrence is neutral.** No BPB improvement from the extra virtual depth. Layer 3 doesn't benefit from being recurred — it's early enough that its representations haven't specialized. Quant gap is actually slightly better (0.0056 vs 0.0060), likely because the additional recurrence acts as implicit regularization. But the +0.0003 sliding is noise. **Not worth the complexity and slower step time.**

### v13f — Hyperparameter Tuning (PR #1493 aligned)
**Config:** Same as v13a but MUON_WD=0.095 (was 0.085), MATRIX_LR=0.022 (was 0.025).

| Metric | v13a (baseline) | v13f | Delta |
|--------|----------------|------|-------|
| post_ema val_bpb | 1.0895 | 1.0897 | +0.0002 |
| int6 roundtrip | 1.1055 | **1.1091** | **+0.0036** |
| sliding BPB | 1.0955 | **1.0985** | **+0.0030** |
| size | 15,844,725 | 15,571,296 | -273,429 |

**Finding: PR #1493's hyperparams are WRONG for our stack.** Pre-quant is neutral, but the lower MLR (0.022) produces weights that quantize worse — the quant gap widens from 0.0060 to 0.0094. Higher MLR (0.025) keeps weight distributions more quantization-friendly. The smaller artifact size is a red herring — smaller models aren't better if they quantize worse. **Keep our original hparams.**

### v13g — Strip Both VE + Bigram
**Config:** VE_ENABLED=0, BIGRAM_VOCAB_SIZE=0 — maximum size reduction.

| Metric | v13a (baseline) | v13g | Delta |
|--------|----------------|------|-------|
| post_ema val_bpb | 1.0895 | 1.0918 | +0.0023 |
| int6 roundtrip | 1.1055 | 1.1053 | -0.0002 |
| sliding BPB | 1.0955 | **1.0959** | **+0.0004** |
| size | 15,844,725 | **15,542,372** | **-302,353** |

**Finding: Stripping both is nearly free on sliding BPB.** Pre-quant is 0.0023 worse (bigram was slightly helping), but the quant gap is actually better (0.0041 vs 0.0060) because fewer parameters need quantizing. Net sliding cost is only +0.0004. Frees 302 KB of headroom — enough for a meaningful capacity addition (wider model, extra layer, etc.).

---

## Key Learnings & Bug Fixes

### 1. Tokenizer Mismatch (CRITICAL)
Initial v13a run used wrong tokenizer file (370,917 bytes from `records/` folder) instead of the correct one (370,952 bytes from `kevclark/parameter-golf` HF repo at `datasets/tokenizers/fineweb_8192_bpe.model`).

- Wrong tokenizer: bytes_per_token = 3.527 → all BPP numbers ~0.06 higher than reality
- Correct tokenizer: bytes_per_token = 3.727 → matches v12's reported numbers

**Lesson:** Always trace the actual tokenizer path from the run script, don't copy from unrelated directories. The v12 script uses `data/tokenizers/fineweb_8192_bpe.model` — verify the file hash matches before running.

### 2. TARGET_MB Units Bug
Code interprets TARGET_MB in MiB (×1,048,576 bytes), but competition cap is 16,000,000 decimal bytes.
- 15.2 MiB = 15,938,355 bytes → 61,645 bytes of headroom under 16MB cap
- TARGET_MB must be ≤15.25 to be safe

### 3. TTT inference_mode Bug (Fixed in train_v13.py)
TTT eval crashed with `RuntimeError: Inference tensors cannot be saved for backward` due to two sources of inference-mode RoPE tensor caching:

**Bug 1:** Phase 1 (scoring) in `eval_val_sliding_ttt` used `torch.inference_mode()`, caching RoPE cos/sin as inference tensors. Phase 2 (training) couldn't backprop through them.
- **Fix:** Changed Phase 1 from `torch.inference_mode()` to `torch.no_grad()` (line 1239).

**Bug 2:** The INT6 roundtrip eval (line 1459) runs under `inference_mode()` before TTT eval is called, also poisoning the RoPE cache.
- **Fix:** Added RoPE cache invalidation (`m._seq_len_cached = 0` on all `Rotary` modules) before Phase 2 training loop.

Both fixes are in `train_v13.py`. These fixed the crash but TTT still diverged catastrophically on BPB.

### 4. VE Strip is Essentially Free
Pre-plan estimates suggested VE strip could cost 0.005-0.015 BPB. Actual cost: **0.000 BPB** on val_loss, with post-EMA actually slightly *better* due to recouped training steps. The VE embedding at layers 9,10 was not load-bearing for this architecture.

### 5. Pruning is the Silent Killer (NEW)
v13b proved that even modest pruning (2.9%) causes disproportionate BPB damage (+0.012 sliding). The pruning algorithm zeros the smallest-magnitude quantized weights, but these are load-bearing for the model's tail-distribution predictions. **Any config that requires pruning to fit the size cap is automatically suspect.** Design for no-pruning first, then optimize within the no-pruning envelope.

### 6. Quant Gap Varies More Than Pre-Quant Quality (NEW)
Across all experiments, pre-quant BPB varies by only 0.0023 (1.0895–1.0918), but the quant gap varies by 0.014 (0.0041–0.0180). The quant gap is the dominant source of variance in final sliding BPB, not training quality. **Optimizing for smaller quant gap matters more than squeezing pre-quant BPB.**

---

## Critical Analysis — Why We're Stuck at 1.095

**Gap to SOTA: 1.0955 - 1.0810 = 0.0145 BPB**

None of the v13 experiments moved the needle. The features we tested from PR #1493 (parallel residuals, 3-layer recurrence, tuned hyperparams) are either neutral or harmful on our stack. The root causes:

1. **TTT is the elephant in the room.** PR #1493 gets ~0.01–0.02 BPB from TTT. We can't use it at all — catastrophic divergence on bankless. This alone may account for most of the 0.0145 gap.

2. **Our architecture is over-featured.** We carry SWA, XSA, PKO, SmearGate, bigram — but each individually seems to contribute near-zero BPB. The complexity may be preventing clean optimization rather than helping.

3. **Our quant pipeline is solid.** The quant gap of 0.006 (v13a) is tight. PR #1493's gap is reportedly ~0.005. We're not losing much to quantization — the gap is pre-quant.

4. **Training time is the binding constraint.** At 600s wallclock, we get ~5100 steps. More steps would directly improve pre-quant quality, but we can't control the clock.

### What might actually work (v14 candidates):

1. **Start from PR #1493's code directly**: Our v12-based code has accumulated technical debt (VE, bigram, SmearGate, custom TTT, custom quant). PR #1493 achieves 1.0810 with a cleaner codebase. Starting fresh from their code and adapting gives us a better foundation.
2. **Explore upstream innovations**: The upstream master has diverged significantly from our fork. New techniques may offer gains we haven't tried.
3. **TTT with Adam instead of SGD**: The SGD divergence might be optimizer-specific. Adam with very low LR could be stable.
4. **Focused hyperparameter search**: Grid search around proven values, not blind adoption.

---

## v13h/v13i — Capacity Expansion Experiments (FAILED)

### v13h — 12 Layers (strip both + extra depth)
**Config:** Same as v13g + NUM_LAYERS=12, TARGET_MB=15.25.

| Metric | v13a (baseline) | v13h (12 layers) | Delta |
|--------|----------------|-----------------|-------|
| post_ema val_bpb | 1.0895 | **1.0886** | **-0.0009 (best ever!)** |
| int6 roundtrip | 1.1055 | 1.1430 | +0.0375 |
| sliding BPB | 1.0955 | **1.1331** | **+0.0376 catastrophic** |
| size | 15,844,725 | 15,990,770 | +146,045 |
| pruned | 0 | 2,352,578 (6.8%) | |
| steps | 5108 | 4612 | -496 (slower steps) |

**Finding: Best pre-quant quality we've ever achieved (1.0886), but 6.8% pruning DESTROYS the quantized model.** The quant gap explodes from 0.006 to 0.044. This confirms the fundamental constraint: you cannot add capacity that pushes the artifact past the size cap, because pruning is exponentially destructive. The relationship between pruning % and quality loss is superlinear — 2.9% costs 0.012 BPB (v13b), 6.8% costs 0.045 BPB (v13h).

### v13i — Wider Model (dim=528, strip both) — CRASH
**Config:** Same as v13g + MODEL_DIM=528.

**Crashed immediately:** Flash Attention on Hopper (H100) requires `head_size` to be a multiple of 8. With dim=528 and 8 heads, head_dim=66 (not divisible by 8). The next valid wider dim is 576 (head_dim=72), but that adds ~4.45 MB int6 — way over budget.

**Lesson:** Model dim is effectively locked to multiples of 64 (8 heads × head_dim multiple of 8). The only valid step up from 512 is 576, which is too large. Capacity expansion must come from depth or MLP width, not attention width.

---

## Fundamental Insights from v13 Series

### 1. The Pruning Cliff
Pruning quality loss is **superlinear** in pruning percentage:

| Pruning % | BPB cost | Experiment |
|-----------|----------|------------|
| 0% | 0.000 | v13a, v13d, v13e, v13f, v13g |
| 2.9% | +0.012 | v13b |
| 6.8% | +0.045 | v13h |

There's a cliff around 2-3% where damage becomes severe. **Any config requiring >1% pruning is almost certainly not worth it.**

### 2. Pre-Quant vs Post-Quant Quality
Pre-quant BPB varies by only 0.0032 across all experiments (1.0886–1.0918), but post-quant sliding varies by 0.0376 (1.0955–1.1331). **The quant pipeline dominates final quality.** Optimizing pre-quant is necessary but not sufficient — you must also preserve quantization-friendliness.

### 3. Features That Don't Pay Their Rent
VE, bigram, parallel residuals, and 3-layer recurrence all failed to improve our best config. Our v12 architecture accumulated features without rigorous A/B testing. PR #1493 achieves 0.0145 BPB better with a **simpler** base architecture plus working TTT.

### 4. The 16MB Cap Is The Real Enemy
At 11 layers × dim=512, we use ~14.8 MB compressed. The gap between 14.8 and 16.0 MB is only 1.2 MB — enough for ~0.6M extra int6 weights. That's not even half a layer. **We're at the capacity frontier for this architecture within the size cap.**

### 5. The Path Forward
Our codebase has diverged too far from the competitive frontier. PR #1493 achieves 1.0810 with different architectural choices (working TTT, no VE/bigram/SmearGate). Rather than patching our v12-derived code, the correct move is to **start from PR #1493's proven codebase** and build improvements on top of their validated baseline.

---

## Current Status

**Best submittable result:** v13a at sliding 1.0955, size 15,844,725 bytes.
**Gap to SOTA:** 1.0955 - 1.0810 = 0.0145 BPB.
**v13 series exhausted.** 9 experiments run, none improved on v13a. The architecture is at its ceiling within the 16MB cap.

### Dead ends (do not revisit):
- **TTT (SGD) on our bankless**: +0.645 BPB catastrophic divergence
- **TTT on bank-based**: +0.089 (v4), +0.046 (v5)
- **Parallel residuals**: -0.0014 BPB, no benefit for our stack
- **3-layer recurrence**: neutral, not worth slower steps
- **PR #1493 hyperparams on our code**: +0.0030 BPP, worse quant gap
- **Strip bigram (keep VE)**: +0.0120 BPB, pruning death spiral
- **12 layers**: best pre-quant ever but pruning destroys it (+0.0376)
- **dim=528**: flash_attn crash (head_dim must be multiple of 8)
- **Any config requiring >1% pruning**: superlinear quality loss

### Next: v14 — Fresh start from PR #1493
1. Pull PR #1493's exact code and run it on our hardware to reproduce their 1.0810
2. Understand their TTT implementation (why does it work for them but not us?)
3. Build improvements on their proven baseline

---

## Files

| File | Purpose |
|------|---------|
| `train_v13.py` | Training script (from v12 bankless + TTT bug fixes) |
| `run_v13a_strip_ve.sh` | Strip VE config — **submittable baseline** |
| `run_v13b_strip_bigram.sh` | Strip bigram config (worse — pruning kills quant) |
| `run_v13c_ttt.sh` | TTT test config (DEAD) |
| `run_v13d_parallel.sh` | Parallel residuals (-0.0014 BPB) |
| `run_v13e_3layer_recur.sh` | 3-layer recurrence (neutral) |
| `run_v13f_hparam_tune.sh` | PR #1493 hparams (worse — quant gap) |
| `run_v13g_strip_both.sh` | Strip both VE+bigram (neutral, 302KB freed) |
| `run_v13h_12layer.sh` | 12 layers (catastrophic — pruning) |
| `run_v13i_wider528.sh` | Wider dim=528 (CRASH — flash_attn) |
| `v13a_strip_ve.log` | v13a run 1 (wrong tokenizer) |
| `v13a_strip_ve_r2.log` | v13a run 2 (correct tokenizer) — **reference run** |
| `v13b_strip_bigram.log` | v13b strip bigram run |
| `v13c_ttt_run3.log` | v13c TTT final run (catastrophic result) |
| `v13d_parallel.log` | v13d parallel residuals run |
| `v13e_3layer_recur.log` | v13e 3-layer recurrence run |
| `v13f_hparam_tune.log` | v13f hyperparameter tuning run |
| `v13g_strip_both.log` | v13g strip both VE+bigram run |
| `v13h_12layer.log` | v13h 12 layers (catastrophic pruning) |
| `v13i_wider528.log` | v13i dim=528 (crash log) |
