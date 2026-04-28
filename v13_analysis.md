# v13 Analysis & Next-Run Plan

**Date:** 2026-04-28
**Branch:** shikhar
**Inputs:** `v13a_strip_ve_r2.log`, `v13c_ttt_run3.log`, `train_v13.py`, updated `v13_plan.md`

---

## Headline

**v13a is real and submittable.** Sliding 1.0955, size 15,844,725 bytes — under the 16,000,000 decimal-byte cap with ~155 KB of headroom. VE strip is genuinely free (post-EMA actually 0.0007 better, dominated by 272 extra training steps).

The two runs (v12 and v13a) both land at sliding **1.09553** — within 8×10⁻⁶ BPB. That isn't noise dampening; that's two runs converging on the **same quantization ceiling** for this stack. To get below 1.0955 we need architectural change, not parameter trimming.

---

## The TARGET_MB units bug — biggest find, undersold in v13_plan.md

The plan treats it as a side note. It retroactively explains everything:

- Code uses `target_mb * 1024 * 1024` (MiB), competition cap is **16,000,000 decimal bytes**
- v12 ran TARGET_MB=15.9 → 15.9 MiB = **16,672,358 decimal bytes** — well over the 16 MB cap
- That's why prune said "already fits, no pruning needed" but `.ptz` was 16.54 MB
- **Every prior run that "fit" via the prune routine was checking against an inflated target** (v9, v10, v11 included)

### Required code fix

The current "fix" of TARGET_MB=15.2 (= 15.94 MB decimal) is sound but cargo-culted. Real fix in `train_v13.py` around line 2018:

```python
# OLD — interprets MiB
target_mb = float(os.environ.get("TARGET_MB", "15.9"))
target_bytes = int(target_mb * 1024 * 1024)

# NEW — interprets decimal MB to match competition cap
target_mb = float(os.environ.get("TARGET_MB", "15.95"))
target_bytes = int(target_mb * 1_000_000)
```

Without this fix, every future run is one config typo away from the same trap.

---

## v13c TTT — the doc's "dead" verdict is overconfident

What we actually showed: **one specific TTT configuration diverges**. Not "TTT on bankless is impossible."

### The smoking gun

`train_v13.py` line 1296 phase 2 forward:
```python
loss = base_model(x, y)
```
This runs with `looping_active=True` (set globally at line 1463 for eval). **Layers 4-5 weights receive double the gradient per backward pass** because they're used twice in the forward. With LR=0.005 and momentum=0.9, runaway updates on exactly the tightest-quantized layers are predictable.

### Other smells in the recipe

- LR=0.005 is huge for fine-tuning **dequantized INT6** weights (their distribution is much tighter than FP)
- 3 epochs × 1238 chunks = ~3700 SGD steps. Cosine schedule reaches LR ≤ 0.0049 only after ~chunk 100 — the "decay" is mostly cosmetic on the front half where damage compounds
- All matrix params with numel>65536 are trainable — including recurred layers
- PR #1493's HPs are tuned for THEIR stack (no SWA, no bigram, no VE, different recurrence depth) — there is no reason they transfer to ours
- 571s eval time is real, but 1-epoch + LR drop should fit under 600s cap

### Things to try before declaring TTT dead (each is a 10-min run)

1. Disable looping during phase 2 forward (toggle `looping_active = False` around the SGD step, restore for phase 1 next chunk)
2. Drop LR 10× → TTT_LR=0.0005
3. Drop epochs to 1 → TTT_EPOCHS=1
4. Exclude recurred layers from TTT params (filter `"blocks.4." not in n and "blocks.5." not in n` at train_v13.py line 1215)
5. TTT only on late layers (head + last 2 blocks)

The current evidence supports **"naive port of PR #1493's TTT recipe diverges on our stack"** — not "TTT is impossible." A 30-minute HP search settles it.

---

## v13a's 1.0955 ceiling — is it real?

Both v12 and v13a land at sliding 1.09553 to within microbits, **with the same SEED=1337**. Two readings:

1. **The ceiling is real** — this stack/data/quantizer combo is converged at 1.0955. To break below, we need architectural change.
2. **The seed isn't actually different enough** — same RNG init, removing VE perturbs the trajectory only mildly. Different seed could land elsewhere.

**Resolution:** run v13a with SEED=42. ~10 minutes. Tells us whether to chase HP/architectural changes (if seeds vary) or whether we've genuinely hit a stack ceiling (if they don't).

---

## What `v13_plan.md` misses or undersells

1. **v13b (bigram strip) was never tested.** Doc says "skipped in favor of TTT." But now TTT is broken and we have size headroom — the question of bigram's BPB cost still matters:
   - If bigram is free (like VE), we have another **~400 KB of headroom** for size-cost additions (larger model_dim, 3-layer recurrence safety margin)
   - If it costs 0.005-0.020, we keep it
   - One run answers it.

2. **v13d (parallel residuals) is the obvious next move and isn't even queued.** Script exists (`run_v13d_parallel.sh`), PARALLEL_START_LAYER=7, zero param/size cost. PR #1493 uses it. Highest-EV remaining move — training-time architectural change won't reopen the quant gap.

3. **No mention of seed variance.** Single-seed v13a result. Gating a submission on a single seed is risky given v7's documented ±0.005 BPB seed variance.

4. **PR #1060's bigram + SP1024 gave 1.1122; our submission uses bigram.** Whether bigram is net-positive on SP8192 + bankless is untested.

5. **The 1.0955 ceiling implies 0.0145 BPB remaining gap to PR #1493 (1.0810)** — and stripping things won't close it. Adding capacity (parallel residuals, 3-layer recurrence, retuned WD/MLR) is the only path.

---

## Run plan (in priority order, all ~10 min each)

### Tier 1 — lock the floor + cheap upside

**1. Submit v13a now.** Confirm leaderboard scores it at 1.0955. Lock the floor before gambling.

**2. Fix TARGET_MB units bug** (code change, no run needed):
```python
target_bytes = int(target_mb * 1_000_000)  # decimal MB, matches comp cap
```
Set TARGET_MB default to 15.95 going forward.

**3. v13d — parallel residuals** (`run_v13d_parallel.sh`):
```bash
PARALLEL_START_LAYER=7 \
# everything else = v13a config
```
Free upside, no size risk. This is the next obvious lever.

### Tier 2 — clear the unknowns

**4. v13a with SEED=42.** Same config, different seed. Tells us if 1.0955 is a stack ceiling or seed artifact.

**5. v13b — strip bigram** (already have script, just need rerun with correct tokenizer):
```bash
BIGRAM_VOCAB_SIZE=0 \
# everything else = v13a config
```
Tells us bigram's BPB cost — gates whether v13b can stack with v13d.

### Tier 3 — TTT salvage attempts (only if Tier 1+2 don't close the gap enough)

Run as a sweep, abort early if first one diverges:

**6. v13c-retry-A:** TTT_LR=0.0005, TTT_EPOCHS=1, looping_active=False during phase 2

**7. v13c-retry-B:** TTT_LR=0.0005, TTT_EPOCHS=1, exclude recurred layers from ttt_params

**8. v13c-retry-C:** TTT_LR=0.0005, TTT_EPOCHS=1, TTT only on last 2 blocks + head

If all three diverge → TTT is genuinely dead for our stack. Mark it and move on.

### Tier 4 — capacity recovery (only if v13d + Tier 2 don't beat 1.0955)

**9. 3-layer recurrence:** RECUR_LAYERS="3,4,5". Only after v13d confirms parallel residuals don't reopen the quant gap. Watch the gap field — if it grows from 0.005 → 0.015+, abort.

**10. Hyperparameter sweep:** WD=0.095, MLR=0.022, EMA=0.9965 from PR #1493. Only meaningful on top of architecture-matched stack (v13d at minimum).

---

## Run scripts that already exist

| Script | Status | Action |
|--------|--------|--------|
| `run_v13a_strip_ve.sh` | Run, submittable result | Submit `models/v13a_strip_ve.int6.ptz` from HF |
| `run_v13b_strip_bigram.sh` | Exists, never run with correct tokenizer | Rerun with verified tokenizer hash |
| `run_v13c_ttt.sh` | Run, catastrophic failure | Modify for retries (Tier 3) |
| `run_v13d_parallel.sh` | Exists, never run | **Run next** |

---

## Pre-flight checklist

- [ ] Confirm `data/tokenizers/fineweb_8192_bpe.model` hash matches `kevclark/parameter-golf` HF copy (not the `records/` variant)
- [ ] Apply TARGET_MB units fix in `train_v13.py` before next run
- [ ] Verify v13a submission was uploaded AND scored on leaderboard at 1.0955
- [ ] For Tier 3 TTT retries: verify `looping_active` toggle is correctly applied around phase 2 forward (line 1296), not just phase 1

---

## Bottom line

v13a is a real win — 1.1117 → 1.0955 is 0.016 BPB closer to SOTA. Submit it. Then run v13d for free upside. TTT is *probably* dead but the doc shouldn't have closed that door without a 30-minute HP search. 3-layer recurrence and HP retuning are valuable but only after we've confirmed the parallel-residual baseline. Don't skip steps.
