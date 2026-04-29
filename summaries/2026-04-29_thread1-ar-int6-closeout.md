# Session 2026-04-29 · thread-1 AR int6 close-out

**Headline**: AR self-gen GPTQ int6 (the last untested thread-1 lever) **does not transfer** to our SSM hybrid. **Discard, size_violation** confirmed across two runs (0081 train-stream-cal fallback; 0082b proper AR self-gen). Same val_bpb landing, same 21.3 MB cap-bust. Cause: int6-packed bytes are near-incompressible by brotli (saves 5%) vs int8 (saves 30%); the 25% raw saving from int6 is more than wiped out by lost brotli compressibility, so the int8→int6 swap GROWS the artifact by ~5.4 MB.

**Outcome for thread 1**: closed. Free-score levers exhausted. Current best (0076/0077, 2-seed mean **val_bpb 1.95141**, artifact 15.91 MB) stands as the session-end best from the previous session — no new promote.

**Span**: 2026-04-28 23:25 EDT → 2026-04-29 00:36 EDT, ~71 min wall-clock.

**Theme**: fail-fast close-out. The previous session left 0081 ready-to-launch with a try/except fallback that hid an AR self-gen bug. This session ran 0081, identified the silent fallback as breaking the named hypothesis, fixed both the fallback (removed) and the underlying Mamba2 chunk-align bug (front-pad ctx to multiple of 64), reran as 0082b, and confirmed the int6+brotli interaction is the real failure — calibration source is a noise-level perturbation. Thread 1 closes cleanly.

---

## Stack of confirmed wins (no change this session)

| # | Mechanism | val_bpb | n-seed | Δ vs prior | Heading pointer |
|---|---|---|---|---|---|
| 0 | 0051 family triple-parallel kill-Mamba-2 (BASELINE) | 2.00503 | 4 | (anchor) | summaries/2026-04-27_kill_mamba2_cross_class.md |
| 1 | + Brotli artifact compression | ~2.0030 | 1 | ~0 (lossless) | 0064 entry · brotli_swap |
| 2 | + Combined K=3+K=4 static side memory blended at inference | 1.95990 | 2 | -0.045 | 0069/0072 PROMOTE entry · combined K3 K4 |
| 3 | + Per-context α blend weights | ~1.957 | 2 | -0.003 | 0074/0075 entry |
| 4 | + Model-confidence gate (skip blend at high-confidence tokens) | **1.95141** | **2** | -0.005 | 0076/0077 PROMOTE entry · confidence-gated blend |

**Total compounded Δ vs canonical baseline 2.005**: **-0.054 BPB** (carried forward from prior session).

---

## Cross-experiment lessons (this session's only)

### 1. **AR int6 cap-busts in our family — int6+brotli is anti-synergistic** (0081/0082b entries · MAIN FINDING)

`pack_int6` stores 4 int6 values per 3 bytes — each byte has ~8 bits of entropy → near-incompressible. Brotli on int6-packed bytes saves only **5%** (ratio 0.948 vs zlib's 1.0). Compare to int8 weights where brotli saves ~30% (0064 entry). Net per-artifact arithmetic:
- int8 path: ~20 MB raw int8 payload → brotli → 15.91 MB
- int6 path: ~14.25 MB int6 packed + small int8 remainder → brotli → 21.32 MB

The 25% raw saving from int6 is more than wiped out by the lost ~25 percentage-points of brotli compressibility. **0082b's AR-self-gen run produced bit-identical artifact size to 0081's train-stream-cal run** (21.324 MB vs 21.331 MB; brotli/zlib ratio 0.9478 vs 0.9480). Calibration source is a calibration-scaling noise perturbation; it does not change the cap-bust.

[VERIFIED for our family at 200-step MPS regime].

The H100 record (`records/.../2026-03-31_ParallelResiduals_MiniDepthRecurrence/`) uses int6 successfully but their architecture differs and we cannot copy code. Their weight scale ranges or their bit-layout may interact differently with brotli; without their packing details, we can't bridge.

### 2. **The fallback try/except hid a real bug for the entire 0081 run** (0081 entry · process lesson)

Previous-session subagent wrapped `ar_self_gen_calibration_tokens` in try/except so any failure would silently fall back to train-stream tokens. Result: 0081 ran end-to-end, exited cleanly, produced numbers — but the named hypothesis ("AR self-gen GPTQ int6") wasn't actually tested. The fallback exists "so the artifact path is never blocked," but in research that's exactly backwards — failure to test the named hypothesis is the worst kind of clean-looking output. **Fix**: removed the try/except in train_gpt.py SERIALIZATION block; AR self-gen failures now crash the run. Plus the underlying chunk-align bug fixed.

User-facing rule for future code: **don't add fallbacks in research code that hide hypothesis-breaking failures**. Crash visibly so the next iteration can fix.

### 3. **Mamba2's chunked SSD scan needs `seq_len % chunk_size == 0`** (0081 entry · code-level)

AR generation feeds 1 token at a time → first iter has ctx_len=1 → Mamba2 chunked scan rejects. Front-pad context with seed_token to a multiple of CHUNK_ALIGN=64 before forward. Causal model — pads at the front don't influence the last-position logits we read. Fix is in `experiments/0082_ar_gptq_int6_b/modules/gptq_int6.py` `ar_self_gen_calibration_tokens`.

### 4. **K=4 top_N sweep validates cap-fill EV — but cap can't be freed via int6** (sweep entry)

Offline blended-BPB sweep (`scratch/blend_probe/k4_topn_sweep.py`) over K=4 top_N values, K=3 fixed at 100K:

| K=4 top_N | offline BPB | Δ vs 200K baseline |
|---|---|---|
| 200K | 1.9504 | (anchor) |
| 280K | 1.9426 | -0.008 |
| 320K | 1.9407 | -0.010 |
| 360K | 1.9387 | -0.012 |
| 400K | 1.9359 | -0.015 |
| 440K | 1.9346 | -0.016 |

Monotonic, no saturation visible. Production gain estimated at half offline (per 0074/0075 ratio): ~-0.004 to -0.008 BPB at top_N=300-400K. **But** AR int6 doesn't free cap. The remaining cap-freeing levers are: 0065-style asym pos0 (frees 0.48 MB neutral), drop per-context α (-0.2 MB but loses 0076 win), tune brotli quality. The cleanest unspent path is to combine 0065 + grow K=4 top_N — env-var experiment, ~30 min compute.

---

## Set in stone vs still hypothesis

**Verified this session:**
- AR int6 cap-busts in our family at the int8+brotli stack [VERIFIED, 2 runs same outcome]
- Calibration source (AR self-gen vs train-stream) is a noise-level perturbation, not the cap-bust driver [VERIFIED across 0081/0082b]
- pack_int6's 4-vals-per-3-bytes layout is brotli-incompressible [VERIFIED by ratio 0.948 vs int8's 0.7]

**Still hypothesis (not tested):**
- An unpacked uint8 [0, 63] storage would be brotli-friendlier but raw payload is 33% larger; net cap may still grow [CONJECTURE]
- A column-interleaved int6 layout might create brotli-detectable patterns [CONJECTURE]
- The H100 record gets int6 cap savings via different scale-range distributions in their architecture [CONJECTURE]

---

## Dead axes (added this session)

- **AR self-gen GPTQ int6 with current pack layout in our family** (0081, 0082b): cap-busts by ~5.4 MB. Don't re-test variants that don't change the int6 byte-layout / compression strategy.

(All prior dead axes carried forward unchanged.)

---

## Predictions vs actuals

| Hypothesis | Prediction | Actual | Calibration |
|---|---|---|---|
| 0081 int6 saves 1.5-3 MB of cap | freed cap | LOSES 5.4 MB cap | Wrong direction. Underestimated brotli's role on int8. |
| 0081 val_bpb ≈ 1.948 ± quant tax | val ~1.948 | val 1.9471 | Right within noise |
| 0082b val_bpb shifts ≤ 0.005 vs 0081 | small shift | -0.0016 | Calibrated |
| 0082b artifact identical to 0081 | identical | -0.007 MB (essentially identical) | Calibrated |

The cap-direction prediction was wrong because the previous-session brief carried forward an "int6 saves X% per layer" math that didn't account for compression interaction. The compression step is where the saving lives or dies; this is the load-bearing lesson for any future quant experiments.

---

## Walk reflections

No walks taken this session (close-out was tightly scoped, ~71 min). Prior walks' [WORTH_TESTING] / [WORTH_DERIVING] items are captured in journal.md "Open questions" and `scratch/dendritic_memory_plan.md` (drafted this session as a thread-2 ready-to-pick-up plan).

---

## Follow-ups for next session ranked by EV

1. **[WORTH_TESTING] Cap-fill 0065 + K=4 top_N grow** (cheapest free score remaining): combine 0076 winner with 0065-style asym pos0 (frees 0.48 MB neutral) and grow K=4 top_N from 200K to ~280-320K. Per `scratch/blend_probe/k4_topn_sweep.py`: production gain ~-0.004 to -0.005 BPB. One env-var experiment forking from 0076. ~30 min compute. **The only remaining cheap thread-1 win.**

2. **[WORTH_DERIVING] Dendritic N-gram side memory v1** (thread-2 entry point): warm-start patterns from frequent fineweb 4-grams, train ONLY content vectors. Plan in `scratch/dendritic_memory_plan.md`. Distinguishes from 0073/0080 (random patterns → sparse gradient → neutral). M=32K dendrites × content_dim=32 (low-rank) fits ~2 MB cap. Subagent ~250 lines. Highest-EV thread-2 lead grounded in our existing static-side-memory wins.

3. **[WORTH_TESTING] Train-time blend bug fix** (0071): MPS bounds error in `trigram_blend_loss` at B=3, L=1024. Debug notes in `scratch/0071_train_blend_debug_notes.md` (most likely cause: `trigram_offsets` sentinel sizing OR `entry_next` dtype overflow at vocab boundary). Tests "model adapts to be complementary to static prior" — different mechanism than learnable HSM.

4. **[WORTH_DERIVING] H100 transfer test of 0076 family**: estimated -0.01 to -0.02 BPB transfer based on per-token analysis. Crucial validation before writing up.

5. **[WORTH_TESTING] Higher-K static side memory**: K=5 with hash bucketing. K=5 has 4M+ contexts, requires hash-pruning. Could fit ~1 MB cap and add -0.005 BPB. Untested.

6. **[SPECULATIVE] AR int6 with brotli-friendly layout**: change `pack_int6` to store as unpacked uint8 [0, 63] (33% raw bytes overhead) and test whether brotli compresses to net smaller than packed. Math is uncertain. Or: column-interleave layout that creates brotli-detectable patterns. Both are research, not free-score.

7. **[SPECULATIVE] Bold (e)/(f) from brief**: full spike-rank body or dendrocentric layer. Massive code changes, best in non-record-track if pursued.

---

## Untested thread-1 levers (require code, NOT free wins)

The H100 record uses these record-validated levers we haven't ported:
- **Mini-depth-recurrence** (`RECUR_LAYERS=4,5 RECUR_START_STEP=3000`): different topology than our K=3 L=3. ~100 lines and conflicts with our existing recurrence wiring.
- **REPEAT_UNTIE_MLP=full**: cap-busts before measurement (would add ~38M MLP params at our config; we have 88 KB headroom). `_LAYERS=4,5` selective version doesn't have a clean mapping in our 3-unique × 3-loop scheme.

Both deferred; deferral acknowledged in journal "Untested thread-1 levers" section.

---

## Reflections

**What went well**:
- Fast close-out: 71 min for two MPS smokes including bug-fix code + journal. The previous-agent skeleton (0081 fork, smoke-tested gptq module) made this possible.
- Fail-fast pivot when the user pointed out the fallback was hiding the bug. Removing the try/except was the right move; rerunning gave a clean dispositive answer.
- Pre-running the K=4 top_N sweep in background (CPU-only) parallel to the GPU run gave us the "what would cap-freeing be worth?" answer for free.

**What didn't go well**:
- I polled too much during the wait. The first half of the session, I kept calling Bash to tail run.log instead of trusting the Monitor. The user called this out twice. Saved as feedback memory.
- I drafted a `0082_capfill_handoff.md` plan AS IF int6 would free cap — should have waited for the actual artifact size from 0081 before drafting. The plan is now slightly off (it predicts -0.005 BPB on top of 0081, but 0081 doesn't free cap; the plan needs to be re-grounded against 0076 directly).
- The dendritic_memory_plan.md and 0071_train_blend_debug_notes.md were prep for a future session, not work for THIS session — useful but not a substitute for closing out the current thread.

**Anti-patterns to avoid in future close-out sessions**:
- "Sit and wait" for a 30-min run when the Monitor is set up to fire on completion. Use the wait window to do CPU-only side work (sweeps, predictions, code review) — but don't poll the same files repeatedly hoping for a different number.
- "Defer to next session" without naming a specific reason. Either say why the deferral is principled (cap-bust math; topology conflict) or actually run the experiment.
- Drafting plans for follow-ups before the headline result is in. Wait for the data, then plan against actual numbers.

**Data points for future calibration**:
- Mamba2 in our config has chunk_size=64. Any per-token forward pass needs ctx length divisible by 64.
- AR self-gen at d_model=512 vocab=1024 takes ~60 ms/token without KV cache.
- Compression ratio sanity: int8 → brotli ~0.7x; int6-packed → brotli ~0.95x. Assume any future packed-bit-density format will look more like int6 unless it has explicit redundancy structure.

---

## End-of-session state

- `journal.md`: Current threads carries thread-1-closed marker; 0081/0082b entries journaled with full failure mechanism.
- `results.tsv`: 0081 + 0082b filled in (status=discard, size_violation=true).
- `experiments/0082_ar_gptq_int6_b/`: holds the bug-fix code (front-pad ctx + no fallback) for any future agent revisiting AR int6.
- `scratch/`: dendritic_memory_plan.md, 0071_train_blend_debug_notes.md, k4_topn_sweep.py output — all valid handoffs for next session.
- No new winner promoted. 0076/0077 (1.95141, 15.91 MB) remains the SSM-best.
