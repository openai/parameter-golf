# Journal · 2026-04-29 · bitnet-ternary-v2-packed

Rotated from journal.md on 2026-04-29 08:04 EDT.

## Entries (newest first)

## 2026-04-29 · session-start framing · thread-2 SNN/temporal-rank pivot

**Question**: Brief asks "Can 1-bit-per-param + ordering carry 8× more usable info, and can a trainable architecture realize that at 16 MB / 200-step MPS?" Two halves: (A) info-theoretic, (B) trainable arch.

**Setup**: re-read program.md + brief + journal + Boahen 2022 in full + prior derivations (UU#1/UU#2 + dendritic_memory_plan).

**Re-derivation in own context** (`scratch/2026-04-29_my_derivation.md`):
- Verified the brief's 8× ratio at the per-event level. At our 16 MB byte budget, rank-coded vs (top-K, prob) gives ~1.5-2× useful bits/byte — smaller than the brief headline.
- Phase-5 lead bet was rank-coded side memory.

**Outside-eyes critique** (called early per program.md guidance):
- Lead bet only addressed (A); the (B) trainable-architecture half is untested.
- Static side memory is what brief warned away from.
- 12-20× cap headroom from binary body sits **completely unspent**.
- Conv1d is already a learnable temporal-rank mechanism inside the body.

**Bold pivot post-outside-eyes**: ternary body (BitNet b1.58) on 0076 winner → 0083. Cheap parallel probe: long-kernel conv1d → 0084.

**Status (01:37)**: 0083 + 0084 launched. Both training healthy — 0083 (ternary) runs ~10 sec/step (vs 0084 ~9), step trajectories within 0.12 train_loss at step 55. ~22 min till both finish.

**Update (01:55)**: 0084 hung at step 55 — MPS contention deadlock (running two PyTorch jobs on single MPS GPU is a bad idea). Killed 0084. Saved feedback memory `feedback_no_concurrent_mps.md`.

## 2026-04-29 · exp 0083 · ternary body trains but cap-busts (BitLinear v1)

**Question**: Can BitNet b1.58 ternary body train on our 0076 winner at MPS 200 steps? Tests brief's (B) trainable architecture half.

**Setup**: BitLinear (absmean γ + STE) replacing CastedLinear in MLP w_*, Mamba2 in_proj/out_proj, Attn qkv/proj. Forward ternarizes on every step. Cleanly verified primitive in `scratch/bitlinear_tiny.py` (worked example, gradient flow, degenerate cases). Wired via TERNARY_BODY=1 env. All other settings inherited from 0076.

**Prediction** [CONJECTURE]: pre-quant val_bpb in [2.0, 2.2] range, healthy trajectory.

**Result**:
- Pre-quant val_bpb (model alone, no side-mem blend): **2.0998**. Vs 0064 (no-side-mem fp16/int8 baseline) 2.003 → ternary penalty +0.097 BPB at 200 steps.
- Post-quant val_bpb (with side-mem blend + int8 round-trip): **2.0303**. Vs 0076 (1.948) → +0.082 BPB net.
- **Artifact: 16.81 MB → CAP-BUST by 812 KB.** Brotli on ternary-body int8 weights compresses worse (ratio 0.89 vs 0076's 0.85) — ternary's discrete outputs from per-row int8 quant have higher entropy than continuous fp16's near-Gaussian distribution.
- Step time 8.97 s (~10% slower than 0076's 8.17 s due to BitLinear absmean recompute).

**Conclusion** [LIKELY]:
1. **(B) is realizable**: ternary body CAN train at our regime. The recipe works without divergence. Expected per BitNet paper: needs more steps to fully converge to fp16-equivalent quality (200 steps too short).
2. **v1 storage is the binding constraint**: int8-of-trained-fp32-BitLinear-weights has high entropy → poor brotli → cap-bust. v2 packed-ternary (2 bits/param direct) is mandatory, not a bonus.
3. **The (A) info-theoretic stack is NOT yet realized** in v1 — we still spend int8 (8 bits) per ternary param. 4× density saving requires v2.

**Disconfirming**: I expected pre-quant in [2.0, 2.2]; landed at 2.10 (mid-range). Recipe trains. But the CAP-BUST surprised me — I'd assumed ternary→int8 would brotli WELL (low entropy, tight distribution). Lesson: the absmean-quantized int8 weights span a wide range (per-row scale set by 0.999-quantile so high tails dominate scale). This loosens the int8 distribution, hurting brotli.

**Mechanism finding** [LIKELY, n=1]: BitLinear with default Muon settings (matrix_lr=0.045) costs ~+0.10 pre-quant val_bpb at 200 MPS steps vs fp16 baseline. The STE shrinks gradient by γ-factor (~0.07 at our regime) — Muon's NS orthogonalization should cancel magnitude differences in theory, but the actual training dynamics are sufficiently different to leave a real gap.

**Next**:
- 0086 (LAUNCHED 02:00): v2 packed-ternary serialization. 2 bits/param storage; should free 4-10 MB cap. Lossless round-trip verified offline. Same architecture as 0083, just different export/import.
- 0087 (READY): MATRIX_LR=0.135 (×3 boost). Tests recipe rescue for the ternary penalty.

## 2026-04-29 · exp 0086 · v2 packed-ternary serialization (HEADLINE INFRASTRUCTURE WIN)

**Setup**: Forked 0083. Added pack_ternary/unpack_ternary helpers (4 vals/byte, 2 bits per ternary). Modified `quantize_state_dict_int8` to detect BitLinear weight names (via `model.named_modules() + isinstance check`) and route them to packed-ternary instead of int8. Custom `unpack_ternary` rescales by 1/frac_nonzero so BitLinear's recompute-gamma-on-forward gives back trained effective weight (subtle: lossless idempotent round-trip verified in `_v2_quant_smoke.py`).

**Result**:
- Pre-quant val_bpb 2.0933 (model-only — matches 0083's 2.0998 within noise; forward path unchanged).
- **Post-quant val_bpb 2.0128** — BETTER than 0083's 2.0303 by -0.018 BPB! Why: the int8-of-fp32-BitLinear path was LOSSY (round-trip flips 1-4% of ternary cells per `bitlinear_int8_roundtrip.py`). Packed-2-bit is lossless to ternary→ternary.
- **Artifact 7.956 MB** vs 0083's 16.812 MB → **freed 8.86 MB cap**.
- size_violation: false (was true for 0083).
- step_avg 8621 ms (5% faster than 0083 — same forward, presumably some random difference).

**Conclusion** [VERIFIED, n=1]: Packed-2-bit ternary serialization is the right export format for any BitNet-style trained model. Frees 56% of artifact cap and IMPROVES post-quant val_bpb by avoiding the int8 lossy round-trip. This is INFRASTRUCTURE — applies to any future low-bit work in this codebase.

**Predicted vs actual**:
- Cap savings: predicted 4-10 MB, actual 8.86 MB ✓
- Post-quant tax: I worried it could be catastrophic from int8 zero-init flips; actual was IMPROVEMENT vs 0083 (-0.018) — even better than expected because v2 sidesteps the problematic path entirely.
- Brotli on packed-ternary: ratio 0.92 (vs 0076's 0.85 on int8). Mild incompressibility but doesn't matter — raw saving dominates.

## 2026-04-29 · exp 0088 · scale d_model 512→640 with packed ternary (NEUTRAL — value-axis saturated)

**Setup**: Fork from 0086. MODEL_DIM=640, NUM_HEADS=10, NUM_KV_HEADS=5. 1.56× body params at 2 bits/param. Tests if scaling capacity alone recovers the +0.10 ternary penalty (per outside-eyes' caution about port-mode reflex).

**Result**:
- Pre-quant val_bpb 2.0881 (vs 0086's 2.0933 = -0.005). Marginal at MPS noise level.
- Post-quant val_bpb 2.0080 (vs 0086's 2.0128 = -0.005). NEUTRAL within MPS noise.
- Artifact 10.77 MB (vs 0086's 7.96 MB = +2.81 MB at 1.25× cap).
- step_avg 11618 ms (35% slower than 0086 due to bigger model).

**Conclusion** [LIKELY, n=1]: Scaling d_model alone at 200 MPS steps does not recover the +0.10 BPB ternary penalty. Below promote threshold. Confirms the BitNet-paper intuition that ternary needs MORE STEPS to converge to fp16-equivalent quality, not just more parameters at the same step budget.

**Outside-eyes catch was right**: pursuing scale-up was port-mode reflex on the value-axis. The 8.86 MB freed cap from 0086 should go to **temporal/ordering** experiments, not more value-axis exploration.

**Pivot**: 0089 (soft-DP fuzzy K-gram match, brief option d) — eval-time only, ~80 lines. Coverage probe (`scratch/softdp_coverage_probe.py`) showed exact match coverage 58.3% → fuzzy 1-edit coverage 98.2% (+40 percentage points). Apply to 0076 winner (1.948 baseline). If wins, this is the first temporal-axis result of the session.

## 2026-04-29 · exp 0089/0091 · soft-DP fuzzy K-gram is dead at our regime

**Setup**: Forked 0076. Added `_apply_fuzzy_kgram_fallback` to trigram_side_memory.py: for K=4 missed contexts (no exact match), generate 3*(V-1)=3069 1-edit-distance neighbors, lookup each, pick best by max log2_backoff (most-frequent neighbor). Blend matched neighbor's predictions with bigram fallback at FUZZY_DOWNWEIGHT.

**Results**:
- 0089 (DOWNWEIGHT=0.5): val 1.9489 vs 0076's 1.9483 = +0.0006 NEUTRAL.
- 0091 (DOWNWEIGHT=0.8): val 1.9502 vs 0076's 1.9483 = +0.002 slightly worse.

**Conclusion** [VERIFIED, n=2]: Coverage probe predicted +40pp coverage gain but BPB didn't follow at any downweight tested. The fuzzy neighbors give predictions noisier than bigram fallback at our regime — higher confidence in fuzzy AMPLIFIES noise. Boahen's "fringing field" robustness analog doesn't translate to BPB at our 200-step / 16MB / sp1024 regime. Soft-DP axis closed.

## 2026-04-29 · exp 0093 · LR rescue + v2 packed compose cleanly (best ternary at full cap)

**Setup**: Forked 0086 (BitLinear ternary body + packed serialization) + MATRIX_LR=0.135 (×3 boost from 0087's finding). Tests if the two confirmed levers stack independently.

**Result**: post-quant val_bpb **1.993 at 8.21 MB**.
- vs 0086 (LR=0.045): 2.013 → -0.020 BPB from LR×3
- vs 0087 (LR×3, no v2 packed, cap-bust 19.83 MB): 1.988 → cap halved at +0.005 BPB
- vs 0076 (fp/int8 baseline): 1.948 → still +0.045 BPB worse, but at HALF the cap

**Conclusion** [VERIFIED, n=1]: LR×3 and v2 packed-ternary serialization are independent levers that compose cleanly. The ternary body story is now coherent for the writeup: trains, costs +0.045 BPB at 200 steps, fits in 8 MB cap, has ~8 MB headroom for downstream mechanisms. BitNet paper says ternary matches fp16 at 20k+ steps; we're 100× short of that and saw a clean LR-sensitivity slope.

## 2026-04-29 · exp 0092/0094 · dendritic v1 + LR rescue split-finds the failure mode

**Setup**: 0092 = warm-start dendritic memory v1 (M=32K patterns from 50M training tokens, learnable d_content=32 + zero-init proj→d_model). 0094 = same + MATRIX_LR=0.135 (test outside-eyes' unified LR hypothesis on dendritic content vectors).

**Results**:
- 0092 (default LR): val 2.020 vs 0086's 2.013 = +0.007 NEUTRAL/slightly worse. Smoke showed 20% fire rate per token (dense gradient flow), but content vectors didn't escape near-zero region in 200 steps.
- 0094 (LR×3): val 2.000 vs 0093's 1.993 (LR×3 baseline) = +0.007 NEUTRAL.

**Conclusion** [VERIFIED, n=2]: The unified "LR was wrong for new components" hypothesis SPLITS:
- BitLinear body weights ARE LR-bound (0083→0087 recovers half the +0.10 ternary penalty at LR×3).
- Dendritic content vectors are NOT LR-bound. They're training-duration-bound. LR×3 doesn't unlock them.

This sharpens the 0073/0080/0092 interpretation: those neutrals are genuine training-duration limits, not LR misconfig. At H100 20k steps, dendritic v1 might unlock — but at MPS 200 steps, learnable side-content needs a different gradient scaffold. Brief option (d-warm) tested.

## 2026-04-29 · exp 0095 · rank-coded blend FALSIFIES brief's strong density form

**Question**: Brief's central claim is "rank-coded bits carry more usable info per byte than (top-K, prob)." Does the per-(ctx, rank) log2p in our existing K=4 side memory matter, or could a global rank template substitute?

**Setup**: Forked 0076. Subagent built Option 3 (in scratch/0095): replace per-(ctx, rank) log2p at K=4 lookup time with `template[r] = mean over contexts of log2p_full[start_ctx + r]`, computed once per call. Storage unchanged; only the decode path changes when RANK_CODED_BLEND=1.

**Result**: post-quant val_bpb **1.9673 vs 0076's 1.9483 = +0.019 REGRESSION**.

**Conclusion** [VERIFIED, n=1]: The strong form of the brief's rank-density claim is FALSE at our regime. Per-context log2p calibration carries irreducible information that the global rank template loses. The current side memory's (top-2 token, top-2 logit) representation is more informative per byte than its rank-only equivalent — at least when the model is trying to consume it via fixed blend weights.

This is a genuine answer to the brief's central density question: at our 200-step / 16 MB / sp1024 regime, ordering alone doesn't substitute for value-with-context. Caveat: this tested rank-only on the SAME storage format (no density payoff), not the brief's full permutation-coded R=8 with int16 token indices (which would test the storage-density claim, but at the cost of 16 vs 8 bytes per entry).



**Side discoveries in scratch/**:
- BitLinear int8 round-trip (`bitlinear_int8_roundtrip.py`): ZERO-INIT weights flip 70% of ternary cells after int8 export. Active layers flip 1-4%. Post-quant val_bpb may significantly degrade. v2 must use packed-ternary serialization (`2026-04-29_v2_packed_ternary.md`).
- BitLinear primitive verified in `scratch/bitlinear_tiny.py` (worked example, degenerate cases, gradient flow all PASS).

