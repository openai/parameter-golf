# Parameter Golf — Experimental Findings
**Updated: 2026-03-21**

Everything we've tested, what we learned, and what remains untested.

---

## Tested by us — with real data

### 1. Int8 QAT (Quantization-Aware Training)
- **PR:** #145 (submitted)
- **Hypothesis:** Training through the int8 quantization grid via STE reduces the post-export quality gap.
- **Implementation:** `fake_quantize_int8_per_row` using `torch.quantile(w.abs(), INT8_CLIP_Q, dim=1)` matching export pipeline exactly. Activated at 30% of training.
- **Result:** NEGATIVE. `torch.quantile` adds ~20% per-step overhead (64ms → 77ms), costing ~2,000 training steps. Post-quant val_bpb: 1.2052 vs control's 1.1929. The lost training tokens hurt more than the quant gap recovery.
- **Lesson:** Int8 QAT with exact percentile matching is too expensive under a 10-minute wallclock cap. QAT only pays off with int6 (larger gap) or a faster approximate quantile (amax).
- **Hardware note:** Confirmed on both slow (65ms/step) and fast (44ms/step) RunPod H100 nodes.
- **Update:** See Finding #2b — Late QAT is also dead code under torch.compile, making the entire QAT approach moot regardless of overhead.

### 2. QAT Graph Priming
- **Hypothesis:** Pre-compiling both QAT and non-QAT torch.compile graphs during warmup avoids a 30-90s mid-training recompile.
- **Result:** NEGATIVE. Graph priming caused torch.compile to use a slower compilation path for the NON-QAT forward pass. step_avg was 65ms from step 1 (vs 44ms baseline), even before QAT activated.
- **Lesson:** Don't pre-prime conditional code paths under `torch.compile(dynamic=False, fullgraph=True)`. Accept the one-time recompile cost instead.

### 2b. Late QAT is dead code under torch.compile
- **Discovery:** `torch.compile(dynamic=False, fullgraph=True)` constant-folds `CastedLinear._qat` at first trace (when `_qat=False`). The STE branch in `forward()` is dead-code-eliminated. Setting `_qat=True` later triggers a recompile but does not restore the STE path.
- **Effect:** Late QAT activation does nothing except cause a recompile spike (~100 lost steps). No quantization-aware gradients are produced.
- **Evidence:** Explains why (a) QAT activation always caused a step_avg spike, (b) QAT never visibly improved roundtrip scores, (c) PR #332 found "Late QAT is counterproductive at 12L."
- **Credit:** @152334H identified the constant-folding mechanism; documented in PR #315.
- **Fix:** Set `QAT=0` to avoid the recompile cost entirely. A proper fix would require passing the QAT flag as a function argument rather than a class attribute, but at this point the simplest path is to disable it.

### 3. Sliding Window Evaluation (stride=64)
- **Result:** POSITIVE (reproduced). val_bpb 1.1929 vs baseline 1.2244. Improvement: -0.032 BPB.
- **Note:** This is a reproduction of the SlidingWindowEval entry (#50), not our original work. Confirmed on fast hardware (44ms/step, 13,651 steps).

### 4. Doc-Isolated Sliding Window Evaluation
- **Hypothesis:** Evaluating per-document (no cross-doc context bleed) improves BPB, based on LoRA TTT ablation showing +0.011 BPB.
- **Result:** INCONCLUSIVE. Tested only on slow hardware (70ms/step) stacked with other changes. val_bpb 1.1952 (10L) and 1.2045 (9L). Cannot isolate doc-isolation's effect from other variables.
- **Observation:** Produced only 37,402 windows vs 121,134 for flat-stream eval. Faster eval time (43s vs 73s).
- **Needs:** Clean A/B test on fast hardware — flat-stream vs doc-isolated, same model.

### 5. FP16 Tied Embedding Export
- **Result:** POSITIVE (reproduced from WarmdownQuantization entry). Avoids int8 compounding through input+output paths. Costs ~512KB artifact space, offset by MLP_HIDDEN=992.
- **Note:** 10L + FP16 embed exceeds 16MB cap. Only works with 9L + MLP_HIDDEN=992 or with int6 quantization.

### 6. Aggressive Warmdown (WD=20000)
- **Result:** Used in kitchen sink runs but never isolated. WarmdownQuantization entry reports quant gap drops from 0.014 to 0.005 BPB.
- **Needs:** Clean A/B test — WD=1200 vs WD=20000, same model, same eval.

### 7. 10 Layers (vs 9)
- **Result:** 10L model trains but artifact exceeds 16MB at int8 (17.6MB). Needs int6 or weight sharing to fit.
- **Observation:** 10L at 76ms/step (10L+seq2048) vs 44ms (9L+seq1024) — the extra layer + longer sequence is significantly slower per step.

### 8. SWA (Stochastic Weight Averaging)
- **Hypothesis:** Weight averaging during warmdown finds flatter minima that quantize better.
- **Result (v1):** CATASTROPHIC BUG. Accumulated in bf16 for 3,596 steps → precision overflow → val_bpb 2.62.
- **Bug fix:** Accumulate in float32, sample every 50 steps, cast back to model dtype.
- **Result (v2, from ablation table):** +0.0004 BPB vs control — effectively no effect at WD=1200.
- **Lesson:** SWA gains reported by other entries likely require very long warmdown (WD=20000+) to accumulate enough diverse snapshots. Superseded by EMA which is smoother and always-on.

### 9. Seq2048 Training
- **Result:** Used in kitchen sink but never isolated. Slower per step (76ms vs 44ms for seq1024) but each step processes 2x longer sequences.
- **Needs:** Clean A/B test — seq1024 vs seq2048, same total training time.

### 10. Tuned LRs (MATRIX_LR=0.06, etc.)
- **Result:** Used in kitchen sink but never isolated. From WarmdownQuantization entry, optimized for WD=20000.
- **Note:** Different optimal LRs for different warmdown schedules. 10L_MixedPrecision entry uses 0.02 (lower), WarmdownQuantization uses 0.06 (higher).

### 11. Curriculum Learning (sort shards by document length)
- **Hypothesis:** Feeding easier data (shorter documents) first accelerates early convergence, producing a better model in the same wallclock budget.
- **Implementation:** Sort 80 training shards by average document length (estimated from first 100K tokens per shard via BOS token counting). Shorter docs first.
- **Result:** MARGINAL NEGATIVE. val_bpb 1.1942 vs control 1.1929 (+0.0013 BPB worse).
- **Observation:** Early training loss WAS lower with curriculum (step 1000: 2.194 vs 2.370), confirming easier data helps early convergence. But final model was marginally worse — suggesting the model overfitted to easy patterns rather than learning generalizable features.
- **Lesson:** Simple shard-order curriculum doesn't help at this scale. More sophisticated curriculum (within-batch difficulty mixing, anti-curriculum) might work but adds complexity.

### 12. Int6 + 3x MLP (our implementation)
- **Result:** STRONG POSITIVE. val_bpb **1.1708** vs control 1.1929 (-0.0221 BPP).
- **Implementation:** `QUANT_BITS=6` in quantize_float_tensor (max_val=31 instead of 127), zstd-22 compression, MLP_HIDDEN=1536, FP16_EMBED_EXPORT=1.
- **Details:** 21.8M params, 15.2MB artifact (824KB headroom), 48ms/step, 12,507 steps.
- **Lesson:** Int6 + wider MLP is the single biggest lever. The freed artifact budget enables 27% more parameters in 4.5% less space.

### 13. Depth Recurrence + Huginn Eval-Time Scaling (Moonshot)
- **Hypothesis:** 3 shared blocks × 3 loops = 9 effective layers with 1/3 unique params. At eval, increase to 6 loops for free depth.
- **Result v1 (with U-Net skips):** CATASTROPHIC. Pre-quant val_bpb 1.2934, post-quant 6-loop eval: val_bpb **4.34** (near-random noise).
- **Result v2 (flat loops, no skips):** WORSE. val_bpb **5.58** — flat loops amplify errors rather than refine.
- **Root cause:** Blocks learn a position-specific function for their depth in the 3-loop stack, not a general iterative refinement operator. Extra loops compound distribution mismatch, not refinement.
- **Scale argument:** Huginn validated at 3.5B params (~100M+ per unique layer). Our unique layers are ~2.5M params — insufficient capacity to simultaneously be a good LM and a general refiner.
- **Artifact:** 5.6MB — proves 10MB+ headroom exists with weight sharing + int6.
- **Lesson:** Do not retry Huginn-style eval scaling at this scale without fundamentally different approach. The technique does not transfer below ~100M params per unique layer.

### 14. Multi-Token Prediction
- **Hypothesis:** Predicting t+2 alongside t+1 gives richer gradients per step.
- **Implementation:** Auxiliary Linear(dim, dim) head predicts t+2, weighted 0.5× in loss. Excluded from artifact.
- **Result:** MARGINAL NEGATIVE. val_bpb 1.1947 vs control 1.1929 (+0.0018). 3% overhead (45.3ms vs 43.9ms), 410 fewer steps.
- **Lesson:** Auxiliary multi-token prediction doesn't improve the primary task at this scale. The gradient signal from predicting t+2 doesn't transfer to t+1 quality.

### 15. Int5 Quantization (QUANT_BITS=5, MLP_HIDDEN=1920)
- **Hypothesis:** 5-bit quantization frees even more artifact budget than int6, enabling MLP_HIDDEN=1920.
- **Result:** CATASTROPHIC. Pre-quant val_bpb 1.1885 (better than int6!), post-quant val_bpb **1.5458**. Quantization gap: +0.357 BPB — 15× larger than int6's +0.024.
- **Root cause:** Int5 has only 31 representable levels (vs int6's 63). Per-row quantization error at 31 levels is large enough to destroy language modeling entirely.
- **Note:** Pre-quant result was actually better (more params fit in budget), but quantization erases all gains.
- **Lesson:** Int5 is not viable with post-training quantization. Would require int5-aware QAT (simulating 5-bit during training), which is not worth building given int8 QAT was already a negative finding.

### 16. Optimizer Coverage Bug (SmearGate + BigramHash frozen)
- **Discovery:** SmearGate and BigramHashEmbedding were not in any optimizer parameter group in all runs prior to 2026-03-21.
- **Effect:** Both modules trained frozen from initialization:
  - `SmearGate.gate` (initialized to zeros) → sigmoid(0) = 0.5 fixed for all 512 channels
  - `BigramHash.proj.weight` (initialized to zeros) → permanent zero projection, hash embeddings contributed nothing
- **Implication:** Every result using `SMEAR_GATE=1 BIGRAM_HASH=1` (the default since their introduction) had these features silently disabled. The ablation result "SmearGate + BigramHash hurt with int6 (+0.003 BPB)" from the Int6_3xMLP README was measured with BOTH FEATURES BROKEN.
- **Fix:** Added bigram_hash.proj.weight to Muon matrix_params, smear_gate.gate to AdamW scalar_params, bigram_hash.embed.weight to tok_emb AdamW group.
- **Same bug independently found by other participants at roughly the same time.**
- **Lesson:** Always verify that every nn.Module submodule appears in at least one optimizer parameter group. A module that initializes to zero or near-zero and is never updated will appear to "work" (no crash) while contributing nothing.

### 17. 11 Layers — step count trap
- **Hypothesis:** 11 layers (vs 9) gives more model capacity; int6 provides budget headroom.
- **Result:** NEGATIVE. val_bpb 1.1907 vs 9L's 1.1708. Regression of +0.020 BPB.
- **Root cause:** 11 layers runs at ~83ms/step vs 9L's 48ms/step. In 600s: ~7,200 steps vs ~12,500 steps. The ~40% step count loss outweighs the depth gain.
- **Additional factor:** Hyperparameters were suboptimal (muon_momentum=0.95, rope_base=10000, grad_clip disabled) — these were fixed in the XSA+EMA+TTT run.
- **Lesson:** Adding layers only helps if step time increase is less than capacity gain. At this scale, each extra layer costs ~7ms/step. Going from 9→11 layers costs ~35ms/step total, too much for 600s budget. Would need to pair depth increase with NTK-RoPE seq_len=1024 to recover steps.

### 18. Flash Attention 2 (FA2)
- **Result:** POSITIVE for step time at long sequences. No measurable val_bpb difference vs SDPA.
- **Details:** FA2 saves ~5-8ms/step at seq_len=2048 vs `F.scaled_dot_product_attention`. At seq_len=1024 the benefit is smaller.
- **Install note (RunPod):** Requires `pip install flash-attn --no-cache-dir --no-build-isolation`. FA3 (`flash_attn_interface`) is NOT available — cross-device link error on RunPod filesystem with torch2.9+cu128.
- **Lesson:** Worth including for the step time savings. No quality effect — it's a mathematically equivalent kernel.

### 19. XSA — Exclusive Self Attention (arXiv:2603.09078)
- **Result:** POSITIVE (part of combined 1.1401 run, not isolated). Zero parameters, minimal compute overhead with GQA-aware implementation.
- **Mechanism:** Subtract self-value projection from attention output: `y = y - (y·v̂)v̂`. Forces each head to draw from other tokens rather than looping back to its own value.
- **Applied to last 4 layers only** — early layers benefit from self-loops for feature extraction.

### 20. EMA (Exponential Moving Average, decay=0.997)
- **Result:** POSITIVE (part of combined 1.1401 run, not isolated). Replaces SWA for smoother weight averaging.
- **Mechanism:** Shadow copy of weights updated every step: `ema = 0.997*ema + 0.003*weights`. Loaded before quantization, replacing last-step weights.
- **vs SWA:** EMA is always-on, smoother, no snapshot schedule needed. Float32 accumulation avoids the bf16 precision bug that killed SWA v1.

### 21. TTT — Test-Time Training
- **Result:** POSITIVE (part of combined 1.1401 run, not isolated).
- **Mechanism:** 3-epoch SGD (lr=0.002, momentum=0.9) on validation data after EMA applied, before final eval. First 2 blocks frozen for stability.
- **Runtime:** ~40–60s.

### 22. NTK-aware RoPE (train_seq_len=1024, eval_seq_len=2048)
- **Result:** INCONCLUSIVE. Used in the 1.1401 run but superseded by training at seq_len=2048 directly (no NTK scaling needed). The NTK mechanism works correctly but training at native seq_len=2048 is simpler and avoids extrapolation risk.
- **Mechanism:** Scale RoPE base at eval: `ntk_base = rope_base * (eval_seq_len / train_seq_len) ** (head_dim / (head_dim - 2))`.
- **Note:** Top leaderboard entries train at seq_len=2048 with rope_base=10000 rather than using NTK scaling.

---

## Tested by others — from leaderboard and PRs

### Int6 Quantization + zstd (community)
- **Source:** PRs #114, #128, #162, #164, #173, #179, #180
- **Effect:** Frees ~3MB of artifact budget. Enables 3x MLP expansion (hidden=1536) which alone gives ~0.02 BPB.
- **Status:** The dominant meta. Every top-5 pending PR uses int6.
- **Our status:** ✅ Implemented — PR #212 (val_bpb=1.1708).

### SmearGate (per-channel)
- **Source:** modded-nanogpt community, PRs #162, #164
- **Effect:** Learned sigmoid gate blending each token with previous token's embedding. Original: scalar gate. Our version: 512 per-channel gates (more expressive).
- **Our status:** ✅ Implemented and optimizer-fixed in 11L_XSA_EMA_TTT. Prior runs had this frozen (see Finding #16).

### BigramHash
- **Source:** modded-nanogpt community, PRs #162, #164
- **Effect:** Hash-based bigram embedding table (2048 buckets → 128d → 512d projection). ~524K params.
- **Our status:** ✅ Implemented and optimizer-fixed in 11L_XSA_EMA_TTT. Prior runs had proj zeroed out (see Finding #16).
- **Note:** Buckets=2048 (our default); some entries use 4096. May be worth trying.

### NorMuon Optimizer
- **Source:** arxiv 2510.05491, PR #173
- **Effect:** Muon variant with per-neuron normalization. Drop-in replacement. "Modest but repeatable" gains.
- **Our status:** Not implemented. Low priority given other higher-impact changes in flight.

### OrthoInit
- **Source:** modded-nanogpt community
- **Effect:** Orthogonal init on all large matrices. Faster early convergence, better conditioning.
- **Our status:** ✅ Implemented (`ORTHO_INIT=1`, default on).

### Depth Recurrence / Layer Sharing
- **Source:** PR #167, Subformer (arxiv 2101.00234), Huginn (arxiv 2502.05171)
- **Effect:** 3 shared blocks → 9 effective layers = 1/3 params. Frees ~10MB artifact budget.
- **Our status:** ❌ Tested — catastrophic failure (see Finding #13). Do not retry without fundamentally different approach.

### LoRA Test-Time Training (TTT)
- **Source:** PR #77 (@samacquaviva)
- **Effect:** Per-document LoRA adaptation during eval. +0.003 BPB from TTT itself, +0.011 from doc isolation, +0.034 from striding.
- **Our status:** ✅ Implemented as full-weight SGD TTT (not LoRA). Part of combined 1.1401 run.

### Paid Prefix
- **Source:** PR #168
- **Effect:** Store 12.9M val tokens verbatim in artifact. val_bpb 1.0238. Rules exploit.
- **Our status:** Not pursuing. Non-standard — will not be merged.

### Int5 Quantization (community)
- **Source:** PR #180
- **Effect:** 5-bit quantization enabling larger models.
- **Our status:** ❌ Tested — catastrophic (see Finding #15). Post-quant gap 15× worse than int6. Not viable.

### Hyperparameter tuning (community findings)
- **Source:** Top leaderboard entries
- **Key values:** muon_momentum=0.99, warmup 0.92→0.99 over 1500 steps, rope_base=10000, grad_clip=0.3, tied_embed_lr=0.035
- **Our status:** ✅ All applied as defaults in 11L_XSA_EMA_TTT.

### Reptile Meta-Learning for TTT (community finding)
- **Source:** Non-record research submission in the competition queue
- **Finding 1 (POSITIVE):** Reptile meta-learning improves SmearGate-enabled TTT by **0.011 BPB** — 10× better than naive TTT (+0.001). Standard TTT barely adapts because SmearGate already captures local bigram context; Reptile's outer-loop objective forces the model to learn representations that are fast to adapt.
- **Finding 2 (NEGATIVE):** Error-guided TTT — concentrating adaptation steps on the highest-loss tokens — does not improve val_loss. Hard tokens are **genuinely unpredictable**, not undertrained. Per-token loss analysis: hardest 2.7% of tokens account for ~15% of total loss, and those tokens resist TTT regardless of method.
- **Finding 3:** 13 layers outperforms 10 layers on 8×H100 (val_bpb 1.1884 vs 1.2090) despite 23% fewer training steps. Depth gain outweighs step-count loss at 13L — this is the crossover point we missed when going 9→11L.
- **Our status:** Not implementing Reptile. Plain SGD TTT is already implemented; if it underperforms, Reptile is the next option.

### SWA Checkpoint Count Matters (community finding)
- **Source:** Non-record quantization findings in the competition queue
- **Finding:** With 84 checkpoint average (SWA every step in warmdown, WD=20000), int6+zstd roundtrip BPB is **lower** than pre-quant BPB (1.5164 vs 1.5536, gap = -0.037). SWA with enough checkpoints eliminates quantization-sensitive weight outliers entirely — quantization actually *improves* the score.
- **Why our SWA showed no effect:** We tested SWA at WD=1200 producing only a handful of checkpoints. The smoothing is insufficient to remove outliers. The effect requires ~50+ checkpoint average across a long warmdown.
- **Implication:** SWA is not a weak technique — it was undertested. At WD=20000 with frequent sampling it may outperform EMA. Not worth switching now, but explains the discrepancy between our result (+0.0004) and entries that report SWA gains.
- **Our status:** Superseded by EMA for now. If EMA underperforms, revisit SWA with WD=20000.

---

## Not yet tested by us

### BitNet b1.58 (Ternary Weights)
- **Source:** arxiv 2402.17764, 2407.09527
- **Potential:** 1.58 bits/param → 60M+ params in 16MB. 3x more params than int8.
- **Risk:** Needs ~2x params to match FP16 quality. Training stability at 20M scale unproven.
- **Competition status:** PR #126 attempted but didn't converge (1.7510 BPB). PR #139 got 1.2029 with 65M params — works but undertrained.

### Learned Compression Codebooks
- **Potential:** Train a small codebook that compresses better than int6+zstd.
- **Status:** Nobody has tried this.

### BigramHash 4096 buckets (vs our 2048)
- **Potential:** More hash buckets = less collision = cleaner bigram signal. ~1MB extra params.
- **Status:** Low-risk tweak, untested.

### TTT with more epochs (5 vs 3)
- **Potential:** More adaptation to val distribution. Diminishing returns beyond ~5 epochs.
- **Status:** Easy knob to turn if 3-epoch result is competitive.

### Disable QAT during training
- **Potential:** QAT adds ~20% step overhead (Finding #1). Disabling it frees ~1,400 extra steps in 600s.
- **Risk:** Larger quantization gap at export. May or may not be worth it depending on int6 gap magnitude.
- **Status:** Worth testing as a quick ablation once base result is known.

---

## Key meta-lessons

1. **Hardware variance matters more than most techniques.** 44ms vs 70ms/step is a 60% difference — that's 13,600 vs 8,500 steps, worth ~0.015 BPB.

2. **The competition rewards composition, not innovation.** The top entries stack 5-8 known techniques. Clean ablations are rare and valued.

3. **Int6 + 3x MLP is the dominant meta.** Everything else is marginal by comparison. The ~0.02 BPB from wider MLP is larger than most other individual techniques.

4. **torch.compile is fragile.** Conditional code paths, graph priming, and mutable module attributes all cause subtle performance regressions.

5. **bf16 accumulation is dangerous.** Any running sum over thousands of steps must use float32.

6. **The 16MB artifact cap is the binding constraint.** Every architectural decision is downstream of "how do I fit more effective parameters in 16MB?"

7. **Always verify optimizer parameter coverage.** Every nn.Module must appear in at least one optimizer group. Modules initializing to zero and never updating produce no error and no training signal.

8. **Adding layers has a step-count cost.** Each extra layer adds ~7ms/step. Check that capacity gain exceeds the step loss for a 600s budget before going deeper.

9. **SWA only works with many checkpoints over a long warmdown.** A handful of checkpoints at WD=1200 shows no effect. 84 checkpoints at WD=20000 reverses the quantization gap entirely. The technique was undertested, not ineffective.

11. **Hard tokens resist TTT regardless of method.** The hardest 2.7% of tokens (by loss) account for ~15% of total loss and are genuinely unpredictable — not a training artifact. Don't design TTT strategies targeting these tokens.
