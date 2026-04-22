# Official SOTA Progression — full tree to #1493

**Scope:** Every merged record-track submission from the first to PR #1493 (current merged SOTA).
**Goal:** Historical understanding — how did the competition evolve?
**Depth:** Deep (arch + tuning + mechanistic why + speculation + ideas that spawn)
**Generated:** 2026-04-21

---

## Overview

The competition ran from 2026-03-19 to 2026-04-09, improving from **1.2197 → 1.0810 bpb** (−0.1387 total).
True SOTA progression has **24 steps** across ~3 weeks.

### The three paradigm shifts

1. **Sliding window eval (#50, March 19)** — pure eval change, −0.0139 instantly, zero training cost. Changed the meaning of "bpb" in the competition.
2. **Legal TTT (#549, March 24)** — introduced eval-time test-time training. Established the legal framework all subsequent TTT built on.
3. **Vocabulary scaling (#1218, April 9)** — 1024 → 4096 vocab, −0.0085 in one shot. The single largest architecture step in the late phase.

### bpb by phase

| Phase | Range | Dominant driver |
|---|---|---|
| Early (Mar 19–20) | 1.22 → 1.14 | Engineering: sliding window, FP16 embed, int6 QAT, MLP width, SWA |
| Middle (Mar 23–30) | 1.14 → 1.11 | Architecture: XSA, EMA, partial RoPE, LN Scale, LeakyReLU², better GPTQ |
| Late (Apr 9) | 1.11 → 1.08 | Composition: vocab scaling, depth recurrence, parallel residuals, TTT |

---

## Step-by-step deep analysis

---

### #42 — fp16 tied embedding + warmdown/LR tuning
**Author:** chonchiog | **bpb: 1.2197** | **Δ: baseline**

**Architectural delta:** Keep the tied embedding (shared input/output weight matrix) in FP16 rather than quantizing it to int8. Trim MLP hidden 1024 → 992 to stay under 16MB.

**Re-tuning required:** Warmdown extended to 3600 steps; matrix LR bumped from default to 0.06 (observation that default schedule doesn't align with actual step count in 10 min).

**Why it worked:** The tied embedding doubles as the output (lm_head) projection — quantization errors in it affect every prediction. FP16 passthrough eliminates the quant gap (~0.007 → ~0.0005 bpb) at the cost of ~500KB, which is recovered by the minor MLP trim.

**Without this change:** Every future submission would have been fighting a ~0.007 bpb quantization tax on the output layer. This is a "must-fix" bug, not an optional improvement.

**Ideas it spawns:** The observation that output-head quantization is disproportionately costly foreshadows GPTQ embedding quantization in #1394.

---

### #49 — SOTA attempt
**Author:** spokane-way | **bpb: 1.2064** | **Δ: −0.0133**

**Architectural delta:** No body text — minimal PR. Likely a LR/schedule sweep on the #42 stack.

**Re-tuning required:** Unknown.

**Why it worked:** Early-stage competition; even basic hyperparameter tuning off the baseline was a record.

**Without this change:** Minor — #50 would have superseded it regardless.

**Ideas it spawns:** Nothing specific.

---

### #50 — Sliding Window Eval (stride=64)
**Author:** mattqlf | **bpb: 1.1925** | **Δ: −0.0139**

**Architectural delta:** Zero changes to training or the model. Evaluation only: replace single-pass (each token sees up to 1023 prior tokens, many see far fewer at document start) with overlapping windows of stride 64. Each scored token now has 960+ tokens of context.

**Re-tuning required:** None — the model was not changed.

**Why it worked:** The transformer's performance degrades sharply when context is short. In single-pass eval, tokens near document boundaries see almost nothing. Sliding window gives every token near-full context, revealing the model's true capability. This is a pure measurement fix, not a model improvement.

**Without this change:** The competition would have been measuring a pessimistic lower bound on model quality. Many future architectural improvements (depth recurrence, parallel residuals) would have appeared smaller because the baseline was artificially inflated.

**Ideas it spawns:** The −0.0139 "free" improvement by just changing eval strategy foreshadowed the entire TTT line of work — if eval context matters this much, what if we can adapt the model to the eval distribution?

---

### #60 — Sliding Window + FP16 Embed + 10L + Muon WD + Overtone Init
**Author:** notapplica | **bpb: 1.1748** | **Δ: −0.0177**

**Architectural delta:** 6 improvements stacked: (1) sliding window eval, (2) FP16 tied embed, (3) 10 layers (up from 9), (4) Muon weight decay (decoupled `p.mul_(1 - wd * lr)`, wd=0.02), (5) Overtone spectral embedding init (SVD power-law shaping), (6) phase-transition resid_mix init.

**Re-tuning required:** The 10th layer is only possible because Muon WD compresses weights enough to stay under 16MB — the two innovations are coupled. Without WD, the extra layer doesn't fit.

**Why it worked:** The WD-compression coupling is the key insight: Muon had no built-in regularization, so adding WD both improves generalization AND makes weights smaller, freeing artifact budget for architectural capacity. This coupling between regularization and artifact budget becomes the dominant design pattern for the entire competition.

**Without this change:** The competition would have stayed at 9 layers for longer. More importantly, the WD-compression insight would have taken longer to crystallize — it's rediscovered/refined in nearly every subsequent record.

**Ideas it spawns:** WD-compression coupling. Every subsequent submission iterates on this. The 10L breakthrough also shows that capacity (more layers) beats width (larger MLP) given the compression constraint.

---

### #63 — 10L Int6 QAT + Zstd MLP2.6x Muon0.99 Sliding Window
**Author:** yahya010 | **bpb: 1.1598** | **Δ: −0.0150**

**Architectural delta:** 11 changes stacked. Key ones: (1) STE int6 QAT — fake-quantize all weights to int6 during training, eliminating the quant gap entirely (pre-quant = post-quant bpb); (2) zstd-22 compression instead of zlib; (3) MLP hidden 1344 (2.625× model dim, funded by int6+zstd savings); (4) Muon momentum 0.92→0.99.

**Re-tuning required:** Lower LRs (MATRIX_LR 0.025→0.02) to accommodate higher momentum; gradient clipping 0.3; warmdown 3600.

**Why it worked:** STE QAT is the big win — by training with fake quantization, the model learns to place weights in positions that quantize well, turning a post-hoc approximation into a first-class training objective. The quant gap (previously 0.01+) drops to 0.000. This frees artifact budget which is immediately reinvested in a wider MLP, which improves quality enough to more than offset the WD cost.

**Without this change:** The competition would have spent much longer fighting the quantization gap rather than improving model quality. This is a genuine paradigm shift in how to think about the 16MB constraint.

**Ideas it spawns:** If QAT can eliminate the quant gap, what happens if you also optimize the compression (brotli vs zstd)? → #1179. What if GPTQ's Hessian-aware quantization is better than per-row clipping? → #1019, #1218.

---

### #65 — Mixed Quant Int6/FP16 + SmearGate + OrthoInit + MLP 3x + Sliding Window
**Author:** aquariouseworkman | **bpb: 1.1556** | **Δ: −0.0042**

**Architectural delta:** SmearGate (learned per-dimension gate blending current token embedding with previous token embedding, ~512 params, init near-identity), BigramHash (4096-bucket hash table, dim=128→512), MLP 3× expansion (1536 hidden), U-Net skip connections (4-layer encoder + 5-layer decoder), OrthoInit.

**Re-tuning required:** Separate LRs per parameter group (embed 0.030, matrix 0.020, scalar 0.020); Muon WD=0.01 with momentum warmup 0.92→0.99 over 1500 steps.

**Why it worked:** SmearGate injects a direct bigram shortcut at the embedding level — the model doesn't have to learn first-order co-occurrence purely through attention. BigramHash adds explicit token-pair statistics. Both are essentially "cheating" (giving the model access to facts that a pure attention mechanism would need to discover), but they're cheap in parameters and highly effective. U-Net skips help the model reuse early representations in later layers.

**Without this change:** The BigramHash line would have taken longer to establish. SmearGate specifically becomes standard equipment across the field.

**Ideas it spawns:** BigramHash grows in every subsequent submission until #1218 removes it (it's less necessary at 4096+ vocab). SmearGate persists all the way to #1493.

---

### #86 — 11L MLP3x + WD=0.04 + zstd-22
**Author:** aruniyer | **bpb: 1.1502** | **Δ: −0.0054**

**Architectural delta:** 11 layers (up from 10), MLP 3× (funded by WD-compression gain), WD raised 0.02→0.04.

**Re-tuning required:** WD sweep (0.01–0.05 range) to find the quantization-compression sweet spot.

**Why it worked:** Confirms the WD-compression-capacity flywheel: higher WD → smaller weights → better compression → more artifact budget → more capacity → better bpb even accounting for WD regularization cost. At WD=0.04, the model compresses enough to fit an 11th layer AND wider MLP.

**Without this change:** The 11-layer architecture that becomes universal might have taken longer to arrive.

**Ideas it spawns:** The WD sweet spot becomes something every competitor tunes. WD=0.085 in #1218, WD=0.090 in #1285 show this keeps compounding.

---

### #162 — Int6 MLP3x + SmearGate + BigramHash + MuonWD + SWA
**Author:** raahilshah | **bpb: 1.1483** | **Δ: −0.0019**

**Architectural delta:** Consolidates the winning stack: per-row int6 (not QAT but cleaner implementation), MLP 3×, SmearGate, BigramHash(4096, dim=128), OrthoInit, Muon WD=0.02. Adds **SWA** (stochastic weight averaging) over last 50% of training.

**Re-tuning required:** Muon momentum warmup 0.92→0.99 over 1500 steps; careful per-parameter WD split.

**Why it worked:** SWA creates a smoother weight manifold by averaging across training checkpoints — averaged weights are flatter (less sharp minima), which is exactly what makes quantization easier. It's a free compression improvement that also generalizes better.

**Without this change:** A small step. The real contribution is consolidating the full stack that becomes the "standard" for the next phase.

**Ideas it spawns:** SWA → EMA (#287), which proves strictly better.

---

### #180 — 10L Int5-MLP + BigramHash(10240) + SWA(0.4) + WD=0.04
**Author:** thwu1 | **bpb: 1.1428** | **Δ: −0.0055**

**Architectural delta:** Key innovation: **mixed int5 MLP / int6 attention**. Int5 ([-16,15]) has 3 zero high bits per byte — zstd-22 compresses it at 1.88× vs 1.51× for int6. This saves 1.86MB, which funds a 10th layer while staying under 16MB. BigramHash table scaled to 10240 buckets.

**Re-tuning required:** WD swept 0.01–0.05; SWA frequency swept 25–200 steps. Careful ablation of each change.

**Why it worked:** This is the first explicit compression-budget arbitrage — trading per-weight quality (int5 is slightly worse than int6) for capacity (extra layer funded by the savings). The net exchange is favorable because depth helps more than per-weight precision at this scale.

**Without this change:** The int5/int6 mixed-precision idea is a precursor to the more sophisticated per-layer precision routing in later records.

**Ideas it spawns:** Per-layer mixed precision becomes standard. The "buy capacity with compression savings" trade becomes the dominant design pattern.

---

### #265 — 11L + Efficient Partial XSA
**Author:** unnir | **bpb: 1.1307** | **Δ: −0.0121**

**Architectural delta:** **Exclusive Self Attention (XSA)** applied to last 3 of 11 layers. XSA projects out the self-attention bias (the component of attention output that points toward the query's own value vector) via an orthogonal projection: `y = y - dot(y, vn) * vn` where vn = normalize(v). Key engineering: efficient GQA-aware implementation using free reshape instead of repeat_interleave, reducing XSA overhead from 7ms/step to 2ms/step.

**Re-tuning required:** Applied only to deepest 3 layers (where self-attention bias is highest, per XSA paper). Efficient implementation required careful GQA reshaping.

**Why it worked:** XSA removes a known degenerate solution: the model can trivially "attend to itself" and return its own value vector. By projecting this out, the model is forced to attend to other tokens, improving the information content of attention heads in deep layers where this bias is most pronounced.

**Without this change:** This is a −0.012 step which is unusually large. XSA plus the efficient GQA implementation becomes a standard component.

**Ideas it spawns:** Apply XSA to all layers, not just last 3 → #1019 (XSA-all). Efficient attention implementations become a recurring theme.

---

### #287 — 11L XSA + EMA + Int6 MLP3x + WD=0.04
**Author:** jfprincz | **bpb: 1.1271** | **Δ: −0.0036**

**Architectural delta:** Replaces SWA with **EMA (exponential moving average)** at decay=0.997, applied every step. Everything else from prior best stack.

**Re-tuning required:** EMA decay tuned (0.997 found to be better than periodic SWA).

**Why it worked:** SWA averages a fixed number of checkpoints taken periodically. EMA is a continuous average that weighs recent checkpoints more heavily. For a model under a LR warmdown schedule, EMA better tracks the model's "optimal" point than periodic checkpoints — it gets the benefit of weight averaging without missing the fast-improvement phase near the warmdown end.

**Without this change:** SWA is a strict downgrade vs EMA here. Once found, EMA becomes universal.

**Ideas it spawns:** The smoothness argument for EMA also applies to quantization (smoother weights → better compression) — confirmed in every subsequent submission.

---

### #315 — 11L Partial RoPE + LN Scale + EMA + XSA4
**Author:** jfprincz | **bpb: 1.1248** | **Δ: −0.0023**

**Architectural delta:** Two zero-parameter changes: (1) **Partial RoPE** — apply rotary position embeddings to only 16 of 64 head dimensions (25%). Remaining 48 dims are position-free. (2) **LN Scale** — RMSNorm output scaled by 1/sqrt(layer_idx+1), damping deeper layers.

**Re-tuning required:** None — both are zero-parameter architectural choices.

**Notable bug:** The code ships with LATE_QAT=1 flag but torch.compile constant-folds the QAT branch, making it dead code. The improvement is entirely from Partial RoPE + LN Scale, not QAT.

**Why it worked:** Partial RoPE allows some dimensions to attend without positional bias, which helps for long-range or position-invariant patterns. LN Scale acts as a depth-aware normalization that prevents deep layers from dominating — a form of learned skip connectivity. Both improve generalization and compressibility at zero parameter cost.

**Without this change:** Small step. But Partial RoPE (16/64 dims) becomes universal equipment through #1493.

**Ideas it spawns:** The 16/64 partial RoPE ratio is adopted directly by all subsequent top submissions. It's never ablated again — accepted as a "free win."

---

### #414 — 11L EMA + GPTQ-lite + warmdown3500 + QAT@0.15
**Author:** signalrush | **bpb: 1.1233** | **Δ: −0.0015**

**Architectural delta:** **GPTQ-lite** — instead of fixed clip (row max), try 5 clip percentiles (0.999, 0.9995, 0.9999, 0.99999, 1.0) per weight matrix row and pick the one minimizing reconstruction MSE. Warmdown extended 3000 → 3500.

**Re-tuning required:** Ablation of each change individually (−0.0006 from GPTQ-lite, −0.0006 from EMA, −0.0002 from warmdown extension).

**Why it worked:** GPTQ-lite is the first application of per-tensor optimal clip search, a technique from the GPTQ literature that adapts the quantization scale to each weight matrix's actual distribution rather than using a fixed heuristic. It's free at inference time (runs once after training) and uniformly better.

**Without this change:** Small. The key contribution is establishing that clip-search is better than fixed-clip, which influences SDClip in #1394.

**Ideas it spawns:** This is the exact per-tensor clip sweep idea we noted in PR #1749 today. It was already in the official tree in March. SDClip (#1394) is a more principled version of the same idea.

---

### #549 — LeakyReLU² + Legal Score-First TTT + Parallel Muon
**Author:** abaybektursun | **bpb: 1.1194** | **Δ: −0.0039**

**Architectural delta:** Two real changes: (1) **LeakyReLU(0.5)²** — replace `relu(x).square()` with `leaky_relu(x, 0.5).square()`. Negative slope 0.5 preserves gradient flow through the MLP for negative activations. One-line change, −0.003 bpb. (2) **Legal Score-First TTT** — for each 32K-token chunk: score under inference_mode FIRST, THEN do 3 SGD epochs with lr=0.002 over the scored tokens.

**Re-tuning required:** TTT LR=0.002, momentum=0.9, 3 epochs. Careful compliance with Issue #1017 (score-before-update).

**Why it worked:**
- LeakyReLU²: allowing negative activation flow preserves gradient signal for ~50% of activations that relu kills. The squared nonlinearity gives a piecewise-quadratic shape with strong expressiveness in the positive regime while not completely zeroing the negative regime.
- TTT: eval-time adaptation to the specific validation distribution. The model has already been trained well; TTT is a small perturbation that closes the train-eval distributional gap. The −0.0025 gain from TTT on this stack will grow as the base model improves.

**Without this change:** LeakyReLU² is adopted universally through #1493. TTT is the seed of the most important eval mechanism in the competition.

**Ideas it spawns:** TTT LR, epochs, and chunk size become key tuning parameters. Every subsequent record uses TTT. The "phased" TTT (split into multiple rounds) and LoRA TTT (#1530/#1610) are direct descendants.

---

### #1019 — AR Self-Gen GPTQ + XSA-all + BigramHash 3072×112
**Author:** abaybektursun | **bpb: 1.11473** | **Δ: −0.0047**

**Architectural delta:** Three changes, drops TTT: (1) **Full Hessian GPTQ** with Cholesky error compensation — prior GPTQ used diagonal Hessian approximation; this uses the full H = X^T X per layer with column reordering for numerical stability. (2) **AR self-generated calibration** — model generates its own 64×2048 calibration sequences autoregressively to avoid using val data. (3) **XSA on all 11 layers** (up from last 4). BigramHash scaled to 3072 buckets × dim=112.

**Re-tuning required:** 186s of calibration generation is carved out of the 600s eval budget.

**Key finding from ablation:** AR self-gen closes 84% of the gap between random-token and val-data calibration (0.0017 of 0.0020 bpb). Using val data for calibration is not meaningfully better than self-generation.

**Why it worked:** Full Hessian GPTQ compensates for quantization error in column j by adjusting all subsequent columns — turning a greedy approximation into a more globally optimal quantizer. The Cholesky trick makes this computationally tractable. The gap between random-calibration and self-gen calibration being 0.0003 bpb is a remarkable result — it means the model's own distribution closely approximates the training distribution.

**Without this change:** The quantization gap would have stayed at ~0.0023 bpb. GPTQ becomes the universal quantizer from here on.

**Ideas it spawns:** AR self-gen calibration solves the "can't use val data" constraint cleanly. This is immediately adopted. TTT is also dropped here (25 failed TTT experiments reported) — showing that architecture improvements can make TTT net-negative on certain stacks.

---

### #1179 — Split-LR + BigramHash(2816×160) + Full GPTQ + Brotli
**Author:** dexhunter | **bpb: 1.1105** | **Δ: −0.0042**

**Architectural delta:** (1) **Split-LR**: different learning rates for early (0.025) vs late (0.030) Muon layers. (2) **Sigmoid-gated U-Net skips** — learnable sigmoid gates on encoder-decoder skip connections. (3) **Soft-round QAT**: temperature-controlled rounding (alpha 1→16) replacing STE. (4) **Brotli-11 + byte-shuffle** instead of LZMA-9 — saves ~400KB artifact budget. (5) Code minification saves another 78KB.

**Re-tuning required:** LR split ratio tuned per-layer. Brotli byte-shuffle requires careful alignment of weight storage.

**Why it worked:** Brotli-11 with byte-shuffle is a pure compression win — by interleaving the high/low bytes of int6-packed weights, the compressor sees more regular patterns and achieves better compression. The ~400KB saving is reinvested in architectural capacity. Split-LR acknowledges that different parts of the network have different learning dynamics (early layers are more stable, later layers are more flexible).

**Without this change:** Brotli+byte-shuffle becomes universal and is in #1493. The artifact budget freed here enables every subsequent "add one more thing."

**Ideas it spawns:** Every subsequent submission uses Brotli+byte-shuffle. The code minification strategy (101KB→23KB) is a pure artifact budget unlock.

---

### #1204 — Parallel Residuals + Mini Depth Recurrence
**Author:** msisovic | **bpb: 1.1063** | **Δ: −0.0042**

**Architectural delta:** Two architecture changes: (1) **Parallel Residuals** (from layer 7) — attention and MLP operate in separate residual streams with learned cross-stream routing weights (`attn_to_attn`, `attn_to_mlp`, `mlp_to_attn`, `mlp_to_mlp`). (2) **Mini Depth Recurrence** — repeat layers 4,5 (around U-Net hinge) twice with shared parameters, with delayed activation (start at step 3000). Untied MLPs, shared attention.

**Re-tuning required:** Sweep of which layers to repeat, whether to untie MLPs (yes) or attention (no), delay start (3000 steps). Careful ablation: one repeated layer helps, two helps more, three loses to step-time penalty.

**Learned routing surprise:** `mlp_to_attn` weights converge near zero (0.009 in deepest layers) — MLP barely writes to the attention lane. Attention-to-attention and MLP-to-MLP are dominant. This means the model essentially discovers that the channels are mostly independent, but the *asymmetry* in how they interact is learned rather than designed.

**Why it worked:** Depth recurrence gives more virtual layers (11 physical → 13+ virtual) within the parameter budget. The looping is applied at the U-Net hinge point where representations are most abstract. Mini recurrence (just 2 layers) avoids the full triple-recurrence overhead while capturing most of the quality gain.

**Without this change:** Depth recurrence (layers 4,5 repeated) becomes universal through #1493. This is a landmark PR.

**Ideas it spawns:** Loop start at step 3000, loop layers 4,5, untied MLPs — all inherited directly. The "loop 3 layers" in #1493 is a direct extension of this work.

---

### #1218 — 4096-Vocab + 4.0-MLP-mult + 0.085-WD + Simplifications
**Author:** clarkkev | **bpb: 1.09785** | **Δ: −0.0085**

**This is the largest single step in the late phase.**

**Architectural delta:** (1) **Vocabulary 1024 → 4096** (SP4096 SentencePiece tokenizer). (2) MLP multiplier 3× → 4× (wider MLP funded by WD-compression gains). (3) WD raised 0.04 → 0.085. (4) Crucially: **removed many things** — removed TTT, QAT, parameter banking, distributed Muon, gated attention, value residuals, hash embeddings, SmearGate. Added GPTQ Hessian quantization, byte-shuffle+brotli, sigmoid skip gates, coprime-stride data loader.

**Re-tuning required:** Full Hessian GPTQ time carved from training. SDClip formula derived mathematically: clipping threshold = k·std(row) where k=12.85 for matrices, k=20 for embeddings. WD swept across several experiments.

**The mathematical insight:** Compressed artifact size ∝ entropy H(q) ≈ b - log₂k + constant, where b is bit-width and k is clip range in σ units. This derivation shows that weight decay directly reduces compressed size (smaller weights → lower entropy after quantization) and that standard-deviation-based clipping is more principled than MSE-minimizing clip search — it directly accounts for what the compressor sees.

**Why it worked:** The 4096-vocab change is the biggest win. More vocabulary tokens per sequence means more training signal per forward pass and longer effective n-gram coverage for the same sequence length. SmearGate and BigramHash become less necessary when the vocab already captures multi-character patterns. The WD increase (0.04 → 0.085) lets the model fit a wider MLP (4×) while staying under 16MB.

**Without this change:** This resets the competition and becomes the foundation for everything from #1285 onward.

**Ideas it spawns:** SP8192 is the next vocab step (#1394). The SDClip mathematical derivation is adopted universally. The "remove things that don't help at this scale" philosophy is important — SmearGate and BigramHash add little value with larger vocab.

---

### #1285 — MuonEq-R + Depth Recurrence + WD=0.090 + All-Int6 GPTQ
**Author:** dexhunter | **bpb: 1.09124** | **Δ: −0.0066**

**Architectural delta:** (1) **MuonEq-R** — row-normalized variant of Muon (arXiv:2603.28254). (2) **Depth recurrence** on layers 4,5 (from #1204). (3) WD 0.085 → 0.090. (4) **All-int6** — all 66 layers at int6 (vs #1218's mixed precision).

**The WD-quantization synergy:** Higher WD (0.090) → weights 5% more compressible → headroom for ALL 66 layers at int6 → quantization quality improvement exceeds WD bpb cost. A careful table shows: WD=0.085 + 60 int6 layers = 1.09217; WD=0.085 + 61 layers = 1.09170; **WD=0.090 + 66 layers = 1.09057** (winner).

**Re-tuning required:** WD tuning is precise — WD=0.090 is a specific local optimum. Above this, the compression gains don't outpace the regularization cost.

**Why it worked:** The WD-compression flywheel reaches a new equilibrium. MuonEq-R's row normalization ensures the Muon step is properly scaled regardless of gradient magnitude, which improves stability at higher WD.

**Without this change:** Without depth recurrence + WD-all-int6 synergy, the competition stays around 1.097.

**Ideas it spawns:** WD tuning becomes increasingly refined. The exact WD optimum changes with each new architectural addition — in #1493 it lands at 0.095.

---

### #1334 — SP4096 + Depth Recurrence + Parallel Residuals + MuonEq-R + QK-Gain 5.0
**Author:** aryanbhosale | **bpb: 1.0897** | **Δ: −0.0015**

**Architectural delta:** Composition of proven winners: 4096-vocab (#1218), depth recurrence layers 4,5 (#1285), parallel residuals from layer 7 (#1204), MuonEq-R (#1285), QK-Gain 5.0.

**Re-tuning required:** QK_GAIN_INIT swept to 5.0 (from 4.0 in #1218) — monotonic improvement found.

**Why it worked:** Pure stacking. Each component is proven; the combination is additive.

**Without this change:** Minor — mostly confirms the stack. The QK-Gain sweep is the key new contribution.

**Ideas it spawns:** QK-Gain 5.0 → 5.25 in #1493. The monotonic improvement relationship suggests 5.25 → 5.5 → ... might continue, though diminishing returns.

---

### #1394 — SP8192 + GPTQ Embeddings + Depth Recurrence + MuonEq-R + SDClip
**Author:** clarkkev | **bpb: 1.08563** | **Δ: −0.0041**

**Architectural delta:** Vocabulary **4096 → 8192** (SP8192 tokenizer). **GPTQ-quantize the embedding matrix** (rather than naive round-to-nearest) — the embedding matrix is both the largest single tensor and the most frequently accessed, so quantization quality here matters disproportionately. Loop layers 4-5 twice (implementing depth recurrence more cleanly). Row-normalized Muon. SDClip with k=12.85/20.

**Re-tuning required:** 5-seed mean for statistical confidence. GPTQ embedding quantization requires Hessian collection for the embedding layer specifically.

**Why it worked:** SP8192 gives the model a larger vocabulary with more context per token. The embedding matrix with GPTQ quantization recovers quality that naive rounding lost. The combination of larger vocab + GPTQ embeddings is the key move — separately, each would give less.

**Without this change:** SP8192 becomes universal through #1493. GPTQ embedding quantization is permanently adopted.

**Ideas it spawns:** Can vocab be scaled further? (SP8192 is where the competition ends up.) GPTQ embedding quantization as a standard practice.

---

### #1413 — SP8192 + QK-Gain 5 + Legal Score-First TTT
**Author:** dexhunter | **bpb: 1.08279** | **Δ: −0.0028**

**Architectural delta:** Built on #1394. Two changes: (1) **QK-Gain 5.0** (vs 4.0 in #1394). (2) **Legal score-first TTT** — SGD lr=0.005, 3 epochs, cosine decay, all blocks unfrozen.

**Re-tuning required:** TTT LR tuned from scratch on the new stack (0.005 vs #549's 0.002 — the larger model can absorb a higher TTT LR).

**Why it worked:** TTT at 1.08 level is stronger than at 1.12 level — the base model is better, so the in-distribution adaptation has more to work with. TTT gain −0.002 on this stack vs −0.0025 in #549 (comparable, suggesting TTT scales with base model quality). QK-Gain 5.0 → sharpens attention, focusing more on the most relevant positions.

**Without this change:** TTT re-enters the competition here after being absent from #1019 onward. It stays through #1493.

**Ideas it spawns:** TTT LR is now 0.005 on the SP8192 stack. The QK-Gain 4.0→5.0→5.25 sweep in #1493 suggests a principled monotonic relationship.

---

### #1477 — SP8192 + Parallel Residuals + Score-First TTT
**Author:** aryanbhosale | **bpb: 1.0822** | **Δ: −0.0006**

**Architectural delta:** Adds **parallel residuals** (from layer 7) to the SP8192+TTT stack from #1413.

**Re-tuning required:** None beyond confirming parallel residuals work with TTT.

**Why it worked:** Parallel residuals + TTT are additive — TTT adapts the base model's prediction direction, while parallel residuals give the model more expressiveness in how it integrates attention and MLP outputs.

**Without this change:** Very small step. The key contribution is demonstrating that parallel residuals + TTT compose cleanly without interference.

**Ideas it spawns:** The full "parallel residuals + TTT + depth recurrence" combination proven additive → sets up #1493.

---

### #1493 — SP8192 + 3-Layer Recurrence + Parallel Residuals + QK-Gain 5.25 + Legal TTT
**Author:** bigbag | **bpb: 1.0810** | **Δ: −0.0012**

**Architectural delta:** Extends depth recurrence from 2-layer (layers 4,5) to **3-layer** (layers 3,4,5), activated at step 35% of training. Parallel residuals from layer 7 (as #1477). QK-Gain **5.25** (up from 5.0 in #1413/#1477). Score-first TTT with SGD lr=0.005, 3 epochs, cosine decay.

**Re-tuning required:** 160+ experiments to find the WD=0.095, MLR=0.022, EMA=0.9965, warmdown=0.72 combination. LZMA code wrapper reduces code footprint to 16.6KB. 3-layer recurrence (L3-5 rather than L4-5) is a deliberate choice — sweeping start layer finds L3 slightly better.

**Why it worked:** 3-layer recurrence (17 virtual layers from 11 physical) vs 2-layer gives more recurrence benefit in the middle of the network. QK-Gain 5.25 continues the monotonic improvement. The precision hyperparameter tuning (WD/MLR/EMA/warmdown) across 160+ experiments is what distinguishes this from #1477 — the architecture is nearly identical, but the training recipe is significantly more polished.

**Without this change:** #1477 (1.0822) would have been the merged SOTA. The −0.0012 gap is almost entirely from hyperparameter tuning rather than architectural innovation.

**Ideas it spawns:** QK-Gain 5.25 → can we go to 5.5 or 6.0? 3-layer recurrence → can we try Loop45 (layers 3-5 × 3 repeats)? This is exactly what #1523 (the first open-PR record after the merge) does.

---

## Synthesis

### Typical Δ per step

| Phase | Typical range | Notes |
|---|---|---|
| Early (Mar 19–20) | −0.003 to −0.018 | High variance — cheap wins being picked up |
| Middle (Mar 23–30) | −0.002 to −0.005 | Architecture stabilizing |
| Late (Apr 9) | −0.001 to −0.009 | Most gains are composition; #1218 is the outlier |

The overall mean Δ is ~−0.006 per step. **Anything >0.008 indicates a paradigm shift** (vocab scaling, sliding window, architecture overhaul). Most steps are −0.002 to −0.005.

### Load-bearing ideas (what actually mattered)

**Genuinely load-bearing (competition wouldn't have progressed without these):**
1. **Sliding window eval (#50)** — changed what "bpb" means
2. **FP16 tied embedding (#42/#60)** — eliminated a 0.007 quant tax
3. **Int6 QAT / STE (#63)** — eliminated the quant gap entirely
4. **WD-compression coupling (#60/#86)** — the fundamental design insight of the competition
5. **XSA (#265)** — architecture improvement with strong theoretical grounding
6. **LeakyReLU² (#549)** — one-line activation change, adopted everywhere
7. **Legal Score-First TTT (#549)** — established the TTT legal framework
8. **Full Hessian GPTQ + AR self-gen (#1019)** — solved the calibration problem
9. **Brotli+byte-shuffle (#1179)** — unlocked 400KB of budget
10. **Vocabulary scaling (#1218)** — largest single architecture step post-March
11. **Depth recurrence (#1204)** — virtual layer scaling within param budget
12. **Parallel residuals (#1204)** — cross-stream routing

**Nice-to-have (incremental but compounding):**
- EMA vs SWA (#287) — improves smoothness, universally adopted
- Partial RoPE 16/64 (#315) — zero params, adopted everywhere
- SDClip / GPTQ-lite (#414, #1394) — quantization quality, meaningful but not paradigm-shifting
- QK-Gain tuning (#1334, #1413, #1493) — monotonic improvement, easy to search

### Three classes of innovation in the official tree

1. **Eval tricks** (sliding window, TTT) — highest leverage, zero parameter cost
2. **Architecture** (XSA, depth recurrence, parallel residuals, LeakyReLU²) — medium leverage, medium complexity
3. **Compression engineering** (FP16 embed, WD-compression, brotli, SDClip) — enables everything else by creating artifact budget

### Gaps on our stack (#1736 baseline)

**Already in #1736:** SP8192, CaseOps, SmearGate, AttnOutGate, QuantGate, Loop45 (exceeds #1493's 3-layer recurrence), phased TTT (exceeds #1493's single-phase TTT), VarLen attn + fused MLP, MuonEq-R/matrix_lr=0.026, per-layer GPTQ clip, int8 embeddings, logit_softcap, XSA, RMSNorm Q/K, LeakyReLU(0.5)², GPTQ mixed int6/int8, EMA, Partial RoPE, Brotli+byte-shuffle.

**Notable from the official tree that's NOT in #1736:**
- **AR self-generated GPTQ calibration** (#1019) — #1736 uses real training-data calibration. AR self-gen is ~0.0003 bpb worse but fully legal. Our current calibration might actually be using val data (needs audit).
- **GPTQ-quantized embedding matrix** (#1394) — #1736 uses int8 round-to-nearest for embeddings, not full Hessian GPTQ. Possible ~0.001 bpb recovery.
- **Coprime-stride data loader** (#1218) — minor data diversity improvement. Low priority.
- **Progressive recurrence start** (#1412) — enable recurrence in two phases rather than all at once. Reduces loss spike at recurrence activation.

**The main conclusion:** #1736 is already ahead of #1493 on every major dimension. Its remaining gaps are all minor engineering details, not missing paradigms.
