# Parameter Golf — Research Journal & Experiment Tracker

*Living document. Updated as experiments run and findings accumulate.*
*Last updated: 2026-04-01*

---

## 1. Current Best Configuration

**exp11_vrl_xsa11_noema.py** — 7000-step run started 2026-04-02:
- 11L/512d/3xMLP, LeakyReLU(0.5)^2, GQA 8H/4KV
- **XSA on ALL 11 layers** (fixed: uses F.normalize, not F.rms_norm)
- **VRL (Value Residual Learning)**: learnable lambda blend of V_1 into all subsequent layers
- Partial RoPE 16/64, LN Scale 1/sqrt(L+1)
- SmearGate + BigramHash(3072x112), U-Net skips, OrthoInit
- EMA(0.997) tracking (NOT applied for quantization — EMA weights kill int6)
- Late QAT (STE @ LR scale<0.15)
- Linear warmdown (our finding), QK gain=2.0 (our finding)
- LZMA-9 compression (beats zstd-22 by ~200KB at high entropy)
- Int6 GPTQ-lite + FP16 embeddings
- Post-training +/-1 pruning for 16MB safety

**7000-step results:**
| Run | fp32 bpb | int6 bpb | Quant gap | Artifact | Key |
|-----|----------|----------|-----------|----------|-----|
| exp11 (optimizer weights) | 1.1874 | **1.1957** | +0.008 | 20.0 MB | Best int6 bpb |
| exp12 (EMA + 3500 QAT) | 1.1877 | 1.1998 | +0.012 | 20.0 MB | EMA+QAT works! |
| exp9 (buggy XSA, reference) | 1.1946 | 1.2022 | +0.008 | 20.5 MB | Old baseline |

**EMA+QAT breakthrough (exp12):** Extended QAT (3500 steps) reduced EMA quantization gap from +0.35 to +0.012. Not quite as good as direct optimizer quantization (+0.008), but viable. On RunPod with full data, EMA version compresses to ~10-13 MB.
**Submission candidate: exp11** (best int6 bpb). exp12 as backup if size is tight.

### Critical Bugs Found & Fixed (2026-04-02)

| Bug | Impact | Fix |
|-----|--------|-----|
| **XSA used F.rms_norm instead of F.normalize** | Over-subtracted by factor of 64 (head_dim). Present in ALL experiments R3 exp1-exp9. | Changed to `F.normalize(v, dim=-1)` |
| **BEMA bias correction was wrong** | Used Adam-style `1/(1-decay^t)` divisor on non-zero-initialized EMA. Mathematically incorrect. | Removed — use plain EMA like SOTA does |
| **EMA switch happened after optimizer step** | Wasted one training step; optimizer momentum became stale | Moved before opt.step, added optimizer state reset |
| **EMA weights catastrophic for int6** | +0.35 bpb gap (1.46→1.81). EMA distributions don't survive int6. | Don't apply EMA for quantization; quantize optimizer weights |

---

## 2. Validated Findings (Empirically Proven)

### 2.1 From Our Experiments (R1–R3, 18+ experiments)

| Finding | Round | Evidence | Ref |
|---------|-------|----------|-----|
| LeakyReLU(0.5)^2: -0.006 bpb | R1 | exp1 vs baseline | experiments/results_summary.txt |
| 10L/3xMLP: -0.042 bpb vs 9L/2xMLP | R1 | exp10 vs baseline | experiments/results_summary.txt |
| Linear warmdown beats cosine: -0.005 bpb | R3 | exp2 vs base | experiments/r3/results_final.md |
| QK gain 2.0 gives small additional gain | R3 | exp5 vs exp2 | experiments/r3/results_final.md |
| zstd-22 saves 5-8% at 500 steps | R3 | exp5 measurement | experiments/r3/results_final.md |
| zstd-22 saves 0% at 7000 steps | exp9 | zstd was -12KB WORSE than zlib | exp9 log |
| EMA without QAT is catastrophic (+0.15-0.22 bpb) | R2 | exp4 | experiments/r2/results.tsv |
| QAT alone has zero effect when quant gap is tiny | R2 | exp12 vs exp11 | experiments/r2/results.tsv |
| SmearGate needs >500 steps to converge | R2, R3 | exp5(R2), exp7(R3) | experiments/r2/results.tsv |
| SOTA stack (XSA-4+RoPE+LN+OrthoInit) starts slower but converges faster | R3 | exp6 convergence curve | experiments/r3/results_final.md |
| DenseFormer matches SOTA stack, smallest artifact (11.79MB) | R3 | exp8 | experiments/r3/results_final.md |
| XSA-all + QK=4.0 simultaneously: FAIL at 500 steps | R3 | exp1 | experiments/r3/results_final.md |
| R2 range regularization breaks torch.compile | R3 | exp4 | experiments/r3/results_final.md |
| More training = worse compression (weight entropy grows) | R2 | 500 vs 1000 step comparison | experiments/r2/results_summary.md |
| Artifact size is dominated by compression ratio, not param count | Analysis | payload identical, compression varies 1.40x-1.94x | compression analysis |

### 2.2 From Competition (Verified via GitHub)

| Technique | ΔBPB | PR/Record | Status |
|-----------|------|-----------|--------|
| XSA on all layers | ~-0.006 | PR #1019 (current SOTA) | Confirmed in merged SOTA |
| VRL (Value Residual Learning) | ~-0.007 to -0.015 | Issue #140, SOTA stack | In SOTA, we haven't added it |
| Sliding window eval stride=64 | ~-0.034 | All submissions | Already in our code |
| Muon optimizer | ~-0.020 vs AdamW | All top-5 | Already in our code |
| EMA(0.997) + XSA switch | ~-0.005 | Top-5 submissions | In our code but switch not implemented |
| BigramHash 3072 | ~-0.012 | Top-3 | In our code |
| Full GPTQ (not just GPTQ-lite) | ~-0.010 | Top-3 | We use GPTQ-lite only |
| AR self-generated GPTQ calibration | ~-0.003 | PR #1019 | Not implemented |
| LZMA-9 beats zstd-22 at high entropy | ~0.5MB savings | SOTA uses LZMA | Not implemented |

### 2.3 From Deep Research Analysis (Paper-Verified, Untested Locally)

| Technique | Paper | Date | Expected Gain | Confidence | Smallest Scale Tested |
|-----------|-------|------|---------------|------------|----------------------|
| **Mousse optimizer** | arXiv:2603.09697 | Mar 2026 | -0.005 to -0.012 bpb | MEDIUM | 160M (6x ours) |
| **BAQ mixed-precision** | arXiv:2506.05664 | Jun 2025 | -0.003 to -0.008 bpb | MEDIUM | 125M (5x ours) |
| **OptRot rotation** | arXiv:2512.24124 | Dec 2025 | -0.002 to -0.005 bpb | HIGH | LLMs (unspecified) |
| **BEMA (bias-corrected EMA)** | arXiv:2508.00180 | Jul 2025 | -0.001 to -0.003 bpb | HIGH | LM fine-tuning |
| **NorMuon** | arXiv:2510.05491 | Oct 2025 | -0.003 to -0.008 bpb | LOW | 1.1B (42x ours) |
| **SLOT (test-time delta)** | arXiv:2505.12392 | May 2025 | -0.003 to -0.008 bpb | HIGH (eval-only) | 7B |
| **Nacrith-style n-gram bias** | arXiv:2602.19626 | Feb 2026 | -0.005 to -0.020 bpb | MEDIUM (eval-only) | 135M |
| **DenseFormer** | arXiv:2402.02622 | Feb 2024 | -0.003 to -0.007 bpb | HIGH (we tested it) | 48L model |

### 2.4 Invalidated / Discarded

| Technique | Why Discarded |
|-----------|--------------|
| MoE at 26M | R1: +0.06-0.08 bpb worse. Confirmed by competition. |
| SwiGLU | 45% slower, net negative. |
| Depth recurrence (full stack) | +0.025 bpb worse. PR #363 confirmed. |
| MoSA (sparse attention) | torch.compile incompatible; sparse attention useless at seq 1024. |
| Cautious optimizer (C-Muon) | N-S orthogonalization destroys element-wise gradient alignment. Competition winners don't use it. |
| MUDDFormer | Smallest test 405M; at 10L static DenseFormer already matches. |
| VGA (value-state gating) | Redundant with XSA; not tested with GQA. |
| Frequency-ordered tokenization | Compresses text, not model weights. Inapplicable. |
| Scylla tokenizer | Closed for byte-counting errors (PR #1143). |
| NorMuon | Smallest test 1.1B — 42x scale gap. |
| nGPT hypersphere | 1.69 bpb — unit-norm conflicts with int6. |

---

## 3. Technique Gaps (In SOTA But Not In Our Stack)

These are proven competition techniques we haven't implemented yet:

| Technique | Priority | Effort | Why Missing |
|-----------|----------|--------|-------------|
| **VRL (Value Residual Learning)** | CRITICAL | Low (~15 lines) | Oversight — should have been in from the start |
| **XSA on ALL layers** (not just last 4) | HIGH | Trivial (1 constant) | We tested XSA-all + QK=4.0 and it failed, but the failure was QK=4.0, not XSA-all |
| **BEMA** (replace standard EMA) | HIGH | Low (~5 lines) | Paper verified, drop-in replacement |
| **EMA→XSA switch at warmdown** | HIGH | Medium (~20 lines) | Not implemented — SOTA switches optimizer weights to EMA at warmdown start |
| **Full GPTQ** (not GPTQ-lite) | MEDIUM | Medium | Our clip search is the "lite" version; full Hessian GPTQ is better |
| **OptRot** (pre-quant rotation) | MEDIUM | Low (~30 lines) | Post-training, zero risk |
| **AR self-gen GPTQ calibration** | MEDIUM | Low (~20 lines) | Generate calibration data from model itself |
| **LZMA-9 compression** | LOW | Trivial | Just swap compressor; may beat zstd at high entropy |
| **11 layers** (instead of 10) | SIZE-DEPENDENT | Trivial | Need to verify it fits with full-data training |

---

## 4. Synergy Map (What Stacks, What Conflicts)

### Positive Synergies (REQUIRES or ENABLES)
- EMA ↔ XSA: EMA REQUIRES XSA for meaningful averaging (without XSA, EMA collapses)
- EMA switch ↔ Warmdown: Switch AT warmdown start for best checkpoint
- OptRot → GPTQ: OptRot PRECEDES GPTQ (rotation reduces outliers, GPTQ then quantizes cleaner weights)
- Weight decay → Quantization: Higher WD → bounded weight RMS → better int6
- XSA ↔ VRL: SYNERGY — XSA handles context, VRL handles token identity (clean separation)
- DenseFormer ↔ VRL: ORTHOGONAL — DenseFormer on hidden states, VRL on value vectors

### Conflicts (REPLACES)
- Mousse REPLACES Muon (not stackable)
- BEMA REPLACES standard EMA (not stackable)
- BAQ REPLACES uniform INT6 (different quantization strategy)
- DenseFormer REPLACES U-Net skips + resid_mix (subsumes both)
- NorMuon REPLACES Muon (not stackable)

### Validated Non-Interactions
- Linear warmdown is orthogonal to all architecture choices
- QK gain tuning is orthogonal to XSA
- SmearGate + BigramHash are orthogonal to attention modifications

---

## 5. Compression Problem Analysis

**The core size challenge:**

| Steps | Payload | zlib-9 | zstd-22 | LZMA-9 | Compression Ratio |
|-------|---------|--------|---------|--------|-------------------|
| 500 (base) | 24.8 MB | 12.76 MB | ~12.03 MB | ~12.0 MB | 1.94x (zlib) |
| 500 (EMA) | 24.8 MB | 10.27 MB | ~9.5 MB | ~9.5 MB | 2.42x (zlib) |
| 1000 | 24.8 MB | 16.11 MB | ~15.2 MB | ~15.0 MB | 1.54x (zlib) |
| 7000 (local, 10 shards) | 28.8 MB | 20.5 MB | 20.5 MB | 18.3 MB | 1.40x (zlib) |
| 7000 (SOTA, full data) | ~28 MB | — | — | ~15.5 MB | 1.81x (LZMA est) |

**Root cause of our 20.5 MB:** Training on 10 shards (each token seen ~35x) causes weight memorization → high entropy → bad compression. Full dataset (100+ shards, each token ~3.5x) produces smoother weights → SOTA compression ratio.

**Mitigation stack (ordered by impact):**
1. Full dataset on RunPod (biggest factor)
2. EMA weight averaging (reduces entropy by smoothing)
3. LZMA-9 instead of zstd-22 (better at high entropy)
4. OptRot pre-quantization rotation (reduces outliers → better quantization → lower entropy)
5. Post-training +/-1 pruning (safety net)
6. 10L instead of 11L (guaranteed to fit even at worst-case compression)

---

## 6. Next Experiment: exp10_vrl_bema_xsa11

### Hypothesis
Adding three proven but missing SOTA techniques — VRL, BEMA, and XSA on all 10 layers — should compound with our existing stack. VRL provides -0.007 bpb from better cross-layer value flow. BEMA eliminates EMA lag. XSA-all gives -0.006 bpb over XSA-4.

### Changes from exp9_full_7000.py
1. **VRL**: Blend first layer's V into all subsequent layers via learned lambdas
2. **BEMA**: Replace `ema_state[k].mul_(decay).add_(v, alpha=1-decay)` with bias-corrected version
3. **XSA on ALL 10 layers** (not just last 4)
4. **EMA→XSA switch**: At warmdown start, load EMA weights into optimizer

### Architecture
```
10L/512d/3xMLP, LeakyReLU(0.5)^2
XSA on ALL 10 layers (changed from last 4)
VRL: V_n = lambda1 * V_1 + lambda2 * H_{n-1} @ W_V_n (NEW)
Partial RoPE 16/64, LN Scale 1/sqrt(L+1)
SmearGate + BigramHash(3072x112)
U-Net skips, OrthoInit
BEMA(0.997) with bias correction (NEW)
EMA switch at warmdown start (NEW)
Late QAT (STE @ scale<0.15)
Linear warmdown, QK gain=2.0
Int6 GPTQ-lite + FP16 emb + LZMA-9 (changed from zstd-22)
Post-training +/-1 pruning
```

### Expected BPB
- At 500 steps: ~1.42-1.44 (slower start from VRL+XSA-all learning)
- At 7000 steps: ~1.18-1.20 (projected from SOTA technique stack)

### Success Criteria
- At 500 steps: must show val_bpb improvement over exp9 at same step count
- Convergence rate (bpb/step) should be steeper than exp9 by step 300+
- Artifact size should be comparable or smaller (BEMA/EMA should improve compression)

### Run Protocol
1. Run 500 iterations locally (RTX 3070, ~50 min)
2. If promising (bpb delta > -0.002 vs exp9 at step 500): continue to 7000 steps (~12 hours)
3. If neutral or worse: analyze which addition hurt, run ablation

---

## 7. Future Experiment Queue (After exp10)

### exp11: OptRot + Full GPTQ (post-training only)
Apply to whatever trained model we have. Zero training impact.
- OptRot: L4-minimizing rotation before quantization
- Full GPTQ with Hessian instead of GPTQ-lite clip search
- AR self-gen calibration data
- LZMA-9 compression
- Expected: -0.003 to -0.010 bpb from better quantization alone

### exp12: Mousse optimizer (if 500-step test shows promise)
Replace Muon with Mousse for hidden-layer matrices.
- Risk: 160M was smallest tested; may not help at 26M
- Need 500-step comparison: Mousse vs Muon on identical architecture
- If step 500 delta > -0.003: continue to 7000

### exp13: Evaluation-time techniques (apply to any trained model)
- SLOT probe vector (512-dim bias, updated per-window)
- Online n-gram bias (trigram model built on already-scored tokens)
- Test on best available checkpoint
- Expected: -0.005 to -0.015 bpb as pure eval-time gain

### exp14: DenseFormer variant (replace U-Net skips)
- Already tested at 500 steps (R3 exp8), matched SOTA stack
- Run at 7000 steps to compare with U-Net skips
- Expected: smaller artifact, comparable bpb

### exp15: BAQ mixed-precision (post-training)
- Per-layer bit allocation based on Hessian sensitivity
- Apply to any trained model alongside OptRot+GPTQ
- Expected: -0.003 to -0.008 bpb over uniform INT6

---

## 8. Competition Intelligence

**Current merged SOTA:** 1.1147 bpb (PR #1019, Mar 30)
**Best open PRs:** ~1.09-1.11 bpb (non-TTT); ~0.93-1.09 bpb (TTT, some disputed)
**Competition deadline:** April 30, 2026
**Our best local result:** 1.2022 bpb (int6, 7000 steps, single 3070)
**Gap to SOTA:** ~0.088 bpb — expected to close significantly on 8xH100 with full data

**Key learnings from competition evolution:**
- The 10-min/8xH100 budget gets ~7000 steps at 83ms/step
- The throughput tax: each 1ms overhead costs ~7 steps = ~0.007 bpb
- Post-training techniques (GPTQ, OptRot, SLOT) are "free" — no training time cost
- EMA + XSA interaction is critical: EMA without XSA collapses

---

*References: See parameter-golf-analysis.md for full paper citations and competition PR links.*
*Experiment data: experiments/results_summary.txt (R1), experiments/r2/ (R2), experiments/r3/ (R3)*
