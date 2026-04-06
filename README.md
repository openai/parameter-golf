# Parameter Golf — Project Context & Battle Plan

## What This Is
OpenAI's **Model Craft: Parameter Golf Challenge** — train the best language model under:
- **16 MB** total artifact size
- **10 minutes** on 8×H100 GPUs
- Metric: **bits-per-byte (BPB)** on FineWeb validation set

## Current Status
- **Our best score**: 1.4031 BPB
- **Leaderboard leader**: ~0.9581 BPB (PR #761)
- **Gap to close**: ~0.45 BPB

## Hardware
- **Primary**: DGX Spark (128GB unified memory) — inference server + local experiments
- **Competition eval**: 8×H100 (OpenAI's infra)
- **Mini PC**: orchestration
- **Windows PC**: management

---

## Leaderboard Intelligence (as of April 2026)

### Top 3 Entries

#### 1. PR #761 — 0.9581 BPB (LEADER)
- 10 unique layers, d=512, 8 heads, MLP 3x (1536)
- **No recurrence** — pure depth
- **Int6 quantization** (post-training, per-row scaling) + **zlib compression**
- **Score-First Test-Time Training (TTT)**: Adapts weights via SGD on already-scored causal context during eval
- **Multi-order N-gram Backoff Cache**: Mixes neural predictions with cached 5-gram history
- **Sliding Window Eval**: stride=64, near-full context for every scored token
- Key insight: TTT + n-gram cache accounts for ~0.15-0.20 BPB of their advantage

#### 2. PR #1333 — 1.0766 BPB
- 6 unique layers × **2 recurrence loops** = 12 effective layers
- d=640, 10 heads
- **Mixed Int8/Int6 QAT** (curvature-weighted variance penalty, CROWN-Q)
- **SentencePiece 4096** vocab (shrinks embedding matrix, frees bytes for wider model)
- **Causal SLOT-16**: Score-first linear output tuning on previously scored tokens
- Parallel residuals for gradient flow across recurrent loops

#### 3. PR #270 — 1.1303 BPB (FarnsworthEngine)
- 11 unique layers, d=512, 8 heads
- **No recurrence**
- Int6 quantization
- **Score-First TTT** (43s of eval budget)
- SmearGate & BigramHash for local token routing
- SWA (Stochastic Weight Averaging) at end of training
- FlashAttention-3 + Muon optimizer with weight decay

### Key Patterns From Leaders
1. **Test-time training is the #1 differentiator** — every top entry uses it
2. **Int6 quantization** (not int8) — ~10.67M params in 16MB vs ~8M at fp16
3. **N-gram caching** at eval time — free BPB improvement, no param cost
4. **Score-first** approach — only train on already-scored tokens (causal legality)
5. **zlib compression** of quantized weights — squeezes extra params into 16MB
6. **d=512 is the sweet spot** — most entries converge on this width
7. **SentencePiece 4096** can free embedding budget for wider models

---

## Theoretical Framework

### Von Neumann Connection
- **Self-reproducing automata** → weight sharing / depth recurrence
- **Statistical brain** → low-precision training with redundancy (int6/int8)
- **Minimum description length** → the competition IS Kolmogorov complexity
- **Cellular automata** → simple recurrent rules generating complex behavior

### Kolmogorov Complexity
- Language modeling = compression (ICLR 2024)
- KoLMogorov Test (ICLR 2025): even 405B LLMs fail 78% of the time
- Our 16MB model = computable approximation within fixed description length
- Below self-reproduction threshold: maximize REUSE (weight sharing) not SIZE

### Key Papers
- "Language Modeling Is Compression" (ICLR 2024)
- "The KoLMogorov Test" (ICLR 2025)
- "Bridging Kolmogorov Complexity and Deep Learning" (2025)
- ALBERT (cross-layer parameter sharing)
- ModernALBERT (Mixture-of-LoRA on shared weights)
- Mamba-3 (complex-valued SSM, MIMO)
- Jetfire (INT8 QAT from scratch)

---

## Architecture Research Summary

### Weight Sharing / Recurrence
- ALBERT: single block shared across 12 layers, 90%+ param reduction
- ModernALBERT: adds LoRA experts on shared backbone, 30B token convergence
- Top competitor (PR #1333): 6 layers × 2 loops with parallel residuals
- Universal Transformer: cross-layer sharing outperforms standard transformer

### Alternative Architectures (from Grok, Gemini)
- **Mamba-3**: 2x more parameter-efficient than transformer at <10M scale
- **RWKV v5/v6**: competitive BPB, heavy regularizer prevents overfitting
- **Looped Mamba** (Grok's bet): 2 blocks × 10 loops + progressive expansion
- **Tiny MoE**: 16 experts × 1M params, top-2 routing
- **KANs, Neural ODEs, Hyena**: less proven at this scale

### Quantization
- **Int8 QAT** (Jetfire/SwitchBack): +0.05-0.1 perplexity penalty, 2x params
- **Int6**: leaders use this — ~10.67M params in 16MB
- **Per-block scaling + INT32 accumulation** required for stability
- **zlib compression** of quantized weights extends effective budget

### Test-Time Techniques (NEW — highest leverage)
- **Score-First TTT**: SGD on already-scored tokens during eval
- **N-gram Backoff Cache**: mix neural logits with cached n-gram history
- **Sliding Window Eval**: stride tuning for near-full context
- **Causal SLOT**: freeze main weights, optimize small delta on scored tokens

---

## Experiment Plan

### Phase 1: Weight Sharing + Recurrence (PRIORITY)
- Modify TinyGPT for N unique blocks × K loops
- Sweep: {3,4,5,6} blocks × {2,3,4} loops
- Add per-loop learned scale factors
- Add parallel residual connections across loops
- **Expected gain**: 0.10-0.15 BPB

### Phase 2: Quantization to Int6/Int8
- Implement QAT with per-row scaling
- Compare int8 vs int6 vs fp16 at same total artifact size
- Add zlib compression to artifact packing
- **Expected gain**: 0.03-0.05 BPB

### Phase 3: Test-Time Training (TTT)
- Implement score-first TTT (SGD on scored tokens during eval)
- Implement multi-order n-gram backoff cache
- Implement sliding window eval with stride tuning
- **Expected gain**: 0.15-0.20 BPB (THIS IS THE BIG ONE)

### Phase 4: Architecture Exploration
- Mamba at 16MB vs transformer baseline
- Hybrid Mamba-Transformer
- Looped Mamba (Grok's suggestion)
- **Expected gain**: 0.01-0.05 BPB (uncertain)

### Phase 5: Optimizer & LR
- Run LR finder on best config
- Compare Adam vs Muon vs Lion vs Sophia
- SWA at end of training (like PR #270)
- **Expected gain**: 0.02-0.05 BPB

---

## Scripts Available (from GPT/Perplexity)

1. `benchmark_tokenizers.py` — BPE/Unigram/ByteLevel comparison on FineWeb
2. `lr_finder_transformer.py` — Leslie Smith LR range test for ~4M param model
3. `sweep_small_transformer_optuna.py` — Optuna hyperparameter sweep under 16MB budget

All scripts use current PyTorch AMP APIs and are single-file, single-GPU.

---

## Delegation Map

| Tool | Best For | Current Tasks |
|------|----------|---------------|
| **Claude Code (DGX Spark)** | Execution, implementation | Run experiments, build repo |
| **Claude (chat)** | Strategy, synthesis, delegation | This document, prompts |
| **GPT/Codex** | Code generation | Weight-sharing TinyGPT, Mamba impl, QAT |
| **Gemini** | Research synthesis | Leaderboard intel, paper surveys |
| **Perplexity** | Quick factual search | Architecture comparisons |
| **Grok** | Contrarian ideas | Unconventional approaches |
| **Aristotle** | Formal proofs | Quantization bounds, packer correctness |

---

## Aristotle (Pending) — Proof Tasks

1. Quantization error bound: |x - q(x)| ≤ Δ/2
2. Artifact size legality: total_bytes ≤ 16MB under given hyperparams
3. Weight-sharing preserves universal approximation
4. Softmax mixing rule produces valid probability distribution
5. Int6 pack/unpack losslessness

---

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-04-05 | Prioritize weight sharing | Every top entry uses recurrence or deep unique layers |
| 2026-04-05 | Target int6 not int8 | Leaders use int6 + zlib, not int8 |
| 2026-04-05 | TTT is highest priority after architecture | 0.15-0.20 BPB gap attributed to TTT+ngram |
| 2026-04-05 | d=512 as default width | Consensus from leaderboard |
| 2026-04-05 | Byte-level default, explore SP4096 | PR #1333 uses SP4096 successfully |

---

## Next Session Checklist
- [ ] Review overnight experiment results
- [ ] Collect Aristotle proofs
- [ ] Implement TTT (Phase 3)
- [ ] Implement n-gram backoff cache
- [ ] Run Mamba comparison (Phase 4)
