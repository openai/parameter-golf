# Parameter Golf — Full Technique Catalog

## Table of Contents
1. [Quantization & Compression](#1-quantization--compression)
2. [Architecture](#2-architecture)
3. [Training Protocol](#3-training-protocol)
4. [Legal TTT Protocols](#4-legal-ttt-protocols)
5. [Custom Tokenizers (Path C)](#5-custom-tokenizers-path-c)
6. [Agent System Prompts](#6-agent-system-prompts)

---

## 1. Quantization & Compression

### Int6 + zstd / brotli / lzma [MEASURED]
- Per-row symmetric int6, single fp16 scale per row
- ~25% artifact reduction vs int8 at ~0.0018 bpb cost on 19M-param models
- Source: PR #70, #78, #65 — now universal at frontier

### GPTQ-lite clip search [MEASURED] (arxiv 2210.17323)
- Per-weight reconstruction minimization via Hessian-guided rounding
- "Lite" = grid search over {0.999, 0.9995, 0.9999, 0.99999, 1.0} picking min-MSE per row
- PR #374 used this → contributed to 1.1228
- Cost: ~30–90s post-training, zero eval cost
- **Status: MANDATORY**

### AR Self-Generated GPTQ Calibration [VERIFIED]
- Feed the model's own generations back as calibration data (vs random training data)
- PR #1019 gained +0.008 bpb on top of base GPTQ
- **Status: MANDATORY**

### Progressive QAT Schedule [MEASURED]
- PR #374 recipe: warmdown3500 + QAT threshold 0.15 (STE activates in last 15% of training)
- ⚠️ **P1 bug risk:** `torch.compile` constant-folds class attributes. Always audit compiled graph. Move flag to `torch.Tensor` buffer if quant op missing.
- **Status: MANDATORY, audit for P1**

### Mixed int5 / int6 / fp16 by Sensitivity [MEASURED]
- PR #65, #180, #219: int5 for MLP weights (1.88× zstd ratio, tolerance high), int6 for attention (1.51× ratio, precision-sensitive), fp16 for embeddings and last-layer key projections
- int5 MLP saves ~1.86 MB vs uniform int6, funding a 10th layer
- **Status: MANDATORY**

### LZMA vs zstd vs brotli [MEASURED]
- PR #549 uses lzma on int6 weights with GPTQ-lite
- Kai's measurement: brotli-11 on int6 weights ≈ 4.96× total pipeline compression
- LZMA typically edges brotli by 3–8% on quantized neural weights
- **Status: try all three at final quantization, pick the best**

### DeepCABAC / NNCodec entropy coding [HYPOTHESIS]
- Context-adaptive binary arithmetic coding tuned for neural weight distributions
- General ML: 2–3× better than zstd on INT8 heavy-tailed weights
- Not tried on parameter-golf leaderboard
- Code cost: NNCodec reference ~2000 LOC; decoder only would be smaller
- **Status: one experiment in week 2 only if brotli/lzma leave headroom**

### LQ-LoRA decomposition [HYPOTHESIS] (arxiv 2311.12023)
- W = Q_lowbit(W - L·R) + L·R where LR is low-rank fp16
- Significant at int2/int3; small gain at int6
- **Status: USE at int4 if attempting sub-int5 ablation; skip at int5/int6**

### Non-power-of-2 bins + rANS [HYPOTHESIS]
- Quantize to N levels where N ≠ power of 2 (e.g., 40 levels ≈ 5.32 bits); encode via range ANS
- PR #1683 (pending) claims "ANS Hybrid" at int4; no measured bpb yet
- **Status: try at int4 if int6 headroom exhausted**

### Hadamard rotation / QuaRot [DE-PRIORITIZE]
- PR #586: near-zero marginal gain when full GPTQ already applied
- Issue #140: "substitutes with GPTQ at int6 — near-zero marginal gain"
- **Status: SKIP — not worth code bytes**

### VPTQ / QuIP# / AQLM vector quantization [HYPOTHESIS]
- Codebook overhead becomes significant fraction of 10–20M param model
- Not tried on leaderboard as of issue #140
- **Status: DEPRIORITIZE — GPTQ at int5 already gets most benefit**

---

## 2. Architecture

### Standard Frontier Shape [MEASURED]
- **11L / 512d / MLP 3× / GQA (8q, 4kv) / tied embeddings**
- Near-universal at frontier since PR #198
- Tied embeddings save ~0.5–1 MB
- Kai's 768d/11L/MLP4× carries 2.25× the parameter density → why artifact is 42MB
- **Downsizing to 512d/MLP3× is the single most important architectural change**

### Muon Optimizer [MEASURED]
- Newton-Schulz orthogonalized momentum for 2D matrix weights
- Settings from PR #549 (proven):
  - `MATRIX_LR=0.025, SCALAR_LR=0.025, TIED_EMBED_LR=0.035`
  - `MUON_MOMENTUM=0.99, MUON_MOMENTUM_WARMUP_START=0.92`
  - `MUON_MOMENTUM_WARMUP_STEPS=1500, MUON_WD=0.04, ADAM_WD=0.04`
  - `WARMDOWN_ITERS=3500`

### Parallel Muon + Parameter Banking [MEASURED]
- PR #399: 4 contiguous 3D `nn.Parameter` banks replace 66 separate `nn.Linear` weights
- Enables batched NS orthogonalization via `torch.bmm`; DDP removed for banks
- async reduce-scatter → local NS → async all-gather
- 83.3 ms/step vs ~85 ms baseline
- **Status: MANDATORY if using Muon**

### XSA — eXtended Sparse Attention [VERIFIED] (arxiv 2603.09078)
- Removes self-value-bias from attention output via orthogonal projection
- Applied ALL layers (PR #1019's key insight vs PR #265's last-4 approach)
- Zero parameters, ~2 ms/step overhead with GQA-aware implementation
- **Status: MANDATORY — use XSA-all, not XSA-last-4**

### Depth Recurrence / Layer Tying [MEASURED — with heavy caveats]
- PR #319, #363, #316: 3+ cycles FAILS at 512d. Quant error amplifies ~900×. 2× recurrence → 0.09 bpb cost
- **Kai's [3,4,5]×2 = P3 risk. Remove unless A/B post-quant shows it wins.**
- EMA combined with recurrence = DISASTER (P2 pattern)

### BigramHash + SmearGate + OrthoInit [MEASURED]
- PR #135: 10240-bucket bigram hash + dim=128 projected to 512 = -0.001–0.008 bpb
- PR #212: SmearGate **without** OrthoInit HURTS. Co-dependency is real.
- PR #609: TrigramHash without gating hurts +0.0049. Needs Engram-style gating.
- **Status: MANDATORY. Use BigramHash(≥3072, dim≥112) + OrthoInit + SmearGate together or not at all.**

### Partial RoPE (16/64 dims) + LN Scale (1/√(L+1)) [MEASURED]
- PR #287, PR #1675 — zero new parameters — worth ~0.003–0.005 bpb combined
- **Status: MANDATORY**

### VE128 — Value Embeddings (layers 9–10) [MEASURED]
- PR #379, #414, #549 — small value-embedding table on last two layers
- ~-0.002 bpb gain, marginal artifact cost
- **Status: RECOMMENDED**

### EMA(0.997) + SWA [MEASURED]
- PR #374 (EMA introduction) + PR #549 (SWA every 50 steps in warmdown)
- **WARNING:** EMA combined with recurrence = 1.42 bpb catastrophe (P2)
- **Status: MANDATORY for non-recurrent stacks. FORBIDDEN if recurrence active.**

### LeakyReLU(0.5)² Activation [MEASURED]
- Single-line change: `F.leaky_relu(x, 0.5).square()` instead of `relu(x).square()`
- -0.003 bpb on non-recurrent stacks
- Source: PR #493, PR #518 (1.0622 record stack), PR #549
- **Status: MANDATORY**

### QK-Gain [MEASURED]
- PR #1176 uses "QK-Gain 4.0" as part of 1.0914 standard-tokenizer best
- Mechanism: scale Q·K before softmax by a learned or fixed gain
- Test gain ∈ {2, 4, 5.25, 8}
- **Status: RECOMMENDED**

### Attention Output Gate [MEASURED]
- PR #1667 (MarioPaerle) used it to hit 1.07139 with SmearGate + Legal TTT
- **Status: RECOMMENDED for Path B stack**

### SLOT — Single Learnable Delta Vector [MEASURED]
- PR #1084: 512-dim delta added to last hidden layer, optimized per-batch during eval
- -0.0008 on PR #549 stack
- Virtue: doesn't touch quantized block weights → avoids GPTQ corruption
- **Status: STACK with SGD TTT. Complementary.**

### Muon Weight Decay [MEASURED]
- PR #60: WD=0.04 → 1.2160 to 1.2094. Smaller weights compress better.
- PR #1218: WD=0.085 + simplification beats complex stacks
- **Status: MANDATORY. Consider WD=0.05–0.09 ablation.**

---

## 3. Training Protocol

### Sequence Length 4096 + torch.compile [MEASURED]
- seq_len=4096 vs 1024 → 71 ms/step vs 43 ms but quality improvement dominates
- **Status: MANDATORY**

### 3-Seed Discipline [VERIFIED]
- Required for record track. std ≈ 0.0005–0.0015 on modern stacks
- 10 unused training shards cost ~0.02 bpb — use full data
- **Status: MANDATORY. Seeds: 1337, 42, 2025**

---

## 4. Legal TTT Protocols

### PR #461 Canonical Recipe [VERIFIED]
```
Val tokens split into 1,893 non-overlapping 32,768-token chunks.
For each chunk:
  SCORE: sliding-window eval under torch.inference_mode(), stride=64.
         Gradients disabled; in-place weight mutation forbidden at the API level.
  TRAIN: SGD(lr=0.002, momentum=0.9) on the already-scored chunk.
         3 epochs, all blocks unfrozen (freeze=0 beats freeze=2 by 0.0004),
         cosine LR decay, grad clip 1.0.
Last chunk is scored but never trained on.
Chunk N is scored by model adapted on chunks 0..N-1 only.
```
- Measured gain on PR #549 base: -0.0025 bpb (1.1218 → 1.1194)
- Eval time: ~410s for 1893 chunks × 3 epochs on 8×H100 (fits in 600s budget)

### LoRA TTT [MEASURED]
- PR #548 @LoquiAuris: 1.0865 via batched per-document LoRA TTT (rank 8–16 adapters)
- Virtue: small state per document, easier to batch; avoids GPTQ weight corruption
- **Status: Try LoRA TTT first (lower risk); full-weight TTT if LoRA plateaus**

### LaCT — Large Chunk TTT [HYPOTHESIS] (arxiv 2505.23884, ICLR 2026 Oral)
- Document-sized chunks → 70% GPU utilization vs <5% for per-token
- Uses Muon as fast-weight optimizer
- Issue #140 est: 0.002–0.008 bpb over current TTT approaches
- **Status: try in week 2 as TTT upgrade**

### TTT × Quantization Interaction [MEASURED]
- PR #601: Full GPTQ + TTT conflicts — updating quantized weights can corrupt GPTQ Hessian
- **Solutions:**
  1. Un-quantize at eval-start (fp16) → TTT → no re-quant (eval is ephemeral) ← **default**
  2. LoRA TTT on top of quantized base (weights stay quantized; only LoRA moves)
  3. SLOT (single last-layer delta; base stays quantized)

### Invalid TTT Patterns
- Multi-pass TTT (PR #573 @Sarimsaljook) — CLOSED as invalid. Do not resubmit.
- Paid-prefix validation storage (#168) — banned
- Error correction (#108) — banned
- Val-only training — banned

---

## 5. Custom Tokenizers (Path C)

### SP1024 Baseline [VERIFIED]
- Vocab=1024, tied embeddings, embed table = 1024×512×2 ≈ 1 MB
- bytes/token ≈ 3.5–4.2 on FineWeb

### SP4096 / SP8192 [MEASURED]
- PR #1218: SP4096 + 4× MLP + high WD = strong simple recipe
- PR #1688, #1689: SP8192-based, 1.080885 and 1.0822 (both open PRs)
- SP8192 × 512 × 2 = 8 MB embed table — crowds weight budget; requires embedding factorization

### Casefold Tokenizer [MEASURED]
- PR #1670 @dexhunter V4: 1.05970 (3-seed mean)
- Mechanism: collapse case-equivalent tokens + case bit in byte-accounting
- Legality hazard: must re-derive `build_sentencepiece_luts()` and prove byte counts match raw UTF-8 stream of canonical `docs_selected.jsonl`
- **Status: Path C primary candidate. Fork dexhunter's PR #1670.**

### Scylla / TokenMonster [MEASURED]
- PR #1184 @icryo: 998-vocab TokenMonster fork + Full GPTQ + XSA-all → **0.9485**
- TokenMonster ungreedy BPE claims 35% better compression vs standard BPE
- Higher DQ risk — requires very rigorous byte-count proof
- **Status: Path C secondary**

### Mandatory Pre-flight for ANY Tokenizer Change
1. Re-derive `build_sentencepiece_luts()` from scratch for the new tokenizer
2. Prove `decoded(token_ids) == original_utf8_stream` for 1000 random validation docs
3. Prove `bytes_per_token` sums to true UTF-8 byte count (not artificially low — the P4 DQ trap)
4. Document proof in `analysis/tokenizer_legality.md`. @critic must sign off.

---

## 6. Agent System Prompts

### @forge (baseline + harness)
Responsible for reference training harness. Gold standard: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Key invariants:
- `torch.compile()` must not constant-fold QAT branches — move flags to buffers
- Seed handling: accept SEED env var, set `torch.manual_seed`, `cuda.manual_seed_all`, `numpy`, Python `random`
- Never modify `data/download_hf_docs_and_tokenize.py` without escalating to @tokenizer-hacker AND @critic

### @shrinker (quantization + artifact)
Priority stack (in order):
1. int5 MLP + int6 attention + fp16 embeddings (MANDATORY)
2. GPTQ-lite per-row clip search (MANDATORY)
3. AR Self-Generated GPTQ calibration (MANDATORY)
4. YAQA rounding (arxiv 2505.22988) — optional, ~80 LOC delta vs GPTQ
5. NNCodec entropy coding — optional one-shot test

Reject result if `quant_gap > 0.010`.

Forbidden: BitNet/ternary from-scratch, Monarch matrices, CERWU, Hadamard rotation standalone.

### @ttt-engineer (Legal TTT)
Implements ONLY legal backward-looking score-first TTT per PR #461/#549.
Must verify all 6 TTT legality invariants before submitting any result.
Default recipe: 32,768-token chunks, SGD(lr=0.002, momentum=0.9), 3 epochs, all blocks unfrozen.
Report: pre_ttt_val_bpb, post_ttt_val_bpb, ttt_delta, ttt_wall_seconds, peak_GPU_memory_GB.

### @tokenizer-hacker (Path C)
Only invoked when @strategist explicitly triggers Path C.
Mandatory pre-flight: re-derive LUT, 1000-doc proof, document in `analysis/tokenizer_legality.md`, @critic sign-off.
Priority: Casefold V4 > Scylla/TokenMonster > SP4096.
Never: byte-level vocab=256, MegaByte/SpaceByte.

### @critic (adversarial red team)
Runs 8-pattern failure scan (P1–P8) against any work product.
Forbidden behaviors: hedging, symmetric analysis, deference to prior decisions.
Response format:
```
## CRITIC VERDICT: [APPROVE | BLOCK]
### Failure patterns triggered: [P1, P3, P7]
### Required fixes before proceeding: [...]
### One-line summary: [...]
```
Cannot be skipped or overridden.

### @submitter (end-stage packaging)
Invoked only at Phase 4. Runs full submission checklist before filing PR.
Must verify 3-seed t-stat, artifact size for all seeds, train/eval timing, standalone compile test.
Invokes @critic one final time before PR submit.
