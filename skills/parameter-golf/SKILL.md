---
name: parameter-golf
description: >
  Comprehensive strategy guide, adversarial critic, and technique oracle for the OpenAI Parameter Golf
  competition (train best LM fitting in 16MB). Invoke this skill whenever the user asks about:
  parameter golf strategy, val_bpb improvement, artifact size reduction, technique selection (GPTQ,
  XSA, TTT, BigramHash, Muon, LeakyReLU², int5/int6, QAT, Parallel Muon), experiment design, phase
  planning, TTT legality, quantization choices, leaderboard SOTA, 3-seed significance, submission
  checklist, agent routing, compression pipeline, or anything related to train_gpt.py. Also trigger
  when the user mentions bpb, artifact bytes, val_bpb, warmdown, SWA, EMA in a training context,
  depth recurrence risks, or asks "what should I work on next" in a competition ML context. This skill
  knows Kai's current state (1.0882 val_bpb, 42.2MB artifact), the 8 mandatory failure patterns, the
  prioritized technique catalog, and the full compression pipeline spec.
---

# Parameter Golf — Omniclaw Strategy

## Hard Constraints (non-negotiable)
- **Artifact** = `len(train_gpt.py) + len(model.int8.ptz)` ≤ **16,000,000 bytes** (decimal, not MiB)
- **Training** ≤ 600s on 8×H100 SXM
- **Evaluation** ≤ 600s separately (TTT included)
- No validation data in training; only backward-looking score-first TTT
- Must beat SOTA 1.1147 by ≥0.005 nats (≈0.0072 bpb) at p<0.01, 3-seed mean

## Kai's Current State [MEASURED]

| Metric | Current | Required | Gap |
|--------|---------|----------|-----|
| val_bpb | 1.0882 | <1.05 | quality is fine |
| Artifact | 42.2 MB | ≤15.7 MB | **26.2 MB over** |
| Architecture | 768d/11L/MLP4×/recurrence[3,4,5]×2 | 512d/11L/MLP3× | must downsize |

**The bottleneck is compression, not quality.** Kai's 1.0882 pre-quant already beats the standard-tokenizer frontier (#1176 at 1.0914). The entire path to a valid record submission is getting this quality into 16MB. Do not spend Phase 1 time on modeling gains.

## Epistemic Labels — Apply to Every Claim
- **[MEASURED]** — confirmed from an actual competition run or public PR result
- **[VERIFIED]** — confirmed from merged PR, repo, or cited arXiv paper
- **[HYPOTHESIS]** — untested, may be wrong — label it and don't present as fact

---

## ⚠️ MANDATORY: 8-Pattern Failure Scan

Run this scan before recommending any technique or approving any phase transition. Block on any triggered pattern.

### P1 — Dead-code QAT (torch.compile constant-folding)
`torch.compile` constant-folds class attributes at first trace, killing QAT/STE branches silently. Logs say "QAT enabled" but compiled graph has only fp32 paths.
**Audit:** dump compiled forward graph via `torch._dynamo.explain(model)(x)`, grep for quant op. If missing, rewrite flag as a `torch.Tensor` buffer or use `torch.compiler.disable` on the branch.

### P2 — EMA + short-run recurrence disaster
EMA(0.997) over 10-min training pulls weights toward early unconverged states. On a recurrent architecture this compounds — the same averaged weights feed themselves twice per layer. Source: substack writer's 1.42 bpb catastrophic regression.
**Rule:** If recurrence OR weight tying is active, EMA is **FORBIDDEN**. Use SWA-only with `start_frac=0.4`.

### P3 — Depth recurrence × TTT × GPTQ triple conflict ← **CRITICAL FOR KAI**
Kai's current `[3,4,5]×2` recurrence + planned TTT + GPTQ post-training all active simultaneously = catastrophic. GPTQ error amplifies ~900× over 3 recurrent cycles (PR #363). TTT updates get 2× effect per recurrent step. Substack writer hit 1.34 bpb regression from exactly this combo.
**Rule:** `recurrence_cycles * gptq_active * ttt_active MUST be 0`.
**First action in Phase 1:** kill the recurrence, prove quality holds on non-recurrent 512d/11L stack, only then add GPTQ+TTT.

### P4 — Tokenizer byte-count bug (instant DQ)
Custom tokenizer changes `bytes_per_token` LUT incorrectly, artificially reducing val_bpb. OpenAI will DQ. Source: @valerio-oai Mar 27 sweep closed 33+ PRs.
**Audit:** `sum(bytes_per_token[tok] for tok in encode(doc)) == len(doc.encode('utf-8'))` on 1000 random docs from `docs_selected.jsonl`.

### P5 — TTT legality drift (pre-eval adaptation)
TTT accidentally updates weights using future validation tokens — classic data leakage. PR #573 (Multi-Pass TTT) closed as invalid for this reason.
**Audit:** SCORE step under `torch.inference_mode()`. `optimizer.step()` called only AFTER chunk is fully scored. No forward-looking adaptation.

### P6 — 3-seed variance > delta
`std ≈ 0.002`, claimed delta 0.0072 bpb, t-stat < 3.0 → fails p<0.01.
**Audit:** `t = (mean_bpb - target_bpb) / (std_bpb / sqrt(3))` with df=2. Require **|t| > 9.92**. If borderline, run 5 seeds; use 3 best.

### P7 — Artifact fragility (15.99 MB)
Artifact_size = 15,998,743 bytes. Any small code change pushes over the limit post-submission. PR gets closed on re-check.
**Rule:** Target ≤ **15,700,000 bytes** with 300 KB safety margin. Alert if > 15.85 MB.

### P8 — Reproducibility failure on reviewer hardware
Your pod hits 1.0899; OpenAI's verifier hits 1.1071. Nondeterminism from FlashAttention, TF32, data shuffling.
**Audit:** Disable TF32, set `torch.use_deterministic_algorithms(True, warn_only=True)`, pin PyTorch/CUDA/Triton versions in README. Run on fresh pod at end of Phase 4.

---

## Phase Plan (deadline 2026-04-30)

| Phase | Days | Goal | Exit Gate |
|-------|------|------|-----------|
| **0** | 0–1 | Baseline reproduction | Reproduce PR #549 (1.1194) ±0.002; confirm Kai's 1.0882 ±0.002; measure compression at each quant level |
| **1** | 1–3 | Compress into 16MB — no quality work | val_bpb ≤ 1.0950 AND artifact ≤ 16,000,000 bytes |
| **2** | 4–7 | Add Legal TTT → target 1.05–1.08 | val_bpb ≤ 1.0600 (target); ≤ 1.0800 minimum; artifact still ≤ 16MB; TTT eval ≤ 500s |
| **3** | 8–9 | Path C (Casefold tokenizer) — only if Phase 2 stuck above 1.05 | val_bpb ≤ 1.010 (Casefold) or ≤ 0.980 (Scylla) |
| **4** | 10–13 | 3-seed finals + PR submission | Full checklist in §Submission |

If Phase 1 exits above 1.10, there is no headroom for Phase 2 to reach sub-1.05 — stop and rethink (stronger quant: int4 MLP + rANS, or 9L shape).

## Three Strategic Paths

- **Path A** — Pure-neural, SP1024 tokenizer. Ceiling ~1.108–1.115. Insufficient alone.
- **Path B** — Path A + Legal TTT (primary). Ceiling ~1.05–1.08. Sub-1.05 realistic.
- **Path C** — Custom tokenizer (Casefold or Scylla). Ceiling ~0.92–0.98. Backup only if A+B stalls above 1.04 by Apr 24.

---

## Technique Priority

Full technique details with mechanisms and source PRs: see `references/techniques.md`.

### MANDATORY
| Technique | Delta | Source |
|-----------|-------|--------|
| Architecture: 512d/11L/MLP3×/GQA(8q,4kv)/tied embeddings | baseline | [MEASURED] |
| int5 MLP + int6 attention + fp16 embeddings | saves ~1.86 MB | [MEASURED] PR #65/#180 |
| GPTQ-lite clip search {0.999, 0.9995, 0.9999, 0.99999, 1.0} | mandatory | [MEASURED] PR #374 |
| AR Self-Generated GPTQ calibration | +0.008 bpb on top of GPTQ | [VERIFIED] PR #1019 |
| Progressive QAT schedule (audit for P1 bug) | mandatory | [MEASURED] PR #374 |
| XSA on ALL layers (not just last 4) | zero params | [VERIFIED] PR #1019 |
| Muon + Parallel Muon + Parameter Banking | 83.3ms/step | [MEASURED] PR #399/#549 |
| LeakyReLU(0.5)² instead of relu(x).square() | -0.003 bpb | [MEASURED] PR #493/#518 |
| BigramHash(≥3072, dim≥112) + OrthoInit + SmearGate | -0.001–0.008 bpb | [MEASURED] PR #135 |
| Partial RoPE (16/64 dims) + LN Scale (1/√(L+1)) | -0.003–0.005 bpb | [MEASURED] PR #287 |
| EMA(0.997) + SWA — **ONLY if no recurrence** | mandatory | [MEASURED] PR #374 |
| Sequence length 4096 + torch.compile | mandatory | [MEASURED] |
| Weight decay WD=0.04–0.085 (smaller weights compress better) | mandatory | [MEASURED] PR #60 |

### RECOMMENDED (strong, test before committing)
| Technique | Delta | Source |
|-----------|-------|--------|
| VE128 (Value Embeddings, layers 9–10) | ~-0.002 bpb | [MEASURED] PR #379/#549 |
| SLOT (single 512-dim last-layer delta) | -0.0008 bpb | [MEASURED] PR #1084 |
| QK-Gain (test ∈ {2, 4, 5.25, 8}) | depends | [MEASURED] PR #1176 |
| Attention Output Gate | used in 1.071 submission | [MEASURED] PR #1667 |

### SKIP — not worth the code bytes or risk
When recommending SKIP, cite the evidence. Evidence is listed per row.

| Technique | Reason | Evidence |
|-----------|--------|----------|
| Hadamard rotation standalone | near-zero marginal gain atop full GPTQ at int6 — GPTQ's per-row clip search already handles the outlier problem Hadamard targets | PR #586 scored 1.1365 WITH Hadamard; Issue #140: "substitutes with GPTQ at int6" [MEASURED] |
| OptRot pre-quantization | same root cause — a substitute for, not complement to, GPTQ | Issue #140 [MEASURED] |
| Monarch matrices | code budget too high, kernel complexity ~2000 LOC, uncertain gain at 10-20M params | no leaderboard evidence [HYPOTHESIS] |
| BitNet b1.58 from scratch | no small-model evidence at 10-min training; requires full retraining | no leaderboard evidence [HYPOTHESIS] |
| CERWU | unverifiable paper — do not cite or implement | [UNVERIFIABLE] |
| Byte-level vocab=256 | loses at this param budget; SP1024 dominates | no measured gain [MEASURED] |
| Depth recurrence [3,4,5]×2 | P3 risk — GPTQ error amplifies ~900× over recurrent cycles | PR #363 [MEASURED] |
| n-gram cache (without full-vocab normalization) | invalidated Mar 27, 33+ PRs closed by @valerio-oai sweep | Mar 27 2026 sweep [VERIFIED] |

---

## Compression Pipeline Spec

Execute post-training in this exact order:

```
1.  Apply EMA weights (if trained with EMA)
2.  Un-quantize any fake-quant state → fp32 master
3.  GPTQ-lite calibration on 8192 tokens of training data
    (AR self-gen variant: feed model's own generations as calibration)
4.  Per-layer bit allocation by sensitivity:
      MLP weights      → int5 per-row
      Attention weights → int6 per-row
      Embeddings (tied) → fp16
      LN scales, biases → fp16
5.  Per-row symmetric quant:
    scale = percentile_search(w, [0.999, 0.9995, 0.9999, 0.99999, 1.0]) → min MSE per row
6.  Pack int5 as 5-bit + int6 as 6-bit (no padding waste)
7.  Try all three codecs: brotli-11, zstd-22, lzma-9 — ship the smallest
8.  Final artifact = len(train_gpt.py.encode('utf-8')) + packed_weights_bytes
9.  Verify artifact < 16,000,000 bytes BEFORE any submission
```

Reject the result if `quant_gap = post_quant_bpb - pre_quant_bpb > 0.010` — try stronger QAT or smaller architecture.

---

## Legal TTT Protocol

From PR #461/#549. ALL six invariants must hold or **ABORT**:

1. SCORE step runs under `torch.inference_mode()` — no gradients possible at the API level
2. TRAIN step operates ONLY on tokens already scored in this eval session
3. Last chunk is scored but never trained on
4. Chunk N is scored by model adapted only on chunks 0..N-1
5. Evaluation order is FIXED (no re-ordering of val set)
6. Val tokens are NEVER written into the 16MB artifact

**Default recipe (PR #549):** 32,768-token chunks · `SGD(lr=0.002, momentum=0.9)` · 3 epochs · all blocks unfrozen · cosine LR decay · grad clip 1.0 · ~410s for 1893 chunks (fits in 600s budget).

**TTT × quantization:** Default to un-quantize at eval-start (fp16) → TTT → no re-quant. Avoids GPTQ Hessian corruption documented in PR #601.

**TTT upgrade candidates** (test ONE per week, in priority order):
1. LoRA TTT (PR #548, rank 8) — avoids weight corruption, achieved 1.0865 [MEASURED]
2. SLOT — stack on top of SGD TTT for additional -0.0008 [MEASURED]
3. LaCT (arxiv 2505.23884) — document-sized chunks, 70% GPU util, est -0.002 to -0.008 [HYPOTHESIS]
4. qTTT (PR #1683) — quantization-aware TTT [HYPOTHESIS]

---

## Submission Checklist

Before filing PR against openai/parameter-golf:

- [ ] 3 seeds {1337, 42, 2025} run on **single 8×H100 SXM Runpod pod**, same day
- [ ] `val_bpb mean ≤ 1.10749` (SOTA 1.1147 − 0.00721)
- [ ] `|t| > 9.92` with N=3, df=2, p<0.01 two-tailed
- [ ] `artifact_bytes ≤ 16,000,000` for ALL 3 seeds
- [ ] `artifact_bytes ≤ 15,700,000` recommended (P7 safety margin)
- [ ] `train_wallclock_seconds ≤ 600` for ALL 3 seeds
- [ ] `eval_wallclock_seconds ≤ 600` for ALL 3 seeds (TTT included)
- [ ] P1 audit passed: compiled graph contains quant op
- [ ] If tokenizer changed: byte-count proof in `analysis/tokenizer_legality.md`, 1000-doc round-trip test passing
- [ ] `train_gpt.py` self-contained, compiles from within record folder, no external script deps
- [ ] `README.md` with ablation table, lineage citations (PR numbers), reproducibility command, hardware spec, timings
- [ ] `submission.json`: `{name, github_id="kailean", val_bpb, artifact_size, prev_record=1.1147, delta_nats}`
- [ ] All 3 seed logs: `train.log`, `train_seed42.log`, `train_seed2025.log`
- [ ] Final @critic sign-off in PR description
- [ ] PR title: `Record: <Technique> val_bpb=<X.XXXX> (3-seed mean)`
- [ ] Branch: `record/<YYYY-MM-DD>_<short-tech>_<bpb>`
- [ ] Folder: `records/track_10min_16mb/YYYY-MM-DD_<TechName>_<val_bpb>/`

---

## Agent Routing

| Situation | Route to |
|-----------|----------|
| New technique request | @critic pre-flight → specialist |
| Compile / distributed error | @forge |
| Artifact > 16 MB | @shrinker |
| TTT regression | @ttt-engineer + @critic jointly |
| Tokenizer proposal | @critic FIRST, then @tokenizer-hacker |
| PR drafting | @submitter → @critic final sign-off |
| Any phase transition | @critic adversarial review FIRST |

Full agent system prompts in `references/agent-prompts.md`.

---

## Experiment Protocol

One experiment = one change = one git branch = one row in `experiments/log.csv`.

```
# Experiment NNN: <one-line hypothesis>
Single variable changed: <e.g., MLP ratio 3 → 2.5>
Base stack: <commit hash or branch>
Claim: <expected delta> [SOURCE: PR#N / arxiv / HYPOTHESIS]
Pre-flight critic review: [APPROVE|BLOCK], date, patterns checked
Run: seed, date, cmd, 8xH100 pod id
Result: pre_quant_bpb | post_quant_bpb | artifact_bytes | wall_time_s
Delta vs base: <number>
Decision: [KEEP|REVERT|ITERATE]
```

Compound stacks of 2+ techniques allowed only AFTER each component individually validated with ≥0.002 bpb gain AND @critic clears the interaction (P3 is common).

## Compute Budget (~$900 total at $20/hr 8×H100 SXM)
- Single 10-min train + 10-min eval: ~$6.67
- 3-seed record run: ~$10
- Use 1×H100 for Phase 1 compression smoke tests (deterministic, no distributed needed, ~$2/hr)
