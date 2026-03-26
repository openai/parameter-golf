# Plan: Innovation opportunities from records (score ↑, constraints respected)

**Mode:** Planning only (no implementation in this document)  
**User request:** Identify spots of innovation in existing **records** to improve **val_bpb** while staying within Parameter Golf **artifact**, **training-time**, and **evaluation** rules.  
**Plan file:** `docs/PLAN-records-innovation-score.md`

---

## Phase -1: Context check

| Check | Result |
|-------|--------|
| `CODEBASE.md` | Not present; assume **macOS** for local commands unless user specifies Linux CUDA host. |
| Prior docs | [PLAN-readme-records-workflow.md](./PLAN-readme-records-workflow.md) defines submission workflow; [WORKFLOW_DOCUMENTATION.md](./WORKFLOW_DOCUMENTATION.md) covers environment, orchestration, and **§ 5** quick index for this plan (changelog tracks doc updates). |
| Project type | **ML training / research** (not web/mobile). Route work per that plan’s subagent matrix. |
| Leaderboard reference | Current top entries (~1.14x **val_bpb**) stack quantization, depth, MLP width, routing, optimizers, compression, and **sliding eval**; see synthesis below. |

**Conversation priority:** User wants a **research plan** mapped to **record-proven** ideas—not a generic app scaffold.

---

## Phase 0: Socratic gate (answer before heavy implementation)

1. **Baseline:** Start from **Naive Baseline** (`records/.../2026-03-17_NaiveBaseline`) or from a **recent SOTA fork** (e.g. SmearGate + BigramHash family)? Starting point sets ablation budget.
2. **Primary axis:** Maximize **training-side** quality (arch + quant + opt), **evaluation-side** (sliding window, doc boundaries), or **test-time** (LoRA TTT within eval budget)—README allows eval innovation if rules on val leakage and eval time are met.
3. **Risk tolerance:** OK to depend on **zstd-22** + extra Python deps? OK to tune **tokenizer/dataset** (higher review burden per README)?
4. **Compute:** Only **8×H100 / 600s** runs, or also local MLX smoke + occasional long runs (non-record track)?

**Default assumptions if unanswered:** (1) fork nearest SOTA record README you can reproduce, (2) prioritize **training + quant + depth** first, **sliding eval** second, (3) avoid tokenizer changes in v1, (4) MLX smoke then 8×H100 for record attempts.

---

## Overview

**What:** A structured map of **where innovation already landed** in this repo’s `records/`, grouped by mechanism, with **constraint callouts** and a **task sequence** to turn ideas into measured improvements.

**Why:** Top scores are not single tricks—they are **stacks** (e.g. mixed int5/int6 + 10L + BigramHash(10240) + SWA + WD + sliding eval). Copying one knob without byte/phase budget awareness fails the 16MB / 10min / eval limits.

**Success criteria (measurable):**

- [ ] Each candidate change has a **hypothesis**, **expected val_bpb direction**, and **artifact size / time / eval** impact called out.
- [ ] Ablations are logged (table or bullet deltas) like `2026-03-20_10L_Int5MLP_MuonWD04_SWA50` README.
- [ ] If targeting leaderboard: improvement vs current SOTA ≥ **0.005 nats** with **p < 0.01** across seeds (per root README), unless systems-only waiver applies.

---

## Constraint summary (non-negotiable)

From [README.md](../README.md)—keep visible while designing experiments:

| Constraint | Implication for “innovation” |
|------------|-------------------------------|
| ≤ **16,000,000** bytes total (code + compressed weights) | Mixed precision / int5 MLP + int6 attn, zstd vs zlib, fewer embedding bits, **trade depth vs bytes** |
| Training ≤ **10 min** on **8×H100 SXM** (record track) | More layers/steps only if step time still fits wallclock |
| Eval ≤ **10 min** on **8×H100** | Sliding window and TTT cost time; must fit budget; document eval wall time |
| No val **cheating** | TTT only on already-scored tokens / doc rules per FAQ |
| SOTA PRs | **0.005 nats** + **p < 0.01** evidence (multi-seed) |

---

## Innovation map (from records): themes and example folders

Use these as **search patterns** when reading `records/track_10min_16mb/*/README.md`, not as a mandate to combine everything.

### A. Compression & quantization (frees parameters for depth/width)

| Idea | What records show | Watchouts |
|------|-------------------|-----------|
| **Per-row int6 + zstd-22** | Widespread (e.g. Int6 MLP3x + SmearGate README); zstd saves MB vs zlib | `zstandard` dep; verify roundtrip metric matches submission |
| **Mixed int5 / int6** | **10L Int5-MLP + BigramHash**: int5 for MLP, int6 for attention; funds 10th layer | Quantization penalty; ablate per component |
| **QAT (STE)** | **11L MLP3x + Int6 QAT**: fake quant during train | Training stability; last-layer fp16 tricks |
| **FP16 export for sensitive weights** | Embeddings / selected projections in fp16 while blocks quantize | Bytes—must still fit cap |

### B. Architecture & capacity

| Idea | What records show | Watchouts |
|------|-------------------|-----------|
| **3× MLP** | Multiple top entries (hidden 1536 @ dim 512) | Parameter and byte budget |
| **Extra layers (10L, 11L)** | Funded by aggressive quant (see int5 + 10L, 11L QAT) | Step time / 600s cap |
| **SmearGate** | Blends current + previous token embedding; tiny param cost | Composes with BigramHash |
| **BigramHash** | 4096 → **10240** buckets in best record; +dim 128 projected to 512 | Collision vs table size |
| **GQA / U-Net skips / RoPE** | Described in multiple READMEs | Match `train_gpt.py` snapshot in record |
| **Longer train seq / 4k** | `TrainingOptSeq4096`, `LongContextSeq2048` | Train time and memory |

### C. Optimization & regularization

| Idea | What records show | Watchouts |
|------|-------------------|-----------|
| **Muon + WD=0.04** | Repeatedly optimal band in READMEs | Pair with AdamW WD for embeds |
| **Muon momentum warmup** | 0.92 → 0.99 over ~1500 steps | |
| **SWA** | `swa_every=50`, `swa_start_frac` 0.4–0.5 | Better quant surfaces; checkpoint storage |
| **LR / warmdown** | Matrix vs tied embed LRs tuned per record | |
| **Pruning** | e.g. 3% magnitude pruning (10L Int5 record) | Interaction with quant |

### D. Evaluation protocol (no train change)

| Idea | What records show | Watchouts |
|------|-------------------|-----------|
| **Sliding window eval, stride=64** | **SlidingWindowEval**: large **post-quant** bpb gain with **same** train run; **pre-quant** nearly unchanged | Eval time ~70s+ vs ~16s; must stay under eval budget |
| **Doc-isolated scoring** | LoRA TTT README ablations: big gain from doc boundaries + stride | Defines “fair” eval interface |

### E. Test-time adaptation (within FAQ)

| Idea | What records show | Watchouts |
|------|-------------------|-----------|
| **LoRA TTT** | Per-chunk adaptation; README stresses **no leakage** across docs | Uses eval budget; must score only allowed tokens |

### F. Non-record / exploratory

| Idea | What records show | Watchouts |
|------|-------------------|-----------|
| **Unlimited train time** | `track_non_record_16mb` (e.g. 4h run) | Not leaderboard record track |

---

## Porter-style priority (suggested order of attack)

1. **Reproduce** one top record end-to-end; confirm **val_bpb**, **bytes**, **600s** on target hardware.  
2. **Lock eval protocol** (sliding stride, batch sizes) so comparisons are apples-to-apples.  
3. **Quant + compression** (int layout, zstd, QAT) to free **bytes** for depth/width.  
4. **Architecture** (MLP3×, layers, SmearGate, BigramHash size).  
5. **Opt + SWA + WD/LR** sweeps in small factorials.  
6. **Eval/TTT** only after train stack plateaus—measure eval seconds carefully.

---

## Task breakdown (task_id, agent, dependencies, INPUT → OUTPUT → VERIFY)

| task_id | name | agent | deps | INPUT → OUTPUT → VERIFY |
|---------|------|-------|------|-------------------------|
| T1 | Corpus of ideas | `explorer-agent` | — | **IN:** `records/track_10min_16mb/*/README.md`, leaderboard table. **OUT:** Table of techniques × record paths × reported Δ. **VERIFY:** Every row cites a file path under `records/`. |
| T2 | Constraint checklist | `project-planner` | T1 | **IN:** README FAQ. **OUT:** Per-idea flags: bytes, train time, eval time, statistical bar. **VERIFY:** Checklist covers all constraints in § Constraint summary. |
| T3 | Pick fork point | `backend-specialist` + user | T2 | **IN:** Repro capacity. **OUT:** Chosen baseline record + reason. **VERIFY:** One command reproduces baseline metrics within tolerance. |
| T4 | Byte budget model | `performance-optimizer` | T3 | **IN:** Param counts, quant scheme, compressor. **OUT:** Rough bytes for next experiment. **VERIFY:** Pre-run estimate + post-run `submission.json` `bytes_total` agree within explained gap. |
| T5 | Single-factor ablations | `test-engineer` | T4 | **IN:** Hypothesis list. **OUT:** Minimal ablation matrix (seed ≥3 for SOTA claims). **VERIFY:** Table like 10L Int5 README “Ablation Summary”. |
| T6 | Integrate best stack | `backend-specialist` | T5 | **IN:** Winning ablations. **OUT:** Single `train_gpt.py` snapshot for record folder. **VERIFY:** Runs in 600s; artifact ≤16MB; logs `final_int8_zlib_roundtrip` / project metric lines. |
| T7 | Record package | `documentation-writer` | T6 | **IN:** [PLAN-readme-records-workflow.md](./PLAN-readme-records-workflow.md) Phase X. **OUT:** `README.md`, `submission.json`, `train.log`. **VERIFY:** Root README submission list complete. |
| T8 | PR & significance | `documentation-writer` + `test-engineer` | T7 | **IN:** SOTA margin rule. **OUT:** PR text with p-value / t-stat or waiver category. **VERIFY:** Meets README acceptance rules. |

---

## Subagent assignments (summary)

| Role | Use for this plan |
|------|-------------------|
| `explorer-agent` | Mine `records/` for techniques and deltas |
| `project-planner` | Phasing, dependency order, constraint matrix |
| `backend-specialist` | `train_gpt.py` changes, env vars, integration |
| `performance-optimizer` | Step time, memory, eval wall time, batching |
| `test-engineer` | Seeds, ablations, significance |
| `documentation-writer` | README + submission narrative |
| `security-auditor` | Light pass on new deps (e.g. `zstandard`) |
| `product-manager` | “Spirit of challenge” / external compute boundaries |

---

## Phase X: Verification checklist

### A. Rules (README)

- [ ] `bytes_total` ≤ 16,000,000; `train_gpt.py` artifact story clear
- [ ] Record track: training ≤600s on 8×H100 SXM in logs
- [ ] Evaluation phase ≤ eval time limit; sliding/TTT documented
- [ ] SOTA: Δ ≥ 0.005 nats vs target baseline with p < 0.01 (or waiver path)

### B. Science quality

- [ ] Ablations distinguish **train** vs **eval** contributions (cf. SlidingWindowEval vs Int5 README)
- [ ] Multi-seed means reported with std

### C. Reproducibility

- [ ] Record folder contains `README.md`, `submission.json`, `train.log`, `train_gpt.py`
- [ ] Command block matches log paths and env

---

## Risks & mitigations

| Risk | Mitigation |
|------|------------|
| Confound sliding eval with train gains | Report **pre-quant** and **post-quant** metrics; cite SlidingWindowEval-style separation |
| Byte budget breaks when adding layer | Pre-calculate with int5/int6 mix and zstd |
| Step time regression | Profile early; reduce batch or seq if needed |
| Tokenizer change for “free” bpb | Defer; extra verification burden |

---

## Next steps

1. Answer **Phase 0** (baseline fork, axes, tokenizer risk, compute).  
2. Run **T1–T3** as a read-only pass through 3–5 READMEs at the Pareto frontier (top of leaderboard).  
3. Run `/create` or direct implementation when ready—**after** you approve this plan.

---

**End of plan**
