# Plan: Comprehensive model improvement toward novel, leaderboard-worthy submissions

**Mode:** Planning only (no implementation in this document)  
**User request:** Synthesize **top-ranked `records/`** for implementable ideas, identify **performance gaps** vs our line (`2026-03-21_OrchestratedStack_10LInt5` / 10L Int5 family), and chart a path to something **novel and distinct**—not a naked copy of others.  
**Plan file:** `docs/PLAN-leaderboard-novel-improve.md`

---

## Phase -1: Context check

| Check | Result |
|-------|--------|
| Current fork baseline | **10L mixed int5/int6 + SmearGate + BigramHash(10240) + SWA + Muon/WD + zstd + sliding eval** (~**1.14x** class when fully run on 8×H100). |
| Leaderboard frontier (repo `README.md` + folders) | **~1.119**–**1.131** band dominated by **11L**, **XSA (partial / efficient)**, **EMA ± SWA**, **Partial RoPE + LN scale**, **GPTQ-lite**, **FlashAttention 3**, **LeakyReLU²**, **legal score-first TTT**, **Parameter Banking + Parallel Muon**, **VE / BigramHash 1536–2048**, **lzma** in some stacks. |
| Honesty bar | Copying a single upstream `train_gpt.py` without **documented deltas + ablations** is weak as “ours.” Novelty = **measurable contribution** (architecture, quant, opt, eval protocol) with **stats**, not folder rename. |

---

## Phase 0: Socratic gate (decide before coding)

1. **Scope of rewrite:** Stay on **10L int5/int6 codebase** and **increment**, or **rebase onto** a current **11L XSA** record and innovate from there? (Frontier math strongly favors **11L + XSA family**.)
2. **TTT appetite:** Willing to implement **eval-time** adaptation under **strict legal** rules (score-first, `inference_mode`, chunk budget) knowing **~400s+ eval** must stay under **10 min eval** total with training?
3. **Novelty thesis:** Pick **one primary hypothesis** (e.g. “activation + quant interaction,” “partial attention variant,” “compression after EMA,” “new legal TTT schedule”)—not five at once.
4. **Compute budget:** How many **8×H100 × 600s** runs can you afford for ablations + 3 seeds?

**Default if unanswered:** Rebase onto **11L XSA + int6 + zstd** line (not 10L fork), pick **one** train-side and **one** post-train or eval-side axis for novelty, require **3 seeds** before any SOTA claim.

---

## Overview

**What:** A structured gap analysis between **our stack** and **top records**, a **transportability matrix** (what we can port vs what needs upstream architecture), and a **research roadmap** ending in a **hypothesis-driven** submission story.

**Why:** The leaderboard moved from “int5 10L + Bigram 10240” to **11L + XSA + EMA + partial RoPE + GPTQ-lite + FA3** and beyond; **1.55 `running_bpb` mid-sliding-eval** on a partial run is **not** comparable to **1.12** final numbers—convergence, quant roundtrip, and **full eval completion** dominate.

**Success criteria:**

- [ ] Written **delta table**: our `train_gpt.py` vs chosen **parent record** (file path + line-level feature list).
- [ ] Each proposed change has **hypothesis**, **expected sign on val_bpb**, **byte/time risk**, **ablation order**.
- [ ] **Novel claim** is one sentence testable by **A/B** on same seed protocol.
- [ ] If targeting SOTA: **≥0.005 nats** vs target + **p < 0.01** (README), else **non-record** narrative with clear **negative result** value.

---

## Gap analysis: top records vs our current line

### A. SOTA cluster themes (from READMEs in repo)

| Theme | Example records | Our 10L orchestrated stack |
|--------|------------------|----------------------------|
| **Depth** | **11L** everywhere in top band | **10L** |
| **XSA (exclusive self-attn)** | Last 3–4 layers, efficient GQA | **Absent** |
| **EMA** | decay 0.997, often + tight SWA | **SWA only** (no EMA) |
| **Partial RoPE + LN scale** | 16/64 dims, 1/√(layer+1) | **Full RoPE** (typical 10L record) |
| **Activation** | **LeakyReLU(0.5)²** cited **~0.003 bpb** in top record | **ReLU²** |
| **Quant post-process** | **GPTQ-lite** row clip search | **Mixed int5/6 + row scale** (no percentile search) |
| **Compression** | zstd-22, **lzma** in some | **zstd** ✓ |
| **Optimizer / infra** | **Parameter Banking + Parallel Muon**, FA3 | **Standard Muon** in fork |
| **TTT** | **Legal score-first** chunk TTT, large eval budget | **None** |
| **Bigram table** | **1536–2048** buckets common in 11L line | **10240** (10L record) |

### B. Performance gap (conceptual)

- **~0.02–0.03+ bpb** separates **~1.14** from **~1.12**; closing that requires **stack upgrades**, not one knob.
- **Sliding eval** alone does not close the gap if **pre-quant / int6** quality lags (see records that report **pre-quant → sliding** breakdown).

### C. “No copy” / distinctiveness

| Approach | Distinct if… |
|----------|----------------|
| Reimplement upstream verbatim | **Low** — cite parent PR; use as **baseline**. |
| **Single well-isolated change** + ablations on same parent | **Medium** — e.g. new **activation**, **QAT schedule**, **GPTQ-lite variant**, **XSA depth/count** sweep. |
| **New combination** with measured **interaction** (e.g. LeakyReLU² × GPTQ-lite clip policy) | **Higher** — publish **factorial** or **sequential ablation** table. |
| **New legal TTT schedule** or **chunking** (still FAQ-compliant) | **High risk / high reward** — heavy eval engineering. |

---

## Recommended strategic fork (not prescriptive)

1. **Parent codebase:** Choose one **11L** record as **merge base**, e.g. **`2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248`** or **`2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`** (read **Late QAT + compile** caveat in that README—avoid dead-code paths).
2. **Reproduce parent** on 8×H100 (1 seed) → lock **val_bpb** and **bytes**.
3. **Novel layer:** Pick **one** primary idea, e.g.:
   - **Activation:** LeakyReLU² with **different slope** or **gated** variant **with ablation** vs parent.
   - **Quant:** **GPTQ-lite** variant (percentile set, per-tensor vs per-layer) **+** byte budget check.
   - **XSA:** **Which layers** / **last-N** sweep with **time** accounting (FA3 path).
   - **RoPE/LN:** **Partial dims** or **LN scale** schedule not identical to published 16/64 + 1/√(layer+1).
   - **TTT:** Only if willing to own **eval budget** and **legal** audit trail.
4. **Secondary knob:** At most **one** small hyper change (e.g. warmdown, Bigram buckets) with bytes tracked.

---

## Task breakdown (task_id, agent, INPUT → OUTPUT → VERIFY)

| task_id | name | agent | INPUT → OUTPUT → VERIFY |
|---------|------|-------|-------------------------|
| T1 | Parent record selection | `explorer-agent` | **IN:** Top 5 READMEs + our `train_gpt.py` diff. **OUT:** Chosen parent path + rationale. **VERIFY:** One paragraph “why this parent.” |
| T2 | Feature diff matrix | `project-planner` | **IN:** Parent vs our fork. **OUT:** Markdown table (XSA, EMA, RoPE, quant, TTT, FA). **VERIFY:** Every row cites file/section. |
| T3 | Novelty hypothesis | `product-manager` + you | **IN:** T2. **OUT:** Single thesis + falsifiable prediction. **VERIFY:** “If false, we’d see ___ on seed 42.” |
| T4 | Byte & time budget | `performance-optimizer` | **IN:** Parent artifact size, eval logs. **OUT:** Headroom for new params/code. **VERIFY:** Pre-run estimate + post-run `submission.json` match. |
| T5 | Implement v1 | `backend-specialist` | **IN:** T3–T4. **OUT:** One branch, minimal diff. **VERIFY:** Compiles; 1×H100 smoke; 8×H100 timed run. |
| T6 | Ablations | `test-engineer` | **IN:** T3. **OUT:** Table: parent, +A, +B, +A+B. **VERIFY:** Same eval protocol every row. |
| T7 | Multi-seed | `test-engineer` | **IN:** Best v1. **OUT:** ≥3 seeds, mean/std, significance note. **VERIFY:** README table + logs. |
| T8 | Record package | `documentation-writer` | **IN:** Challenge checklist. **OUT:** `README.md`, `submission.json`, `train.log`. **VERIFY:** Phase X in [PLAN-readme-records-workflow.md](./PLAN-readme-records-workflow.md). |

---

## Subagent matrix

| Phase | Agent |
|-------|--------|
| Record mining | `explorer-agent` |
| Sequencing / risk | `project-planner` |
| `train_gpt.py` | `backend-specialist` + `python-patterns` |
| Speed / memory / eval wall | `performance-optimizer` |
| Ablations / seeds | `test-engineer` |
| PR narrative | `documentation-writer` |
| Rules / TTT fairness | `product-manager` + human FAQ |

---

## Phase X: Verification checklist

- [ ] Parent + our delta documented; **novelty sentence** in README.
- [ ] **Sliding eval** (if used) completes; **final_int8_zlib_roundtrip_exact** (or record’s equivalent) logged.
- [ ] **bytes_total** ≤ 16,000,000; train **≤600s** on 8×H100 SXM for record track.
- [ ] SOTA: **0.005 nats** + **p < 0.01** or non-record justification.
- [ ] No **validation leakage**; TTT (if any) matches **legal** protocol in spirit of top TTT record.

---

## Risks & mitigations

| Risk | Mitigation |
|------|------------|
| **Late QAT + torch.compile** dead branch | Grep/trace compiled path; verify STE runs or remove claim. |
| **Eval timeout** (TTT + sliding) | Time phases separately; trim chunk count before full runs. |
| **OOM on 1×H100** | Not diagnostic of 8×; use **nproc=8** for real comparison. |
| **“Novel” rejected as incremental** | Front-load **ablation + interaction** evidence in README. |

---

## Next steps

1. Answer **Phase 0** (parent choice, TTT yes/no, thesis).  
2. Run **T1–T2** as read-only diff against chosen **`records/.../train_gpt.py`**.  
3. Execute **T5+** after plan approval—outside this planning document.

---

**End of plan**
