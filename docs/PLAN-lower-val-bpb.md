# Plan: Lower `val_bpb` using themes from the current leaderboard frontier

**Mode:** Planning only (no implementation in this document)  
**User request:** Brainstorm ways to improve score (`val_bpb`, lower is better) informed by **recent top submissions**, then a concrete **plan of attack**.  
**Plan file:** `docs/PLAN-lower-val-bpb.md`  
**Context:** After merging `upstream/main` (April 2026), the record leaderboard **frontier** sits near **~1.08** (SP8192 stacks + recurrence + legal TTT + strong quant), vs older **~1.12–1.14** 11L / 10L lines documented in `docs/PLAN-leaderboard-novel-improve.md`.

---

## Phase -1: Context check

| Check | Result |
|-------|--------|
| Metric | **`val_bpb`** (bits per byte on FineWeb val; lower is better). |
| Hard constraints (record track) | **≤16 MB** artifact, **≤10 min train** on **8×H100**, **≤10 min eval** total (FAQ); tokenizer-agnostic evaluation. |
| Current README frontier (merged `main`) | Top rows: **~1.081–1.086** — **SP8192**, **parallel residuals**, **depth / mini recurrence**, **QK-gain ~5–5.25**, **legal score-first TTT**, **MuonEq-R**, **GPTQ-style clipping** (Hessian-aware / SDClip / embeddings), **progressive recurrence**, **4096 vocab** experiments, **AR self-gen GPTQ calibration + XSA** (~1.1147). |
| Your orchestrated 10L line | Competitive **10L int5/int6** story is **~1.14** class — large gap to frontier; **incremental knobs alone** unlikely to reach **~1.08** without **stack change** (seq length, recurrence, TTT, quant pipeline). |
| Related doc | `docs/PLAN-leaderboard-novel-improve.md` (gap vs 11L line; still useful for **themes**, but **numbers and top records** have moved — prefer **current `README.md` table**). |

---

## Phase 0: Socratic gate (answer before a big port)

1. **Target band:** Chase **~1.12** (older 11L+XSA) vs **~1.08** (current SP8192 frontier)? The latter implies **different tokenizer/seq pipeline** and **much heavier** engineering.
2. **TTT:** Willing to implement **legal score-first TTT** knowing **eval-time** dominates wall clock and must stay **FAQ-legal**?
3. **Seq length / vocab:** Accept **SP4096 / SP8192** + new data cache scripts and **4096 vocab** tradeoffs (params elsewhere), or stay **SP1024**?
4. **Merge base:** Fork from a **specific record folder** (e.g. top SP8192 README) vs evolve **OrchestratedStack 10L** only?
5. **Compute budget:** How many full **8×H100 × 600s** + eval runs for ablations and **multi-seed** stats (README SOTA rule)?

**Default if unanswered:** Pick **one** parent record in the **~1.09–1.11** band with **clear `train_gpt.py` + README** (manageable port), add **one** novel axis; defer **full SP8192 + TTT** until a **second** funding phase.

---

## Brainstorm summary (options from recent low scores)

Themes that **recur** in the **lowest** recent scores (see `README.md` leaderboard after merge):

| Theme | What winners did | Tradeoff |
|-------|------------------|----------|
| **Longer context (SP4096 / SP8192)** | Train/eval at **longer seq** than SP1024; often paired with recurrence / special residuals. | More memory, different **data + tokenizer** pipelines; not a drop-in on 2048-sp1024 code. |
| **Depth recurrence / parallel residuals** | **Looped blocks**, **parallel attention/MLP residual lanes**, **mini recurrence** on subset of layers. | Implementation complexity; must stay within **byte + time** caps. |
| **Strong quant story** | **All-int6 GPTQ**, **GPTQ embeddings**, **Hessian / SDClip**, **AR self-generated calibration** (ValCalib record ~1.1147). | Export path heavy; easy to break **roundtrip exact** checks. |
| **MuonEq-R / QK-gain** | Optimizer and attention scaling tweaks tied to deep stacks. | Needs stable parent **stack** to tune. |
| **Legal score-first TTT** | **Eval-time** adaptation that only uses **legal** information flow; large gains when done right. | **Eval seconds** budget; correctness vs FAQ. |
| **Higher WD / simplified blocks** | Some top runs **remove** SmearGate / value residuals / hash — **simpler** model + **higher WD** (e.g. 0.085–0.090). | Ablation needed; may conflict with “keep BigramHash” intuition. |

**Unconventional / high-risk:** Vocab 4096 + 4× MLP + simplified removals (see `2026-04-01_Vocab4096_MLPMult4_WD085`); **ternary / 1-bit** tracks sit in different leaderboards.

---

### Option A: Rebase onto **SP4096 / recurrence / MuonEq-R** parent (~1.09)

**Idea:** Use `2026-04-04_SP4096_DepthRecurrence_ParallelResid_MuonEqR` (or adjacent) as **merge base**; reproduce parent metric, then **one** new hypothesis (e.g. calibration clip policy, LeakyReLU², TTT schedule).

**Pros:** Moves you into **modern** architecture family without full SP8192 + TTT immediately.  
**Cons:** Still **major** port from 10L sp1024 orchestrated fork.  
**Effort:** High.

---

### Option B: **Val-calib GPTQ + XSA** line (~1.1147)

**Idea:** Parent `2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072` — self-gen calibration + XSA; closer to older 11L tooling than SP8192 monsters.

**Pros:** Strong **quant + attention** story; more contained than full recurrence + TTT stacks.  
**Cons:** Still not **~1.08**; requires **BigramHash 3072** + XSA discipline.  
**Effort:** Medium–High.

---

### Option C: **Incremental** on current **Orchestrated 10L** (LeakyReLU², GPTQ-lite clip, EMA)

**Idea:** Stay on **10L int5/int6**; ship **LeakyReLU A/B**, optional **EMA**, **export-time clip search** — align with `docs/PLAN-h100-novel-budget.md`.

**Pros:** Low port cost; clear **ablation** narrative.  
**Cons:** Ceiling **~1.14** class; **won’t** match **~1.08** frontier.  
**Effort:** Low–Medium.

---

### Option D: Full **SP8192 + legal TTT + parallel residuals** chase (~1.08)

**Idea:** Treat `2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT` (or #1394 lineage) as **target stack**.

**Pros:** **Same ballpark as SOTA** on README.  
**Cons:** **Maximum** engineering + eval risk; **not** a weekend project.  
**Effort:** Very high.

---

## Recommendation (planning default)

**Short term:** **Option C** if the goal is **shippable science** on your **existing** fork quickly.  
**Competitive:** **Option A or B** as **next** merge base once you choose a parent README and reproduce its **published** `val_bpb`.  
**Frontier:** **Option D** only with **explicit** time budget and **FAQ/legal TTT** review.

---

## Task breakdown (task_id → output → verify)

| task_id | Task | Owner | Output | Verify |
|---------|------|-------|--------|--------|
| T1 | Lock **target band** (1.14 vs 1.11 vs 1.08) | you | One paragraph goal | Matches budget + skills |
| T2 | Select **parent record** folder | you | Path under `records/track_10min_16mb/...` | README + `train_gpt.py` exist |
| T3 | **Delta table** vs parent (features, bytes, eval) | explorer / you | Markdown table | Matches parent README |
| T4 | **Port or branch** from parent | implementer | Clean fork | Runs smoke on 8×GPU |
| T5 | **One hypothesis** + ablation order | you | 3–5 runs max | Same seed protocol |
| T6 | **3 seeds** on best config | you | Mean ± stderr | README statistical rule |
| T7 | **Submission** PR | you | Record README + logs | `final_int8_zlib_roundtrip_exact` captured |

---

## Phase X: Verification checklist

- [ ] Parent **`train_gpt.py`** runs to completion on **8×H100** (or documented 1× debug path).
- [ ] **Sliding eval** + **roundtrip** lines logged for submission.
- [ ] **bytes_total** ≤ 16,000,000; **train + eval** within challenge time limits.
- [ ] Claims cite **correct** leaderboard row and **honest** comparison (SP1024 vs SP8192 not apples-to-apples).

---

## References (refresh after each `upstream` merge)

- `README.md` — **Leaderboard** table (authoritative ordering).
- `docs/PLAN-leaderboard-novel-improve.md` — older gap analysis (still useful for **technique** names).
- `docs/PLAN-h100-novel-budget.md` — **$25 / smoke** workflow for **incremental** 10L experiments.

---

**End of plan**
