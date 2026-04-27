# Plan: SP8192 parent fork + one novel knob (legal TTT, budget-conscious)

**Mode:** Planning + partial repo artifacts (see §0)  
**User request:** Anchor on **SP8192**, **fork a parent** record, add **one novel axis**, target **competitive leaderboard** with **limited compute**; identify the **most compelling** single knob from **past entries**.  
**Plan file:** `docs/PLAN-sp8192-novel-knob.md`

---

## 0. Implementation status (repo)

| Item | Status |
|------|--------|
| **Parent P0** | `records/track_10min_16mb/2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT/` (README **~1.0810**). |
| **`train_gpt.py` format** | **LZMA-compressed** wrapper (`exec(L.decompress(B.b85decode(...)), ...)`). The published one-liner still embeds **hardcoded** `negative_slope=.5` (no env there). |
| **`train_gpt_plain.py`** | Decompressed trainer with **`LEAKY_RELU_SLOPE`** on the MLP (`≤0` → ReLU², default `0.5` → parent). **Use for A/B runs** (`NOVEL_KNOB.md`). |
| **Decompress helper** | `scripts/decompress_record_train_gpt.py` — writes plain Python **without executing** the trainer. |
| **Novel-knob docs + spec** | P0 folder: `NOVEL_KNOB.md`, `mlp_activation_spec.py`; tests: `tests/test_sp8192_p0_mlp_activation_spec.py`. |
| **Fork workflow** | Run **`train_gpt_plain.py`** for experiments; keep **`train_gpt.py`** wrapped unless you recompress for layout. |

Remaining **outside this repo** unless you add runs: S1 reproduction + S2 A/B on hardware, new record folder + logs.

---

## Phase -1: Context check

| Check | Result |
|-------|--------|
| Goal | **`val_bpb` ↓** on record track; **≤16 MB**, **≤10 min train** (8×H100), **≤10 min eval** total; **legal** TTT only. |
| Anchor | **SP8192** — current README **frontier** (~**1.081–1.086**) is dominated by **PR #1394**-line stacks + **parallel residuals** + **legal score-first TTT** + **QK-gain ~5–5.25** + strong **GPTQ / SDClip** variants. |
| Budget stance | **Reproduce parent first** (1 run), **one ablation** (1–2 runs), optional **third seed** only if Δ is promising — **no** multi-axis grids. |
| Novelty bar | One **hypothesis-driven** change with **A/B** on **fixed seed** and identical protocol — not a blind stack of features. |

---

## Phase 0: Socratic gate (locked for this plan)

| Question | Chosen answer |
|----------|----------------|
| Seq length | **SP8192** (anchor). |
| TTT | **Yes** — parent includes **legal score-first TTT**; do not invent TTT from scratch. |
| Parent selection | **Fork one folder** below; read its README + PR refs before editing. |
| Novel knob count | **Exactly one** train- or export-time axis (see §3). |
| Compute | **≥2** full runs minimum (reproduce + novel); **+1** seed if budget allows. |

---

## 1. Recommended parent (pick one — same PR family)

All are **SP8192** and **legal TTT**-class per `README.md` (newest first):

| Priority | Record folder | Score (README) | Notes |
|----------|----------------|---------------|--------|
| **P0** | `records/track_10min_16mb/2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT/` | **1.0810** | Newest top row; 3-layer recurrence + parallel residuals + QK 5.25 + legal TTT. **Default parent** if `train_gpt.py` is self-contained and documented. |
| **P1** | `2026-04-08_SP8192_ParallelResid_ScoreFirstTTT/` | 1.0822 | Slightly simpler headline than P0 — fallback if P0 is hard to run. |
| **P2** | `2026-04-06_SP8192_QK5_LegalTTT_1.0828/` | 1.0828 | Good if you want a **smaller** diff from “canonical” PR #1413 stack. |

**Action:** Open the **README** of your chosen folder, confirm **data/tokenizer pipeline (SP8192)**, **dependencies**, and **logged** `final_*` lines. **Freeze** that commit as **baseline** before any edit.

---

## 2. Most compelling single novel knob (recommendation)

### Primary recommendation: **LeakyReLU² MLP** (e.g. `negative_slope = 0.5`) vs **ReLU²**

**Why this is the strongest “one knob” story given leaderboard history**

1. **Named precedent:** The challenge already lists **“LeakyReLU² + Legal Score-First TTT + Parallel Muon”** (`2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`, **1.1194**) — i.e. **LeakyReLU² combined with a TTT stack** is an **accepted** competitive pattern, not a random tweak.
2. **Hypothesis is clear:** Non-zero gradient through **negative MLP pre-activations** **before** squaring can improve **optimization** and **quantization-friendly** weight surfaces (often cited in record READMEs).
3. **A/B is clean:** Same arch, same TTT legality, same seed — only **MLP activation** changes (if parent uses **ReLU²** today).
4. **Budget-efficient:** Usually a **small localized** change vs adding recurrence layers or new quant pipelines.

**When *not* to use it:** If the parent you fork **already** uses **LeakyReLU²** (or equivalent) everywhere relevant — then pick **Alternative A** or **B** below instead.

---

### Alternative A — **GPTQ-lite–style clip / percentile search at export** (no extra train)

**Precedent:** `11L EMA + GPTQ-lite` row; **ValCalib** / **self-gen calibration** records.  
**Pros:** Can improve **roundtrip `val_bpb`** without retraining if the parent export path supports it.  
**Cons:** More **engineering** in export; easier to break **exact roundtrip** checks if mishandled.

---

### Alternative B — **EMA weights for export** (train EMA shadow; quantize EMA vs raw)

**Precedent:** Multiple **EMA** rows (e.g. 11L EMA + GPTQ-lite).  
**Pros:** Smoother weights before int export.  
**Cons:** Larger patch + optimizer state; more hyperparameters.

---

## 3. Execution strategy (maximize limited runs)

| Step | Run | Purpose |
|------|-----|---------|
| S0 | **Smoke** (optional, `SMOKE_MODE`-style if parent supports) | Paths, compile, NCCL — **not** for leaderboard numbers. |
| S1 | **Parent reproduction** | Match published order of magnitude; save **full log**. |
| S2 | **Novel knob only** (e.g. LeakyReLU² **on** vs baseline **off**) | Same `SEED`, same env except the knob. |
| S3 | **Second seed** (optional) | Same winner config from S2 vs baseline — variance check. |

**Do not** change **TTT legality**, **eval stride**, or **quant scheme** in the same run as the activation experiment unless the parent README says they are coupled.

---

## Task breakdown

| ID | Task | Output | Verify |
|----|------|--------|--------|
| T1 | Select **one** parent folder (P0–P2) | Path in repo | README + `train_gpt.py` exist |
| T2 | Document **parent** hyperparams & activation type | Short table in your record README | Matches parent logs |
| T3 | Implement **one** knob (prefer **LeakyReLU²** toggle) | Diff isolated to MLP (or export path) | Unit test or shape test if feasible |
| T4 | Run **S1** reproduction | `train.log` | `final_*` lines captured |
| T5 | Run **S2** ablation | `train.log` | Δ `val_bpb` tabulated |
| T6 | **Bytes + time** check | `submission.json` | ≤16 MB; train/eval within limits |
| T7 | PR / submission narrative | README paragraph | Honest: parent + **one** delta |

---

## Phase X: Verification checklist

- [ ] **Legal TTT:** Implementation matches FAQ (score-first, `inference_mode`, chunk budget as per parent).
- [ ] **SP8192 data** cached and paths correct for the parent script.
- [ ] **Novel claim** = **one** sentence testable by **S1 vs S2**.
- [ ] **Statistics:** If claiming beat parent, follow README guidance (multi-seed / significance as applicable).

---

## References

- `README.md` — leaderboard ordering (authoritative).
- `records/.../2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md` — **LeakyReLU² + TTT** precedent.
- `docs/PLAN-lower-val-bpb.md` — broader option ladder.
- Chosen SP8192 parent README — **single source of truth** for PR stack (#1394, #1413, #1493, etc.).

---

**End of plan**
