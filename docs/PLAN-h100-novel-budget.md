# Plan: Novel features on 8×H100 under ~$25 compute budget

**Mode:** Planning + **implementation note** (below) — trainer behavior lives in code.  
**User request:** A **trainable model** direction for **8×H100** with **smoke + full** runs, **new/novel** elements that target **lower `val_bpb`**, while **minimizing spend** (~**$25**).  
**Plan file:** `docs/PLAN-h100-novel-budget.md`

---

## Phase -1: Context check

| Check | Result |
|-------|--------|
| Base trainer | `records/track_10min_16mb/2026-03-21_OrchestratedStack_10LInt5/train_gpt.py` — **10L**, mixed int5/int6, SmearGate, BigramHash, SWA, zstd, sliding eval. |
| Already implemented hook | **`LEAKY_RELU_SLOPE`** env (default `0` = ReLU²; **`0.5`** = LeakyReLU²) — **zero extra runs** to “ship” a first novel axis; only **A/B** runs needed. |
| Budget reality | **$25** at typical **8×H100** cloud rates (~**$15–25+/hr** variable) ≈ **~1–1.5 hours** of pod time if dominated by GPU — fits **~2–4** end-to-end jobs if each is **~10 min train + eval**, or **1–2** full jobs plus **smokes** if eval is long / retries happen. |
| Goal metric | Minimize **`val_bpb`** (bits per byte); user wrote “rbpb” — interpret as **`val_bpb`**. |

---

## Phase 0: Socratic gate (answer before spending)

1. **Actual $/hr** on your provider for **8×H100**? (Sets exact run cap — plug into § Budget ledger.)
2. **Pod already has FineWeb cached?** If yes, skip download cost/time; if no, **first minutes** burn on I/O.
3. **Non-negotiable:** Must every run use **full sliding eval** + **full int6 roundtrip**, or is a **smoke** allowed with **`EVAL_STRIDE=1024`** (disjoint chunks) **only for compile/OOM checks** — **not** for leaderboard numbers?
4. **Novelty bar:** Is **“LeakyReLU² A/B on our stack + table”** enough for your narrative, or do you require a **second** mechanism in the same $25 window?

**Default if unanswered:** Assume **~$20/hr** → **~75 min** GPU budget; **smoke uses shortened wallclock + no/one val**; **full runs** use **production eval**; **primary novel** = **LeakyReLU²**; **second** = **only if first shows ≥0.002 bpb** on one seed.

---

## Overview

**What:** A **run schedule** and **feature bundle** that maximizes **information per dollar**: one **implemented** train-side novelty (**LeakyReLU²**), optional **post-train** ideas **deferred** until code exists, and **strict** smoke → full ordering to avoid burning full 10-minute jobs on trivial failures.

**Why:** $25 **cannot** support a full **11L XSA + EMA + TTT + GPTQ-lite** port **and** multi-seed grid; the plan **sequences** cheap checks first and **one** clear hypothesis test second.

**Success criteria (within budget):**

- [ ] **Smoke:** 8 processes start, **≥1 training step**, no immediate OOM, log shows **`leaky_relu_slope`** line.
- [ ] **Full A/B:** Same **`SEED`**, same env except **`LEAKY_RELU_SLOPE` ∈ {0, 0.5}`**, **final `val_bpb`** (and **artifact bytes**) recorded for both.
- [ ] **README row:** “Baseline relu² vs LeakyReLU² (0.5) on our stack.”
- [ ] **Stretch:** If Δ meets your internal bar, **third run** (second seed) on **winner** only.

---

## Novel feature bundle (prioritized for $25)

| Priority | Feature | Status in repo | Hypothesis | Risk |
|----------|---------|----------------|------------|------|
| **P0** | **LeakyReLU(0.5)² vs ReLU²** | **Implemented** (`LEAKY_RELU_SLOPE`) | Matches top-record motivation: gradient through negative pre-acts before **square**; may improve **quant-friendly** surfaces. | May **hurt** if stack was tuned for ReLU²; **must A/B**. |
| **P1** | **GPTQ-lite–style clip search** (extra percentiles / per-layer) | Not in orchestrated fork | Better int6 scales → lower **roundtrip bpb** at **same** train; **zero extra train** if done at export. | **Needs code**; debug time burns budget — **defer** unless P0 flat. |
| **P2** | **EMA (decay ~0.997)** + keep SWA | Not in orchestrated fork | Smoother weights before quant (seen in 11L records). | **Larger patch** + compile interactions; **high risk** for $25 unless you already merged parent. |
| **P3** | **Partial RoPE (e.g. 16/64)** | Not in orchestrated fork | Strong in 11L line; **not** free to port under time. | **Defer** to next funding. |
| **P4** | **Legal score-first TTT** | Not in orchestrated fork | Gains on top record; **~400s eval** — eats budget **fast**. | **Only** if P0 wins big and you add eval-time cap discipline. |

**Bundled “story” for submissions (honest):** *On the 10L int5/int6 orchestrated stack, we isolate **MLP activation (LeakyReLU²)** vs **ReLU²** under identical training and eval protocol; future work: export-time clip search.*

---

## Budget ledger (template — fill your $/hr)

Assume **T_full** = wall minutes per **full** job (train cap 600s + sliding eval + export). Assume **T_smoke** ≈ **3–8 min** if wallclock **60–120s** and eval reduced/disabled for smoke only.

| Line item | Count | Minutes each | Notes |
|-----------|-------|--------------|--------|
| Pod bring-up + `pip` + data (if any) | 1 | 10–30 | **Fixed cost** |
| Smoke **8×proc** | 1–2 | 3–8 | OOM/NCCL debug |
| Full run **A** (slope=0) | 1 | T_full | Baseline |
| Full run **B** (slope=0.5) | 1 | T_full | Novel |
| Retry (OOM / crash) | 0–1 | T_full | **Reserve** |
| Optional seed-2 on winner | 0–1 | T_full | If budget remains |

**Rule:** If **first smoke fails twice**, switch to **`nproc=1`** only to debug **imports/data paths**, then return to **8×** — do **not** burn **8× full** jobs on **dataset path** errors.

---

## Smoke-test protocol (8×H100)

**Purpose:** NCCL, CUDA, paths, **torch.compile**, **zstd**, **no OOM** — **not** trustworthy `val_bpb`.

Suggested env (or equivalent):

- `SMOKE_MODE=1` — **implemented** in `2026-03-21_OrchestratedStack_10LInt5/train_gpt.py`: skips **all** val during train and skips **int export + sliding final eval** (exits after SWA block).
- `MAX_WALLCLOCK_SECONDS=120` (or **60** if stable)
- `ITERATIONS=5000` (hit wall first; harmless if wall wins)
- `VAL_LOSS_EVERY=0` — redundant with `SMOKE_MODE=1` for mid-train val; smoke still skips final eval via `SMOKE_MODE`.
- Shell helper: `bash scripts/smoke_orchestrated_8xh100.sh`

**Full runs:** restore **`MAX_WALLCLOCK_SECONDS=600`**, production **`EVAL_STRIDE=64`**, **`VAL_LOSS_EVERY`** as in parent record for fair comparison.

---

## Full A/B protocol (novelty test)

1. **Fix** `SEED` (e.g. `42`), **fix** all other env vars across A/B.
2. **Run A:** `LEAKY_RELU_SLOPE=0` (or unset).
3. **Run B:** `LEAKY_RELU_SLOPE=0.5`.
4. Log: **`final_int8_zlib_roundtrip_exact val_bpb`**, **`bytes_total`**, **train wall**, **eval wall** if printed.
5. **Decision:** If **B better** and **bytes ≤ 16M**, consider **one** extra run: **seed 1337** on **B** only (variance check).

---

## Task breakdown (task_id, agent, INPUT → OUTPUT → VERIFY)

| task_id | name | agent | INPUT → OUTPUT → VERIFY |
|---------|------|-------|-------------------------|
| T1 | $/hr + minutes | you | **IN:** provider invoice estimate. **OUT:** max **N_full** runs. **VERIFY:** Written in this doc or lab notebook. |
| T2 | Smoke script/env | `devops-engineer` | **IN:** § Smoke protocol. **OUT:** `scripts/smoke_orchestrated_8xh100.sh` + `SMOKE_MODE` in trainer. **VERIFY:** Completes without traceback. |
| T3 | Full A | `backend-specialist` | **IN:** Frozen env A. **OUT:** `train.log` excerpt + metrics. **VERIFY:** Under **16MB**, train **≤600s**. |
| T4 | Full B | `backend-specialist` | **IN:** Frozen env B. **OUT:** same. **VERIFY:** Same checks; **Δ bpb** computed. |
| T5 | README + claim | `documentation-writer` | **IN:** T3–T4. **OUT:** A/B table + honesty on smoke vs full. **VERIFY:** Matches logs. |
| T6 | Next hypothesis | `project-planner` | **IN:** T4 result. **OUT:** P1 vs stop. **VERIFY:** One sentence rationale. |

---

## Subagent matrix

| Role | Use |
|------|-----|
| `performance-optimizer` | Choose smoke vs full eval to fit minutes |
| `backend-specialist` | Env parity, OOM triage |
| `test-engineer` | Local unittest already exists for MLP; optional pre-flight on laptop |
| `documentation-writer` | Record results |
| `project-planner` | Stop/go for P1+ |

---

## Phase X: Verification checklist

- [ ] **Smoke** logs show **8 ranks** (or expected **world_size**), **no NCCL hang** at start.
- [ ] **Full** runs: **`final_int8_zlib_roundtrip_exact`** lines captured for **A** and **B**.
- [ ] **Artifact** ≤ **16,000,000** bytes (decimal) per run.
- [ ] **Train** ≤ **600s** on **8×H100 SXM** for record-track claims.
- [ ] **No val leakage**; smoke shortcuts **not** presented as leaderboard scores.

---

## Risks & mitigations

| Risk | Mitigation |
|------|------------|
| **Sliding eval** dominates wall | Smoke **without** full slide; full runs **once** each for A/B |
| **OOM** on 8× | Same code as SOTA parent — if OOM, lower **`TRAIN_BATCH_TOKENS`** **only** after smoke, **document** change (breaks strict A/B — prefer **fix infra**) |
| **Budget blown** on setup | Pre-bake **Docker** or **RunPod template** with `torch+zstandard` |

---

## What this plan does *not* claim

- It does **not** guarantee **leaderboard** placement on **$25**.
- It does **not** replace **11L XSA** family for raw competitiveness — it **isolates one novel lever** on **your** stack **efficiently**.

---

## Next steps

1. Fill **Phase 0** ($/hr, data cached?, smoke eval rules).  
2. Run **smoke** once on **8×H100**.  
3. Run **full A/B** (`LEAKY_RELU_SLOPE` 0 vs 0.5).  
4. Run **`/create`** only if implementing **P1+** (GPTQ-lite, EMA, etc.).

---

**End of plan**

---

## Implementation status (repo)

| Item | Location |
|------|----------|
| `SMOKE_MODE` | `records/.../OrchestratedStack_10LInt5/train_gpt.py` (`Hyperparameters.smoke_mode`, early exit before serialization) |
| Smoke script | `scripts/smoke_orchestrated_8xh100.sh` |
| Full A/B helper | `scripts/run_orchestrated_full_ab.sh` (`baseline` \| `leaky`) |
| Tests | `python -m unittest tests.test_orchestrated_mlp_activation tests.test_orchestrated_smoke_hyperparams` |
