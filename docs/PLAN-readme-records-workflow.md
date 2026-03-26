# Plan: Parameter Golf workflow from README, data docs, and records

**Mode:** Planning only (no implementation in this document)  
**Created for:** Operating in this repository using authoritative guidelines (`README.md`, `data/README.md`, `records/**/README.md`) and coordinating Cursor subagents.  
**Plan file:** `docs/PLAN-readme-records-workflow.md`

---

## Phase -1: Context check

| Check | Result |
|-------|--------|
| `CODEBASE.md` (per project-planner) | **Not present** in this workspace. OS from user info: **macOS** (darwin). |
| Existing plan files | None required before this plan; treat this document as the canonical workflow plan unless superseded. |
| Conversation context | User asked for a **comprehensive** plan, **`docs/`** output, and explicit **subagent** assignments. |

**Project type (for agent routing):** This repo is primarily **ML training / research** (PyTorch MLX, CUDA), plus **submission packaging** under `records/`. It is **not** a typical web-app or mobile CRUD project. Use agent assignments below instead of the default “web → frontend-specialist” mapping when work touches `train_gpt.py`, datasets, or record folders.

---

## Phase 0: Socratic gate (clarifying questions & assumptions)

Ask or confirm with the user before large implementation efforts:

1. **Target track:** Leaderboard **10 min / 8×H100 / 16 MB** (`records/track_10min_16mb/`) vs **non-record** unlimited compute (`records/track_non_record_16mb/`) vs local-only experimentation?
2. **Hardware path:** Apple Silicon MLX only, remote CUDA (e.g. Runpod), or both?
3. **Submission intent:** PR-ready record folder vs internal experiments only (no PR)?

**Assumptions baked into this plan (override if the user answers differently):**

- You want a **reproducible** path from clone → data → train → metrics → optional **record PR**.
- You will **not** violate challenge rules (no validation leakage, artifact budget, statistical significance for SOTA).

---

## Overview

**What:** A single reference for *how to work* in Parameter Golf: where rules live, how existing **records** model successful submissions, and which **Cursor subagents** fit each phase.

**Why:** The root `README.md` defines the challenge, constraints, and submission process; `data/README.md` defines dataset/tokenizer layout; each `records/.../README.md` shows a concrete, accepted pattern. Aligning work with these avoids rejected PRs and wasted compute.

**Success criteria (measurable):**

- [ ] Data layout matches `data/README.md` (paths, manifests, validation split understanding).
- [ ] Training commands and env vars are documented like existing records (command block + key metrics).
- [ ] If submitting: folder contains `README.md`, `submission.json`, `train.log`, and runnable `train_gpt.py` per root README.
- [ ] `val_bpb` and size metrics are taken from the prescribed logging lines (`final_int8_zlib_roundtrip`, etc., as in baseline record).

---

## Authoritative sources (read in this order)

| Priority | Document | Use |
|----------|----------|-----|
| 1 | [README.md](../README.md) | Challenge objective, 16 MB artifact definition, 10-minute training rule, FAQ, **submission process**, leaderboard table → links to records. |
| 2 | [data/README.md](../data/README.md) | Download/rebuild data, tokenizer variants, shard math, env knobs for export. |
| 3 | `records/<track>/<date>_<Name>/README.md` | **Gold standard** for how to describe configs, commands, metrics, file list, and narrative. |
| 4 | Same folder `submission.json` | Required metadata shape (`author`, `github_id`, `val_bpb`, sizes, etc.). |
| 5 | `.cursor/ARCHITECTURE.md` | How the Cursor kit (agents, skills, workflows) is organized in this repo. |
| 6 | `.cursor/agents/project-planner.md` | Generic planning phases, task format, verification mindset (adapt paths: this repo may not have `.agent/scripts/`; use project scripts and README checks). |

---

## Constraint summary (from README)

Copy into every plan/checklist for submission work:

- **Artifact:** Code bytes + compressed model bytes ≤ **16,000,000** decimal bytes; counted code should live in `train_gpt.py` for the artifact story.
- **Training time (leaderboard):** ≤ **10 minutes** on **8×H100 SXM** for record track.
- **Evaluation time:** Also capped (see FAQ in README).
- **SOTA PRs:** Beat existing SOTA by ≥ **0.005 nats** with **p < 0.01** evidence (e.g. multiple runs / logs) unless ML-unchanged systems optimization (per README).
- **Tokenizer/dataset changes:** Higher scrutiny; must prove `val_bpb` correctness.

---

## Subagent matrix (Cursor specialists → Parameter Golf work)

Use this instead of web/mobile defaults from `project-planner.md` when work is training- or submission-focused.

| Phase | Primary subagent | Supporting skills / notes |
|-------|------------------|---------------------------|
| Repo & record survey | `explorer-agent` | Map `train_gpt.py`, `records/` layout, differences vs baseline. |
| Planning & milestones | `project-planner` | `plan-writing`, `brainstorming`; align tasks with README submission checklist. |
| Python training code | `backend-specialist` (API/script-heavy logic) or treat as core ML | Pair with **`python-patterns`** skill; challenge code is script-style, not a REST API. |
| Numerics / perf of training loop | `performance-optimizer` | Throughput, memory, batching; not the same as Lighthouse/web Vitals—interpret as **GPU/step time**. |
| Reproducibility & tests | `test-engineer` | Smoke tests, deterministic logging, comparing metrics across runs. |
| Debugging failed runs | `debugger` | `systematic-debugging`; use `train.log` evidence. |
| Infrastructure (Runpod, `torchrun`, env) | `devops-engineer` | SSH, process layout, `NCCL` hints as in record READMEs. |
| Security / supply chain for deps | `security-auditor` | `requirements.txt`, pip packages—light touch unless importing unusual deps. |
| Documentation for PR | `documentation-writer` | Record `README.md` clarity, tables, commands. |
| “Is this fair?” / rules | `product-manager` or human review | Challenge FAQ: external compute, good-faith hyperparameter search. |
| Large refactors of legacy `train_gpt.py` | `code-archaeologist` | If merging ideas from multiple records. |

**Explicit non-defaults:**

- **`frontend-specialist` / `mobile-developer`:** Skip unless you add a real web/mobile surface (out of scope for core challenge).
- **`seo-specialist`:** Skip unless editing public docs pages for the repo’s discoverability (optional).

---

## Task dependency graph (high level)

```mermaid
flowchart LR
  A[Read README + data README] --> B[Pick track and baseline record]
  B --> C[Data download / verify paths]
  C --> D[Local smoke train MLX or small CUDA]
  D --> E[Full timed run on target hardware]
  E --> F[Capture train.log + metrics]
  F --> G[Package record folder]
  G --> H[Open PR]
```

---

## Task breakdown (task_id, agent, dependencies, INPUT → OUTPUT → VERIFY)

| task_id | name | agent | deps | INPUT → OUTPUT → VERIFY |
|---------|------|-------|------|-------------------------|
| T1 | Ingest challenge rules | `explorer-agent` | — | **IN:** Root README sections on artifact, time limits, submission. **OUT:** Short bullet list of hard constraints for your run. **VERIFY:** Every later task cites these bullets or marks an exception with owner approval. |
| T2 | Align data pipeline | `backend-specialist` + `python-patterns` | T1 | **IN:** `data/README.md`, desired `--variant` and `--train-shards`. **OUT:** Local `data/datasets/` and `data/tokenizers/` populated as documented. **VERIFY:** Scripts run without path errors; validation split behavior matches README (full val). |
| T3 | Baseline reproduction | `debugger` / `test-engineer` | T2 | **IN:** A chosen baseline record’s command (e.g. NaiveBaseline README). **OUT:** Comparable metric ballpark on comparable hardware (allow variance). **VERIFY:** `val_bpb` and size reported in same log format as reference. |
| T4 | Experiment design | `project-planner` + `product-manager` | T3 | **IN:** Hypothesis (arch/quantization/eval). **OUT:** Written experiment plan + success/fail metrics. **VERIFY:** Plan explicitly avoids forbidden leakage (FAQ). |
| T5 | Implement `train_gpt.py` changes | `backend-specialist` + `code-archaeologist` | T4 | **IN:** Single-file constraint awareness. **OUT:** Runnable script with clear env toggles. **VERIFY:** Local or remote smoke run completes; artifact size still in budget if applicable. |
| T6 | Scale & tune on GPU | `performance-optimizer` + `devops-engineer` | T5 | **IN:** Target machine (1× vs 8× GPU). **OUT:** Stable `torchrun` command, wallclock under cap for record track. **VERIFY:** `train.log` shows wallclock and step times; training stops per cap. |
| T7 | Statistical evidence for SOTA | `test-engineer` | T6 | **IN:** README threshold (0.005 nats, p&lt;0.01). **OUT:** Multiple runs or sufficient logging. **VERIFY:** Numbers support claim before PR title asserts SOTA. |
| T8 | Record package | `documentation-writer` | T6, T7 | **IN:** Template from existing `records/...` folders. **OUT:** `README.md`, `submission.json`, `train.log`, `train_gpt.py`. **VERIFY:** Checklist in root README “Submission Process” satisfied. |
| T9 | PR readiness | `documentation-writer` + `security-auditor` | T8 | **IN:** Repo diff scope (add folder only). **OUT:** PR description linking leaderboard comparison. **VERIFY:** No accidental core-only changes unless separately justified per README. |

---

## Phase X: Verification checklist (definition of done)

Use before declaring work “ready” or merging a submission PR.

### A. Rule compliance (from README)

- [ ] 16,000,000-byte total artifact story understood; sizes reported match `submission.json` fields.
- [ ] Record track: training ≤ 10 min on 8×H100; logs show wallclock.
- [ ] No training on validation data; FAQ on test-time training understood if applicable.
- [ ] SOTA: improvement ≥ 0.005 nats with p &lt; 0.01 evidence (or waiver category documented).

### B. Filesystem completeness (submission)

- [ ] New folder only under correct `records/track_*` subtree.
- [ ] `README.md` explains approach, configuration, and command (like existing records).
- [ ] `submission.json` fields consistent with README and logs.
- [ ] `train.log` included; metrics reproducible from stated command.
- [ ] `train_gpt.py` runs from record folder context.

### C. Technical sanity

- [ ] Data paths and tokenizer match documented layout (`data/README.md`).
- [ ] If tokenizer/dataset changed: extra correctness checks documented (per README warning).

### D. Cursor / process

- [ ] Subagent assignments above reviewed for the actual work performed (adjust retroactively in PR description if different).
- [ ] Assumptions in Phase 0 confirmed or updated.

### E. Phase X completion marker (fill when verified)

```markdown
## PHASE X COMPLETE
- Rules checklist: [ ] pass
- Record folder completeness: [ ] pass
- Repro command verified: [ ] pass
- Date: YYYY-MM-DD
```

---

## Risks & mitigations

| Risk | Mitigation |
|------|------------|
| Metric bug after tokenizer change | Extra validation passes; compare against known baselines; document hashes (`docs_selected.source_manifest.json` per data README). |
| Over-fit to val via eval tricks | Re-read FAQ; keep training/eval boundary clean. |
| PR rejected for insufficient significance | Pre-calculate margin vs current SOTA; run multiple seeds if variance high. |
| `project-planner.md` references `.agent/` scripts | This repo may use different paths; prefer README + `npm`/`python` checks actually present here. |

---

## Appendix: Example record structure (reference)

See `records/track_10min_16mb/2026-03-17_NaiveBaseline/`:

- `README.md` — configuration, command, key metrics, file list  
- `submission.json` — author metadata and scores  
- `train.log` — ground truth for verification  
- `train_gpt.py` — frozen code snapshot  

---

## Next steps for the user

1. Answer **Phase 0** questions (track, hardware, PR intent).  
2. Re-read sections of [README.md](../README.md) that match your phase (Getting Started vs Submission Process).  
3. Skim one **record README** closest to your idea.  
4. Run **`/create`** (or equivalent implementation workflow) when ready to execute—**after** approving this plan.

---

**End of plan**
