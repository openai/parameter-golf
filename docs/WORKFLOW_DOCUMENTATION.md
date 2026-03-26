# Parameter Golf — workflow & session documentation

**Purpose:** Single living document for *how we work* in this repo: environment, authoritative sources, orchestrated agent roles, process steps, and a **changelog** updated as the project evolves.

**Related plans:**

| Plan | Use |
|------|-----|
| [PLAN-readme-records-workflow.md](./PLAN-readme-records-workflow.md) | End-to-end workflow: data → train → record folder → PR; tasks **T1–T9**; Phase X checklist. |
| [PLAN-records-innovation-score.md](./PLAN-records-innovation-score.md) | **Score improvement:** innovation themes mined from `records/`, constraints, suggested ablation order, tasks **T1–T8**. |

---

## 1. Environment (venv + uv)

Use the project virtualenv and install dependencies with **uv** (as requested):

```bash
cd /path/to/parameter-golf
source .venv/bin/activate
uv pip install -r requirements.txt
# or for one-off packages:
# uv pip install <package>
```

**Note:** `requirements.txt` is the reference for CUDA/remote training; local Apple Silicon quickstart packages are listed in the root [README.md](../README.md) (MLX path).

---

## 2. Authoritative sources (read order)

| Order | Document | Role |
|-------|-----------|------|
| 1 | [README.md](../README.md) | Challenge rules, 16 MB artifact, 10 min / 8×H100, submission process, FAQ |
| 2 | [data/README.md](../data/README.md) | Dataset/tokenizer layout, download, shard math |
| 3 | `records/<track>/<date>_<Name>/` | Gold-standard **README.md**, **submission.json**, **train.log**, **train_gpt.py** |
| 4 | [.cursor/ARCHITECTURE.md](../.cursor/ARCHITECTURE.md) | Cursor kit layout (agents, skills, workflows) |
| 5 | [PLAN-records-innovation-score.md](./PLAN-records-innovation-score.md) | When improving **val_bpb**: theme map (quant, arch, opt, eval, TTT) and task list |

---

## 3. Constraint summary (from README)

Copy into experiment notes when preparing a submission:

- **Artifact:** `train_gpt.py` + compressed model bytes ≤ **16,000,000** bytes (decimal); self-contained at eval time.
- **Record track training:** ≤ **10 minutes** on **8×H100 SXM**; evaluation time also capped (see FAQ).
- **SOTA PRs:** Beat prior SOTA by ≥ **0.005 nats** with **p < 0.01** evidence (multiple runs / logs), unless ML-unchanged systems-only case per README.
- **Tokenizer/dataset changes:** Extra scrutiny; prove `val_bpb` correctness.

---

## 4. Process at a glance

1. **Read** README + `data/README.md`, pick **track** (`track_10min_16mb` vs `track_non_record_16mb`) and a **reference record** (e.g. NaiveBaseline).
2. **Download data** via `data/cached_challenge_fineweb.py` with intended `--variant` and `--train-shards`.
3. **Smoke train** (MLX locally or 1×GPU CUDA), then **full timed run** on target hardware.
4. **Capture** `train.log` and final metrics (`val_bpb`, `final_int8_zlib_roundtrip` lines, sizes).
5. **Package** a new `records/.../` folder: `README.md`, `submission.json`, `train.log`, `train_gpt.py`.
6. **Open PR** that adds only the record folder (core script changes follow README rules separately).

---

## 5. Innovation roadmap (records → better score)

Full detail lives in [PLAN-records-innovation-score.md](./PLAN-records-innovation-score.md). Use this section as a **quick index**; update the changelog when you complete phases.

**Socratic questions (before big runs):** baseline fork (Naive vs SOTA snapshot), primary axis (train vs eval vs TTT), tokenizer risk, compute budget—see Phase 0 in that plan.

**Suggested order (Porter-style):**

1. Reproduce one **frontier** record (leaderboard-linked README) on target hardware.  
2. **Fix eval protocol** (e.g. `EVAL_STRIDE`, batching) so comparisons match.  
3. **Quant + compression** (int6/int5 split, QAT, zstd) to free bytes for depth/width.  
4. **Architecture** (MLP 3×, layers, SmearGate, BigramHash table size).  
5. **Optimization** (Muon, WD, SWA, LRs, warmdown).  
6. **Eval / TTT** only after train stack plateaus—watch eval wall time.

**Theme → where to read in `records/` (examples, not exhaustive):**

| Theme | Examples in repo |
|-------|-------------------|
| Mixed int5 MLP + int6 attn, 10L | `2026-03-20_10L_Int5MLP_MuonWD04_SWA50` |
| **Ready-run orchestrated fork** (default `EVAL_BATCH_SEQS=64`, zstd warning) | `2026-03-21_OrchestratedStack_10LInt5` — run: `bash scripts/run_orchestrated_stack_8xh100.sh` |
| SmearGate + BigramHash + Muon WD + SWA | `2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA` |
| 11L + QAT + sliding eval | `2026-03-19_MLP3x_QAT_Int6_SlidingWindow` |
| Sliding eval only (train unchanged) | `2026-03-19_SlidingWindowEval` |
| Doc boundaries / LoRA TTT | `2026-03-17_LoRA_TTT` |

**Plan tasks (innovation plan):** **T1** corpus of ideas → **T2** constraint matrix → **T3** pick fork → **T4** byte budget → **T5** ablations → **T6** integrated stack → **T7** record package → **T8** PR + significance.

---

## 6. Orchestrated subagents (Parameter Golf–specific)

Default web/mobile routing from generic templates does **not** apply. Use:

| Work | Primary agent | Notes |
|------|----------------|-------|
| Map repo / records | `explorer-agent` | `train_gpt.py`, `records/` layout |
| Milestones & checklist | `project-planner` | Align with workflow plan + [PLAN-records-innovation-score.md](./PLAN-records-innovation-score.md) when tuning score |
| Training script / Python | `backend-specialist` + `python-patterns` | Script-style, not REST |
| Step time / memory | `performance-optimizer` | GPU throughput, not Lighthouse |
| Reproducibility | `test-engineer` | Logs, multi-seed where needed |
| Remote / `torchrun` | `devops-engineer` | Runpod, NCCL, env |
| Record PR prose | `documentation-writer` | README + `submission.json` |
| Rules / fairness | `product-manager` + human | FAQ on external compute |

Skip `frontend-specialist` / `mobile-developer` unless you add a real web/mobile app.

---

## 7. Repository layout (discovery snapshot)

**Relevant for workflow (not exhaustive):**

- `train_gpt.py`, `train_gpt_mlx.py` — entry points for CUDA and MLX
- `data/` — download helpers, datasets, tokenizers
- `records/track_10min_16mb/`, `records/track_non_record_16mb/` — submissions
- `docs/` — this file, [PLAN-readme-records-workflow.md](./PLAN-readme-records-workflow.md), [PLAN-records-innovation-score.md](./PLAN-records-innovation-score.md)
- `.cursor/` — skills, agents, workflows (orchestration tooling)

---

## 8. Verification scripts (workspace)

Paths in this clone use `.cursor/skills/` (not `.agent/`):

```bash
python3 .cursor/skills/vulnerability-scanner/scripts/security_scan.py .
python3 .cursor/skills/lint-and-validate/scripts/lint_runner.py .
```

**Interpretation:** The stock scanner flags many **false positives** on ML code (e.g. `model.eval()`, `mx.eval()`). Treat as **advisory** unless the finding is clearly sensitive (real secrets, unsafe `eval` of strings). `lint_runner` may include files under `.cursor/`; failures there do not necessarily reflect challenge code.

---

## 9. How to update this file (iterative)

On each meaningful change in the conversation or repo:

1. Append a **Changelog** entry (date, what changed, files touched).
2. If the plan or workflow shifts, adjust **§ 4** or **§ 5** and add a one-line pointer in the changelog.
3. Keep **§ 2** in sync with any new canonical doc paths.
4. When completing innovation phases, add a line under **§ 5** or only in the changelog—avoid duplicating full ablation tables here (keep those in record `README.md`).

---

## 10. Changelog

### 2026-03-21 — `torchrun` debug: run script uses venv + `python -m torch.distributed.run`

- **Fixed:** `scripts/run_orchestrated_stack_8xh100.sh` — resolves `.venv/bin/python`, checks `import torch` and **CUDA** before launch; falls back from `torchrun` to `python -m torch.distributed.run`; clear errors if PyTorch missing or on macOS without CUDA (points to `train_gpt_mlx.py`).

### 2026-03-21 — Competitive trainer record + run script (orchestrated implementation)

- **Added:** `records/track_10min_16mb/2026-03-21_OrchestratedStack_10LInt5/` — `train_gpt.py` fork of `2026-03-20_10L_Int5MLP_MuonWD04_SWA50` with docstring gap analysis, default `EVAL_BATCH_SEQS=64`, zstd-missing warning, `requirements-record.txt`, README (gap table + command), placeholder `submission.json`.
- **Added:** `scripts/run_orchestrated_stack_8xh100.sh` — 8×GPU `torchrun` wrapper with standard env defaults.
- **Updated:** This doc **§ 5** theme table with the new record + script pointer.
- **Agents (conceptual):** `explorer-agent` (root vs SOTA gap), `backend-specialist` (trainer), `performance-optimizer` (eval batching), `documentation-writer` (README + workflow doc).

### 2026-03-21 — Innovation plan integrated (`/create`)

- **Linked:** [PLAN-records-innovation-score.md](./PLAN-records-innovation-score.md) from this workflow doc (header table + § 2 sources + § 7 layout).
- **Added:** **§ 5 Innovation roadmap** — Porter-style order, theme→record examples, cross-reference to plan tasks T1–T8.
- **Updated:** Subagent table to point `project-planner` at both plans when tuning score; **§ 9** (how to update) includes innovation changelog guidance.
- **Roles:** `documentation-writer` (edits), `project-planner` (structure alignment with existing plans).

### 2026-03-21 — Initial documentation + `/create` orchestration

- **Added:** `docs/WORKFLOW_DOCUMENTATION.md` (this file): environment (`source .venv/bin/activate`, `uv pip install`), source order, constraints, process, subagent matrix, discovery snapshot, verification commands, update protocol.
- **Linked:** [PLAN-readme-records-workflow.md](./PLAN-readme-records-workflow.md) as the detailed plan (tasks T1–T9, Phase X).
- **Orchestration (conceptual roles):**
  - **explorer-agent:** Confirmed `docs/`, `records/` layout, script locations under `.cursor/skills/`.
  - **documentation-writer:** Authored and structured this markdown.
  - **project-planner:** Aligned sections with the existing plan file and workflow phases.
- **Verification:** Ran `security_scan.py` and `lint_runner.py` from `.cursor/skills/`; report summarized in **§ 11** (scanner/lint noise on ML code and `.cursor/` helpers).

---

## 11. Last verification run (2026-03-21)

| Script | Result | Notes |
|--------|--------|--------|
| `.cursor/skills/vulnerability-scanner/scripts/security_scan.py .` | Exit 0 | JSON report: many pattern hits on `model.eval()` / MLX `eval`; one false “Bearer” in scanner script itself — **not** introduced by this doc. |
| `.cursor/skills/lint-and-validate/scripts/lint_runner.py .` | Fail (ruff) | Issues in `.cursor/.shared/ui-ux-pro-max/scripts/design_system.py` (unused import, f-string style) — **not** introduced by this doc. |

**Action:** No change to training code from this doc-only deliverable. If you want a clean `lint_runner` pass, scope ruff to `train_gpt.py` / `train_gpt_mlx.py` / `data/` or exclude `.cursor/.shared` in project config (future task).

---

**End of document** (update **§ 10** (changelog) and **§ 11** (verification) as you continue.)
