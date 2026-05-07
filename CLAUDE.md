# CLAUDE.md — Arch1 Mamba-Hybrid Dev Workflow

> This file is read automatically by Claude Code. Follow every rule here on every interaction.

---

## Project Context

- **Branch:** `arch1/mamba-hybrid`
- **Goal:** Replace most attention layers with Mamba-2 SSM layers → 18 total layers (15 Mamba + 3 GQA Attn) → target BPB ≤ 1.110
- **Baseline:** SOTA 1.11473 BPB, 11L transformer
- **Baseline file:** `records/track_10min_16mb/2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072/train_gpt.py`
- **Working file:** `train_gpt.py` (root)
- **Stack:** Python, PyTorch, Mamba-SSM, H100 GPU(s)

---

## Workflow Rules (Follow Strictly)

### 1. One Epic at a Time
- Work through epics in order: Epic 1 → 2 → 2B → 3 → 4 → 5 → 6 → 7
- Do NOT start the next epic until the current one is fully complete
- Before moving on, always output an **Epic Summary** (see format below)

### 2. Epic Summary (Required Before Moving On)
Before starting any new epic, output a summary block in this exact format:

```
## Epic [N] Complete — Summary

**What was implemented:**
- [bullet list of every story and task completed]

**Tests passing:**
- [list all tests that passed]

**GitHub commits pushed:**
- [list commit hashes or messages]

**Key decisions made:**
- [any Go/No-Go or architectural decisions]

**Ready to proceed to:** Epic [N+1] — [Epic Name]
```

### 3. GitHub Push After Every Task
- After every task with a **passing test**, immediately push to GitHub
- Commit message format: `[Epic N | Task X.X.X] <description>`
- Example: `[Epic 1 | Task 1.1.1] Copy SOTA baseline to root train_gpt.py`
- **NEVER push if any test is failing**

### 4. Test Discipline — No Exceptions
- **Never mark a task as `[x]` complete if its tests are failing**
- If tests fail, fix the code — do not skip or workaround the test
- Mark tasks as `[~]` (In Progress) while tests are not yet passing
- Only flip to `[x]` after you confirm: tests pass + push succeeds

### 5. Go/No-Go Gates
- **Epic 1 Gate:** Step time on 1xH100 must be <= 100ms. If > 150ms → ABORT, switch to arch2/arch3
- **Epic 7 Gate:** Mean BPB <= 1.1097, p < 0.01 (Welch's t-test vs SOTA)
- Always document the Gate decision in `specs/` before proceeding

---

## Backlog Reference

### Status Legend
| Symbol | Meaning |
|--------|---------|
| `[ ]`  | To Do |
| `[~]`  | In Progress |
| `[x]`  | Done |
| `[!]`  | Blocked |
| `[—]`  | Cancelled / Skipped |

### Epic Order & Dependencies
```
Epic 1  →  Go/No-Go  →  Epic 2  →  Epic 3  →  Epic 4
                     ↘  Epic 2B (runs in parallel, continuous)
                     ↘  Epic 5  (parallel with Epic 4)
                     ↘  Epic 6  (parallel with 4 & 5)
                                              ↓
                                           Epic 7
```

| Epic | Name | Priority | Depends On |
|------|------|----------|------------|
| 1 | Foundation & Smoke Test | P0 | — |
| 2 | Core Hybrid Architecture | P0 | Epic 1 PROCEED |
| 2B | Testing & Validation | P0 | Parallel with all epics |
| 3 | Performance Optimization | P0 | Epic 2 |
| 4 | Quantization & Artifact | P0 | Epic 3 |
| 5 | Training Stack Integration | P1 | Epic 2 |
| 6 | Ablation & Hyperparameter Tuning | P1 | Epic 2 |
| 7 | Final Evaluation & Submission | P0 | Epics 1–6 |

---

## Test Rules Per Epic

| Epic | Test Stories | When to Run | Must Pass Before |
|------|-------------|-------------|-----------------|
| 1 | 2B.1 (MambaBlock unit tests) | Every commit | Pushing any task |
| 2 | 2B.2, 2B.3, 2B.5 | After impl changes | Epic 2 summary |
| 3 | 2B.6 (benchmarks) | After perf changes | Epic 3 summary |
| 4 | 2B.4 (quant tests) | After quant changes | Epic 4 summary |
| Pre-7 | 2B.7 (E2E smoke tests) | Before milestones | Starting Epic 7 |

---

## Commit & Push Protocol

```bash
# After every passing task:
git add <specific files>
git commit -m "[Epic N | Task X.X.X] <short description>"
git push origin arch1/mamba-hybrid
```

**Never:**
- `git push` with failing tests
- Skip the commit message format
- Batch multiple tasks into one commit (one task = one commit)

---

## File Structure

```
.
├── train_gpt.py              # Working file (copy of SOTA baseline, modified)
├── CLAUDE.md                 # This file
├── requirements.txt          # Must include mamba-ssm>=2.2.0, causal-conv1d>=1.4.0
├── tests/
│   ├── test_mamba_block.py   # Story 2B.1
│   ├── test_hybrid_gpt.py    # Story 2B.2
│   ├── test_training_loop.py # Story 2B.3
│   ├── test_quantization.py  # Story 2B.4
│   └── test_regression.py    # Story 2B.5
└── specs/
    ├── MASTERPLAN.md          # Branch masterplan
    ├── arch1_mamba_hybrid.md  # Tech spec
    ├── BACKLOG.md             # Full agile backlog
    ├── smoke_test_results.md  # Task 1.3.4 output
    └── FINAL_CONFIG.md        # Task 7.1.1 output
```

---

## Key Architecture Targets

| Parameter | Value |
|-----------|-------|
| Total layers | 18 |
| Mamba layers | 15 |
| Attention layers | 3 (GQA) |
| d_model | 512 |
| d_state | 32 |
| d_conv | 4 |
| expand | 1.5 |
| Params per Mamba layer | ~1.27M |
| Total params | ~27.8M |
| Target BPB | <= 1.110 |
| Artifact size | <= 16MB |
| Step time target | <= 85ms |

---

## How to Start Each Session

When resuming work, always:
1. State the **current epic and task** you are on
2. Show the **test status** (passing/failing)
3. Show the **last GitHub commit** that was pushed
4. Ask for confirmation before proceeding to the next task

Example opening:
```
Current: Epic 1 | Task 1.2.2 — MambaBlock.forward [~]
Last commit: [Epic 1 | Task 1.2.1] MambaBlock.__init__ implemented
Tests: test_mamba_block.py::test_param_count PASSING
Next: Implement forward pass, then run test_forward_shape
```

---

## Git Authentication

- Remote uses HTTPS with PAT (token must be set in remote URL for push)
- If push fails with 403, re-set remote URL with valid PAT:
  `git remote set-url origin https://<TOKEN>@github.com/johnlennyt5/parameter-golf.git`
