# arch1/mamba-hybrid — Current Status Assessment
**Date:** 2026-04-23
**Branch:** arch1/mamba-hybrid

---

## Executive Summary

**Branch Status:** Core implementation EXISTS but needs validation and testing.

**Key Finding:** Epic 1 and Epic 2 core implementation is COMPLETE in code, but:
- ✗ BACKLOG.md not updated (all tasks still marked `[ ]`)
- ✗ No tests directory (Epic 2B claimed in commits but missing)
- ✗ No smoke test results documented
- ✗ Multiple hotfixes indicate unresolved issues
- ✗ No Go/No-Go gate decision recorded

**Recommendation:** Before proceeding to new work, we must:
1. Validate current implementation works
2. Document smoke test results
3. Make Go/No-Go decision for Epic 1
4. Create missing tests (Epic 2B)
5. Update BACKLOG.md to reflect actual status

---

## Code Implementation Status

### ✅ COMPLETED — Epic 1 & 2 Core Code

**MambaBlock (lines 710-832):**
- ✓ Full Mamba-2 SSM implementation
- ✓ CUDA fast path with mamba-ssm kernels
- ✓ Sequential fallback for CPU
- ✓ Proper initialization (dt_proj, A_log, D)
- ✓ CastedLinear for QAT support
- ✓ Gradient checkpointing support
- ✓ RMSNorm pre-normalization

**GPT Hybrid Architecture (lines 935-1034):**
- ✓ Mamba/Attention dispatch maps (mamba_idx_map, attn_idx_map)
- ✓ Parameter banks sized for attention layers only
- ✓ mamba_blocks ModuleList instantiation
- ✓ Hybrid layer initialization

**Forward Pass Dispatch (lines 1101-1141):**
- ✓ `_forward_layer()` method with Mamba/Attention switching
- ✓ U-Net skip connections preserved across 18 layers
- ✓ Encoder/decoder loops use hybrid dispatch

**Hyperparameters (lines 109-114):**
- ✓ MAMBA_LAYERS (comma-separated indices)
- ✓ MAMBA_D_STATE (default 32)
- ✓ MAMBA_D_CONV (default 4)
- ✓ MAMBA_EXPAND (default 1.5)
- ✓ MAMBA_MATRIX_LR (default 0.015)
- ✓ MAMBA_GRAD_CHECKPOINT (default 0)

**Dependencies (requirements.txt):**
- ✓ mamba-ssm>=2.2.0
- ✓ causal-conv1d>=1.4.0
- ✓ pytest

---

## ✗ MISSING — Documentation & Validation

### Epic 1 Deliverables

**Task 1.3.4: Smoke Test Results**
- ✗ No `specs/smoke_test_results.md` file
- ✗ Step time on 1×H100 not documented
- ✗ Memory usage not documented
- ✗ **Go/No-Go Decision NOT recorded**

**Required:**
```markdown
# Smoke Test Results

**Test Date:** YYYY-MM-DD
**Hardware:** 1×H100 (80GB)
**Config:** 1 Mamba layer at position 0

## Metrics
- Step time: XX.X ms/step (baseline: 86.7ms)
- GPU memory: XX.X GB (peak)
- Loss at step 50: X.XXX
- Param count: XX.XM

## Go/No-Go Decision
- [X] PROCEED: Step time ≤ 100ms
- [ ] INVESTIGATE: Step time 100-150ms
- [ ] ABORT: Step time > 150ms

## Notes
[Any observations, issues, recommendations]
```

---

## ✗ MISSING — Epic 2B Tests

**Git history shows commits claiming tests exist, but NO test files found:**
```
* 52240df [Epic 2B | Task 2B.7.2] Add E2E CPU smoke tests — 31 tests, full pipeline validation
* 56dc532 [Epic 2B | Tasks 2B.3.3-2B.3.4, 2B.5.1-2B.5.3, 2B.7.1, 4.1.2] Complete remaining CPU tests
* 4a4d67d [Epic 2B | Task 2.1.4+2.2.4] Add all-params gradient and optimizer step tests
```

**Reality:** `ls tests/` → **No such directory**

**Required Tests (from BACKLOG.md Epic 2B):**
1. `tests/test_mamba_block.py` — 6 unit tests
2. `tests/test_hybrid_gpt.py` — 5 unit tests
3. `tests/test_training_loop.py` — 5 integration tests
4. `tests/test_quantization.py` — 5 integration tests
5. `tests/test_regression.py` — 3 regression tests
6. **Total: 24 tests minimum**

---

## 🔥 ISSUES — Recent Hotfixes

**Commit analysis shows unresolved problems:**

1. **dtype Issues (4 commits):**
   - `9b775e1` Restore .bfloat16() on flash_attn calls — graph breaks lose dtype context
   - `7145c4e` Fix FlashAttention dtype: cast q/k/v to bf16 before flash_attn call
   - `3284b59` Fix FlashAttention fp32 error: cast MambaBlock output to input dtype
   - `727c492` Fix causal_conv1d dtype mismatch: cast bias to match weight dtype

2. **torch.compile Issues (2 commits):**
   - `d9ba42d` Disable torch.compile for hybrid Mamba models to fix dtype errors
   - `cb8023b` Fix torch.compile fullgraph=True incompatibility with Mamba sequential scan

3. **Step Time Issue:**
   - `83acf8f` Phase 1: Fix step time bottlenecks (613ms -> ~85ms target)
   - **Problem:** 613ms is **7× slower than target** (85ms) and **7× slower than SOTA baseline** (86.7ms)
   - **Epic 1 Go/No-Go Gate:** Step time must be ≤100ms. **613ms = ABORT criteria!**

---

## Critical Questions to Answer NOW

### Question 1: What is the CURRENT step time?

**Why it matters:** Epic 1 Go/No-Go gate requires ≤100ms/step.

**How to check:**
```bash
# Run 50-step smoke test on 1×H100
export MAMBA_LAYERS="0"  # Single Mamba layer at position 0
export ITERATIONS=50
python train_gpt.py --seed 42 > smoke_test.log 2>&1

# Extract step time from logs
grep "step_avg" smoke_test.log
```

**Decision tree:**
- ≤100ms → ✅ PROCEED to Epic 3 (Performance Optimization)
- 100-150ms → ⚠️ INVESTIGATE (profile bottlenecks, try CUDA kernels)
- >150ms → ❌ ABORT arch1/mamba-hybrid (switch to arch2 or arch3)

---

### Question 2: Are the dtype fixes stable?

**Why it matters:** Multiple hotfixes suggest fragile implementation.

**How to check:**
```bash
# Run 100 steps with full hybrid config (15M + 3A)
export MAMBA_LAYERS="0,1,2,3,4,5,6,7,8,9,10,11,14,15,16"
export ITERATIONS=100
python train_gpt.py --seed 42 > full_hybrid_test.log 2>&1

# Check for errors
grep -i "error\|nan\|inf\|dtype" full_hybrid_test.log
```

**Pass criteria:**
- No dtype errors
- No NaN/Inf in loss
- Loss decreases smoothly
- All 100 steps complete

---

### Question 3: Does torch.compile work?

**Why it matters:** Epic 3 requires torch.compile for ≤85ms/step target.

**Current state:** Commit `d9ba42d` says "Disable torch.compile for hybrid Mamba models"

**How to check:**
```python
# Check train_gpt.py for torch.compile usage
grep -n "torch.compile" train_gpt.py
```

**If disabled:** Need to re-enable with `fullgraph=False` (from MEMORY.md: Mamba causes graph breaks)

---

## Recommended Immediate Actions

### Step 1: Run Smoke Test (30 min)

**File: `scripts/run_smoke_test.sh`**
```bash
#!/bin/bash
export MAMBA_LAYERS="0"
export ITERATIONS=50
export SEED=42
export DATA_PATH="./data/datasets/fineweb10B_sp1024"
export TOKENIZER_PATH="./data/tokenizers/fineweb_1024_bpe.model"
export VOCAB_SIZE=1024

python train_gpt.py > logs/smoke_test_$(date +%Y%m%d_%H%M%S).log 2>&1
```

**Extract results:**
- Step time
- Memory usage
- Loss at step 50
- Any errors

**Document in:** `specs/smoke_test_results.md`

---

### Step 2: Make Go/No-Go Decision

**Based on smoke test:**
- If ≤100ms → PROCEED
- If >150ms → ABORT (pivot to arch2 or other plan)

**Update:** `specs/smoke_test_results.md` with formal decision

---

### Step 3: Create Missing Tests (if PROCEED)

**Priority order:**
1. `tests/test_mamba_block.py` — Unit tests for MambaBlock
2. `tests/test_hybrid_gpt.py` — Validate dispatch logic
3. `tests/test_training_loop.py` — 10-step convergence test

**Run tests:**
```bash
pytest tests/ -v
```

**Workflow:** ALL tests must pass before proceeding to Epic 3

---

### Step 4: Update BACKLOG.md

**Mark completed tasks:**
- Epic 1: All tasks → `[x]`
- Epic 2: Stories 2.1-2.4 → `[x]`
- Epic 2B: Mark as `[~]` (in progress) until tests pass

**Commit:**
```bash
git add specs/smoke_test_results.md specs/CURRENT_STATUS.md specs/BACKLOG.md
git commit -m "[Epic 1 | Complete] Document smoke test results and Go/No-Go decision"
git push origin arch1/mamba-hybrid
```

---

## Current Epic/Task Status (Best Guess)

**Based on code + commits:**

| Epic | Status | Evidence |
|------|--------|----------|
| Epic 1 | `[~]` IN PROGRESS | Code complete, but no Go/No-Go decision documented |
| Epic 2 | `[~]` IN PROGRESS | Core architecture exists, but validation incomplete |
| Epic 2B | `[!]` BLOCKED | Tests claimed in commits but missing from repo |
| Epic 3 | `[ ]` TODO | Depends on Epic 1 PROCEED decision |
| Epic 4-7 | `[ ]` TODO | Sequential dependencies |

**Actual current task (per CLAUDE.md):**
- **Epic 1, Task 1.3.4:** Document smoke test results
- **Epic 1, Story 1.3:** Make Go/No-Go decision

---

## Next Session Opening Statement (Template)

```
Current: Epic 1 | Task 1.3.4 — Document smoke test results [~]

Last commit: 9b775e1 Restore .bfloat16() on flash_attn calls

Code status:
- MambaBlock implementation: COMPLETE
- Hybrid GPT architecture: COMPLETE
- Forward dispatch: COMPLETE
- Tests: MISSING (Epic 2B)
- Smoke test results: NOT DOCUMENTED

Tests: NO tests directory exists

Next: Run smoke test on 1×H100 to measure step time, then make Go/No-Go decision.
```

---

## Risk Assessment

| Risk | Status | Mitigation |
|------|--------|------------|
| Step time >150ms (ABORT criteria) | 🔴 CRITICAL | Last known: 613ms (7× over limit) — must re-test |
| dtype instability | 🟡 MEDIUM | Multiple fixes applied, needs validation |
| torch.compile disabled | 🟡 MEDIUM | Required for Epic 3, needs re-enable with fullgraph=False |
| Missing tests | 🟡 MEDIUM | Create tests per Epic 2B before proceeding |
| BACKLOG.md out of sync | 🟢 LOW | Documentation issue, easy to fix |

---

## Files to Review Next

1. `train_gpt.py:710-832` — MambaBlock implementation
2. `train_gpt.py:1101-1141` — Forward dispatch logic
3. Recent commit diffs — Understand what the hotfixes actually changed
4. MEMORY.md — torch.compile + Mamba CUDA kernel guidance

---

## Summary for User

**Good news:** Core Mamba hybrid architecture is implemented.

**Bad news:** No validation, no tests, no Go/No-Go decision. Last known step time was 613ms (7× too slow).

**Critical next step:** Run smoke test NOW to determine if arch1/mamba-hybrid is viable or if we need to pivot.
