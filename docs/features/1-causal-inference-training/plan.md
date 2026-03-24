# Plan: Causal Inference for Parameter-Efficient LM Training

## Implementation Order

The plan follows the design's discovery-adjust cycle architecture. Steps are grouped into phases with explicit dependencies. Each step follows **test-first ordering**: define interface → write tests → implement until tests pass. Test framework: **pytest**.

```
Phase A: Foundation
  S1 → S2 (common.py depends on project setup)

Phase B: DAG Discovery Pipeline
  S3 → S4 → S5 (sequential: parse → estimate → check)

Phase C: Experiment Pipeline
  S6 → S7 (runner → analysis, can start after S2)

Phase D: Diagnostic Probes (parallel, can start after S2)
  S8, S9, S10, S11 (independent of each other)

Phase E: Integration
  S12 (after at least one cycle of B+C completes)
```

## Steps

### S1: Project Setup & Dependencies
**Complexity**: Simple
**Components**: Infrastructure
**Dependencies**: None
**Why this item**: All scripts depend on the Python environment and directory structure.
**Why this order**: Must be first — nothing else can run without dependencies.
**Inputs**: pyproject.toml (if exists), uv toolchain
**Outputs**: Working Python environment with all dependencies, test framework ready

1. Verify pyenv + uv are available
2. Run `uv add causal-learn statsmodels networkx graphviz scipy pytest`
3. Verify graphviz system binary: `dot -V` exits 0. If not: `brew install graphviz`
4. Create directory structure: `scripts/causal/`, `results/causal/`, `results/causal/diagnostics/`, `tests/causal/`
5. Add `results/causal/` to `.gitignore`
6. Create empty `scripts/causal/__init__.py` (enables imports)

**Verification**: `python -c "import causallearn; import statsmodels; import networkx; import graphviz; import scipy; import pytest"` exits 0 AND `dot -V` exits 0

---

### S2: common.py — Shared Utilities (C1)
**Complexity**: Medium
**Components**: C1
**Dependencies**: S1
**Why this item**: All downstream scripts depend on shared model loading and statistical utilities.
**Why this order**: Must precede all scripts that load checkpoints or compute statistics.
**Inputs**: train_gpt_mlx.py (for model import), sentencepiece tokenizer
**Outputs**: `scripts/causal/common.py`, `tests/causal/test_common.py`

**Step 0 — Import safety check** (prerequisite for all subsequent steps):
- Run `python -c "import train_gpt_mlx"` and confirm it exits 0 with no training activity, no data loading, and no MLX device allocation. If it fails, extract model classes into a separate `scripts/causal/model.py` module as fallback.

**Step 1 — Write tests** (`tests/causal/test_common.py`):
- Test `load_submission_json` with a known submission.json
- Test `load_model` returns a model object that can compute a forward pass
- Test `compute_bpb` matches known baseline value (±0.001)
- Test `paired_ttest` with synthetic data (known effect)
- Test `holm_bonferroni` adjusts p-values upward
- Test `decision_gate` returns correct classification for each threshold
- Test `log_experiment` appends entry and reads back
- Test `get_cycle_dir` creates directory
- Test `dag_diff` detects added/removed edges

**Step 2 — Implement** (each function until corresponding test passes):
1. `load_submission_json(path)` — parse records' submission.json
2. `load_model(checkpoint_path, config_overrides)` — import GPT from train_gpt_mlx.py via `__name__` guard (TD-7), construct model, load weights via `mx.load()`, load tokenizer
3. `compute_bpb(model, val_tokens, sp_model)` — reuse eval_val logic from train_gpt_mlx.py
4. `paired_ttest(treatment, control)` — scipy.stats.ttest_rel + bootstrap CI (10,000 resamples)
5. `holm_bonferroni(p_values, alpha)` — wrap `statsmodels.stats.multitest.multipletests`
6. `decision_gate(effect_size, p_value, mde)` — returns "confirmed"/"suggestive"/"null"
7. `log_experiment(path, entry)` — append to experiment_log.json
8. `get_cycle_dir(base_path, cycle)` — create and return `results/causal/cycle_{N}/`
9. `dag_diff(old_dag_path, new_dag_path)` — compare adjacency matrices, return edges added/removed/strengthened

**Verification**: `pytest tests/causal/test_common.py` all green.

---

### S3: extract_interventions.py — Record Parser (C2)
**Complexity**: Medium
**Components**: C2
**Dependencies**: S2
**Why this item**: DAG estimation requires structured intervention data from leaderboard records.
**Why this order**: Must run before C3/C4 which consume its output.
**Inputs**: `records/track_10min_16mb/` (dynamically discovered submission directories)
**Outputs**: `results/causal/interventions.json`, `tests/causal/test_extract.py`

**Step 1 — Write tests**: Test each parser format with sample README snippets. Test field_coverage computation. Test --append-experiment mode with mock raw_runs.json.

**Step 2 — Implement**:
1. `submission.json` pass — extract 6 core fields from all discovered records
2. Format A parser — regex for `| Change | ... | Impact |` tables
3. Format B parser — regex for `| | Base | This |` comparison tables, compute delta from val_bpb row
4. Format C fallback — extract from headings, bullet lists, blurb field
5. Cross-reference pass — compute total delta from base_bpb where available
6. `--append-experiment` mode — convert raw_runs.json to submission format and append
7. Compute field_coverage metric and metadata counts

**Verification**: `pytest tests/causal/test_extract.py`. Integration: run on actual records, assert field_coverage ≥ 0.90, assert all discovered submissions parsed (count > 0), spot-check 3 records.

---

### S4a: estimate_dag.py — Expert Skeleton + Encoding (C3, part 1)
**Complexity**: Medium
**Components**: C3
**Dependencies**: S3 (needs interventions.json)
**Why this item**: DAG is the navigation tool for the discovery-adjust cycle.
**Why this order**: Depends on structured intervention data from S3.
**Inputs**: `results/causal/interventions.json`
**Outputs**: `results/causal/cycle_0/causal_dag.json`, `results/causal/cycle_0/dag.png`

**Step 1 — Write tests**: Test expert skeleton has expected edges. Test binary encoding produces correct matrix dimensions. Test DOT output renders.

**Step 2 — Implement**:
1. Expert-guided skeleton builder — hardcode known causal relationships, tag as `expert_imposed`
2. Binary encoding of interventions (presence/absence matrix)
3. Marginal effect estimation per node (mean BPB of records with/without each intervention)
4. DOT visualization output via graphviz
5. Output causal_dag.json with schema (cycle=0, method="expert_guided")

**Verification**: `pytest tests/causal/test_estimate_dag.py`. Run on actual interventions.json, assert ≥5 nodes, assert dag.png renders.

---

### S4b: estimate_dag.py — FCI Validation + Cycle Updates (C3, part 2)
**Complexity**: Complex (FCI integration is highest-risk sub-step)
**Components**: C3
**Dependencies**: S4a
**Why this item**: Data validation reveals where expert priors are wrong.
**Why this order**: Expert skeleton must exist before FCI can compare against it.

**Step 1 — Write tests**: Test FCI on synthetic data with known structure. Test degenerate detection. Test edge tagging logic. Test --previous-dag diff.

**Step 2 — Implement**:
1. FCI validation (causal-learn, Fisher-Z, alpha=0.01)
2. Degenerate graph detection (empty or fully-connected → skip FCI, keep expert)
3. Edge tagging logic (expert_imposed → data_confirmed/data_contradicted/uncertain)
4. `next_intervention` recommendation (highest expected BPB improvement among uncertain edges)
5. `--previous-dag` mode for cycle updates (dag_diff integration, edge stability tracking)

**Verification**: `pytest tests/causal/test_estimate_dag.py`. Assert edge tags are valid values. Assert next_intervention is non-null.

---

### S5: identifiability_check.py — Data Quality (C4)
**Complexity**: Simple
**Components**: C4
**Dependencies**: S3, S4a (needs interventions.json + causal_dag.json)
**Why this item**: Decision gate for whether architecture ablation is feasible.
**Why this order**: Depends on both parsed data and DAG structure.
**Inputs**: `results/causal/interventions.json`, `results/causal/cycle_0/causal_dag.json`
**Outputs**: `results/causal/cycle_0/identifiability_report.json`, `tests/causal/test_identifiability.py`

**Step 1 — Write tests**: Test with synthetic interventions (known single-variable count). Test proceed/skip threshold. Test unexplored combination enumeration.

**Step 2 — Implement**:
1. Count single-variable records (exactly 1 intervention differs from baseline)
2. Count multi-variable records (3+ simultaneous changes)
3. Compute identifiability score (fraction of testable edges)
4. Identify confounded pairs (always co-occurring interventions)
5. Generate proceed/skip recommendation (>50% multi-variable → skip)
6. Enumerate unexplored combinations with expected effects × interaction priors

**Verification**: `pytest tests/causal/test_identifiability.py`. Assert recommendation is "proceed" or "skip".

---

### S6: experiment_runner.py — Paired Seed Ablation (C5)
**Complexity**: Complex
**Components**: C5
**Dependencies**: S2 (common.py for log_experiment, get_cycle_dir)
**Why this item**: Runs the actual causal experiments that produce BPB measurements.
**Why this order**: Can start after S2; runs in parallel with Phase B.
**Inputs**: Treatment config JSON, control config JSON
**Outputs**: `results/causal/cycle_N/raw_runs.json`, `tests/causal/test_experiment_runner.py`

**Step 1 — Write tests**: Test config validation (reject invalid env_overrides). Test stdout BPB parsing with mock output. Test partial failure handling (1 of 3 seeds fails). Test timeout behavior.

**Step 2 — Implement**:
1. Config loading and validation (env_overrides schema)
2. Subprocess invocation with SEED env var injection, timeout = MAX_WALLCLOCK_SECONDS + 120s buffer
3. Stdout parsing for val_bpb (regex: `val_bpb:<float>`), capture stderr for crash diagnostics
4. Fallback parsing from training log file
5. **Error handling**: On subprocess crash/timeout: capture error, write partial result with `"error": "..."` field. If 1 of 3 seeds fails, report with n=2 and flag `"reduced_power": true`. If 2+ seeds fail, mark entire condition as failed.
6. Per-run metrics capture (JSON-lines from stdout)
7. Checkpoint/log path capture for downstream scripts
8. Run 3 seeds × 2 conditions, collect into raw_runs.json schema
9. Call log_experiment to append to experiment_log.json

**Verification**: `pytest tests/causal/test_experiment_runner.py`. Integration: dry-run with ITERATIONS=10, MAX_WALLCLOCK_SECONDS=30. Assert raw_runs.json schema valid.

---

### S7: statistical_analysis.py — Effect Estimation (C6)
**Complexity**: Medium
**Components**: C6
**Dependencies**: S2 (common.py for statistical functions), S6 output
**Why this item**: Produces the decision gate output that drives the cycle.
**Why this order**: Consumes experiment runner output.
**Inputs**: `results/causal/cycle_N/raw_runs.json`
**Outputs**: `results/causal/cycle_N/ablation_results.json`, `tests/causal/test_statistical.py`

**Step 1 — Write tests**: Test with synthetic data (known effect). Assert CI contains true effect. Assert Holm-Bonferroni adjusts p-values upward. Assert decision_gate returns correct classification per threshold.

**Step 2 — Implement**:
1. Load raw_runs.json, extract per-seed BPB pairs (handle partial failures gracefully)
2. Compute paired differences, mean effect, bootstrapped 95% CI
3. Compute paired t-test p-value
4. Apply Holm-Bonferroni correction (if multiple comparisons)
5. Classify via decision_gate (confirmed/suggestive/null)
6. Compute platform transfer coefficient if both MLX + H100 data present

**Verification**: `pytest tests/causal/test_statistical.py`.

---

### S8: token_loss_decompose.py — Per-Token Analysis (C7)
**Complexity**: Medium
**Components**: C7
**Dependencies**: S2 (common.py for load_model, compute_bpb)
**Why this item**: Reveals which token categories drive loss — informs intervention selection.
**Why this order**: Parallel diagnostic, runs after S2.

**Step 1 — Write tests**: Test decomposition check with mock model output. Test frequency bucketing with known vocab. Test BPB contribution summation.

**Step 2 — Implement**:
1. Load model and validation data via common.py
2. Forward pass with reduction='none' for per-token losses
3. Decomposition verification: mean(per_token) matches aggregate within 1e-6
4. Build frequency buckets from tokenizer vocab (top-100, 100-500, 500-1024)
5. Classify boundary vs. mid-sequence tokens
6. Compute per-category statistics (mean loss, std, BPB contribution)

**Verification**: `pytest tests/causal/test_token_loss.py`. Assert decomposition_check.passed, buckets sum to total, BPB contributions sum to aggregate (±0.001).

---

### S9: quant_gap_analysis.py — Pre/Post Quantization (C8)
**Complexity**: Medium
**Components**: C8
**Dependencies**: S2 (common.py for load_model)
**Why this item**: Determines if quantization gap dominates causal effects.
**Why this order**: Parallel diagnostic, runs after S2.

**Step 1 — Write tests**: Test gap computation with mock BPB values. Test threshold check logic.

**Step 2 — Implement**:
1. Load model via common.py, eval pre-quant BPB
2. Import quantize/dequantize functions from train_gpt_mlx.py
3. Quantize → dequantize → eval post-quant BPB
4. Compute gap and threshold check (gap > 3× largest training effect)
5. Optional: per-token category comparison pre/post quant

**Verification**: `pytest tests/causal/test_quant_gap.py`. Assert quant_gap > 0, assert gap_exceeds_3x_threshold is boolean.

---

### S10: influence_proxy.py — Gradient Inner Product (C9)
**Complexity**: Medium
**Components**: C9
**Dependencies**: S2 (common.py for load_model)
**Why this item**: Scores training data shards by causal influence on validation loss.
**Why this order**: Parallel diagnostic, runs after S2.

**Step 1 — Write tests**: Test dot product computation with small mock gradients. Test CV calculation. Test skip recommendation threshold.

**Step 2 — Implement**:
1. Load model via common.py
2. Compute validation gradient via plain nn.value_and_grad (trainable params only)
3. **Memory check**: After first shard gradient, check peak memory. If extrapolated total > 80% available, warn and reduce --max-shards.
4. Iterate shards (sample 4096 tokens per shard), compute per-shard gradient
5. Compute dot product with validation gradient, call mx.eval() after each (hard requirement)
6. Sort scores descending
7. Compute variance check (CV threshold < 0.1 → skip recommendation)
8. Write influence_scores.json with variance_check and curriculum_skipped fields

**Verification**: `pytest tests/causal/test_influence.py`. Integration: run with --max-shards 5, assert scores sorted, assert CV non-negative.

---

### S11: gradient_attribution.py — Per-Layer Logging (C10)
**Complexity**: Complex
**Components**: C10
**Dependencies**: S1, S2 (common.py validates import compatibility; S11 operates on source file, S2 on module — both must work with same train_gpt_mlx.py version)
**Why this item**: Per-layer gradient norms reveal which components drive loss reduction.
**Why this order**: Independent diagnostic, but depends on S2 for compatibility assurance.

**Step 1 — Write tests**: Test LAST occurrence targeting on mock source. Test sentinel validation passes on current train_gpt_mlx.py. Test sentinel fails on modified source. Test JSON-lines parsing.

**Step 2 — Implement**:
1. Prerequisite check: verify sentinel strings still exist in current train_gpt_mlx.py
2. Read train_gpt_mlx.py source, find LAST occurrence of `accumulate_flat_grads`
3. Validate sentinel: assert BOTH `train_loss = train_loss + loss` AND `lr_mul` within ±5 lines (substring match)
4. Insert gradient norm logging code after the call site
5. Write `train_gpt_mlx_instrumented.py`
6. Execute via subprocess with short training (for testing)
7. Parse JSON-lines output, compute phase boundaries from lr_mul transitions
8. Compute per-phase correlations between layer norms and val_loss

**Verification**: `pytest tests/causal/test_gradient_attr.py`. Assert sentinel validation passes on current source. Short training produces valid JSON-lines.

---

### S12: Submission Assembly (Phase 5)
**Complexity**: Medium
**Components**: Integration
**Dependencies**: At least one confirmed effect from S6→S7 cycle, OR engineering fallback
**Why this item**: Translates causal findings into a competition submission.
**Why this order**: Terminal step — requires all prior work.
**Inputs**: Best causal finding (or SOTA baseline), competition constraints
**Outputs**: `records/track_10min_16mb/YYYY-MM-DD_causal_BPB/` submission package

1. Map confirmed causal findings to train_gpt.py code changes
2. Verify artifact ≤ 16MB and training ≤ 10min on 8×H100
3. Run 3-seed validation on H100
4. Produce README.md with ablation table documenting causal findings
5. Produce submission.json with metadata
6. If no causal findings: compose existing tricks (engineering fallback R5.3)

**Verification**: Assert artifact_size ≤ 16,000,000 bytes. Assert 3-seed BPB reported with mean and std. Assert all required submission files present.

## Dependency Graph

```
S1 ──→ S2 ──→ S3 ──→ S4a ──→ S4b ──→ S5
        │                                │
        ├──→ S6 ──→ S7                   │
        │                                │
        ├──→ S8 (parallel diagnostic)    │
        ├──→ S9 (parallel diagnostic)    │
        ├──→ S10 (parallel diagnostic)   │
        │                                │
        └──→ S11 (parallel diagnostic, also depends on S2 for compatibility)

        S7 result → feed back to S3 (--append-experiment) → S4a/S4b → repeat

        S12 ← after cycle produces confirmed effect OR time gate
```

## Discovery-Adjust Cycle Execution

**This cycle is researcher-driven (manual).** The scripts/causal/README.md (produced in S4a) documents the cycle protocol. The researcher reads each output JSON, checks the `recommendation` and `cycle_status` fields, and decides the next action.

```
Cycle 0 (build scripts S1-S11, then run):
  1. Run S3 (extract_interventions.py) on records/
  2. Run S4a+S4b (estimate_dag.py) → cycle_0/causal_dag.json
  3. Run S5 (identifiability_check.py) → cycle_0/identifiability_report.json
  4. Run S8-S11 in parallel (diagnostic probes)
  5. Read next_intervention from causal_dag.json
  6. Researcher creates treatment + control config JSONs

Cycle 1+ (researcher-driven loop):
  1. Run S6 (experiment_runner.py) → cycle_N/raw_runs.json
  2. Run S7 (statistical_analysis.py) → cycle_N/ablation_results.json
  3. Check decision_gate: confirmed → S12, null → continue
  4. Run S3 --append-experiment cycle_N/raw_runs.json
  5. Run S4a+S4b --previous-dag cycle_N-1/causal_dag.json → cycle_N/causal_dag.json
  6. Read updated next_intervention
  7. Check stop conditions (max 4 cycles, 3 null streak, 2-day gate)
  8. Update experiment_log.json with cycle_status

Final:
  Run S12 (submission assembly) with best findings
```

## Risk Mitigations in Plan

| Risk | Plan Mitigation |
|------|----------------|
| FCI degenerate at n=20 | S4 uses expert DAG as primary (TD-9). FCI is validation only. |
| Most records lack structured tables | S3 has three-tier parser (Format A/B/C). Field coverage is computed, not assumed. |
| MLX findings don't transfer to H100 | S6 supports --platform flag. S7 computes transfer coefficient. BC-4 prevents false claims. |
| Influence scores have low variance | S10 includes CV check. CV < 0.1 → skip recommendation, document as null. |
| Patch site drifts in train_gpt_mlx.py | S11 uses LAST occurrence targeting + dual sentinel (train_loss + lr_mul). |
| All causal angles null | S12 includes engineering fallback (R5.3). Null results documented as scientific contribution. |
