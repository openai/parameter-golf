# Tasks: Causal Inference for Parameter-Efficient LM Training

## Phase A: Foundation

### T1: Install dependencies and create directory structure [S1]
- [ ] Run `uv add causal-learn statsmodels networkx graphviz scipy pytest`
- [ ] Run `brew install graphviz` if `dot -V` fails
- [ ] Create directories: `scripts/causal/`, `results/causal/`, `results/causal/diagnostics/`, `tests/causal/`
- [ ] Add `results/causal/` to `.gitignore`
- [ ] Create `scripts/causal/__init__.py`
- [ ] Run `uv pip check` — no dependency conflicts
- [ ] Verify: `python -c "import causallearn; import statsmodels; import networkx; import graphviz; import scipy"` exits 0
**Deps**: None | **Done**: All imports succeed, `dot -V` exits 0

### T2: Validate train_gpt_mlx.py import safety [S2]
- [ ] Run `python -c "import train_gpt_mlx; print(train_gpt_mlx.GPT)"` — must exit 0 in <15s, print class ref
- [ ] If fails: extract GPT/Block/CausalSelfAttention/MLP/Rotary into `scripts/causal/model.py`
**Deps**: T1 | **Done**: GPT class accessible via import

### T3: Write tests for common.py [S2]
- [ ] Create `tests/causal/test_common.py` with one test per function:
  - test_load_submission_json (known submission.json)
  - test_load_model (returns model, computes forward pass)
  - test_compute_bpb (matches known baseline ±0.001)
  - test_paired_ttest (synthetic data, known effect)
  - test_holm_bonferroni (adjusts p-values upward)
  - test_decision_gate (each threshold: confirmed/suggestive/null)
  - test_log_experiment (append + read back)
  - test_get_cycle_dir (creates directory)
  - test_dag_diff (detects added/removed edges)
**Deps**: T2 | **Done**: All tests written, all fail (RED)

### T4a: Implement common.py — model loading [S2]
- [ ] Implement load_submission_json, load_model, compute_bpb
- [ ] If load_model requires model.py fallback from T2, complete extraction first
- [ ] `pytest tests/causal/test_common.py -k "load or bpb"` green
**Deps**: T3 | **Done**: Model loading + BPB tests pass

### T4b: Implement common.py — statistics + utilities [S2]
- [ ] Implement paired_ttest, holm_bonferroni, decision_gate
- [ ] Implement log_experiment, get_cycle_dir, dag_diff
- [ ] `pytest tests/causal/test_common.py` all green
**Deps**: T4a | **Done**: All common.py tests pass (GREEN)

## Phase B: DAG Discovery

### T5: Write tests for extract_interventions.py [S3]
- [ ] Create test fixtures from 3 actual READMEs (one per format A/B/C)
- [ ] Create `tests/causal/test_extract.py`:
  - test_format_a_parser (Change/Impact table)
  - test_format_b_parser (Base/This comparison)
  - test_format_c_fallback (prose only)
  - test_field_coverage_computation
  - test_append_experiment_mode (mock raw_runs.json)
  - test_unknown_format_fallback
**Deps**: T4b | **Done**: All tests written, all fail

### T6: Implement extract_interventions.py [S3]
- [ ] Implement submission.json pass (6 core fields)
- [ ] Implement Format A, B, C parsers
- [ ] Implement cross-reference pass and --append-experiment mode
- [ ] Compute field_coverage metric
- [ ] `pytest tests/causal/test_extract.py` all green
- [ ] Integration: run on actual records, assert field_coverage ≥ 0.90
**Deps**: T5 | **Done**: Tests pass, field_coverage ≥ 0.90 on real records

### T7: Write tests for estimate_dag.py [S4]
- [ ] Create `tests/causal/test_estimate_dag.py`:
  - test_expert_skeleton_edges
  - test_binary_encoding_dimensions
  - test_fci_synthetic_data (known structure)
  - test_fci_near_degenerate (n=20, correlated binary cols)
  - test_linalg_error_fallback
  - test_edge_tagging_logic
  - test_previous_dag_diff
  - test_dot_renders
**Deps**: T4b | **Done**: All tests written, all fail

### T8: Implement estimate_dag.py — expert skeleton + encoding [S4]
- [ ] Expert-guided skeleton builder (known causal edges, tagged `expert_imposed`)
- [ ] Binary encoding (presence/absence matrix)
- [ ] Marginal effect estimation per node
- [ ] DOT visualization via graphviz
- [ ] Output causal_dag.json (cycle=0, method="expert_guided")
**Deps**: T7 | **Done**: Expert skeleton tests pass, dag.png renders

### T9: Implement estimate_dag.py — FCI validation + cycle [S4]
- [ ] FCI validation (causal-learn, Fisher-Z, alpha=0.01) with try/except
- [ ] Degenerate detection (empty/fully-connected → keep expert)
- [ ] Edge tagging (data_confirmed/data_contradicted/uncertain)
- [ ] next_intervention recommendation
- [ ] --previous-dag mode (dag_diff, edge stability)
- [ ] `pytest tests/causal/test_estimate_dag.py` all green
**Deps**: T8 | **Done**: All estimate_dag tests pass

### T10: Write scripts/causal/README.md [S4]
- [ ] Document cycle protocol (Cycle 0, Cycle 1+, stop conditions)
- [ ] CLI usage for each script
- [ ] Prerequisites and data requirements
**Deps**: T9 | **Done**: README covers all 10 scripts with examples

### T11a: Write tests for identifiability_check.py [S5]
- [ ] Create `tests/causal/test_identifiability.py` (synthetic interventions, threshold, combinations)
**Deps**: T4b | **Done**: All tests written, all fail (RED)

### T11b: Implement identifiability_check.py [S5]
- [ ] Implement: single/multi-variable counts, identifiability score, confounded pairs
- [ ] Proceed/skip recommendation, unexplored combinations
- [ ] `pytest tests/causal/test_identifiability.py` all green
**Deps**: T11a, T9 | **Done**: Tests pass, recommendation is "proceed" or "skip"

## Phase C: Experiment Pipeline

### T12: Write tests for experiment_runner.py [S6]
- [ ] Create `tests/causal/test_experiment_runner.py`:
  - test_config_validation (reject invalid env_overrides)
  - test_stdout_bpb_parsing (mock output, LAST occurrence)
  - test_partial_failure (1/3 seeds fails → reduced_power)
  - test_timeout_handling
  - test_truncated_stdout (no val_bpb line)
**Deps**: T4b | **Done**: All tests written, all fail

### T13: Implement experiment_runner.py [S6]
- [ ] Config loading + validation
- [ ] Subprocess invocation with SEED, timeout = wallclock + 120s
- [ ] Parse LAST `val_bpb:<float>` from complete stdout, capture stderr
- [ ] Fallback log parsing, error handling (crash/partial/timeout)
- [ ] Per-run metrics, checkpoint/log path capture
- [ ] Support --platform flag (mlx|h100) — selects train_gpt_mlx.py vs train_gpt.py
- [ ] 3 seeds × 2 conditions → raw_runs.json, append to experiment_log.json
- [ ] `pytest tests/causal/test_experiment_runner.py` all green
- [ ] Integration: dry-run ITERATIONS=10, valid raw_runs.json
**Deps**: T12 | **Done**: Tests pass, dry-run produces valid schema

### T14: Implement statistical_analysis.py [S7]
- [ ] Write `tests/causal/test_statistical.py` (synthetic data, CI, Holm-Bonferroni, decision gate)
- [ ] Implement: load raw_runs, paired differences, bootstrap CI, t-test, correction, gate
- [ ] Platform transfer coefficient
- [ ] `pytest tests/causal/test_statistical.py` all green
**Deps**: T4b | **Done**: Tests pass

## Phase D: Diagnostic Probes (parallel)

### T15: Implement token_loss_decompose.py [S8]
- [ ] Write `tests/causal/test_token_loss.py` (decomposition check, bucketing, summation)
- [ ] Implement: load model, forward pass reduction='none', decomposition verify
- [ ] Frequency buckets, boundary/mid-sequence classification, per-category stats
- [ ] `pytest tests/causal/test_token_loss.py` all green
**Deps**: T4b, checkpoint exists | **Done**: decomposition_check.passed, buckets sum correctly

### T16: Implement quant_gap_analysis.py [S9]
- [ ] Write `tests/causal/test_quant_gap.py` (gap computation, threshold check)
- [ ] Implement: load model, pre-quant BPB, quantize→dequantize→post-quant BPB
- [ ] Gap and threshold check (import quantize_state_dict_int8/dequantize_state_dict_int8 from train_gpt_mlx.py via `__name__` guard, same pattern as T4a load_model)
- [ ] `pytest tests/causal/test_quant_gap.py` all green
**Deps**: T4b, checkpoint exists | **Done**: quant_gap > 0, gap_exceeds_3x_threshold is boolean

### T17: Implement influence_proxy.py [S10]
*Note: R4.2 (shard variance check) is integrated into this script per plan consolidation; no separate shard_variance_check.py is produced.*
- [ ] Write `tests/causal/test_influence.py` (dot product mock, CV, skip threshold)
- [ ] Implement: load model, val gradient, memory check, shard iteration + dot product
- [ ] mx.eval() after each, sort scores, CV check, skip recommendation
- [ ] `pytest tests/causal/test_influence.py` all green
**Deps**: T4b, checkpoint + shards exist | **Done**: --max-shards 5 produces sorted scores, CV non-negative

### T18: Implement gradient_attribution.py [S11]
- [ ] Write `tests/causal/test_gradient_attr.py` (LAST occurrence, sentinel, ast.parse() syntax, JSON-lines)
- [ ] Implement: find LAST accumulate_flat_grads, validate dual sentinel
- [ ] Insert logging, write instrumented copy, ast.parse() validation
- [ ] Execute short training, parse JSON-lines, phase boundaries, correlations
- [ ] `pytest tests/causal/test_gradient_attr.py` all green
**Deps**: T2, T4 | **Done**: Sentinel passes on current source, patched file is valid Python

## Phase E: Integration

### T19: Implement submission assembly [S12]
- [ ] Write `tests/causal/test_submission.py` (schema, size check, required sections, files present)
- [ ] Map causal findings → train_gpt.py changes
- [ ] Verify artifact ≤ 16MB, training ≤ 10min
- [ ] 3-seed H100 validation (using S6 --platform h100)
- [ ] README.md with ablation table, submission.json
- [ ] Engineering fallback if no causal findings
**Deps**: Confirmed cycle effect OR time gate | **Done**: All submission files pass validation

## Parallel Groups

Tasks that can execute simultaneously:
- **Group 1** (after T4b): T5, T7, T11a, T12, T14, T15, T16, T17, T18
- **Group 2** (after T6): T8 → T9 → T10 → T11b (sequential within DAG pipeline)
- **Group 3** (after T13): cycle execution begins. T14 can start with synthetic data after T4b (Group 1); integration with real raw_runs.json from T13 is an optional follow-up, not a hard dependency.

## Summary

21 tasks across 5 phases. 3 Complex (T9, T13, T18), 11 Medium, 7 Simple.
Critical path: T1 → T2 → T3 → T4a → T4b → T5 → T6 → T7 → T8 → T9 → T11b (DAG pipeline).
Widest parallelism: 9 tasks after T4b completes.
