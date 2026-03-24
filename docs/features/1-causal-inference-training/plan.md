# Plan: Causal Inference for Parameter-Efficient LM Training

## Implementation Order

Each step follows **test-first**: write pytest tests → implement until tests pass.

```
Phase A: Foundation
  S1 → S2

Phase B: DAG Discovery
  S3 → S4 → S5

Phase C: Experiment Pipeline (parallel with B after S2)
  S6 → S7

Phase D: Diagnostic Probes (parallel with B+C after S2)
  S8, S9, S10, S11

Phase E: Integration
  S12
```

## Steps

### S1: Project Setup
**Complexity**: Simple | **Deps**: None

1. `uv add causal-learn statsmodels networkx graphviz scipy pytest`
2. `brew install graphviz` if `dot -V` fails
3. Create `scripts/causal/`, `results/causal/`, `results/causal/diagnostics/`, `tests/causal/`
4. Add `results/causal/` to `.gitignore`
5. Create `scripts/causal/__init__.py`

**Verify**: All imports succeed, `dot -V` exits 0.

---

### S2: common.py (C1)
**Complexity**: Medium | **Deps**: S1

**Prerequisite**: `python -c "import train_gpt_mlx"` must exit 0 in <5s with no MLX warnings. If fails, extract model classes into `scripts/causal/model.py`.

**Tests** (`tests/causal/test_common.py`): One test per function — load_submission_json, load_model, compute_bpb, paired_ttest, holm_bonferroni, decision_gate, log_experiment, get_cycle_dir, dag_diff.

**Implement** (`scripts/causal/common.py`):
1. `load_submission_json(path)` — parse submission.json
2. `load_model(checkpoint_path, config_overrides)` — import GPT via `__name__` guard (TD-7), load weights, load tokenizer
3. `compute_bpb(model, val_tokens, sp_model)` — reuse eval_val logic
4. `paired_ttest(treatment, control)` — scipy.stats.ttest_rel + bootstrap CI
5. `holm_bonferroni(p_values, alpha)` — wrap statsmodels multipletests
6. `decision_gate(effect_size, p_value, mde=0.002)` — "confirmed"/"suggestive"/"null"
7. `log_experiment(path, entry)` — append to experiment_log.json
8. `get_cycle_dir(base_path, cycle)` — create `results/causal/cycle_N/`
9. `dag_diff(old_dag, new_dag)` — edges added/removed/strengthened

---

### S3: extract_interventions.py (C2)
**Complexity**: Medium | **Deps**: S2

**Tests** (`tests/causal/test_extract.py`): Each parser format with sample README snippets. Field coverage computation. --append-experiment with mock data.

**Implement**:
1. submission.json pass — 6 core fields from all discovered records
2. Format A parser — `| Change | ... | Impact |` tables (~2 records)
3. Format B parser — `| | Base | This |` tables (~8 records), compute delta
4. Format C fallback — headings, bullets, blurb (~10 records)
5. Cross-reference pass — compute total delta from base_bpb
6. `--append-experiment` mode — convert experiment results to submission format
7. Field coverage metric

**Verify**: field_coverage ≥ 0.90 on actual records. All discovered submissions parsed.

---

### S4: estimate_dag.py (C3)
**Complexity**: Complex | **Deps**: S3

**Tests** (`tests/causal/test_estimate_dag.py`): Expert skeleton edges. Binary encoding dimensions. FCI on synthetic data. Near-degenerate case (n=20, correlated binary cols). LinAlgError caught → expert fallback. Edge tagging. --previous-dag diff. DOT renders.

**Implement**:
1. Expert-guided skeleton — hardcode known causal relationships, tag as `expert_imposed`
2. Binary encoding of interventions (presence/absence matrix)
3. FCI validation (causal-learn, Fisher-Z, alpha=0.01) with try/except for numerical errors
4. Degenerate detection (empty/fully-connected → keep expert only)
5. Edge tagging (expert_imposed → data_confirmed/data_contradicted/uncertain)
6. Marginal effect estimation per node
7. `next_intervention` recommendation (highest expected BPB improvement among uncertain edges)
8. `--previous-dag` mode (dag_diff, edge stability tracking)
9. DOT visualization via graphviz
10. Write `scripts/causal/README.md` — cycle protocol and CLI usage

**Verify**: ≥5 nodes. dag.png renders. next_intervention non-null. Edge tags valid.

---

### S5: identifiability_check.py (C4)
**Complexity**: Simple | **Deps**: S3, S4

**Tests** (`tests/causal/test_identifiability.py`): Synthetic interventions with known counts. Proceed/skip threshold. Combination enumeration.

**Implement**:
1. Count single-variable and multi-variable records
2. Identifiability score (fraction of testable edges)
3. Confounded pairs (always co-occurring interventions)
4. Proceed/skip recommendation (>50% multi-variable → skip)
5. Unexplored combinations with expected effects × interaction priors

---

### S6: experiment_runner.py (C5)
**Complexity**: Complex | **Deps**: S2

**Tests** (`tests/causal/test_experiment_runner.py`): Config validation. Stdout BPB parsing with mock output. Partial failure (1/3 seeds). Timeout handling.

**Implement**:
1. Config loading + validation (env_overrides schema)
2. Subprocess invocation with SEED env var, timeout = wallclock + 120s
3. Stdout parsing for `val_bpb:<float>`, stderr capture
4. Fallback parsing from training log file
5. Error handling: crash → partial result with error field; 1/3 fail → reduced_power flag; 2+/3 fail → condition failed
6. Per-run JSON-lines metrics capture
7. Checkpoint/log path capture
8. 3 seeds × 2 conditions → raw_runs.json
9. Append to experiment_log.json

**Verify**: Dry-run (ITERATIONS=10, MAX_WALLCLOCK_SECONDS=30). Valid raw_runs.json schema.

---

### S7: statistical_analysis.py (C6)
**Complexity**: Medium | **Deps**: S2, S6 output

**Tests** (`tests/causal/test_statistical.py`): Synthetic data with known effect. CI contains true effect. Holm-Bonferroni adjusts upward. Decision gate classification.

**Implement**:
1. Load raw_runs.json, extract per-seed BPB pairs (handle partial failures)
2. Paired differences, mean effect, bootstrapped 95% CI
3. Paired t-test p-value
4. Holm-Bonferroni correction
5. Decision gate classification
6. Platform transfer coefficient if MLX + H100 data

---

### S8: token_loss_decompose.py (C7)
**Complexity**: Medium | **Deps**: S2 | **Prereq**: Saved checkpoint exists

**Tests** (`tests/causal/test_token_loss.py`): Decomposition check with mock output. Frequency bucketing. BPB contribution summation.

**Implement**:
1. Load model + validation data via common.py
2. Forward pass with reduction='none' for per-token losses
3. Decomposition verification: mean(per_token) matches aggregate within 1e-6
4. Frequency buckets (top-100, 100-500, 500-1024)
5. Boundary vs. mid-sequence classification
6. Per-category statistics

**Verify**: decomposition_check.passed. Buckets sum to total. BPB contributions sum to aggregate (±0.001).

---

### S9: quant_gap_analysis.py (C8)
**Complexity**: Medium | **Deps**: S2 | **Prereq**: Saved checkpoint exists

**Tests** (`tests/causal/test_quant_gap.py`): Gap computation with mock BPB. Threshold check logic.

**Implement**:
1. Load model, eval pre-quant BPB
2. Quantize → dequantize → eval post-quant BPB (reuse train_gpt_mlx.py functions)
3. Gap and threshold check (gap > 3× largest training effect)
4. Optional: per-token category comparison pre/post quant

---

### S10: influence_proxy.py (C9)
**Complexity**: Medium | **Deps**: S2 | **Prereq**: Saved checkpoint + training shards

**Tests** (`tests/causal/test_influence.py`): Dot product with small mock gradients. CV calculation. Skip threshold.

**Implement**:
1. Load model via common.py
2. Validation gradient via plain nn.value_and_grad (trainable params only)
3. Memory check after first shard — warn and reduce --max-shards if needed
4. Iterate shards (4096 tokens/shard), per-shard gradient + dot product
5. `mx.eval()` after each (hard requirement)
6. Sort scores, compute CV, skip recommendation if CV < 0.1

**Verify**: --max-shards 5, scores sorted, CV non-negative.

---

### S11: gradient_attribution.py (C10)
**Complexity**: Complex | **Deps**: S1, S2

**Tests** (`tests/causal/test_gradient_attr.py`): LAST occurrence targeting. Sentinel validation on current source. Sentinel fails on modified source. JSON-lines parsing.

**Implement**:
1. Verify sentinel strings exist in current train_gpt_mlx.py
2. Find LAST `accumulate_flat_grads`, validate dual sentinel (train_loss + lr_mul within ±5 lines)
3. Insert gradient norm logging, write `train_gpt_mlx_instrumented.py`
4. Execute via subprocess (short training for test)
5. Parse JSON-lines, compute phase boundaries from lr_mul transitions
6. Per-phase correlations between layer norms and val_loss

---

### S12: Submission Assembly
**Complexity**: Medium | **Deps**: Confirmed effect from cycle OR engineering fallback

**Tests** (`tests/causal/test_submission.py`): submission.json schema. Artifact size ≤ 16MB. README.md required sections. All files present.

**Implement**:
1. Map causal findings to train_gpt.py code changes
2. Verify artifact ≤ 16MB, training ≤ 10min on 8×H100
3. 3-seed validation on H100
4. README.md with ablation table
5. submission.json with metadata
6. Engineering fallback (R5.3) if no causal findings

## Dependency Graph

```
S1 → S2 → S3 → S4 → S5
      │
      ├→ S6 → S7           S7 → feed back to S3 → S4 (cycle)
      │
      ├→ S8, S9, S10       (parallel diagnostics)
      └→ S11

      S12 ← confirmed effect OR time gate
```

## Discovery-Adjust Cycle

**Manual, researcher-driven.** Protocol documented in `scripts/causal/README.md`.

```
Cycle 0: S3 → S4 → S5 + S8-S11 in parallel → read next_intervention → create configs
Cycle 1+: S6 → S7 → decision gate → S3 --append → S4 --previous-dag → repeat
Stop: confirmed effect | 3 null streak | 4 cycles max | 2-day gate
Final: S12 with best findings
```

## Risks

| Risk | Mitigation |
|------|-----------|
| FCI degenerate (n=20) | Expert DAG primary. FCI validation only. |
| Records lack structured tables | Three-tier parser. Coverage computed, not assumed. |
| MLX→H100 transfer fails | S6 --platform flag. S7 transfer coefficient. |
| Low influence variance | S10 CV check. CV < 0.1 → documented null. |
| Patch site drifts | S11 LAST occurrence + dual sentinel. |
| All causal angles null | S12 engineering fallback. Null results = scientific contribution. |
