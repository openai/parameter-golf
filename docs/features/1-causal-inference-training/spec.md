# Specification: Causal Inference for Parameter-Efficient LM Training

## Overview

Apply formal causal inference methods to the parameter golf challenge to discover principled mechanisms of loss reduction, then translate findings into a competitive submission (target: ≤1.12 BPB, 3-seed mean). Four causal angles are tested sequentially with hard decision gates; null results trigger engineering fallback.

## Definitions

| Term | Definition |
|------|-----------|
| BPB | Bits-per-byte: tokenizer-agnostic compression metric = (mean_loss / ln(2)) × (total_tokens / total_bytes), computed as aggregate over all validation batches (see train_gpt_mlx.py:812-813) |
| MDE | Minimum detectable effect: ~0.002 BPB (n=3 seeds, σ=0.0005, α=0.01, power=0.80) |
| Causal effect | BPB difference between treatment and control, averaged over ≥3 seeds, with p<0.01 via paired t-test |
| Null result | Effect size < MDE or p ≥ 0.01; reported as suggestive only |
| Decision gate | Binary proceed/skip checkpoint at end of each phase |
| Retrospective DAG | Directed acyclic graph over architectural choices → BPB, estimated from existing leaderboard records |
| Influence score | Gradient inner product between training shard gradient and validation loss gradient |

## Requirements

### Phase 1: Retrospective Causal DAG (FR-1)

**Input:** All submission directories in `records/track_10min_16mb/` (20 as of 2026-03-24), each containing `README.md` with ablation tables and `submission.json` with BPB outcomes. The extraction script dynamically discovers all subdirectories rather than assuming a fixed count.

**Phase ordering rationale:** Phases follow the pre-mortem recommendation of highest-ROI first (architecture > loss decomposition > gradient probes > data curriculum), not the hypothesis numbering (H1-H4) in the PRD decomposition.

**R1.1 — Data extraction script**
- Parse each submission's README.md to extract intervention-outcome pairs into a structured CSV/JSON table
- Fields: `submission_id`, `date`, `author`, `base_bpb`, `final_bpb`, `interventions[]` (each with `name`, `category`, `delta_bpb`, `delta_source`)
- Categories: `architecture` (layers, dims, attention), `quantization` (int5/6/8, GPTQ-lite), `optimization` (EMA, SWA, LR, warmdown), `encoding` (RoPE, positional), `data` (vocab, tokenizer)
- Handle missing delta values (mark as `null`, note simultaneous changes)

**R1.2 — DAG estimation**
- Fit a structural causal model over the extracted intervention-outcome pairs
- Nodes: architectural choices (num_layers, mlp_mult, attention_variant, positional_encoding, weight_averaging, quantization_method, compression)
- Edges: directed causal effects on BPB
- Method: PC algorithm or equivalent constraint-based causal discovery, with significance threshold α=0.05
- Contingency: If PC algorithm produces degenerate output (empty or fully-connected graph due to small sample size n≈20), fall back to expert-guided DAG based on temporal ordering of submissions and domain knowledge. Document which edges are data-driven vs. expert-imposed.
- Output: adjacency matrix + visualization (networkx/graphviz DOT)

**R1.3 — Identifiability check**
- Count records where only 1 variable changes (cleanest quasi-experimental data)
- Count records where 3+ variables change simultaneously
- If >50% of records change 3+ variables: DAG is under-identified, skip Phase 2
- Report: identifiability score, confounded pairs, partial identification bounds where possible

**R1.4 — Interaction discovery**
- From DAG, identify unexplored component combinations (pairs not co-occurring in any submission)
- Rank by expected causal effect (sum of individual effects × interaction prior). Interaction prior defaults to 1.0 (no prior expectation of synergy or antagonism). Set prior to 1.2 if domain knowledge suggests synergy (e.g., quantization method + MLP width). Document all non-default priors with rationale.
- Output: ranked list of candidate combinations for Phase 2

**Acceptance Criteria:**
- [ ] AC-1.1: Extraction script dynamically discovers and parses all submissions into structured format with ≥90% field coverage. Coverage = (non-null fields extracted) / (submissions × 6 core fields: submission_id, date, author, base_bpb, final_bpb, interventions). Submissions with no parseable ablation table count as 0/6 in the denominator.
- [ ] AC-1.2: DAG produced with ≥5 nodes and interpretable edges; adjacency matrix serialized to JSON
- [ ] AC-1.3: Identifiability report produced with proceed/skip recommendation
- [ ] AC-1.4: ≥3 unexplored combinations identified (or documented reason why fewer exist)

### Phase 2: Architecture Ablation (FR-3)

**Precondition:** Phase 1 decision gate passes (DAG reveals unexplored interactions).

**R2.1 — Experiment runner**
- Script that runs paired experiments: 3 seeds × 2 conditions (treatment vs. control)
- Treatment: top DAG-suggested component combination from R1.4
- Control: current SOTA configuration (11L, 3x MLP, XSA, SmearGate, BigramHash, Partial RoPE, LN Scale, EMA, GPTQ-lite)
- Seeds: fixed set [42, 137, 256] for reproducibility
- Platform: MLX first (local iteration), then H100 validation (3 seeds)

**R2.2 — Statistical analysis**
- Paired t-test (treatment vs. control) per seed
- Report: mean effect, 95% CI (bootstrapped, 10,000 resamples), p-value
- Apply Holm-Bonferroni correction if testing multiple combinations
- Effect size threshold: ≥0.002 BPB for "confirmed"; <0.002 for "suggestive"

**R2.3 — MLX-to-H100 validation**
- If MLX shows promising signal (effect > 0.001 BPB): validate on H100 with 3 seeds
- Compare MLX and H100 effect sizes; report platform transfer coefficient
- If H100 effect < MDE: report MLX-only finding, do not claim H100 validity

**Acceptance Criteria:**
- [ ] AC-2.1: Experiment runner executes 3-seed paired design with reproducible seeds
- [ ] AC-2.2: Statistical report includes mean, CI, p-value with Holm-Bonferroni correction
- [ ] AC-2.3: Decision gate produces proceed (effect > MDE) or skip recommendation
- [ ] AC-2.4: If H100 validation run, platform transfer coefficient reported (ratio of H100 effect to MLX effect). Coefficient < 0.5 flags platform divergence requiring investigation.

### Phase 3: Loss Decomposition (FR-4)

**Precondition:** Phase 2 null or skipped.

**R3.1 — Per-token loss attribution**
- Add per-token loss logging to train_gpt_mlx.py (no reduction in cross_entropy)
- Decompose validation loss into per-token contributions
- Classify tokens by frequency bucket (top-100, 100-500, 500-1024) and context type (boundary vs. mid-sequence)
- Verification: per-token losses must aggregate (mean) to within 1e-6 of the existing reduced loss value on the same batch, confirming decomposition correctness

**R3.2 — Pre-quant vs. post-quant causal pathway**
- Measure BPB with raw weights vs. quantized weights at same checkpoint
- Quantization gap = post_quant_bpb - pre_quant_bpb (~0.017 expected)
- If quant gap > 3× largest training-side causal effect: restrict all causal claims to post-quant outcomes
- Compare which token categories lose most from quantization

**R3.3 — Causal transfer analysis**
- Identify tokens where training loss reduction transfers to validation loss reduction (high causal transfer)
- Identify tokens where training loss reduction does NOT transfer (confounded/overfitting)
- Report: fraction of tokens in each category, BPB contribution of each

**Acceptance Criteria:**
- [ ] AC-3.1: Per-token loss decomposition runs on the full validation set (all val shards) from one checkpoint on Apple Silicon MLX within 2 hours
- [ ] AC-3.2: Quantization gap measured and threshold applied to restrict claims
- [ ] AC-3.3: Token categories identified with causal transfer scores

### Phase 4: Gradient/Data Probes (FR-2, FR-5)

**Precondition:** Phase 3 null or skipped.

**R4.1 — Gradient inner product influence proxy**
- Script (~50 lines MLX) computing gradient inner product between:
  - Training shard gradient: ∇_θ L_train(shard_k)
  - Validation gradient: ∇_θ L_val
- Uses `nn.value_and_grad` API (train_gpt_mlx.py line 910-912)
- Runs on saved checkpoint (no training required)
- Output: influence score per shard, sorted descending

**R4.2 — Shard influence variance check**
- Compute variance of influence scores across shards
- If coefficient of variation < 0.1: all shards similarly influential → data curriculum unlikely to help
- If CV ≥ 0.1: sufficient heterogeneity for reordering experiment

**R4.3 — Causal curriculum retraining**
- Reorder training shards by influence score (highest first)
- Retrain with 3 seeds, compare BPB vs. default shard order
- Paired t-test, same statistical protocol as R2.2

**R4.4 — Per-layer gradient attribution**
- Log per-layer gradient norms at validation checkpoints during training
- Correlate per-layer gradient norm trajectories with val_loss reduction across training phases
- Identify layers with high causal signal (high correlation) vs. noise (low correlation)
- Stratify by training phase: warmup (first 20 steps), main (step 20 to warmdown onset), warmdown (warmdown onset to end). Phase boundaries determined at runtime from lr_mul transitions. MLX supports both step-based (when max_wallclock_seconds ≤ 0) and wallclock-based warmdown; the detection script must check which mode is active and compute boundaries accordingly.

**Acceptance Criteria:**
- [ ] AC-4.1: Influence proxy script runs on MLX, produces per-shard scores in <2 hours
- [ ] AC-4.2: Variance check produces proceed/skip recommendation
- [ ] AC-4.3: If curriculum experiment run, statistical report follows R2.2 protocol
- [ ] AC-4.4: Per-layer attribution logged and correlated with val_loss trajectory

### Phase 5: Engineering Fallback & Submission (FR-3)

**Precondition:** Always reached (either with causal findings or without).

**R5.1 — Finding integration**
- For each phase that produced confirmed causal effect (>MDE):
  - Map finding to concrete code change in train_gpt.py
  - Verify change runs within 10min on 8×H100
  - Verify artifact ≤ 16MB (16,000,000 bytes decimal)

**R5.2 — Submission assembly**
- Compose best causal findings with existing SOTA tricks (EMA, GPTQ-lite, XSA)
- Run 3-seed validation on H100
- Produce: train_gpt.py, submission.json, README.md with ablation table, train_log.txt

**R5.3 — Engineering-only fallback**
- If no causal angle produces confirmed effect: compose existing tricks without causal novelty
- Document null results as scientific contribution
- Target: match or beat current SOTA (1.1233 BPB) via engineering stacking

**Acceptance Criteria:**
- [ ] AC-5.1: Each causal finding maps to a code change that passes constraint checks
- [ ] AC-5.2: Submission package complete with all required files
- [ ] AC-5.3: 3-seed BPB results reported with mean and std. Target: ≤1.12 BPB (from PRD). Stretch: ≤1.1233 BPB (current SOTA). Acceptance does not require hitting target — null causal results with documented findings and engineering submission are acceptable per R5.3.

## Non-Functional Requirements

| ID | Requirement | Verification |
|----|-------------|-------------|
| NFR-1 | All causal diagnostic scripts run on Apple Silicon MLX within 2 hours | Timed execution on M-series Mac |
| NFR-2 | No causal instrumentation overhead in final submission runs | Diff between submission train_gpt.py and instrumented version shows no logging code |
| NFR-3 | All causal claims include effect size, 95% CI, seed count, and p-value | Review of output reports |
| NFR-4 | Multiple comparison correction (Holm-Bonferroni) applied across all 4 hypotheses (derived from PRD Review 0 correction) | Statistical report includes corrected thresholds |
| NFR-5 | Final submission reproducible: same seeds produce same BPB within ±0.0002 (derived from competition reproducibility requirements) | 3 independent runs with fixed seeds |

## Behavioral Constraints

| ID | Must NOT | Rationale |
|----|----------|-----------|
| BC-1 | Claim causal effects without paired seed design (≥3 seeds per condition) | Inter-seed σ ~0.0005 BPB makes single-run comparisons meaningless |
| BC-2 | Use IHVP-based influence functions inside the 10-min training loop | Computationally infeasible at 85ms/step with no headroom |
| BC-3 | Train on validation data | Competition rules explicitly prohibit this |
| BC-4 | Claim H100 validity from MLX-only results | Quantization, FlashAttention 3, multi-GPU behavior differ |
| BC-5 | Report sub-MDE effects as confirmed | Statistical honesty; effects < 0.002 BPB are suggestive only |

## Scope Boundaries

**In scope:**
- Offline causal analysis scripts (Python, MLX)
- Modifications to train_gpt.py and train_gpt_mlx.py for instrumentation (diagnostic only)
- Paired seed experiment runner
- Competitive submission assembly

**Out of scope:**
- General-purpose causal inference library
- Full IHVP-based influence functions
- Multi-token prediction (MTP) causal analysis
- Test-time training (TTT) causal analysis
- Theoretical guarantees about causal identification

## Data Flow

```
records/track_10min_16mb/  ──[R1.1 extract]──▶  interventions.json
                                                      │
                                                      ▼
                                               [R1.2 DAG estimate]
                                                      │
                                                      ▼
                                               causal_dag.json + dag.png
                                                      │
                                               [R1.3 identifiability]
                                                      │
                                            ┌─────────┴─────────┐
                                         proceed              skip
                                            │                    │
                                     [R2 architecture]           │
                                            │                    │
                                      ┌─────┴─────┐             │
                                   confirmed    null             │
                                      │           │              │
                                      │           └──────┬───────┘
                                      │                  │
                                      │           [R3 loss decomp]
                                      │                  │
                                      │            ┌─────┴─────┐
                                      │         confirmed    null
                                      │            │           │
                                      │            │    [R4 gradient probes]
                                      │            │           │
                                      │            │     ┌─────┴─────┐
                                      │            │  confirmed    null
                                      │            │     │           │
                                      ▼            ▼     ▼           ▼
                                   [R5 submission + integration]  [R5.3 fallback]
                                                      │
                                                      ▼
                                          records/track_10min_16mb/
                                            YYYY-MM-DD_causal_BPB/
```

## File Structure

```
docs/features/1-causal-inference-training/
├── prd.md
├── spec.md (this file)
├── .meta.json

scripts/causal/
├── extract_interventions.py    # R1.1: Parse leaderboard READMEs
├── estimate_dag.py             # R1.2: Structural causal model
├── identifiability_check.py    # R1.3: DAG identifiability analysis
├── experiment_runner.py        # R2.1: Paired seed ablation runner
├── statistical_analysis.py     # R2.2: Paired t-test + Holm-Bonferroni
├── token_loss_decompose.py     # R3.1: Per-token loss attribution (MLX)
├── quant_gap_analysis.py       # R3.2: Pre/post-quant BPB comparison
├── influence_proxy.py          # R4.1: Gradient inner product (~50 lines)
├── shard_variance_check.py     # R4.2: Influence score heterogeneity
├── gradient_attribution.py     # R4.4: Per-layer gradient logging
└── README.md                   # Script documentation and usage

results/causal/
├── interventions.json          # R1.1 output
├── causal_dag.json             # R1.2 output
├── dag.png                     # R1.2 visualization
├── identifiability_report.json # R1.3 output
├── ablation_results.json       # R2.2 output
├── token_analysis.json         # R3.3 output
├── influence_scores.json       # R4.1 output
└── gradient_attribution.json   # R4.4 output
```

## Decision Gate Protocol

Each phase ends with a binary gate:

```python
def decision_gate(phase_results):
    if phase_results.max_effect_size >= MDE:  # 0.002 BPB
        if phase_results.p_value < 0.01:
            return "proceed_to_engineering"  # Confirmed effect → integrate
        else:
            return "suggestive_only"  # Below significance → next phase
    else:
        return "null_result"  # Below MDE → next phase
```

On confirmed effect, the phase result is carried forward to Phase 5 integration. Subsequent phases are skipped — stacking multiple causal findings across phases is out of scope for this iteration.

After Phase 4 null: trigger engineering fallback (Phase 5 with R5.3).

Note: FR-4 (per-layer gradient attribution) is deprioritized if earlier phases confirm effects. R4.4 may be run as an optional standalone diagnostic regardless of gate outcomes, but is not required if Phase 2 or 3 produces a confirmed result.
