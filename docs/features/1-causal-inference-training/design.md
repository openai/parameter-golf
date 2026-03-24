# Design: Causal Inference for Parameter-Efficient LM Training

## Prior Art Research

### Codebase Patterns
- **Training loop hooks**: No callback API — instrumentation must be inserted inline. MLX: between `loss_and_grad_chunked` accumulation and `opt.step()` (train_gpt_mlx.py:1031-1044). PyTorch: between `backward()` and `opt.step()` (train_gpt.py:1018-1033).
- **Gradient access**: MLX `nn.value_and_grad` returns raw grad trees; `accumulate_flat_grads` produces a flat dict keyed by parameter name — natural insertion point for per-layer norm logging.
- **Quantization**: `quantize_state_dict_int8` / `dequantize_state_dict_int8` reusable for pre/post-quant BPB comparison. GPTQ-lite's MSE-optimal clip search is a self-contained ~20-line function.
- **BPB eval**: `eval_val()` + `build_sentencepiece_luts()` is the complete pipeline. Returns aggregate BPB; per-token decomposition requires skipping `.mean()` reduction.
- **Data shards**: 195 train shards (~100M tokens each), 1 val shard (~62M tokens). `load_data_shard()` reads uint16 binary format. Val fits in memory on Apple Silicon.
- **Records format**: `submission.json` (8 fields: author, github_id, name, blurb, date, val_loss, val_bpb, bytes_total) + `README.md` in three format variants: (a) Change/Impact ablation tables with signed deltas (~2 records), (b) Base/This comparison tables showing config diffs (~8 records), (c) prose-only configuration descriptions (~10 records). The NaiveBaseline has no table at all.
- **No existing analysis scripts** — entirely greenfield.

### External Research
- **Causal discovery (small n)**: FCI preferred over PC when latent confounders may exist (causal-learn). NOTEARS outperforms GES in low-data regime. For n≈20, Fisher-Z conditional independence tests are underpowered; KCI (kernel) is cubic in n but feasible.
- **Expert priors are load-bearing for n<30**: VaMSL + BED querying (arXiv:2510.06735) shows purely data-driven discovery is unreliable at this scale. "Imaginary observations" framework maps expert beliefs to beta distributions as pseudo-data. PC-stable with conservative alpha (0.01) reduces false positive edges.
- **Iterative causal discovery**: CBO with Unknown Causal Graphs (arXiv:2503.19554) simultaneously optimizes target and discovers causal parents. Key insight: only need to identify direct causal parents of BPB, not the full graph. Bayesian Intervention Optimization (arXiv:2406.10917) uses Bayes factors to pick the most decisive intervention at each cycle.
- **Influence scoring**: TracIn (gradient inner product summed over checkpoints) is the standard. TRAK is more scalable but PyTorch-only. Influence functions work reliably at small model scale (EMNLP 2025 shows they fail on LLMs but our model is ~28M params). MATES (NeurIPS 2024) uses adaptive influence models retrained during pretraining — relevant for data selection.
- **Paired Seed Evaluation** (PSE, arXiv:2512.24145): Matched seeds exploit positive correlation for variance reduction — directly applicable to our 3-seed paired design.
- **Per-token loss**: Rho-1 demonstrates heterogeneous per-token landscapes; selective loss on high-score tokens yields large gains.
- **Statistics**: `statsmodels.stats.multitest.multipletests(pvals, method='holm')` for Holm-Bonferroni.
- **Pipeline**: causal-learn for structure discovery + statsmodels for tests + DoWhy optional for refutation.

## Architecture Overview

The system uses a **discovery-adjust cycle** rather than a linear pipeline. Each experiment's results feed back into the DAG, refining causal beliefs and guiding the next intervention. The goal is pragmatic: if an intervention improves validation BPB, integrate it — the causal structure serves as a navigation tool, not a theoretical end.

```
                    ┌─────────────────────────────────┐
                    │         DISCOVERY-ADJUST CYCLE    │
                    │                                   │
                    │  ┌──────────┐    ┌────────────┐  │
  leaderboard ────→ │  │ extract  │───→│ estimate   │  │
  records           │  │ + DAG    │    │ + identify │  │
                    │  └──────────┘    └─────┬──────┘  │
                    │                        │         │
                    │          ┌──────────────┘         │
                    │          ▼                        │
                    │  ┌──────────────┐                 │
                    │  │ select next  │ ◄── expert      │
                    │  │ intervention │     priors      │
                    │  └──────┬───────┘                 │
                    │         │                         │
                    │         ▼                         │
                    │  ┌──────────────┐                 │
                    │  │  experiment  │ 3 seeds ×       │
                    │  │  runner      │ 2 conditions    │
                    │  └──────┬───────┘                 │
                    │         │                         │
                    │         ▼                         │
                    │  ┌──────────────┐                 │
                    │  │ statistical  │──→ decision     │
                    │  │ analysis     │    gate         │
                    │  └──────┬───────┘                 │
                    │         │                         │
                    │    ┌────┴────┐                    │
                    │ confirmed  null                   │
                    │    │         │                    │
                    │    │    feed results back ────┐   │
                    │    │    into DAG (re-run      │   │
                    │    │    estimate with new     │   │
                    │    │    data point)       ────┘   │
                    │    │                              │
                    └────┼──────────────────────────────┘
                         ▼
                  ┌──────────────┐
                  │  submission  │
                  │  assembly    │
                  └──────────────┘

  PARALLEL DIAGNOSTICS (run alongside any cycle iteration):
  ┌────────────────┐  ┌────────────────┐  ┌──────────────┐
  │ token_loss     │  │ quant_gap      │  │ gradient     │
  │ decompose      │  │ analysis       │  │ attribution  │
  └────────────────┘  └────────────────┘  └──────────────┘
  │ influence_proxy │
  └────────────────┘
```

### Key Architectural Change: Iterative DAG

Instead of a one-shot "discover → intervene → done" pipeline, the DAG is a living artifact:

1. **Cycle 0**: Build initial DAG from 20 leaderboard records + expert priors
2. **Cycle 1**: Pick highest-value intervention from DAG, run 3-seed experiment, update DAG with new data point (n=21)
3. **Cycle 2+**: Re-estimate DAG with augmented dataset, pick next intervention, repeat
4. **Stop condition**: Confirmed effect found (integrate into submission) OR 2-day time gate expires OR 3 consecutive null results

This is inspired by CBO with Unknown Causal Graphs (arXiv:2503.19554) — we only need to identify causal parents of BPB, not the full graph. Each cycle adds one data point, progressively disambiguating uncertain edges.

The diagnostic scripts (token loss, quant gap, gradient attribution, influence proxy) run as parallel observability probes at any point in the cycle, providing supporting evidence for intervention selection.

**Spec-design reconciliation**: The spec defines linear phases (1→2→3→4→5). The design maps these as: Cycle 0 = Phase 1 (R1.1-R1.4), Cycles 1-4 = Phase 2 (R2.1-R2.3) with iterative DAG feedback, diagnostics = Phases 3-4 running in parallel alongside any cycle. Phase 5 (submission) is unchanged. The spec's sequential decision gates are preserved within each cycle iteration. This is a refinement of the spec's execution model, not a contradiction — the spec allows engineering fallback and does not prohibit iterating.

### Design Principles

1. **Script independence**: Each script is self-contained with CLI args for inputs/outputs. Shared utilities in `common.py`.
2. **JSON as interchange + experiment log**: All results in `results/causal/`. Each cycle's results are versioned: `results/causal/cycle_N/`. A master `experiment_log.json` tracks all runs across cycles.
3. **MLX-first diagnostics**: Diagnostic probes run on Apple Silicon MLX. Experiment runner supports both MLX and H100.
4. **No training loop modification for final submission**: All diagnostics are offline scripts. C10 creates a separate instrumented copy.
5. **Iterative over linear**: The DAG evolves with each experiment. The researcher runs the cycle manually, using the DAG + diagnostics to pick interventions. Each output JSON includes `recommendation` + `confidence` fields.
6. **Maximum observability**: Every experiment logs: full config, per-step metrics, gradient norms (if instrumented), per-token loss breakdown, and decision rationale. Nothing is "fire and forget" — every run must produce enough data to explain why BPB moved (or didn't).

## Observability Infrastructure

All experiments produce structured logs that enable post-hoc edge discovery:

### Experiment Log (results/causal/experiment_log.json)
Append-only log of all experimental runs across all cycles:
```json
{
  "experiments": [{
    "cycle": int,
    "experiment_id": str,          # Unique ID (timestamp-based)
    "hypothesis": str,             # What we're testing
    "intervention": str,           # What changed
    "control_config": {...},
    "treatment_config": {...},
    "results": {
      "seeds": [int],
      "control_bpb": [float],
      "treatment_bpb": [float],
      "mean_effect": float,
      "p_value": float,
      "decision": str
    },
    "diagnostics": {               # Optional: attached diagnostic probes
      "gradient_norms": str,       # Path to gradient log if collected
      "token_analysis": str,       # Path to per-token breakdown if collected
      "quant_gap": float | null
    },
    "dag_update": {                # How DAG changed after this experiment
      "edges_added": [str],
      "edges_removed": [str],
      "edges_strengthened": [str]
    },
    "timestamp": str,
    "notes": str                   # Researcher notes / rationale
  }]
}
```

### Per-Run Metrics (written by experiment_runner)
Each training run produces a JSON-lines metrics file:
```
{"step": 0, "loss": 2.5, "val_loss": null, "lr_mul": 0.05, "elapsed_ms": 0}
{"step": 100, "loss": 2.1, "val_loss": 2.3, "lr_mul": 1.0, "elapsed_ms": 8500}
...
```
This enables post-hoc analysis of training dynamics differences between treatment and control — not just final BPB but the trajectory.

## Components

### C1: common.py — Shared Utilities

Utility module providing:
- `load_submission_json(path) -> dict`: Parse submission.json
- `load_model(checkpoint_path, config_overrides=None) -> (model, tokenizer)`: Import GPT class from train_gpt_mlx.py, construct model, load weights via mx.load(), load tokenizer. Single entry point for all offline inference scripts (C7, C8, C9, C10). See TD-7. Confirmed: train_gpt_mlx.py line 1103 contains `if __name__ == "__main__":` guard.
- `compute_bpb(model, val_tokens, sp_model) -> float`: BPB computation reusing eval_val logic
- `paired_ttest(treatment, control) -> (mean_diff, ci_lo, ci_hi, p_value)`: Paired t-test with bootstrap CI
- `holm_bonferroni(p_values, alpha) -> (reject_mask, adjusted_p)`: Wraps statsmodels multipletests
- `decision_gate(effect_size, p_value, mde=0.002) -> str`: Returns "confirmed"/"suggestive"/"null"
- `log_experiment(experiment_log_path, entry) -> None`: Append an experiment entry to the master experiment_log.json
- `get_cycle_dir(base_path, cycle) -> Path`: Returns `results/causal/cycle_{N}/`, creating if needed
- `dag_diff(old_dag_path, new_dag_path) -> dict`: Compare two causal_dag.json files, return `{"edges_added": ["A -> B"], "edges_removed": [...], "edges_strengthened": [...]}`. Edge format: `"source -> target"`. Used by C3 when `--previous-dag` is provided.

### C2: extract_interventions.py — Record Parser (R1.1)

Parses `records/track_10min_16mb/*/README.md` and `submission.json` to build a structured intervention-outcome dataset.

**Strategy**: Three-tier extraction:
1. **submission.json pass**: Extract `author`, `date`, `val_bpb`, `blurb` (always available, 20/20 records)
2. **README.md pass**: Multi-format table parser handling three variants:
   - **Format A** (~2 records): `| Change | Base ref | This | Impact |` — extract `delta_bpb` directly from Impact column
   - **Format B** (~8 records): `| | Base ref | This |` comparison tables with rows like `val_bpb | 1.1271 | 1.1248` — compute `delta_bpb` as `this - base` from val_bpb row; extract intervention names from "What's new" / "Changes from" sections
   - **Format C** (~10 records): No table — extract interventions from markdown headings ("### Key Innovations"), bullet lists, and `blurb` field. `delta_bpb` set to `null`.
3. **Cross-reference pass**: For records with `base_bpb` available (from table or "Previous:" reference), compute total delta. For records without, use `final_bpb` only.
4. **Experiment append** (cycle 1+): C2 also supports `--append-experiment raw_runs.json` which converts experiment results into the same submission format and appends them to the existing interventions.json. This is the feedback mechanism: experiment results become new "submissions" in the dataset for C3 to re-estimate the DAG. The appended entry includes the experiment's intervention as a single intervention, its BPB as final_bpb, and the control's BPB as base_bpb.

**Output schema** (interventions.json):
```json
{
  "submissions": [
    {
      "submission_id": "2026-03-22_11L_EMA_..._1.1233",
      "date": "2026-03-22",
      "author": "signalrush",
      "base_bpb": 1.1239,
      "final_bpb": 1.1233,
      "interventions": [
        {"name": "GPTQ-lite clip", "category": "quantization", "delta_bpb": -0.0006, "delta_source": "ablation_table"},
        {"name": "EMA decay=0.997", "category": "optimization", "delta_bpb": -0.0006, "delta_source": "ablation_table"}
      ],
      "parse_quality": "structured"
    }
  ],
  "field_coverage": 0.93,
  "metadata": {"total_submissions": 20, "format_a_count": 2, "format_b_count": 8, "format_c_count": 10}
}
```

### C3: estimate_dag.py — Causal Structure Discovery (R1.2)

**Algorithm choice**: Expert-guided DAG as primary, with FCI as optional validation.

Given n≈20 (growing with each cycle), purely data-driven discovery is unreliable (arXiv:2510.06735). The approach:
1. **Expert-guided skeleton**: Build initial DAG from domain knowledge — temporal ordering of submissions, known causal relationships (e.g., quantization → artifact size → BPB), community consensus from README discussions
2. **Data validation**: Run FCI (causal-learn, Fisher-Z, alpha=0.01 conservative) on the data. Compare with expert DAG. Flag disagreements for investigation.
3. **Uncertainty marking**: Edges are tagged as `expert_imposed`, `data_confirmed`, `data_contradicted`, or `uncertain`. Only `data_confirmed` edges are used for high-confidence intervention selection.
4. **Cycle update**: After each experiment, re-run with n+1 data points. Track edge stability across cycles — edges that survive multiple re-estimations gain confidence.

**Encoding**: Convert categorical interventions to binary presence/absence per submission. Nodes: `{num_layers, mlp_mult, attention_variant, rope_variant, weight_avg_method, quant_method, compression}` + `bpb` outcome.

**Output**: Versioned adjacency matrix (JSON, one per cycle) + DOT visualization + edge stability history.

### C4: identifiability_check.py — Data Quality Assessment (R1.3)

Analyzes the intervention matrix from C2 to determine:
- **Single-variable records**: Submissions where exactly 1 intervention differs from a known baseline
- **Multi-variable records**: Submissions with 3+ simultaneous changes
- **Identifiability score**: Fraction of edges in the DAG that are testable via single-variable records
- **Confounded pairs**: Interventions that always co-occur (never independently varied)

Decision gate: If >50% of records have 3+ simultaneous changes → recommend skip Phase 2.

Also produces interaction discovery (R1.4): enumerate all node pairs, check co-occurrence across submissions, rank unexplored combinations by sum of individual marginal effects × interaction prior (default 1.0).

**Note**: Spec R4.2 (shard_variance_check.py) is merged into C9 (influence_proxy.py) because the variance check requires the influence scores already computed by C9, making a separate script unnecessary. The spec file structure lists it separately but the design consolidates for simplicity.

### C5: experiment_runner.py — Paired Seed Ablation (R2.1)

Wrapper script that:
1. Takes treatment config JSON and control config JSON
2. Runs `train_gpt_mlx.py` (or `train_gpt.py` for H100) with 3 seeds [42, 137, 256]
3. Collects BPB results per seed per condition
4. Outputs raw results for C6 statistical analysis

Uses subprocess to invoke training scripts — does not import or modify them. Seeds passed via `SEED` environment variable (existing convention in records).

**Config schema**: Since train_gpt_mlx.py uses a Hyperparameters dataclass populated from environment variables (line 48+), configs map field names to env var overrides:
```json
{
  "script": "train_gpt_mlx.py",
  "env_overrides": {
    "NUM_LAYERS": "11",
    "MLP_MULT": "3",
    "MODEL_DIM": "512",
    "MAX_WALLCLOCK_SECONDS": "600"
  },
  "description": "11L 3x MLP baseline"
}
```
The runner sets each `env_overrides` key as an environment variable before subprocess invocation. Only fields in `env_overrides` differ between treatment and control; all other env vars are inherited from the parent process. Allowed override fields: any Hyperparameters class attribute that reads from `os.environ`.

**Platform note**: For MLX experiments, use root `train_gpt_mlx.py`. For H100 experiments, use root `train_gpt.py`. Both scripts share the same env var interface for core hyperparameters (NUM_LAYERS, MLP_MULT, MODEL_DIM, SEED, MAX_WALLCLOCK_SECONDS). SOTA record scripts in `records/` are NOT used directly — they contain submission-specific code (GPTQ-lite, custom quantizers) that differs from the baseline.

### C6: statistical_analysis.py — Effect Estimation (R2.2)

Takes raw BPB results from C5. Computes:
- Per-seed paired differences
- Mean effect size with bootstrapped 95% CI (10,000 resamples)
- Paired t-test p-value
- Holm-Bonferroni correction across all tested hypotheses
- Decision gate classification (confirmed/suggestive/null)
- Platform transfer coefficient if both MLX and H100 results available

### C7: token_loss_decompose.py — Per-Token Analysis (R3.1)

MLX script that:
1. Loads a checkpoint + validation data
2. Runs forward pass with `reduction='none'` in cross_entropy to get per-token losses
3. Classifies tokens by frequency bucket (top-100, 100-500, 500-1024) using tokenizer vocab frequencies
4. Classifies tokens by context type (boundary: first token after whitespace; mid-sequence: all others)
5. Verifies decomposition: `mean(per_token_losses)` matches aggregate loss within 1e-6 (spec R3.1)
6. Outputs per-category statistics (mean loss, std, BPB contribution)

### C8: quant_gap_analysis.py — Pre/Post Quantization (R3.2)

Reuses existing `quantize_state_dict_int8` and `dequantize_state_dict_int8` from train_gpt_mlx.py:
1. Load raw checkpoint → eval BPB (pre-quant)
2. Quantize → dequantize → eval BPB (post-quant roundtrip)
3. Compute gap and compare against training-side causal effects
4. Optionally: per-token loss comparison pre/post quant to identify which token categories lose most

### C9: influence_proxy.py — Gradient Inner Product (~50 lines) (R4.1)

TracIn-inspired single-checkpoint influence proxy:
```
For each shard k:
  grad_train_k = grad(loss(shard_k_batch, checkpoint))
  grad_val     = grad(loss(val_batch, checkpoint))
  influence_k  = dot(flatten(grad_train_k), flatten(grad_val))
```

Uses plain (non-compiled) `nn.value_and_grad` API, which returns gradients only for trainable parameters by default. This means the influence inner product is computed only over trainable parameters (RoPE frequencies, embedding norms, etc. are excluded — desirable since they are not learned). Operates on a single checkpoint (not summed across checkpoints like full TracIn — one checkpoint is sufficient for ranking, per TracIn paper Section 3.2).

**Per-shard batch size**: Sample 4096 tokens (4 sequences of 1024) per shard, not the full ~100M tokens. This gives a representative gradient direction while keeping each forward+backward pass under 1 second on Apple Silicon.

**Memory budget**: For the 11L, 512-dim, 3× MLP SOTA config: ~28M params ≈ ~56MB in bf16 + 2× gradient buffers ~56MB each + shard batch ~8KB + val batch ~8KB ≈ ~170MB peak. Well within 16GB Apple Silicon.

**Runtime estimate**: 195 shards × ~1s per shard ≈ ~4 minutes total. With `--max-shards 20` default for initial iteration: ~20 seconds.

**Hard requirement**: `mx.eval()` must be called after each inner product computation to force materialization and prevent lazy eval memory buildup. This is not optional.

### C10: gradient_attribution.py — Per-Layer Logging (R4.4)

Creates a patched copy of train_gpt_mlx.py at runtime. The mechanism: C10 reads train_gpt_mlx.py as text, inserts gradient-norm logging code after the `accumulate_flat_grads` call site (line ~1036), writes the result to `train_gpt_mlx_instrumented.py`, and executes it via subprocess. The copy is regenerated each time C10 runs, ensuring it tracks upstream changes. **Patch targeting**: C10 targets the LAST occurrence of `accumulate_flat_grads` in the file (the main training loop call at line ~1036, not the function definition or warmup loop calls). **Sentinel validation**: After patching, C10 asserts that the surrounding context (within ±5 lines) includes BOTH `train_loss = train_loss + loss` (unique to the main loop, not present in warmup) AND `lr_mul` assignment, confirming the correct call site was patched. If validation fails, aborts with an error indicating the patch site has drifted. The original script is never modified. The patched copy adds logging at each validation checkpoint:
1. After `loss_and_grad_chunked` returns `(loss, grads)`, iterate flat grad dict
2. Compute L2 norm per named parameter group (attention, MLP, embedding, skip weights)
3. Log to JSON-lines file: `{step, elapsed_ms, val_loss, layer_norms: {name: norm}}`
4. Phase boundary detection: monitor `lr_mul` return value; transition from 1.0 → <1.0 marks warmdown onset. Check `max_wallclock_seconds` to determine step-based vs. wallclock-based mode.

## Technical Decisions

### TD-1: FCI as Validation, Expert DAG as Skeleton
**Choice**: Expert-guided DAG (primary) + FCI validation (secondary) with Fisher-Z conditional independence test, alpha=0.01 conservative
**Rationale**: PC/FCI alone are unreliable at n<30 (high degeneracy risk). Expert priors from domain knowledge (leaderboard READMEs, competition community) provide the skeleton. FCI validates or contradicts specific edges. Edges are tagged by source for transparency.
**Trade-off**: Expert bias persists where data is insufficient. Mitigated by edge stability tracking across cycles and requiring data confirmation for high-confidence intervention selection.

### TD-2: TracIn-Style Single-Checkpoint Influence over TRAK
**Choice**: Simple gradient inner product at one checkpoint
**Rationale**: TRAK has no MLX port. Full TracIn requires multiple checkpoints. Single-checkpoint ranking is sufficient for shard ordering (we need relative ranking, not absolute attribution).
**Trade-off**: Less accurate than multi-checkpoint TracIn or TRAK. Mitigated by the variance check (R4.2) — if CV < 0.1, the signal is too weak regardless of method accuracy.

### TD-3: Subprocess Invocation for Experiment Runner
**Choice**: experiment_runner.py calls training scripts via subprocess, not import
**Rationale**: Training scripts use global state, torch.distributed, compiled functions — importing them would require refactoring. Subprocess preserves their exact execution environment and avoids interference.
**Trade-off**: Slower startup, no shared memory. Acceptable since each run is 10+ minutes; startup cost is negligible.

### TD-4: JSON Interchange over Shared Database
**Choice**: JSON files in results/causal/ as interchange format
**Rationale**: Scripts run independently, potentially days apart. JSON is human-readable, git-diffable, and requires no server. Each phase reads the previous phase's output file.
**Trade-off**: No query capability, no schema enforcement at read time. Mitigated by keeping schemas simple and validating inputs at script entry.

### TD-5: causal-learn + statsmodels over DoWhy-Only
**Choice**: causal-learn for discovery, statsmodels for statistical tests, DoWhy optional for refutation
**Rationale**: causal-learn has the broadest algorithm set for structure discovery. statsmodels has battle-tested Holm-Bonferroni. DoWhy adds value for effect estimation refutation but is not required for the core pipeline.
**Trade-off**: More dependencies. Mitigated by keeping each script's requirements minimal and documented.

### TD-7: Model Instantiation for Offline Scripts
**Choice**: common.py provides `load_model(checkpoint_path, config) -> (model, tokenizer)` that imports the GPT class from train_gpt_mlx.py with an import guard.
**Rationale**: train_gpt_mlx.py defines all model classes (GPT, Block, CausalSelfAttention, MLP, etc.) but also has a `main()` training loop that runs on import if not guarded. The model classes themselves are clean and importable. common.py will: (1) `import train_gpt_mlx as tgm` (relies on existing `if __name__ == "__main__"` guard at the bottom of the script), (2) construct `tgm.GPT(tgm.Hyperparameters())` with config overrides, (3) load state dict via `mx.load()` and apply to model, (4) load tokenizer via sentencepiece. This avoids duplicating the model definition.
**Trade-off**: Couples offline scripts to train_gpt_mlx.py's internal API. If the model class changes, common.py must be updated. Acceptable because we control both files and changes are infrequent.
**Scope of common.py load_model**: Returns a fully constructed model object (not just state dict). All scripts needing inference (C7, C8, C9, C10) use this single entry point.

### TD-8: Discovery-Adjust Cycle over Linear Pipeline
**Choice**: Iterative DAG refinement with experiment feedback loop
**Rationale**: A one-shot DAG from 20 observational records is unreliable (high degeneracy risk). Each experiment we run adds a controlled data point that disambiguates uncertain edges. CBO with Unknown Graphs (arXiv:2503.19554) shows that jointly optimizing and discovering is more efficient than discovering-then-optimizing. We only need to identify the causal parents of BPB (5-7 variables), not the full graph.
**Cycle protocol**: (1) Estimate DAG, (2) Select intervention with highest expected BPB improvement among uncertain edges, (3) Run 3-seed experiment, (4) Update DAG with new data point, (5) Repeat until confirmed effect or time gate. Maximum 4 cycles (matching the 2-day decision gate per phase).
**Trade-off**: More complex than linear phases. Mitigated by the experiment_log.json that tracks all cycles and the versioned DAG outputs.

### TD-9: Expert-Guided DAG as Primary (not Fallback)
**Choice**: Expert priors are the starting point, not the contingency
**Rationale**: With n<30, expert priors are load-bearing (arXiv:2510.06735). FCI validation catches cases where data contradicts expert beliefs — but the expert DAG is the skeleton we navigate by. This inverts the original design (FCI primary, expert fallback).
**Trade-off**: Expert bias could persist if data is insufficient to contradict wrong edges. Mitigated by explicit edge tagging (expert_imposed vs. data_confirmed) and requiring data confirmation before high-confidence intervention selection.

### TD-6: Flat Gradient Dict for Per-Layer Attribution
**Choice**: Use MLX's existing `accumulate_flat_grads` pattern (tree_flatten → flat dict) for gradient norm logging
**Rationale**: The flat dict already exists in the training loop (train_gpt_mlx.py:1031-1044). Computing norms per key requires zero structural changes — just iterate the dict and log.
**Trade-off**: Logs parameter-level norms, not semantic-layer-level (e.g., "layer_3.attn" vs. "attention layers 0-5"). Post-processing in analysis can group by naming convention.

## Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| FCI produces degenerate graph (n=20 too small) | Data-driven validation unavailable for cycle 0 | High | Expert DAG is primary (TD-9). FCI is validation only. Each cycle adds 1 data point, improving FCI reliability over time. |
| Only ~10 of 20 records have structured tables | R1.1 field coverage below 90% for delta_bpb | Medium | Three-tier parser (Format A/B/C). Accept null delta_bpb for prose-only records. Field coverage computed over 6 core fields, not just delta. |
| Gradient inner products have low variance across shards (CV < 0.1) | Phase 4 data curriculum branch skipped | Medium | This is a valid scientific finding (null result). CV threshold is the explicit gate. |
| MLX-to-H100 effect transfer coefficient < 0.5 | MLX findings do not replicate | Medium | BC-4 prevents claiming H100 validity from MLX-only. Phase 2 explicitly includes H100 validation step. |
| Per-token loss decomposition is trivially dominated by high-frequency tokens | Phase 3 yields uninteresting signal | Low | Frequency bucketing separates the effect. If top-100 tokens dominate >80% of loss, report and move to Phase 4. |
| Holm-Bonferroni overcorrects with 4 hypotheses, reducing power | Confirmed effects downgraded to suggestive | Low | Sequential phase design means typically 1-2 hypotheses tested (not all 4). Effective correction is mild. |

## Interfaces

### I1: Script CLI Interface Standard

All scripts follow a consistent CLI pattern:

```
python scripts/causal/<script>.py \
  --input <input_path>     \  # Required: input JSON/directory
  --output <output_path>   \  # Required: output JSON path
  [--config <config.json>] \  # Optional: overrides
  [--verbose]              \  # Optional: debug logging
  [--dry-run]                 # Optional: validate inputs without executing
```

### I2: extract_interventions.py

```
Input:  records/track_10min_16mb/  (directory)
Output: results/causal/interventions.json

CLI (initial extraction):
  python scripts/causal/extract_interventions.py \
    --input records/track_10min_16mb/ \
    --output results/causal/interventions.json

CLI (append experiment results for cycle 1+):
  python scripts/causal/extract_interventions.py \
    --input records/track_10min_16mb/ \
    --output results/causal/interventions.json \
    --append-experiment results/causal/cycle_1/raw_runs.json

Output Schema:
{
  "submissions": [{
    "submission_id": str,        # Directory name
    "date": str,                 # ISO date from submission.json
    "author": str,               # From submission.json
    "base_bpb": float | null,    # Previous SOTA BPB (from ablation table "Base" column)
    "final_bpb": float,          # From submission.json val_bpb
    "interventions": [{
      "name": str,               # Intervention description
      "category": str,           # architecture|quantization|optimization|encoding|data
      "delta_bpb": float | null, # Signed BPB change (null if unknown)
      "delta_source": str        # ablation_table|readme_prose|submission_blurb
    }],
    "parse_quality": str         # structured|prose|minimal
  }],
  "field_coverage": float,       # (non-null fields) / (submissions × 6)
  "metadata": {
    "total_submissions": int,
    "structured_count": int,
    "prose_only_count": int,
    "extraction_timestamp": str
  }
}
```

### I3: estimate_dag.py

```
Input:  results/causal/interventions.json
Output: results/causal/cycle_N/causal_dag.json, results/causal/cycle_N/dag.png

CLI:
  python scripts/causal/estimate_dag.py \
    --input results/causal/interventions.json \
    --output-dir results/causal/cycle_0/ \
    [--previous-dag results/causal/cycle_N-1/causal_dag.json]  \  # For cycle 1+: computes edge diff
    [--alpha 0.01]                # Significance threshold (conservative for n<30)

Output Schema (causal_dag.json — versioned per cycle):
{
  "cycle": int,                  # 0 = initial, 1+ = after experiments
  "method": str,                 # expert_guided+fci_validation
  "nodes": [str],                # Variable names
  "adjacency_matrix": [[int]],   # 0=no edge, 1=directed, 2=bidirected, 3=uncertain
  "edges": [{
    "from": str,
    "to": str,
    "type": str,                 # directed|bidirected|uncertain
    "source": str,               # expert_imposed|data_confirmed|data_contradicted|uncertain
    "stability": int,            # Number of consecutive cycles this edge has survived
    "effect_estimate": float | null
  }],
  "estimated_effects": [{
    "node": str,
    "marginal_effect_on_bpb": float,
    "n_observations": int,
    "confidence": str            # high (data_confirmed) | medium (expert_imposed) | low (uncertain)
  }],
  "next_intervention": {         # Recommended next experiment
    "variable": str,
    "suggested_value": str,
    "expected_bpb_delta": float,
    "rationale": str
  } | null,
  "metadata": {
    "n_samples": int,
    "alpha": float,
    "fci_degenerate": bool,
    "expert_edges_count": int,
    "data_confirmed_count": int,
    "data_contradicted_count": int
  }
}
```

### I4: identifiability_check.py

```
Input:  results/causal/interventions.json, results/causal/cycle_N/causal_dag.json
Output: results/causal/cycle_N/identifiability_report.json

CLI:
  python scripts/causal/identifiability_check.py \
    --interventions results/causal/interventions.json \
    --dag results/causal/cycle_0/causal_dag.json \
    --output results/causal/cycle_0/identifiability_report.json

Output Schema:
{
  "single_variable_records": int,
  "multi_variable_records": int,   # 3+ simultaneous changes
  "total_records": int,
  "identifiability_score": float,  # Fraction of testable edges
  "confounded_pairs": [{"a": str, "b": str, "co_occurrence_count": int}],
  "recommendation": "proceed|skip",
  "unexplored_combinations": [{
    "pair": [str, str],
    "expected_effect": float,      # sum(individual) × interaction_prior
    "interaction_prior": float,
    "prior_rationale": str | null  # null if default 1.0
  }]
}
```

### I5: experiment_runner.py

```
Input:  Treatment config JSON, Control config JSON
Output: results/causal/raw_runs.json

CLI:
  python scripts/causal/experiment_runner.py \
    --treatment treatment_config.json \
    --control control_config.json \
    --output results/causal/raw_runs.json \
    --seeds 42,137,256 \
    --platform mlx|h100

Output Schema:
{
  "platform": str,
  "seeds": [int],
  "treatment": {
    "config": {...},
    "results": [{"seed": int, "val_bpb": float, "val_loss": float, "wall_time_s": float, "checkpoint_path": str | null, "train_log_path": str | null}]
  },
  "control": {
    "config": {...},
    "results": [{"seed": int, "val_bpb": float, "val_loss": float, "wall_time_s": float, "checkpoint_path": str | null, "train_log_path": str | null}]
  }
}

Note: The --seeds argument sets the SEED environment variable for each subprocess invocation, separate from env_overrides which apply uniformly across all seed runs.
```

### I6: statistical_analysis.py

```
Input:  results/causal/raw_runs.json (or multiple)
Output: results/causal/ablation_results.json

Output Schema:
{
  "comparisons": [{
    "name": str,
    "platform": str,
    "n_seeds": int,
    "mean_effect": float,          # treatment_bpb - control_bpb (negative = improvement)
    "ci_lo": float,
    "ci_hi": float,
    "p_value": float,
    "p_value_corrected": float,    # After Holm-Bonferroni
    "decision": str                # confirmed|suggestive|null
  }],
  "correction_method": "holm-bonferroni",
  "alpha": 0.01,
  "mde": 0.002,
  "platform_transfer": {           # Present only if both MLX and H100 data
    "mlx_effect": float,
    "h100_effect": float,
    "transfer_coefficient": float,
    "divergence_flag": bool
  } | null
}
```

### I7: token_loss_decompose.py

```
Input:  Checkpoint path, validation data path, tokenizer path
Output: results/causal/token_analysis.json

CLI:
  python scripts/causal/token_loss_decompose.py \
    --checkpoint <path> \
    --val-data data/datasets/fineweb10B_sp1024/ \
    --tokenizer data/tokenizers/sp_bpe_1024.model \
    --output results/causal/diagnostics/token_analysis.json

Output Schema:
{
  "aggregate_bpb": float,
  "decomposition_check": {
    "mean_per_token_loss": float,
    "aggregate_loss": float,
    "delta": float,               # Must be < 1e-6
    "passed": bool
  },
  "by_frequency_bucket": {
    "top_100": {"n_tokens": int, "mean_loss": float, "std": float, "bpb_contribution": float},
    "mid_100_500": {"n_tokens": int, "mean_loss": float, "std": float, "bpb_contribution": float},
    "tail_500_1024": {"n_tokens": int, "mean_loss": float, "std": float, "bpb_contribution": float}
  },
  "by_context_type": {
    "boundary": {"n_tokens": int, "mean_loss": float, "bpb_contribution": float},
    "mid_sequence": {"n_tokens": int, "mean_loss": float, "bpb_contribution": float}
  }
}
```

### I7b: quant_gap_analysis.py

```
Input:  Checkpoint path, validation data path, tokenizer path
Output: results/causal/quant_report.json

CLI:
  python scripts/causal/quant_gap_analysis.py \
    --checkpoint <path> \
    --val-data data/datasets/fineweb10B_sp1024/ \
    --tokenizer data/tokenizers/sp_bpe_1024.model \
    --output results/causal/diagnostics/quant_report.json

Output Schema:
{
  "pre_quant_bpb": float,
  "post_quant_bpb": float,
  "quant_gap": float,                    # post - pre (~0.017 expected)
  "largest_training_effect": float | null, # From prior phase results
  "gap_exceeds_3x_threshold": bool,       # quant_gap > 3 × largest_training_effect
  "restrict_to_post_quant": bool,         # Recommendation based on threshold
  "per_category_deltas": {                # Optional: per-token category quant impact
    "top_100": {"pre_loss": float, "post_loss": float, "delta": float},
    "mid_100_500": {"pre_loss": float, "post_loss": float, "delta": float},
    "tail_500_1024": {"pre_loss": float, "post_loss": float, "delta": float}
  } | null
}
```

### I8: influence_proxy.py

```
Input:  Checkpoint path, training shard directory, validation data path
Output: results/causal/influence_scores.json

CLI:
  python scripts/causal/influence_proxy.py \
    --checkpoint <path> \
    --train-data data/datasets/fineweb10B_sp1024/ \
    --val-data data/datasets/fineweb10B_sp1024/ \
    --output results/causal/diagnostics/influence_scores.json \
    [--max-shards 20]           # Limit for faster iteration

Output Schema:
{
  "checkpoint": str,
  "n_shards_scored": int,
  "scores": [{"shard": str, "influence_score": float}],  # Sorted descending
  "variance_check": {
    "mean": float,
    "std": float,
    "cv": float,                 # Coefficient of variation
    "recommendation": str        # proceed|skip
  },
  "curriculum_skipped": bool,    # True if CV < 0.1
  "reason": str | null           # "CV < 0.1" if skipped
}
```

### I9: gradient_attribution.py

```
Input:  Training script path, config overrides
Output: results/causal/gradient_attribution.json

CLI:
  python scripts/causal/gradient_attribution.py \
    --script train_gpt_mlx.py \
    --output results/causal/diagnostics/gradient_attribution.json \
    [--val-every 100]           # Log norms every N steps

Output Schema:
{
  "phase_boundaries": {
    "warmup_end_step": int,
    "warmdown_start_step": int,      # Detected from lr_mul transition
    "warmdown_mode": str             # step_based|wallclock_based
  },
  "per_step_norms": [{
    "step": int,
    "elapsed_ms": float,
    "val_loss": float,
    "lr_mul": float,
    "layer_norms": {
      "<param_name>": float          # L2 norm per named parameter
    }
  }],
  "phase_correlations": {
    "warmup": {"<param_group>": {"correlation": float, "p_value": float}},
    "main": {"<param_group>": {"correlation": float, "p_value": float}},
    "warmdown": {"<param_group>": {"correlation": float, "p_value": float}}
  }
}
```

## Dependencies

### Python Packages (new, not in current requirements.txt)
| Package | Version | Used By | Purpose |
|---------|---------|---------|---------|
| causal-learn | >=0.1.3 | C3 | FCI/PC causal discovery algorithms |
| statsmodels | >=0.14 | C6, common.py | Holm-Bonferroni multipletests |
| networkx | >=3.0 | C3 | DAG representation and analysis |
| graphviz | >=0.20 | C3 | DOT visualization rendering |
| scipy | >=1.10 | C6, common.py | paired t-test, bootstrap |

### Existing Dependencies (already available)
- mlx, mlx.nn, mlx.optimizers — for all MLX-based scripts
- numpy — used throughout for array operations
- sentencepiece — for tokenizer and BPB computation
- json, argparse, pathlib, subprocess — stdlib

## File Structure (Final)

```
scripts/causal/
├── common.py                   # C1: Shared utilities
├── extract_interventions.py    # C2: Record parser (R1.1)
├── estimate_dag.py             # C3: Causal structure discovery (R1.2)
├── identifiability_check.py    # C4: Data quality + interaction discovery (R1.3, R1.4)
├── experiment_runner.py        # C5: Paired seed ablation (R2.1)
├── statistical_analysis.py     # C6: Effect estimation (R2.2)
├── token_loss_decompose.py     # C7: Per-token analysis (R3.1)
├── quant_gap_analysis.py       # C8: Pre/post quant BPB (R3.2)
├── influence_proxy.py          # C9: Gradient inner product (R4.1) + shard variance check (R4.2, consolidated from spec's shard_variance_check.py)
├── gradient_attribution.py     # C10: Per-layer gradient logging (R4.4)
├── requirements.txt            # Phase-specific dependencies
└── README.md                   # Usage guide with per-script examples

results/causal/                 # Created by scripts, gitignored
├── experiment_log.json         # Master log spanning all cycles
├── interventions.json          # Grows: initial records + experiment results appended
├── cycle_0/                    # Initial discovery from leaderboard records
│   ├── causal_dag.json
│   ├── dag.png
│   └── identifiability_report.json
├── cycle_1/                    # After first experiment
│   ├── causal_dag.json         # Re-estimated with n+1 data points
│   ├── dag.png
│   ├── raw_runs.json
│   └── ablation_results.json
├── cycle_N/                    # ...
├── diagnostics/                # Probes (run at any point, not cycle-specific)
│   ├── token_analysis.json
│   ├── quant_report.json
│   ├── influence_scores.json
│   └── gradient_attribution.json
└── submission/                 # Final submission artifacts
    └── ...
```
