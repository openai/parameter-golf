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
- **Influence scoring**: TracIn (gradient inner product summed over checkpoints) is the standard approach. TRAK is more scalable but PyTorch-only. No MLX ports exist — custom implementation required.
- **Paired Seed Evaluation** (PSE, arXiv:2512.24145): Matched seeds exploit positive correlation for variance reduction — directly applicable to our 3-seed paired design.
- **Per-token loss**: Rho-1 demonstrates heterogeneous per-token landscapes; selective loss on high-score tokens yields large gains — validates R3 hypothesis.
- **Statistics**: `statsmodels.stats.multitest.multipletests(pvals, method='holm')` for Holm-Bonferroni. More powerful than Bonferroni, valid under arbitrary dependence.
- **Pipeline**: causal-learn for structure discovery + DoWhy for effect estimation and refutation tests.

## Architecture Overview

The system consists of 5 independent script modules corresponding to the 5 spec phases, connected by JSON artifacts. No shared runtime state — each script reads inputs from disk, writes outputs to disk.

```
┌─────────────────────────────────────────────────────────┐
│                   scripts/causal/                        │
│                                                          │
│  Phase 1: DAG Discovery                                  │
│  ┌──────────────┐  ┌─────────────┐  ┌────────────────┐  │
│  │ extract_      │→│ estimate_   │→│ identifiability_ │  │
│  │ interventions │  │ dag         │  │ check           │  │
│  └──────────────┘  └─────────────┘  └────────────────┘  │
│         ↓                ↓                  ↓            │
│   interventions.json  causal_dag.json  ident_report.json │
│                                                          │
│  Phase 2: Architecture Ablation                          │
│  ┌──────────────┐  ┌──────────────────┐                  │
│  │ experiment_   │→│ statistical_      │                  │
│  │ runner        │  │ analysis          │                  │
│  └──────────────┘  └──────────────────┘                  │
│                           ↓                              │
│                    ablation_results.json                  │
│                                                          │
│  Phase 3: Loss Decomposition                             │
│  ┌──────────────┐  ┌──────────────────┐                  │
│  │ token_loss_   │→│ quant_gap_        │                  │
│  │ decompose     │  │ analysis          │                  │
│  └──────────────┘  └──────────────────┘                  │
│         ↓                  ↓                             │
│   token_analysis.json  quant_report.json                 │
│                                                          │
│  Phase 4: Gradient/Data Probes                           │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────┐  │
│  │ influence_    │→│ shard_        │→│ gradient_       │  │
│  │ proxy         │  │ variance_    │  │ attribution     │  │
│  │               │  │ check        │  │                 │  │
│  └──────────────┘  └──────────────┘  └────────────────┘  │
│         ↓                                    ↓           │
│   influence_scores.json          gradient_attribution.json│
└─────────────────────────────────────────────────────────┘
```

### Design Principles

1. **Script independence**: Each script is self-contained with CLI args for inputs/outputs. No shared library beyond a thin `common.py` utility module.
2. **JSON as interchange**: All intermediate results are JSON files in `results/causal/`. Human-readable, diffable, versionable.
3. **MLX-first diagnostics**: Phases 1, 3, 4 run entirely on Apple Silicon MLX. Phase 2 starts on MLX, validates on H100.
4. **No training loop modification for final submission**: All diagnostics are offline scripts. C10 creates a separate instrumented copy, never modifies the original.
5. **Manual phase orchestration**: The researcher runs scripts sequentially, checks each phase's output JSON for the decision gate recommendation field, and decides the next step. No automated orchestrator — the decision gates involve scientific judgment that should not be automated. Each output JSON includes a `recommendation` field ("proceed"/"skip"/"null") for the researcher to act on.

## Components

### C1: common.py — Shared Utilities

Thin utility module providing:
- `load_submission_json(path) -> dict`: Parse submission.json
- `load_model(checkpoint_path, config_overrides=None) -> (model, tokenizer)`: Import GPT class from train_gpt_mlx.py, construct model, load weights via mx.load(), load tokenizer. Single entry point for all offline inference scripts (C7, C8, C9, C10). See TD-7.
- `compute_bpb(model, val_tokens, sp_model) -> float`: BPB computation reusing eval_val logic
- `paired_ttest(treatment, control) -> (mean_diff, ci_lo, ci_hi, p_value)`: Paired t-test with bootstrap CI
- `holm_bonferroni(p_values, alpha) -> (reject_mask, adjusted_p)`: Wraps statsmodels multipletests
- `decision_gate(effect_size, p_value, mde=0.002) -> str`: Returns "confirmed"/"suggestive"/"null"

### C2: extract_interventions.py — Record Parser (R1.1)

Parses `records/track_10min_16mb/*/README.md` and `submission.json` to build a structured intervention-outcome dataset.

**Strategy**: Three-tier extraction:
1. **submission.json pass**: Extract `author`, `date`, `val_bpb`, `blurb` (always available, 20/20 records)
2. **README.md pass**: Multi-format table parser handling three variants:
   - **Format A** (~2 records): `| Change | Base ref | This | Impact |` — extract `delta_bpb` directly from Impact column
   - **Format B** (~8 records): `| | Base ref | This |` comparison tables with rows like `val_bpb | 1.1271 | 1.1248` — compute `delta_bpb` as `this - base` from val_bpb row; extract intervention names from "What's new" / "Changes from" sections
   - **Format C** (~10 records): No table — extract interventions from markdown headings ("### Key Innovations"), bullet lists, and `blurb` field. `delta_bpb` set to `null`.
3. **Cross-reference pass**: For records with `base_bpb` available (from table or "Previous:" reference), compute total delta. For records without, use `final_bpb` only.

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

**Algorithm choice**: FCI (Fast Causal Inference) from causal-learn, preferred over PC because:
- Handles latent confounders (unobserved factors like data ordering, hardware variance)
- Outputs a Partial Ancestral Graph (PAG) with explicit uncertainty markers (o→, ↔)
- Available with Fisher-Z test (continuous variables), suitable for BPB outcomes

**Contingency** (from spec R1.2): If FCI produces degenerate output (empty or fully-connected), fall back to expert-guided DAG using temporal ordering + domain priors. Document data-driven vs. expert edges.

**Encoding**: Convert categorical interventions to binary presence/absence per submission. Nodes: `{num_layers, mlp_mult, attention_variant, rope_variant, weight_avg_method, quant_method, compression}` + `bpb` outcome.

**Output**: Adjacency matrix (JSON) + DOT visualization (graphviz).

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

Uses `nn.value_and_grad` API. Operates on a single checkpoint (not summed across checkpoints like full TracIn — one checkpoint is sufficient for ranking, per TracIn paper Section 3.2).

**Per-shard batch size**: Sample 4096 tokens (4 sequences of 1024) per shard, not the full ~100M tokens. This gives a representative gradient direction while keeping each forward+backward pass under 1 second on Apple Silicon.

**Memory budget**: For the 11L, 512-dim, 3× MLP SOTA config: ~28M params ≈ ~56MB in bf16 + 2× gradient buffers ~56MB each + shard batch ~8KB + val batch ~8KB ≈ ~170MB peak. Well within 16GB Apple Silicon.

**Runtime estimate**: 195 shards × ~1s per shard ≈ ~4 minutes total. With `--max-shards 20` default for initial iteration: ~20 seconds.

**Hard requirement**: `mx.eval()` must be called after each inner product computation to force materialization and prevent lazy eval memory buildup. This is not optional.

### C10: gradient_attribution.py — Per-Layer Logging (R4.4)

Creates a patched copy of train_gpt_mlx.py at runtime. The mechanism: C10 reads train_gpt_mlx.py as text, inserts gradient-norm logging code after the `accumulate_flat_grads` call site (line ~1036), writes the result to `train_gpt_mlx_instrumented.py`, and executes it via subprocess. The copy is regenerated each time C10 runs, ensuring it tracks upstream changes. **Sentinel validation**: After patching, C10 asserts that the string `accumulate_flat_grads` appears within ±5 lines of the inserted code; if not found, aborts with an error indicating the patch site has drifted. The original script is never modified. The patched copy adds logging at each validation checkpoint:
1. After `loss_and_grad_chunked` returns `(loss, grads)`, iterate flat grad dict
2. Compute L2 norm per named parameter group (attention, MLP, embedding, skip weights)
3. Log to JSON-lines file: `{step, elapsed_ms, val_loss, layer_norms: {name: norm}}`
4. Phase boundary detection: monitor `lr_mul` return value; transition from 1.0 → <1.0 marks warmdown onset. Check `max_wallclock_seconds` to determine step-based vs. wallclock-based mode.

## Technical Decisions

### TD-1: FCI over PC for DAG Estimation
**Choice**: FCI (Fast Causal Inference) with Fisher-Z conditional independence test
**Rationale**: PC assumes no latent confounders — unrealistic when hardware variance, data ordering, and training stochasticity are unobserved. FCI produces a PAG that explicitly marks uncertain edge orientations.
**Trade-off**: FCI may produce more ambiguous edges (o→ instead of →), requiring expert disambiguation. Acceptable given the contingency fallback.

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

### TD-6: Flat Gradient Dict for Per-Layer Attribution
**Choice**: Use MLX's existing `accumulate_flat_grads` pattern (tree_flatten → flat dict) for gradient norm logging
**Rationale**: The flat dict already exists in the training loop (train_gpt_mlx.py:1031-1044). Computing norms per key requires zero structural changes — just iterate the dict and log.
**Trade-off**: Logs parameter-level norms, not semantic-layer-level (e.g., "layer_3.attn" vs. "attention layers 0-5"). Post-processing in analysis can group by naming convention.

## Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| FCI produces degenerate graph (n=20 too small) | Phase 1 yields no actionable DAG | High | Expert-guided fallback DAG (TD-1 contingency). Document data-driven vs. expert edges. |
| Only ~5 of 20 records have structured ablation tables | R1.1 field coverage below 90% | Medium | Two-pass parser: structured table → prose fallback → submission.json blurb. Accept lower coverage for prose-only records. |
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

CLI:
  python scripts/causal/extract_interventions.py \
    --input records/track_10min_16mb/ \
    --output results/causal/interventions.json

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
Output: results/causal/causal_dag.json, results/causal/dag.png

CLI:
  python scripts/causal/estimate_dag.py \
    --input results/causal/interventions.json \
    --output results/causal/causal_dag.json \
    --viz results/causal/dag.png \
    [--method fci|pc|notears]  \  # Default: fci
    [--alpha 0.05]                # Significance threshold

Output Schema (causal_dag.json):
{
  "method": str,                 # fci|pc|notears|expert_guided
  "nodes": [str],                # Variable names
  "adjacency_matrix": [[int]],   # 0=no edge, 1=directed, 2=bidirected, 3=uncertain
  "edges": [{
    "from": str,
    "to": str,
    "type": str,                 # directed|bidirected|uncertain
    "source": str                # data_driven|expert_imposed
  }],
  "estimated_effects": [{
    "node": str,
    "marginal_effect_on_bpb": float,
    "n_observations": int
  }],
  "metadata": {
    "n_samples": int,
    "alpha": float,
    "degenerate": bool,
    "fallback_used": bool
  }
}
```

### I4: identifiability_check.py

```
Input:  results/causal/interventions.json, results/causal/causal_dag.json
Output: results/causal/identifiability_report.json

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
    --output results/causal/token_analysis.json

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
    --output results/causal/quant_report.json

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
    --output results/causal/influence_scores.json \
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
    --output results/causal/gradient_attribution.json \
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
├── interventions.json
├── causal_dag.json
├── dag.png
├── identifiability_report.json
├── raw_runs.json
├── ablation_results.json
├── token_analysis.json
├── quant_report.json
├── influence_scores.json
└── gradient_attribution.json
```
