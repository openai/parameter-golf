# Causal Inference Training Pipeline

Scripts for causal discovery and intervention-guided BPB optimization. The pipeline uses a discovery-adjust cycle: build a causal DAG from leaderboard data, pick high-value interventions, run paired-seed experiments, update the DAG, and repeat.

## Prerequisites

### Dependencies

```bash
uv add causal-learn statsmodels networkx graphviz scipy pytest
brew install graphviz   # required for DAG rendering
```

Verify:

```bash
python -c "import causallearn; import statsmodels; import networkx; import graphviz; import scipy"
dot -V
```

### Data and Checkpoints

- **Leaderboard records**: `records/track_10min_16mb/` (submission directories with `submission.json` and `README.md`)
- **Training data**: `data/fineweb10B/` (195 train shards + 1 val shard, uint16 binary format)
- **Tokenizer**: `tokenizers/tok32000.model` (SentencePiece)
- **Model checkpoint**: `.safetensors` file from a trained run (needed for diagnostic scripts)

## Directory Structure

```
scripts/causal/
  common.py                  # Shared utilities (model loading, BPB, statistics, logging)
  extract_interventions.py   # C2: Parse leaderboard records into intervention matrix
  estimate_dag.py            # C3: Build/update causal DAG (expert + FCI)
  identifiability_check.py   # C4: Data quality assessment and identifiability scoring
  experiment_runner.py       # C5: Run paired-seed ablation experiments
  statistical_analysis.py    # C6: Effect estimation with decision gates
  token_loss_decompose.py    # C7: Per-token loss decomposition (diagnostic)
  quant_gap_analysis.py      # C8: Pre/post quantization gap (diagnostic)
  influence_proxy.py         # C9: Gradient inner product shard scoring (diagnostic)
  gradient_attribution.py    # C10: Per-layer gradient norm logging (diagnostic)

results/causal/
  interventions.json         # Extracted interventions from all records
  experiment_log.json        # Master log of all experiments across cycles
  cycle_0/                   # Cycle 0 outputs
    causal_dag.json          # DAG structure, edges, effects, recommendations
    dag.png                  # DAG visualization
    identifiability_report.json
  cycle_1/                   # Cycle 1+ outputs (after experiments)
    raw_runs.json            # Per-seed BPB results
    ablation_results.json    # Statistical analysis with decision gates
    causal_dag.json          # Updated DAG
    dag_diff.json            # Edge changes from previous cycle

tests/causal/                # Test suite for all scripts
```

## Cycle Protocol

### Cycle 0: Initial Discovery

Build the DAG from existing leaderboard records without running any experiments.

1. **Extract interventions** from all leaderboard records
2. **Estimate DAG** using expert-guided skeleton + FCI validation
3. **Check identifiability** to assess data quality
4. **Review** DAG edges, marginal effects, and next-intervention recommendation

```bash
# Step 1: Extract
python scripts/causal/extract_interventions.py \
  --input records/track_10min_16mb/ \
  --output results/causal/interventions.json

# Step 2: Estimate DAG
python scripts/causal/estimate_dag.py \
  --input results/causal/interventions.json \
  --output-dir results/causal/cycle_0/

# Step 3: Check identifiability
python scripts/causal/identifiability_check.py \
  --interventions results/causal/interventions.json \
  --dag results/causal/cycle_0/causal_dag.json \
  --output results/causal/cycle_0/identifiability_report.json
```

### Cycle 1+: Experiment and Update

Each subsequent cycle runs one intervention experiment, analyzes results, and updates the DAG.

1. **Select intervention** from the DAG's `next_intervention` recommendation
2. **Write configs** for treatment and control (JSON with `env_overrides`)
3. **Run experiment** with 3 seeds x 2 conditions
4. **Analyze statistically** to get confirmed/suggestive/null decision
5. **Update DAG** with new data point (append experiment, re-estimate)

```bash
# Step 1-2: Author treatment/control configs (manual)

# Step 3: Run experiment
python scripts/causal/experiment_runner.py \
  --treatment treatment_config.json \
  --control control_config.json \
  --output results/causal/cycle_1/raw_runs.json \
  --seeds 42,137,256 \
  --platform mlx

# Step 4: Statistical analysis
python scripts/causal/statistical_analysis.py \
  --input results/causal/cycle_1/raw_runs.json \
  --output results/causal/cycle_1/ablation_results.json

# Step 5: Append new data and re-estimate DAG
python scripts/causal/extract_interventions.py \
  --input records/track_10min_16mb/ \
  --output results/causal/interventions.json \
  --append-experiment results/causal/cycle_1/raw_runs.json

python scripts/causal/estimate_dag.py \
  --input results/causal/interventions.json \
  --output-dir results/causal/cycle_1/ \
  --previous-dag results/causal/cycle_0/causal_dag.json
```

### Stop Conditions

Stop the cycle when any of:

- **Confirmed effect found**: `decision == "confirmed"` (|effect| >= MDE and p < 0.01) -- integrate into submission
- **Time gate expires**: 2-day wall-clock limit reached
- **3 consecutive null results**: No detectable effect after 3 interventions

## CLI Reference

### extract_interventions.py

Parse leaderboard records into a structured intervention matrix.

```bash
python scripts/causal/extract_interventions.py \
  --input records/track_10min_16mb/ \
  --output results/causal/interventions.json

# Append experiment results as a new submission entry:
python scripts/causal/extract_interventions.py \
  --input records/track_10min_16mb/ \
  --output results/causal/interventions.json \
  --append-experiment results/causal/cycle_1/raw_runs.json
```

### estimate_dag.py

Build or update the causal DAG from interventions data.

```bash
# Cycle 0 (initial):
python scripts/causal/estimate_dag.py \
  --input results/causal/interventions.json \
  --output-dir results/causal/cycle_0/

# Cycle 1+ (with previous DAG for diff/stability tracking):
python scripts/causal/estimate_dag.py \
  --input results/causal/interventions.json \
  --output-dir results/causal/cycle_1/ \
  --previous-dag results/causal/cycle_0/causal_dag.json \
  --alpha 0.01
```

Output: `causal_dag.json`, `dag.png` (visualization).

### identifiability_check.py

Assess data quality: single/multi-variable record counts, identifiability score, confounded pairs, and unexplored combinations.

```bash
python scripts/causal/identifiability_check.py \
  --interventions results/causal/interventions.json \
  --dag results/causal/cycle_0/causal_dag.json \
  --output results/causal/cycle_0/identifiability_report.json
```

Output: `identifiability_report.json` with recommendation `proceed` or `skip`.

### experiment_runner.py

Run paired-seed ablation experiments (3 seeds x 2 conditions).

```bash
python scripts/causal/experiment_runner.py \
  --treatment treatment_config.json \
  --control control_config.json \
  --output results/causal/cycle_1/raw_runs.json \
  --seeds 42,137,256 \
  --platform mlx
```

Config format (treatment or control):

```json
{
  "script": "train_gpt_mlx.py",
  "env_overrides": {
    "NUM_LAYERS": "11",
    "MLP_MULT": "3",
    "MAX_WALLCLOCK_SECONDS": "600"
  },
  "description": "11L 3x MLP treatment"
}
```

### statistical_analysis.py

Compute effect sizes, bootstrap CIs, p-values, and decision gate classifications.

```bash
python scripts/causal/statistical_analysis.py \
  --input results/causal/cycle_1/raw_runs.json \
  --output results/causal/cycle_1/ablation_results.json \
  --mde 0.002 \
  --alpha 0.01
```

### token_loss_decompose.py

Per-token loss decomposition by frequency bucket and position (diagnostic).

```bash
python scripts/causal/token_loss_decompose.py \
  --checkpoint path/to/model.safetensors \
  --val-data data/fineweb10B/ \
  --tokenizer tokenizers/tok32000.model \
  --output results/causal/diagnostics/token_loss.json
```

### quant_gap_analysis.py

Measure the BPB gap introduced by int8 quantization (diagnostic).

```bash
python scripts/causal/quant_gap_analysis.py \
  --checkpoint path/to/model.safetensors \
  --val-data data/fineweb10B/ \
  --tokenizer tokenizers/tok32000.model \
  --output results/causal/diagnostics/quant_gap.json \
  --largest-training-effect 0.005
```

### influence_proxy.py

Score training data shards by gradient inner product with validation gradient (diagnostic).

```bash
python scripts/causal/influence_proxy.py \
  --checkpoint path/to/model.safetensors \
  --train-data data/fineweb10B/ \
  --val-data data/fineweb10B/ \
  --output results/causal/diagnostics/influence_scores.json \
  --max-shards 20
```

### gradient_attribution.py

Instrument training script to log per-layer gradient norms (diagnostic).

```bash
python scripts/causal/gradient_attribution.py \
  --script train_gpt_mlx.py \
  --output results/causal/diagnostics/gradient_attr.json \
  --val-every 100 \
  --gradient-log results/causal/diagnostics/gradient_norms.jsonl
```
