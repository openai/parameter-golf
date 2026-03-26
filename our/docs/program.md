# Autonomous Experiment Loop — Instructions for Claude

You are running autonomous experiments for the Parameter Golf challenge.
Your goal: minimize val_bpb while keeping the compressed model under 16MB.

## Current Best

Check `experiments.csv` for the latest results. Beat the best val_bpb.

## Workflow

1. **Read** `experiments.csv` to see what's been tried and what the current best is
2. **Think** about what to try next. Pick ONE change at a time so you can attribute improvements
3. **Run** the experiment using `run_experiment.sh`
4. **Evaluate** — compare the new val_bpb to the current best
5. **Log** — results auto-append to `experiments.csv`
6. **Repeat** — try the next idea

## Experiment Ideas (ordered by expected impact)

### Architecture search (use train_gpt_mlx_recurrent.py)
- Vary num_unique_layers: 2, 3, 4, 5
- Vary num_recurrences: 2, 3, 4, 5, 6
- Vary model_dim: 640, 768, 896, 1024
- Vary num_heads: 8, 12, 16
- Vary num_kv_heads: 2, 4, 8
- Vary mlp_mult: 1, 2, 3

### Training hyperparameters
- Learning rates: MATRIX_LR, SCALAR_LR, TIED_EMBED_LR
- Batch size: TRAIN_BATCH_TOKENS
- Warmup/warmdown: WARMUP_STEPS, WARMDOWN_ITERS
- Optimizer: BETA1, BETA2, MUON_MOMENTUM

### Compression-aware
- Wider models compress better with int8 (more rows = better per-row quantization)
- Models near 16MB boundary: check compressed size carefully

## Constraints

- ALWAYS check that compressed model < 16,000,000 bytes
- Use ITERATIONS=200 for quick A/B tests, ITERATIONS=500+ for promising configs
- One change at a time for attribution
- Name experiments descriptively: `recurrent_3x4_dim896`, `baseline_wider_640`

## What NOT to do

- Don't modify the validation/evaluation code
- Don't change the dataset or tokenizer (yet — that's a separate track)
- Don't run more than one experiment at a time on MacBook (memory)
