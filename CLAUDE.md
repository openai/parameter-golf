# Parameter Golf Project

## What is this?

OpenAI's Parameter Golf challenge: train the best LM that fits in 16MB (int8+zlib compressed), trains in <10 min on 8×H100s, scored by bits-per-byte (BPB) on FineWeb validation. Lower BPB = better.

## Repo Structure

```
parameter-golf/
├── train_gpt.py                    # upstream baseline (PyTorch/CUDA)
├── train_gpt_mlx.py                # upstream baseline (MLX/Mac)
├── data/                           # FineWeb dataset + tokenizers
├── records/                        # upstream submission records
│
├── our/                            # === OUR CODE ===
│   ├── models/
│   │   ├── train_gpt_mlx_recurrent.py   # depth-recurrent model
│   │   └── train_gpt_mlx_enhanced.py    # + smear gate, backout
│   ├── scripts/
│   │   ├── run_experiment.sh             # single experiment runner
│   │   ├── adaptive_sweep.py             # smart adaptive sweep
│   │   └── sweep.py                      # grid sweep (legacy)
│   └── docs/
│       ├── APPROACH.md                   # strategy document
│       └── program.md                    # autonomous loop instructions
│
├── results/                        # === ALL OUTPUTS ===
│   ├── experiments.csv             # leaderboard
│   ├── pipeline_log.txt            # decision log
│   ├── logs/                       # per-experiment training logs
│   ├── configs/                    # per-experiment config JSONs
│   └── model_artifacts/            # saved model weights
```

## Running Experiments

```bash
# Single experiment
SCRIPT=our/models/train_gpt_mlx_recurrent.py ./our/scripts/run_experiment.sh my_test \
    NUM_UNIQUE_LAYERS=3 NUM_RECURRENCES=3 MODEL_DIM=768

# Adaptive sweep (learns from past results)
python3 our/scripts/adaptive_sweep.py --max-experiments 20

# Full eval (set VAL_SAMPLE_FRAC=1.0, default is 0.1 for fast A/B)
VAL_SAMPLE_FRAC=1.0 SCRIPT=our/models/train_gpt_mlx_recurrent.py \
    ./our/scripts/run_experiment.sh final_eval
```

## Key Constraints

- 16,000,000 bytes total artifact (code + compressed weights)
- Submission must be a single train_gpt.py (PyTorch, not MLX)
- Evaluation: BPB on fixed 50K-document FineWeb validation split
- 10 min training + 10 min eval on 8×H100s

## Current Best

Check `results/experiments.csv` for latest. Best so far: depth recurrence (3×3, dim=768) at 2.2714 BPB (200 iters, 1 shard — not comparable to leaderboard's 1.22 BPB which uses full data).
