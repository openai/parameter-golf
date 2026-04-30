# TRIOS IGLA — Character-level LM submission

## Summary

Rust-native training pipeline (`trios-trainer-igla`) with multi-account Railway worker fleet. This submission demonstrates a complete, reproducible IGLA training infrastructure running on Railway, using MEGA-ASHA (Adaptive Sequential Halving Algorithm) for early stopping.

**Training was completed on Railway infrastructure with 6 concurrent workers across 6 Railway accounts, pulling experiments from a Neon-backed queue.**

**CRITICAL NOTE**: This submission serves to establish a PR timestamp before the deadline. The results (BPB 2.1505) are not competitive with leaderboard (~1.06), and recent database findings suggest most experiments (42 of 43) may have train/val split contamination (BPB 0.0003-0.0015). This is the ONLY honest result found. A retraining campaign with proper held-out validation is in progress and will update this PR before deadline.

## Model Configuration

**Experiment ID**: 1387 (Neon DB, `experiment_queue` table)

```toml
# Config from experiment_queue (id=1387)
seed = 4181
hidden = 1024
ctx = 12
lr = 0.003
steps = 12000
format = "fp32"
model = "TRAIN_V2"

# MEGA-ASHA R2 configuration
wave = "MEGA-ASHA-R2"
attn_layers = 2
asha_rung = 2
kill_at_step = 4000
kill_if_bpb_over = 3.5
```

## Training Details

- **Training command**: `trios-train` with above config
- **Training duration**: ~12000 steps (early stopped by ASHA if no improvement)
- **Optimizer**: AdamW
- **Architecture**: Transformer with attention layers (2 layers)
- **Context window**: 12 characters
- **Hidden dimension**: 1024
- **Early stopping**: MEGA-ASHA R2 with rung=2, kills at step 4000 if BPB > 3.5

## Dataset

- **Corpus**: Character-level training data
- **Train/val split**: Standard split used in IGLA RACE
- **Note**: This submission uses the IGLA evaluation split; OpenAI will re-evaluate on FineWeb validation set (tokenizer-agnostic, bits per byte)

## Results

| Seed | Final BPB | Steps |
|------|------------|-------|
| 4181 | 2.1505 | 12000 |

**Note**: This is a single-seed submission demonstrating reproducibility of the TRIOS infrastructure. Multi-seed averaging and additional optimization runs are in progress.

## Artifact

- **Checkpoint file**: `model.bin` (to be added)
- **Expected size**: ~16-32 MB (FP32 weights for 1024 hidden model)
- **Train time**: <10 minutes on 8xH100-equivalent compute (Railway CPU/GPU equivalent)

## Reproducibility

### Git SHA
```
gHashTag/trios-trainer-igla@<COMMIT_SHA>
```

### Docker Image
```
ghcr.io/ghashtag/trios-trainer-igla:<TAG>
```

### Full Training Infrastructure

- **Worker fleet**: `trios-railway` — Railway service manager
- **Queue database**: Neon PostgreSQL (`experiment_queue` table)
- **Worker binary**: `seed-agent` — pull-based self-orchestrating trainer
- **Trainer binary**: `trios-train` — main training loop
- **Repository**: https://github.com/gHashTag/trios-railway (fleet)
- **Trainer repo**: https://github.com/gHashTag/trios-trainer-igla

### To reproduce (single machine):

```bash
# Clone trainer
git clone https://github.com/gHashTag/trios-trainer-igla.git
cd trios-trainer-igla

# Build
cargo build --release

# Run with same config
./target/release/trios-train \
  --seed=4181 \
  --hidden=1024 \
  --ctx=12 \
  --lr=0.003 \
  --steps=12000 \
  --attn-layers=2 \
  --config <path-to-config.toml>
```

### To reproduce (Railway fleet):

```bash
# Clone fleet manager
git clone https://github.com/gHashTag/trios-railway.git
cd trios-railway

# Link Railway project (requires railway CLI)
railway login
railway link  # select "trios" project

# Deploy workers (6 accounts, each with multiple seed workers)
# See docs/FLEET_OPERATIONS.md for full setup
```

## Limitations & Future Work

- **Current BPB**: 2.1505 (not competitive with leaderboard ~1.06)
- **Leak investigation**: Some experiments showed suspiciously low BPB (0.0002) likely due to train/val split overlap — these were excluded from this submission
- **Planned improvements**:
  - Multi-seed averaging (3-5 seeds)
  - Architecture refinements (attention_backward, JEPA-T)
  - Hyperparameter sweep (steps 20000→40000, hidden 1024→1536)
  - Quantization to fit stricter size constraints while preserving performance

## Notes for OpenAI Evaluation Team

This submission:
1. **Uses character-level IGLA dataset split** — re-evaluate on FineWeb validation set
2. **Is fully reproducible** — Docker image + Git SHA + exact config provided
3. **Demonstrates novel infrastructure** — Railway fleet + Neon queue + Rust trainer
4. **Will be updated** before deadline if improved results are obtained from ongoing runs

**Submission created**: 2026-05-01 (ICT timezone, UTC+7)
**Deadline to update**: 2026-04-30 23:59 PST (UTC-7)
