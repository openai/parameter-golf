# Arman SP8192 4ep Split-WD Legal TTT 40K

Status: pending final Armenia 8xH100 FineWeb seed scores.

This record mirrors the root `train.py` entrypoint used for the Armenia final submission branch. It is not a final leaderboard claim until the required 8xH100 seed runs finish and the score placeholders are replaced.

## Required Score Placeholders

The Armenia form asks for regular FineWeb validation scores on these seeds:

| Seed | FineWeb val_bpb |
|---:|---:|
| 1337 | TBD |
| 42 | TBD |
| 2025 | TBD |

## Candidate Stack

- FineWeb SP8192 cached data and tokenizer
- 11-layer 512d Transformer, 8 query heads, 4 KV heads
- 3-layer depth recurrence
- parallel residual lanes
- QK gain 5.25
- Muon-style optimizer with split weight decay
- EMA
- GPTQ int6/int8 compression plus Brotli
- legal score-first TTT

Primary environment values:

```text
VOCAB_SIZE=8192
QK_GAIN_INIT=5.25
TTT_ENABLED=1
TTT_LR=0.005
TTT_EPOCHS=4
TTT_MOMENTUM=0.9
TTT_CHUNK_TOKENS=40960
MIN_LR=0.1
MUON_WD=0.095
MUON_WD_MLP=0.115
EMA_DECAY=0.9965
WARMDOWN_FRAC=0.72
GPTQ_RESERVE_SECONDS=0.5
MAX_WALLCLOCK_SECONDS=590
```

## Rerun Plan

The final submission should replace the placeholders with three valid 8xH100 logs:

```text
seeds: 1337, 42, 2025
hardware: 8x H100
required: train_ms < 600,000, eval_ms < 600,000, artifact_bytes < 16,000,000
```

Queued 8xH100 jobs on YSU at the time this draft was updated:

```text
32652 pg_r40_1337  SEED=1337  TTT_CHUNK_TOKENS=40960  MAX_WALLCLOCK_SECONDS=590
32653 pg_r40_42    SEED=42    TTT_CHUNK_TOKENS=40960  MAX_WALLCLOCK_SECONDS=590
32654 pg_r40_2025  SEED=2025  TTT_CHUNK_TOKENS=40960  MAX_WALLCLOCK_SECONDS=590
```

These are the only runs intended for final leaderboard evidence.

## Lineage

This is a public-stack remix. It intentionally attributes the main ingredients to prior public OpenAI Parameter Golf work:

- SP8192 / GPTQ / SDClip line from Kevin Clark and related public PRs
- depth recurrence from dexhunter lineage
- parallel residuals from Robby Sneiderman / Marko Sisovic lineage
- legal score-first TTT from abaybektursun and dexhunter lineage
- hyperparameter remix around 4-epoch TTT, split WD, EMA, and wallclock reserve

The final submission should include the completed logs and a short ablation note before being marked ready for review.
