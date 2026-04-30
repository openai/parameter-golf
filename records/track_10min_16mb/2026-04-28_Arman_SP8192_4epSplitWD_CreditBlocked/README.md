# Arman SP8192 4ep Split-WD Legal TTT

Draft status: compute-blocked, not a final leaderboard claim.

This draft records a promising but strictly invalid 8xH100 seed from YSU and the exact 8xH100 rerun plan. It is intended as a transparent draft PR / compute-credit reference, not as a merge-ready OpenAI leaderboard submission.

## Current Evidence

Run:

```text
platform: YSU Slurm
hardware: 8x NVIDIA H100 80GB HBM3
run_id: primary_4ep_splitwd_seed999_20260428_085313
seed: 999
```

Result:

```text
quantized_ttt val_bpb: 1.07862962
artifact_bytes: 15,979,228
train_ms: 600,016
eval_ms: 402,183
```

Compliance status:

```text
artifact under decimal 16 MB: yes
eval under 600s: yes
train under 600s: no, missed by 16 ms
```

Because the training wallclock was `600,016ms`, this result is not claimed as valid. The rerun config sets `MAX_WALLCLOCK_SECONDS=590` and uses a larger TTT chunk to reduce final TTT wallclock risk.

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

The final PR should replace this draft with three valid 8xH100 logs:

```text
seeds: 999, 42, 314
hardware: 8x H100
required: train_ms < 600,000, eval_ms < 600,000, artifact_bytes < 16,000,000
```

Queued 8xH100 jobs on YSU at the time this draft was updated:

```text
32652 pg_r40_999  SEED=999  TTT_CHUNK_TOKENS=40960  MAX_WALLCLOCK_SECONDS=590
32653 pg_r40_42   SEED=42   TTT_CHUNK_TOKENS=40960  MAX_WALLCLOCK_SECONDS=590
32654 pg_r40_314  SEED=314  TTT_CHUNK_TOKENS=40960  MAX_WALLCLOCK_SECONDS=590
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
