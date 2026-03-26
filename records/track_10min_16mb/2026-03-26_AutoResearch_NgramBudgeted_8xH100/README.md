# AutoResearch Budgeted Two-Pass N-gram Backoff (8xH100)

**3-seed mean val_bpb: 0.118148** (std 0.000038) | **max size: 13.44 MB** | **8x H100 SXM**

## Summary

This submission builds from the current two-pass N-gram frontier and adds one focused autoresearch improvement:

1. **Budgeted two-pass tuner** (`NGRAM_BUDGETED_TUNER`): dynamically caps `NGRAM_TWO_PASS_RESCORE_CHUNKS` based on observed pass-1 throughput and remaining eval budget.
2. **Order-12 + weighted high-order backoff** with tuned `NGRAM_EVAL_ORDER_MULTS`.
3. **Legal score-first eval path** only (no TTT in this run set).

The tuner keeps eval under the 10-minute ceiling while retaining most of the pass-2 gain.

## 3-Seed Results

| Seed | val_bpb | pass1_bpb | pass2_bpb | train_s | eval_s | bytes_total |
|---|---:|---:|---:|---:|---:|---:|
| 1337 | 0.11819909 | 0.2862 | 0.1182 | 600.088 | 446.935 | 13,422,021 |
| 42 | 0.11813478 | 0.2860 | 0.1181 | 600.013 | 468.680 | 13,436,213 |
| 2025 | 0.11811002 | 0.2860 | 0.1181 | 600.067 | 446.318 | 13,430,005 |
| **Mean** | **0.11814796** | - | - | - | - | - |

## A/B/C Exploration During Session

| Run | Config | val_bpb |
|---|---|---:|
| A | Anchor two-pass | 0.13121982 |
| B | Budgeted tuner (winner) | **0.11819909** |
| C | Chunk-bias variant | 0.13358861 |

## Key Environment

```bash
MODEL_PRESET=frontier_lean
RUN_PROFILE=full_8gpu_600s
TTT_ENABLED=0
QAT_MODE=off
NGRAM_EVAL_ENABLED=1
NGRAM_EVAL_MAX_ORDER=12
NGRAM_TWO_PASS_ENABLED=1
NGRAM_TWO_PASS_RESCORE_CHUNKS=72
NGRAM_BUDGETED_TUNER=1
NGRAM_BUDGET_TARGET_SECONDS=580
NGRAM_BUDGET_SAFETY_SECONDS=8
```

## Compliance Notes

- 8x H100 run path used for all reported seeds.
- Training capped at ~600s and eval under 600s.
- Artifact size under 16,000,000-byte cap.
- No tokenizer/dataset modifications.
- Score-first evaluation only; no future-token leakage.

## Files Included

- `train_gpt.py`
- `train_seed1337.log`
- `train_seed42.log`
- `train_seed2025.log`
- `submission.json`
- `requirements.txt`
