# Fast Full-Rescore N-gram (8xH100)

**3-seed mean val_bpb: 0.09420444** (std 0.00002598) | **max size: 13.44 MB** | **8x H100 SXM**

**Reference neural roundtrip (same runs): mean val_bpb 1.15945860** (std 0.00060298)

## Summary

This submission replaces selective two-pass chunk rescoring with a score-first **full-rescore** path:

1. Pass 1 scores validation normally and records per-token neural probability/entropy.
2. The N-gram cache is fully built from scored tokens.
3. Pass 2 rescoring is applied to **all chunks** using cached token statistics (no second neural forward pass).

The code also adds two robustness knobs:
- `NGRAM_SELF_EXCLUDE` for leave-one-out style subtraction.
- `NGRAM_COUNT_CONF_GAIN` to downweight low-support contexts.

Winner config in this submission keeps both robustness knobs off, because it was best in measured score.

## 3-Seed Results (Winner: `A_fullrescore_anchor`)

| Seed | final val_bpb | reference roundtrip val_bpb | pass1_bpb | pass2_bpb | train_s | eval_s | bytes_total |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1337 | 0.09423413 | 1.15987619 | 0.2830 | 0.0942 | 600.086 | 373.439 | 13,439,385 |
| 42 | 0.09417085 | 1.15860591 | 0.2827 | 0.0942 | 600.015 | 373.898 | 13,443,809 |
| 2025 | 0.09420833 | 1.15989369 | 0.2829 | 0.0942 | 600.089 | 373.760 | 13,433,689 |
| **Mean** | **0.09420444** | **1.15945860** | - | - | - | - | - |
| **Std** | **0.00002598** | **0.00060298** | - | - | - | - | - |

## A/B/C Exploration

| Run | Config | val_bpb |
|---|---|---:|
| A | Full-rescore anchor | **0.09423413** |
| B | Capacity tuned (8M buckets) | 0.12161267 |
| C | Robust (`self_exclude=1`, confidence gating) | 0.29024345 |

## Key Environment (Winner)

```bash
MODEL_PRESET=frontier_lean
RUN_PROFILE=full_8gpu_600s
TTT_ENABLED=0
QAT_MODE=off
NGRAM_EVAL_ENABLED=1
NGRAM_TWO_PASS_ENABLED=1
NGRAM_FULL_RESCORE=1
NGRAM_EVAL_MAX_ORDER=12
NGRAM_EVAL_CHUNK_TOKENS=262144
NGRAM_EVAL_BUCKETS=4194304
NGRAM_EVAL_ALPHA_MAX=0.70
NGRAM_EVAL_ORDER_MULTS="0.3,0.3,0.97,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0"
NGRAM_SELF_EXCLUDE=0
NGRAM_COUNT_CONF_GAIN=0.0
```

## Compliance Notes

- Train and eval each run are under the 10-minute caps on 8x H100.
- Artifact size is below 16,000,000 bytes.
- No tokenizer/dataset modification.
- Score-first order is preserved (no hindsight/min-loss selection).

## Files Included

- `train_gpt.py`
- `train_seed1337.log`
- `train_seed42.log`
- `train_seed2025.log`
- `submission.json`
- `requirements.txt`
