# Non-Record: PR1471 Hessian Half-Group Clip Negative Result

**val_bpb = 1.09503234** on seed 42 | **15,993,125 bytes** | 8xH100 SXM

This is a non-record full-run negative result on the PR1471 SP8192 no-TTT stack. It tests a conservative Hessian-aware/group SDClip variant inspired by the PR1412 and PR1689 quantization directions. The run completed cleanly under the 10-minute training cap and under the 16MB artifact cap, but it regressed versus the PR1471 public same-lineage reference.

## Result

| Run | Seed | Stop step | Pre-quant BPB | Sliding BPB | Roundtrip BPB | Artifact bytes | Train time |
|-----|------|-----------|---------------|-------------|---------------|----------------|------------|
| PR1471 public reference | 3-seed mean | - | - | 1.0866 | - | ~15.98 MB | ~590s |
| Hessian half-group variant | 42 | 4017 | 1.09444822 | 1.09503234 | 1.11193162 | 15,993,125 | 590.110s |

The tested variant is worse than the PR1471 public reference by `+0.00843234` BPB. This is not a SOTA claim and not an optimization result; it is a documented full-scale negative result.

## What Changed

The base is PR1471:

`records/track_10min_16mb/2026-04-08_SP8192_SDClip_3LayerRecur_EMA0.9965_1.0866`

The tested quantization controls were:

```bash
HESSIAN_CLIP_LAMBDA=0.0875
CLIP_MULT_EARLY=1.03
CLIP_MULT_LOOP=1.015
CLIP_MULT_MID=1.0
CLIP_MULT_LATE=0.985
```

No tokenizer, data, architecture, training schedule, evaluation method, TTT path, or artifact format changed.

## Mechanism Tested

The patch extends SDClip with two default-off controls:

1. Row-level Hessian modulation:

```text
c_i = k * std(row_i) * (1 + lambda * (row_importance_i - 1))
```

2. Coarse layer-group clip multipliers:

```text
early: 1.03
loop:  1.015
mid:   1.0
late:  0.985
```

The intent was to allocate slightly different quantization slack across rows and layer groups while staying inside the 16MB cap. At full scale, the setting stayed under cap but did not improve quality.

## Compliance

- Train under 600s on 8xH100 SXM: yes, `590.110s`.
- Eval under 600s: yes, `277.205s`.
- Artifact under 16,000,000 bytes: yes, `15,993,125` bytes.
- No tokenizer change.
- No dataset change.
- No TTT, SLOT, ETLB, n-gram cache, or eval-time adaptation.
- Non-record submission: this is a single-seed negative result, not a SOTA claim.

## Reproduction

```bash
# from repo root after downloading the SP8192 FineWeb cache
REPO_DIR=$PWD
RECORD_DIR=$REPO_DIR/records/track_non_record_16mb/2026-04-30_PR1471_HessianHalfGroup_NegativeResult

cd "$RECORD_DIR"
DATA_PATH="$REPO_DIR/data/datasets/fineweb10B_sp8192/" \
TOKENIZER_PATH="$REPO_DIR/data/tokenizers/fineweb_8192_bpe.model" \
SEED=42 \
HESSIAN_CLIP_LAMBDA=0.0875 \
CLIP_MULT_EARLY=1.03 \
CLIP_MULT_LOOP=1.015 \
CLIP_MULT_MID=1.0 \
CLIP_MULT_LATE=0.985 \
torchrun --standalone --nproc_per_node=8 train_gpt.py \
  > train_seed42.log 2>&1
```

## Files

- `README.md`
- `submission.json`
- `train_gpt.py`
- `train_seed42.log`

## Attribution

- PR1471 base stack: @X-Abhishek-X.
- SP8192/GPTQ/SDClip lineage: @clarkkev.
- Hessian/group SDClip direction: @Robby955 and PR1412 lineage.
- Adaptive Hessian-sensitivity clipping direction: PR1689 lineage.
- Depth recurrence: @dexhunter.
- Parallel residuals: @Robby955 and @msisovic.
