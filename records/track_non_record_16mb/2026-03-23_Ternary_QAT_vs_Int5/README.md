# Non-Record Submission: Ternary QAT vs Int5 Proxy Study

This folder contains a non-record submission documenting a BitNet b1.58 style ternary-QAT exploration built from the published `2026-03-20` top-1 `train_gpt.py`.

The work is intentionally scoped as an interesting negative/partial result rather than a leaderboard claim:

- training and evaluation were run on `1x RTX 2060 Max-Q`
- development used a FineWeb-based proxy split rather than the full challenge validation set
- the objective was to test whether ternary MLP compression can buy enough byte headroom to scale model size under the `16,000,000` byte cap

## Submission Summary

The central result is that `full ternary` was not competitive locally, but `mixed ternary` remained viable when the saved bytes were reinvested into a larger model.

Key runs:

| Run | Config | Final val_bpb | Total bytes | Notes |
|---|---|---:|---:|---|
| `proxy_int5_10l_seed42_20260323` | matched `10L` int5 | 3.26066687 | 15,406,268 | reference proxy baseline |
| `proxy_mixed_10l_seed42_20260323` | matched `10L` mixed ternary | 3.26632616 | 8,705,415 | worse quality, much smaller artifact |
| `proxy_mixed_11l640_seed42_20260329` | mixed ternary, `11L d=640` | 3.26090698 | 13,609,639 | best run so far |
| `proxy_mixed_11l640_seed1337_20260329` | mixed ternary, `11L d=640` | 3.26105237 | 13,572,709 | confirms seed stability |

For the two larger mixed runs, the mean post-quant proxy score is `3.26097967` with standard deviation `0.00007269`. That is only about `+0.00031` bpb behind the matched int5 proxy baseline while still saving about `1.8 MB` of total artifact size.

## What Changed

Relative to the copied top-1 script, this submission adds:

- `QUANT_MODE=int5|ternary|mixed`
- `TernaryLinear` with STE ternary forward weights in `{-1, 0, +1}`
- `pack_ternary()`, `unpack_ternary()`, and generic quantized export/dequantize paths
- post-export quantized roundtrip evaluation for mixed and ternary modes
- local portability knobs:
  - `COMPUTE_DTYPE`
  - `USE_TORCH_COMPILE`
  - `USE_FUSED_OPTIM`
- ternary zero-fraction logging and abort protection

The experiment script stays under the repo's `1500` line limit.

## Why This Is Interesting

The useful result is not "ternary beats int5 at the same model shape." It does not.

The useful result is narrower:

1. Same-shape mixed ternary compresses far more aggressively than int5.
2. That byte savings can fund a much larger mixed model.
3. On the proxy split, the larger mixed model nearly closes the quality gap to int5 while still staying comfortably under the artifact cap.

This makes mixed ternary a plausible budget-reallocation technique, even though full ternary currently looks too lossy.

## Reproducing the Proxy Runs

Install the repo dependencies plus `zstandard`:

```bash
pip install -r requirements.txt
```

Generate the FineWeb-based proxy split:

```bash
python3 make_proxy_dataset.py
export PROXY_DATA_PATH=/abs/path/data/datasets/fineweb10B_sp1024_proxy
```

Best configuration so far:

```bash
env DATA_PATH="$PROXY_DATA_PATH" \
TOKENIZER_PATH=/abs/path/data/tokenizers/fineweb_1024_bpe.model \
RUN_ID=proxy_mixed_11l640_seed42 \
QUANT_MODE=mixed SEED=42 \
COMPUTE_DTYPE=fp32 USE_TORCH_COMPILE=0 USE_FUSED_OPTIM=0 \
NUM_LAYERS=11 MODEL_DIM=640 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=3 \
ITERATIONS=600 TRAIN_LOG_EVERY=50 SPARSITY_LOG_EVERY=100 \
TRAIN_BATCH_TOKENS=4096 TRAIN_SEQ_LEN=256 VAL_BATCH_SIZE=8192 \
EVAL_STRIDE=0 EVAL_BATCH_SEQS=4 MATRIX_LR=0.0005 SCALAR_LR=0.001 \
TIED_EMBED_LR=0.003 HEAD_LR=0.001 WEIGHT_DECAY=0.0 MUON_MOMENTUM=0.95 \
GRAD_CLIP_NORM=0.1 python3 train_gpt.py
```

The helper launcher used for the recorded runs is [run_proxy_experiment.sh](/home/alpha/dev/research/competition/parameter-golf/records/track_non_record_16mb/2026-03-23_Ternary_QAT_vs_Int5/run_proxy_experiment.sh).

## Included Files

- `train_gpt.py`: submission code snapshot
- `submission.json`: metadata for the non-record track
- `requirements.txt`: repo requirements plus `zstandard`
- `logs/*.txt`: train logs for the smoke matrix and larger proxy comparisons
- `results_summary.md`: compact result table
- `technical_report.md`: detailed analysis of the current findings
- `make_proxy_dataset.py`: local helper to build the proxy split

## Limitations

- These runs do not claim an official challenge score on the full FineWeb validation split.
- The larger mixed result is promising, but still local and proxy-based.
- Mixed runs consistently saturated the last MLP projection to `1.000` zero fraction, which is a likely quality bottleneck.
- `author` and `github_id` in `submission.json` still need to be filled before opening a PR.
