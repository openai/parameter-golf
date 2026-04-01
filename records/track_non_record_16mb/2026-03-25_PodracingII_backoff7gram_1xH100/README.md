# Non-Record Submission: Podracing II Backoff7gram on 1xH100 with Zlib Fallback

This submission captures an off-spec but interesting **1xH100** run of the compacted `#753`-style root lane.

It is **not** a leaderboard-valid record submission. The run used `torchrun --nproc_per_node=1`, only reached `899` train steps inside the 600-second wallclock cap, and the final legal 7-gram evaluation took about `970s`, so it does not satisfy the challenge's `8xH100 / 10-minute eval` record conditions.

It is also **not** an exact `#753` donor reproduction. `flash-attn` was available on the pod, but `zstandard` was not installed, so the export path fell back to `int6+zlib` rather than the intended `int6+zstd`.

Even so, the result is notable: the same legal score-first adaptive `2..7`-gram backoff path from `#753` still reaches **`0.92092798 val_bpb`** on the final n-gram exact metric with a **`7,772,644` byte** total artifact, despite the badly undertrained dense base.

## Why This Is Interesting

This run is a signs-of-life demonstration for the `#753` eval mechanism, not a claim of a new training recipe or an exact donor repro.

- Hardware: `1xH100 80GB`
- Training wallclock: `600302ms`
- Steps completed: `899`
- Base post-export sliding exact: `2.10939936 val_bpb`
- Final legal score-first `7`-gram exact: **`0.92092798 val_bpb`**
- Total submission size: `7,772,644 bytes`

The dense model is clearly undertrained on a single GPU, but the evaluation tail is strong enough to drive the final score far below the current merged leaderboard values. That makes this a useful non-record data point for the challenge's requested-weird-ideas spirit, even though it is off-spec for acceptance as a record.

## Configuration

This run used the compact `#753`-style root settings, but on **one GPU** instead of `8xH100`, and with `int6+zlib` fallback export because `zstandard` was unavailable on the pod:

```bash
RUN_ID=root-pr753-repro \
SEED=2045 \
ADAM_WD=0.04 \
BIGRAM_DIM=128 \
BIGRAM_VOCAB_SIZE=1536 \
COMPILE_ENABLED=1 \
EVAL_SEQ_LEN=2048 \
EVAL_STRIDE=64 \
ITERATIONS=20000 \
LATE_QAT_THRESHOLD=0.5 \
LN_SCALE=1 \
MATRIX_LR=0.025 \
MAX_WALLCLOCK_SECONDS=600 \
MLP_ACT=leaky_relu_sq \
MLP_LEAKY_SLOPE=0.5 \
MUON_MOMENTUM=0.99 \
MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 \
MUON_WD=0.04 \
NGRAM_EVAL_ADAPTIVE=1 \
NGRAM_EVAL_ALPHA=0.30 \
NGRAM_EVAL_ALPHA_MAX=0.60 \
NGRAM_EVAL_ALPHA_MIN=0.05 \
NGRAM_EVAL_BUCKETS=4194304 \
NGRAM_EVAL_ENTROPY_CENTER=4.0 \
NGRAM_EVAL_ENTROPY_SCALE=2.0 \
NGRAM_EVAL_MAX_SECONDS=0.0 \
NGRAM_EVAL_MIN_COUNT=2 \
NGRAM_EVAL_MIN_ORDER=2 \
NGRAM_EVAL_ORDER=7 \
NUM_LAYERS=11 \
QAT_ENABLED=0 \
ROPE_DIMS=24 \
SCALAR_LR=0.025 \
TIED_EMBED_LR=0.035 \
TRAIN_BATCH_TOKENS=786432 \
TRAIN_SEQ_LEN=2048 \
VE_DIM=128 \
VE_ENABLED=1 \
VE_LAYERS=9,10 \
WARMUP_STEPS=20 \
WARMDOWN_ITERS=3500 \
XSA_LAST_N=4 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

## Key Metrics

From `train.log`:

- `world_size:1 grad_accum_steps:8`
- `flash_attn_interface: FOUND`
- `zstandard: MISSING`
- `step:899/20000 val_bpb:1.3749 train_time:600302ms step_avg:667.74ms`
- `Serialized model int6+zlib: 7698444 bytes`
- `Code size: 74200 bytes`
- `Total submission size int6+zlib: 7772644 bytes`
- `final_int6_roundtrip_exact val_bpb:2.11530408`
- `final_int6_sliding_window_exact val_bpb:2.10939936`
- `final_int6_sliding_window_ngram7_exact val_bpb:0.92092798`
- `final_int6_sliding_window_ngram7 eval_time:970441ms`

## Interpretation

This should be read as a **non-record 1xH100 sign-of-life**:

- It does **not** prove leaderboard-valid performance.
- It **does** show that the legal score-first `#753` backoff evaluator remains extremely strong even when the underlying dense model is far from the intended `8xH100` training regime and the export path is on the `zlib` fallback.
- The very small artifact size also suggests room for more ambitious model-side experimentation in off-spec tracks.

## Included Files

- `train_gpt.py`: pre-hybrid root snapshot used for this run
- `train.log`: original 1xH100 run log
- `submission.json`: metadata for this non-record submission
