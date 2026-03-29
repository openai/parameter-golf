# Non-Record Submission: Compiled LeakyReLU2 + Slide64 Eval (1xRTX4090)

This is a non-record submission for the 16MB track documenting a single-GPU 600-second confirmation run on **1x RTX 4090**. The submission keeps the compiled leaky 9-layer proxy fixed and changes only the evaluation strategy: instead of scoring disjoint 1024-token chunks, it scores the validation set with a sliding window at `EVAL_STRIDE=64`.

The final post-quant roundtrip score is **1.33207082 val_bpb** with a total submission artifact size of **14,864,024 bytes**, which is under the 16MB cap.

This is **not** a leaderboard record claim. The run was not executed on the official 8xH100 setup, and the sliding-window evaluation is much slower than flat evaluation on a single 4090. The point of this submission is to document a reproducible, interesting non-record result and the surrounding sweep that established the eval-side win on this branch.

## Key Idea

The training setup is the same compiled leaky baseline used for the current 1x4090 proxy:

1. `MLP_ACTIVATION=leakyrelu2`
2. `LEAKY_SLOPE=0.5`
3. `TRAIN_BATCH_TOKENS=131072`
4. `WARMDOWN_ITERS=450`
5. `MATRIX_LR=0.028`
6. `SCALAR_LR=0.028`
7. `torch.compile` enabled

The only scoring change for the best run is:

1. `EVAL_STRIDE=64`
2. `EVAL_BATCH_SEQS=32`

With `TRAIN_SEQ_LEN=1024`, each validation token is scored once but with much richer left context than the flat baseline.

## Best Run

Run ID:

- `baseline_leaky_wd450_lr028_slide64_compiled_tb131072_uc600`

Key metrics from `train.log`:

- `final_int8_zlib_roundtrip_exact val_bpb`: **1.33207082**
- `final_int8_zlib_roundtrip_exact val_loss`: **2.24914289**
- pre-quant eval at stop: **1.3303 val_bpb**, **2.2462 val_loss**
- `step_stop`: **2301**
- `wallclock_seconds`: **600.007**
- `eval_time_seconds`: **990.944**
- `Total submission size int8+zlib`: **14,864,024 bytes**
- `Serialized model int8+zlib`: **14,801,183 bytes**
- `Code size`: **62,841 bytes**
- peak memory: **2664 MiB allocated / 3224 MiB reserved**

The `train_gpt.py` included in this folder is the exact source snapshot embedded at the top of the submitted run log.

## Flat Control vs Slide64

The most relevant comparison on this branch is the same training setup evaluated two ways:

| Run | Eval mode | Post-quant val_bpb | Eval time |
| --- | --- | --- | --- |
| `baseline_leaky_wd450_lr028_evalctl_compiled_tb131072_uc600` | flat | `1.36461125` | `27.916s` |
| `baseline_leaky_wd450_lr028_slide64_compiled_tb131072_uc600` | slide64 | **`1.33207082`** | `990.944s` |
| `baseline_leaky_wd450_lr028_slide64_dociso_compiled_tb131072_uc600` | slide64 + doc isolation | `1.34040978` | `1009.880s` |

So on this compiled leaky proxy, the score win is large and clear, but the runtime cost is also large. That makes this a good non-record submission and a useful reference point for later contest-hardware validation.

## Exact Configuration

```bash
RUN_ID=baseline_leaky_wd450_lr028_slide64_compiled_tb131072_uc600
DATA_PATH=./data/datasets/fineweb10B_sp1024/
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model
VOCAB_SIZE=1024
NUM_LAYERS=9
MODEL_DIM=512
NUM_HEADS=8
NUM_KV_HEADS=4
MLP_MULT=2
TRAIN_SEQ_LEN=1024
TRAIN_BATCH_TOKENS=131072
MAX_WALLCLOCK_SECONDS=600
VAL_LOSS_EVERY=0
TRAIN_LOG_EVERY=25
WARMUP_STEPS=0
WARMDOWN_ITERS=450
MLP_ACTIVATION=leakyrelu2
LEAKY_SLOPE=0.5
MATRIX_LR=0.028
SCALAR_LR=0.028
EVAL_STRIDE=64
EVAL_BATCH_SEQS=32
uv run train_gpt.py
```

Other logged settings of note:

- `tie_embeddings=True`
- `embed_lr=0.05`
- `head_lr=0.0`
- `muon_weight_decay=0.0`
- `adam_weight_decay=0.0`
- `compile_disabled=0`

## Sweep Context

This folder also includes `results.tsv`, the broader experiment table from the March 27 sweep. The main takeaways around this run were:

1. Sliding-window evaluation was the first clear 600-second win that held on this branch.
2. A stricter document-isolated variant still improved over the flat control, but not as much as the plain slide64 run.
3. Several training-side follow-ups on the same baseline did not beat the flat 600-second control, including late SWA, keep-float export tweaks, SmearGate, XSA, BigramHash, and ortho-init plus muP-style projection scaling.

## Included Files

- `train_gpt.py` - exact source snapshot for the submitted run
- `train.log` - full training log with final metric lines
- `results.tsv` - broader March 27 experiment table from the same pod workflow
- `submission.json` - metadata for this non-record submission
