# Compiled LeakyReLU2 + Slide64 Eval (1xRTX4090)

**val_bpb: 1.3321** | **14.86 MB** | **1x RTX 4090** | **600s train + 991s eval**

This is a non-record 16MB submission documenting a single-GPU confirmation run where the compiled leaky 9-layer proxy is kept fixed and only the evaluation strategy changes. Instead of scoring disjoint 1024-token chunks, the submitted run uses sliding-window evaluation with `EVAL_STRIDE=64`.

This is not a leaderboard record claim. The run was not executed on the official 8xH100 setup, and the eval path is too expensive on a single 4090 to claim contest-ready timing. The point of this submission is to document a reproducible, interesting result and the surrounding March 27 sweep that showed the eval-side win clearly on this branch.

## Results

| Run | Eval mode | Pre-quant bpb | Post-quant bpb | Gain vs flat | Eval time | Artifact |
| --- | --- | --- | --- | --- | --- | --- |
| `baseline_leaky_wd450_lr028_evalctl_compiled_tb131072_uc600` | flat | `1.3628` | `1.36461125` | baseline | `27.916s` | `14,851,789` |
| `baseline_leaky_wd450_lr028_slide64_compiled_tb131072_uc600` | slide64 | `1.3303` | **`1.33207082`** | **-0.0325** | `990.944s` | `14,864,024` |
| `baseline_leaky_wd450_lr028_slide64_dociso_compiled_tb131072_uc600` | slide64 + doc isolation | `1.3390` | `1.34040978` | `-0.0242` | `1009.880s` | `14,861,683` |

The submitted run is `baseline_leaky_wd450_lr028_slide64_compiled_tb131072_uc600`.

## Key Idea: Score With More Context

The training setup is the same compiled leaky baseline used for the current 1x4090 proxy:

- `MLP_ACTIVATION=leakyrelu2`
- `LEAKY_SLOPE=0.5`
- `TRAIN_BATCH_TOKENS=131072`
- `WARMDOWN_ITERS=450`
- `MATRIX_LR=0.028`
- `SCALAR_LR=0.028`
- `torch.compile` enabled

The only scoring change for the submitted run is:

```python
EVAL_STRIDE = 64
EVAL_BATCH_SEQS = 32
```

With `TRAIN_SEQ_LEN=1024`, each validation token is still scored once, but it is usually scored with much richer left context than in the flat baseline. On this branch, that alone was enough to turn the same 600-second training run from `1.36461125` to `1.33207082` post-quant.

## Training Architecture

| Component | Setting |
| --- | --- |
| Layers | 9 |
| Width | 512 |
| Heads / KV heads | 8 / 4 |
| MLP | 2x with `leakyrelu2` |
| Tokenizer | SentencePiece 1024 |
| Batch | `131072` tokens |
| Sequence length | `1024` |
| Wallclock | `600s` |
| Warmdown | `450` |
| Optimizer | Muon + Adam, no extra weight decay |
| Compile | enabled |

### Best Run Metrics

| Metric | Value |
| --- | --- |
| `final_int8_zlib_roundtrip_exact val_bpb` | **`1.33207082`** |
| `final_int8_zlib_roundtrip_exact val_loss` | `2.24914289` |
| `step_stop` | `2301` |
| `wallclock_seconds` | `600.007` |
| `eval_time_seconds` | `990.944` |
| `Serialized model int8+zlib` | `14,801,183` |
| `Code size` | `62,841` |
| `Total submission size int8+zlib` | `14,864,024` |
| Peak memory | `2664 MiB allocated / 3224 MiB reserved` |

The `train_gpt.py` included in this folder is the exact source snapshot embedded at the top of the submitted run log.

## Run Command

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

This folder also includes `results.tsv`, the broader March 27 experiment table from the same pod workflow. The main takeaways around this run were:

1. Sliding-window evaluation was the first clear 600-second win that held on this branch.
2. A stricter document-isolated variant still improved over the flat control, but not as much as the plain slide64 run.
3. Several training-side follow-ups on the same baseline did not beat the flat 600-second control, including late SWA, keep-float export tweaks, SmearGate, XSA, BigramHash, and ortho-init plus muP-style projection scaling.

## Included Files

- `train_gpt.py` - exact source snapshot for the submitted run
- `train.log` - full training log with final metric lines
- `results.tsv` - broader March 27 experiment table from the same pod workflow
- `submission.json` - metadata for this non-record submission
