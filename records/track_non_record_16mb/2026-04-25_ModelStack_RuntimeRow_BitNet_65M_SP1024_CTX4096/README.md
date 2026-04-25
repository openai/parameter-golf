# Non-record: Model Stack Runtime-Row BitNet 65M, SP1024, Context 4096

This is a non-record submission focused on a kernel-compatible ternary layout for Model Stack's packed BitNet runtime.

The run trains a 64.5M parameter BitNet-style ternary transformer under the 16MB artifact limit using runtime-row ternary scales. The resulting artifact can be exported exactly into Model Stack's packed BitNet runtime format with no skipped packed tensors and zero packed-weight reconstruction error.

## Result

| Metric | Value |
|---|---:|
| Final roundtrip val_loss | 2.0515 |
| Final roundtrip val_bpb | 1.2350 |
| Pre-roundtrip val_bpb at stop | 1.2341 |
| Training steps | 5,630 |
| Training time | 600,095 ms |
| Step average at stop | 106.6 ms |
| Artifact bytes | 14,330,708 |
| Current code bytes | 74,361 |
| Total submission bytes | 14,405,069 |
| Budget | 14.40MB / 16.00MB |

## Configuration

- Tokenizer/data: `sp1024`, cached FineWeb challenge split, 80 train shards.
- Context: `TRAIN_SEQ_LEN=4096`, `YARN_MAX_LEN=4096`.
- Model: `MODEL_DIM=1024`, `NUM_LAYERS=7`, `NUM_HEADS=16`, `NUM_KV_HEADS=4`, `MLP_MULT=2`.
- Ternary layout: `BITNET_SCALE_LAYOUT=runtime_row`, `BITNET_GROUP_SIZE=64`.
- Training: `TRAIN_BATCH_TOKENS=524288`, `MAX_WALLCLOCK_SECONDS=599`, `ITERATIONS=20000`.
- Hardware: 8x NVIDIA H100 80GB HBM3.

## Run Command

The exact run script is included as `run_65m_full.sh`. The command used was:

```bash
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
TRAIN_SEQ_LEN=4096 \
YARN_MAX_LEN=4096 \
RUN_ID=ternary_runtime_row_sp1024_ctx4096_8xh100_1024x7_full \
MAX_WALLCLOCK_SECONDS=599 \
ITERATIONS=20000 \
VAL_LOSS_EVERY=0 \
VAL_BATCH_SIZE=524288 \
TRAIN_BATCH_TOKENS=524288 \
TRAIN_LOG_EVERY=50 \
WARMUP_STEPS=1 \
BITNET_GROUP_SIZE=64 \
BITNET_SCALE_LAYOUT=runtime_row \
FP_STORAGE=0 \
SEQ_SCHEDULE_FRACTION=0.33 \
BATCH_SCHEDULE_FRACTION=0.33 \
MATRIX_LR=0.04 \
SCALAR_LR=0.04 \
TIED_EMBED_LR=0.05 \
HEAD_LR=0.008 \
MUON_BACKEND_STEPS=5 \
LOGIT_SOFTCAP=30 \
QK_GAIN_INIT=1.5 \
MODEL_DIM=1024 \
NUM_LAYERS=7 \
NUM_HEADS=16 \
NUM_KV_HEADS=4 \
MLP_MULT=2 \
TRAINING_DEPTH_RECURRENCE=1 \
EVAL_DEPTH_RECURRENCE=1 \
ROPE_TYPE=yarn \
torchrun --nproc_per_node=8 train_gpt.py
```

## Model Stack Runtime Export

After training, the generated `final_model.ternary.ptz` was exported through Model Stack's Parameter Golf BitNet exporter:

```bash
python3 tests/bench_parameter_golf_bitnet_export.py \
  --pg-script /root/parameter_golf/train_gpt_cuda_ternary.py \
  --artifact /root/parameter_golf/final_model.ternary.ptz \
  --export-packed /root/transformer_10_h100/artifacts/parameter_golf/ternary_1024x7_ctx4096_8xh100.runtime_bitnet.pt \
  --verify-export \
  --summary-json
```

Export verification summary:

- Packed ternary params: `62,390,272`
- Packed ternary tensors: `28`
- Floating params: `1,081,456`
- Floating tensors: `31`
- Skipped tensors: `0`
- Max packed absolute error: `0.0`
- Max floating absolute error: `0.0`

The exported runtime bundle is not required for this Parameter Golf folder, but it demonstrates that the trained ternary weights are exactly representable by Model Stack's packed BitNet runtime path.

## Included Files

- `train_gpt.py`: standalone training/evaluation script used for this run.
- `train.log`: full canonical 8xH100 run log.
- `run_65m_full.sh`: exact launcher script used on the 8xH100 machine.
- `submission.json`: metadata for this non-record submission.
