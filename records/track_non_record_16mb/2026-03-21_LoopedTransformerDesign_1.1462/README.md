This record captures a non-record 10-minute submission built from a PR315-style trainer with an explicitly looped transformer core.

This run is intentionally submitted as a notable non-record result rather than a record attempt. It stays under the `16,000,000` byte artifact cap and the 10-minute training cap on `8xH100`, but it does not beat the current SOTA and was not exhaustively tuned. The goal of the submission is to document a clean recurrent-depth design point that meaningfully works without heavy optimization.

This result is far from optimized. It should be read as a working looped-transformer baseline, not a polished endpoint. If anyone wants to keep pushing this direction, the most useful next step would be to run proper sweeps over loop geometry, shared-vs-untied allocation, attention cadence, optimizer settings, and schedule choices.

Configuration:
- Track: `non-record`, 10-minute, under the `16,000,000` byte artifact cap
- Base stack: PR315-style `11L` feature set adapted into a looped transformer
- Model shape: `NUM_LAYERS=6 MODEL_DIM=640 NUM_HEADS=10 NUM_KV_HEADS=5`
- Loop design: `LOOP_CORE_LAYERS=2 LOOP_REPEATS=5 LOOP_ATTN_EVERY=2 LOOP_ADAPTER_DIM=64 LOOP_REPEAT_EMBED=1`
- Effective executed layers: `14`
- Positional/normalization changes: `ROPE_DIMS=16 LN_SCALE=1`
- Quantization path: `LATE_QAT=1 QAT_THRESHOLD=0.1`
- Attention extras: `XSA_LAST_N=4`
- Token-side features: `BIGRAM_VOCAB_SIZE=2048 BIGRAM_DIM=128`
- Batching: `TRAIN_BATCH_TOKENS=786432 TRAIN_SEQ_LEN=2048`
- Optimizer line: `MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 MUON_WD=0.04`
- EMA: enabled

Command (track-relevant params):
```bash
OMP_NUM_THREADS=1 \
TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
RUN_ID=pr315_loop_20260321_065138 \
DATA_PATH=/home/user/parameter-golf/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/home/user/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
NUM_LAYERS=6 \
MODEL_DIM=640 \
NUM_HEADS=10 \
NUM_KV_HEADS=5 \
MLP_MULT=3.0 \
LOOP_CORE_LAYERS=2 \
LOOP_REPEATS=5 \
LOOP_ATTN_EVERY=2 \
REFINE_MLP_MULT=1.0 \
REFINE_LOCAL_MIX=1 \
LOOP_ADAPTER_DIM=64 \
LOOP_REPEAT_EMBED=1 \
BIGRAM_VOCAB_SIZE=2048 \
BIGRAM_DIM=128 \
XSA_LAST_N=4 \
ROPE_DIMS=16 \
LN_SCALE=1 \
LATE_QAT=1 \
QAT_THRESHOLD=0.1 \
TRAIN_BATCH_TOKENS=786432 \
TRAIN_SEQ_LEN=2048 \
EVAL_SEQ_LEN=2048 \
WARMUP_STEPS=20 \
MATRIX_LR=0.025 \
SCALAR_LR=0.025 \
TIED_EMBED_LR=0.035 \
MUON_WD=0.04 \
EMA_ENABLED=1 \
EMA_DECAY=0.997 \
MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Key metrics (from `train.log`):
- Timed training stopped at `4851/20000` steps due to the 10-minute wallclock cap.
- Best pre-quant eval at stop: `val_loss:1.9559`, `val_bpb:1.1584`
- Post-quant roundtrip exact eval: `val_loss:1.97599793`, `val_bpb:1.17029728`
- Post-quant sliding exact eval: `val_loss:1.93531268`, `val_bpb:1.14620421`
- Train time: `600090ms` (`step_avg:123.70ms`)
- Peak memory: `31126 MiB allocated`, `31896 MiB reserved`
- Serialized model int6+zstd: `15509240 bytes`
- Code size: `79859 bytes`
- Total submission size int6+zstd: `15589099 bytes`

Training volume:
- Global batch: `786432` tokens/step
- Total train tokens seen: `3814981632`

Why this is notable:
- It demonstrates a working recurrent-depth / shared-core transformer submission under the official 10-minute and 16MB constraints.
- The design meaningfully improved over earlier loop baselines, but still left obvious tuning headroom.
- It is being submitted specifically as an architectural reference point, not as a polished SOTA attempt.
- It is intentionally offered as a base for further sweeps rather than as a tuned final answer.

Included files:
- `train_gpt.py` (exact code snapshot used for the run)
- `train.log` (exact remote training log)
- `submission.json` (submission metadata)
