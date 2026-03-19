This record captures an unlimited-compute non-record Apple Silicon MLX run using a deeper/narrower SP-1024 configuration.

The main idea was to trade some width for depth while keeping the artifact comfortably under the `16,000,000` byte cap:
- Layout: `VOCAB_SIZE=1024 NUM_LAYERS=14 MODEL_DIM=416 NUM_HEADS=8 NUM_KV_HEADS=2 MLP_MULT=2`
- Trainer: root `train_gpt_mlx.py` snapshot copied into this record folder as `train_gpt.py`
- Hardware: Apple Silicon (`Apple M5 Max`, local MLX run)
- Dataset/tokenizer: published `fineweb10B_sp1024` export with `10` train shards downloaded locally plus the full fixed validation split
- Track: non-record, unlimited compute, still under the `16,000,000` byte artifact cap

This run also increases validation batch size and enables logit chunking so the full validation pass can complete in a reasonable amount of time on local hardware without changing the measured metric:
- `VAL_BATCH_SIZE=8388608`
- `LOGIT_CHUNK_TOKENS=65536`

Command (run from this record folder so it uses the archived trainer snapshot):
```bash
cd records/track_non_record_16mb/2026-03-19_Deep14x416_KV2_SP1024_MLX_750it

RUN_ID=deep14_416_kv2_full_750 \
ITERATIONS=750 \
MAX_WALLCLOCK_SECONDS=0 \
TRAIN_BATCH_TOKENS=16384 \
VAL_LOSS_EVERY=0 \
VAL_BATCH_SIZE=8388608 \
LOGIT_CHUNK_TOKENS=65536 \
TRAIN_LOG_EVERY=50 \
WARMUP_STEPS=10 \
DATA_PATH=../../../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../../../data/tokenizers/fineweb_1024_bpe.model \
NUM_LAYERS=14 \
MODEL_DIM=416 \
NUM_HEADS=8 \
NUM_KV_HEADS=2 \
MLP_MULT=2 \
python3 ./train_gpt.py
```

Key metrics (from `train.log`):
- Final pre-quant eval at step `750`: `val_loss:3.1118`, `val_bpb:1.8430`
- Post-quant roundtrip eval: `val_loss:3.11359052`, `val_bpb:1.84404368`
- Exact printed metric: `final_int8_zlib_roundtrip_exact val_bpb:1.84404368`
- Train time: `529553ms` (`step_avg:706.07ms`)
- Saved FP model: `63982252 bytes`
- Serialized model int8+zlib: `12339367 bytes`
- Code size: `49622 bytes`
- Total submission size int8+zlib: `12388989 bytes`

Training volume:
- Global batch: `16384` tokens/step
- Total train tokens seen: `12288000`
- Local dataset subset: `10/195` train shards from `fineweb10B_sp1024`

Included files:
- `train_gpt.py` (exact MLX trainer snapshot used for the run, copied under the conventional record entrypoint name)
- `train.log` (exact local training log)
- `submission.json` (leaderboard metadata)
