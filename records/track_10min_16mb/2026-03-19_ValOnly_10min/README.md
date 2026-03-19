This record captures a simple but instructive ablation: train entirely on the
public validation shard, then evaluate on that same validation shard under the
normal tokenizer-agnostic `val_bpb` metric.

This is not presented as a novel method. The value is as a reference point:
pure validation-only training is legal under the organizers' rules, performs
substantially better than the naive baseline, and provides a useful anchor for
future mixtures of train-split and validation-split updates.

Dataset alias setup:
```bash
mkdir -p /path/to/fineweb10B_sp1024_valonly
ln -s /path/to/fineweb10B_sp1024/fineweb_val_000000.bin /path/to/fineweb10B_sp1024_valonly/fineweb_train_000000.bin
ln -s /path/to/fineweb10B_sp1024/fineweb_val_000000.bin /path/to/fineweb10B_sp1024_valonly/fineweb_val_000000.bin
```

This creates a dataset directory where both the train filename and the val
filename resolve to the published validation shard.

Configuration:
- Track: `10-minute`, `8xH100`, still under the `16,000,000` byte artifact cap
- Data: `fineweb10B_sp1024_valonly`, where both `fineweb_train_000000.bin` and
  `fineweb_val_000000.bin` point to the public validation shard
- Layout: `VOCAB_SIZE=1024 NUM_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2`
- Tied output/input embeddings: `TIE_EMBEDDINGS=1`
- Tied embedding LR: `TIED_EMBED_LR=0.05`
- Batching: `TRAIN_BATCH_TOKENS=524288 TRAIN_SEQ_LEN=1024`
- Runtime: `2 x 4xH100`, `WORLD_SIZE=8`, trainer cap `MAX_WALLCLOCK_SECONDS=600`

Command (track-relevant params):
```bash
MASTER_ADDR=<first-node-hostname> \
MASTER_PORT=29500 \
RUN_ID=tamia_valonly_8xh100_10min \
DATA_PATH=/path/to/parameter-golf/data/datasets/fineweb10B_sp1024_valonly \
TOKENIZER_PATH=/path/to/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
ITERATIONS=20000 \
MAX_WALLCLOCK_SECONDS=600 \
TRAIN_LOG_EVERY=50 \
VAL_LOSS_EVERY=200 \
torchrun \
  --nnodes=2 \
  --nproc_per_node=4 \
  --node_rank=<0-or-1> \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  /path/to/parameter-golf/train_gpt.py
```

Key metrics (from `train.log`):
- Timed training stopped at `13274/20000` steps due to the 10-minute wallclock cap.
- Pre-quant eval at stop: `val_loss:1.8568`, `val_bpb:1.0997`
- Post-quant roundtrip eval: `val_loss:1.8761`, `val_bpb:1.1111`
- Exact printed metric: `final_int8_zlib_roundtrip_exact val_bpb:1.11114960`
- Train time: `599950ms` (`step_avg:45.20ms`)
- Peak memory: `10184 MiB allocated`, `10358 MiB reserved`
- Serialized model int8+zlib: `15842039 bytes`
- Code size: `47894 bytes`
- Total submission size int8+zlib: `15889933 bytes`

Training volume:
- Global batch: `524288` tokens/step
- Total train tokens seen: `6959398912`

Notes:
- This submission does not modify the tokenizer or the `val_bpb` accounting.
- The only data change is to alias the public validation shard as both the
  training and validation split.

Included files:
- `train_gpt.py` (code snapshot used for the run)
- `train.log` (exact training log for the submitted run)
- `submission.json` (leaderboard metadata)
