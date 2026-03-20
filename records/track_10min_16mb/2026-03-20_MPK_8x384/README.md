This record captures an MPK-style multi-path causal language model submission candidate for the 10-minute 16 MB track.

Final submission score: `val_bpb:1.01558307` (`val_loss:2.28798722`) after the int8+zlib roundtrip.

Additional supporting seeds now included:
- `seed=1338`: `val_bpb:1.04330173` (`val_loss:2.35043404`)
- `seed=1339`: `val_bpb:1.03281422` (`val_loss:2.32680692`)
- `seed=1340` (clean rerun): `val_bpb:1.01452726` (`val_loss:2.28560860`)
- 3-seed mean across `1337/1338/1339`: `val_bpb:1.03056634`
- 2-seed mean across matched-environment clean runs `1337/1340`: `val_bpb:1.01505517`

Trainer/model changes in this snapshot:
- MPK model family enabled in `train_gpt.py`
- `8` layers at width `384` with `8` attention heads and `4` KV heads
- MPK temporal strides `k=2`, `m=4`
- tied embeddings with tuned lower learning rates
- full FineWeb SP-1024 validation and `80` training shards

Configuration:
- Layout: `MODEL_FAMILY=mpk NUM_LAYERS=8 MODEL_DIM=384 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2`
- MPK strides: `MPK_K_STRIDE=2 MPK_M_STRIDE=4`
- Tied embeddings: `TIE_EMBEDDINGS=1`
- Learning rates: `TIED_EMBED_LR=0.03 MATRIX_LR=0.02 SCALAR_LR=0.02`
- Batching: `TRAIN_BATCH_TOKENS=524288 TRAIN_SEQ_LEN=1024`

Command (track-relevant params):
```bash
RUN_ID=mpk_real_8x384_80shards \
MODEL_FAMILY=mpk \
NUM_LAYERS=8 \
MODEL_DIM=384 \
NUM_HEADS=8 \
NUM_KV_HEADS=4 \
MLP_MULT=2 \
MPK_K_STRIDE=2 \
MPK_M_STRIDE=4 \
TIED_EMBED_LR=0.03 \
MATRIX_LR=0.02 \
SCALAR_LR=0.02 \
COMPILE_MODEL=0 \
ITERATIONS=20000 \
WARMUP_STEPS=0 \
VAL_LOSS_EVERY=1000 \
MAX_VAL_TOKENS=0 \
MAX_WALLCLOCK_SECONDS=600 \
TRAIN_BATCH_TOKENS=524288 \
VAL_BATCH_SIZE=524288 \
TRAIN_LOG_EVERY=200 \
DATA_PATH=/workspace/parGolfMPK/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/workspace/parGolfMPK/data/tokenizers/fineweb_1024_bpe.model \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Key metrics (from `train.log`):
- Timed training stopped at `4108/20000` steps due to the wallclock cap.
- Pre-quant eval at stop: `val_loss:2.2827`, `val_bpb:1.0132`
- Post-quant roundtrip eval: `val_loss:2.28798722`, `val_bpb:1.01558307`
- Rounded log line: `final_int8_zlib_roundtrip val_loss:2.2880 val_bpb:1.0156`
- Timed training: `599527ms` (`step_avg:145.94ms`)
- Final eval time after stop: `6088ms`
- Peak memory: `19867 MiB allocated`, `20514 MiB reserved`
- Serialized model int8+zlib: `14518775 bytes`
- Code size: `59498 bytes`
- Total submission size int8+zlib: `14578273 bytes`

Training volume:
- Global batch: `524288` tokens/step
- Total train tokens seen before cap: `2153775104`

Supporting seeds:
- `mpk_seed_1338.txt`: slower host / non-template rerun with a compatibility patch for hosts whose PyTorch lacked native `enable_gqa`; stopped at `2146` steps and finished at `val_bpb:1.04330173`
- `mpk_seed_1339.txt`: same slower environment / compatibility setup; stopped at `2480` steps and finished at `val_bpb:1.03281422`
- These runs were gathered after accidentally provisioning a fresh 8x H100 instance without the exact prior par-golf template environment, which reduced throughput versus the original `145.94ms` step average run
- `mpk_seed_1340_clean.txt`: clean rerun on the correct native-GQA environment; stopped at `4094` steps with `step_avg:146.50ms` and finished at `val_bpb:1.01452726`, closely matching the original `1.01558307` run

Included files:
- `train_gpt.py` (MPK trainer snapshot used for the run)
- `train.log` (remote training log for the timed run)
- `mpk_seed_1338.txt` (supporting seed log)
- `mpk_seed_1339.txt` (supporting seed log)
- `mpk_seed_1340_clean.txt` (clean matched-environment supporting seed log)
- `submission.json` (leaderboard metadata)
