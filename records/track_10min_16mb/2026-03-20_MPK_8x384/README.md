This record captures an MPK-style multi-path causal language model submission candidate for the 10-minute 16 MB track.

Correction: earlier reported BPB values in this record were invalid due to a SentencePiece leading-space accounting bug in `build_sentencepiece_luts`, where the leading-space marker check used `?` instead of `▁`.

Corrected rerun score:
- `seed=1341` (bug-fixed rerun): `val_bpb:1.35172182` (`val_loss:2.28232567`) after the int8+zlib roundtrip
- Pre-quant at stop: `val_bpb:1.3495` (`val_loss:2.2785`)

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

Original submission log:
- `train.log` is kept for audit/history, but its BPB values should be treated as invalid because of the marker bug above

Corrected rerun metrics (from `mpk_seed_1341_bpbfix.txt`):
- Timed training stopped at `4101/20000` steps due to the wallclock cap
- Pre-quant eval at stop: `val_loss:2.2785`, `val_bpb:1.3495`
- Post-quant roundtrip eval: `val_loss:2.28232567`, `val_bpb:1.35172182`
- Timed training: `599986ms` (`step_avg:146.30ms`)
- Final eval time after stop: `6173ms`
- Peak memory: `19867 MiB allocated`, `20514 MiB reserved`
- Serialized model int8+zlib: `14529900 bytes`
- Code size: `59500 bytes`
- Total submission size int8+zlib: `14589400 bytes`

Training volume:
- Global batch: `524288` tokens/step
- Total train tokens seen before cap: `2153775104`

Superseded logs:
- `mpk_seed_1338.txt`, `mpk_seed_1339.txt`, `mpk_seed_1340_clean.txt`, and `mpk_seed_1341_clean.txt` were generated before the SentencePiece marker bug was corrected, so their BPB values are also invalid
- They are retained only as audit artifacts showing the pre-correction evaluation history

Included files:
- `train_gpt.py` (corrected trainer snapshot with the `▁` marker fix)
- `train.log` (original timed run log; BPB values invalid due to the bug)
- `mpk_seed_1338.txt` (pre-correction audit log)
- `mpk_seed_1339.txt` (pre-correction audit log)
- `mpk_seed_1340_clean.txt` (pre-correction audit log)
- `mpk_seed_1341_clean.txt` (pre-correction audit log)
- `mpk_seed_1341_bpbfix.txt` (corrected bug-fixed rerun log)
- `submission.json` (corrected leaderboard metadata)
