This non-record run explores mirrored depth recurrence on Apple Silicon MLX.

Idea:
- Keep the baseline parameter budget almost unchanged by reusing `9` unique transformer blocks across `18` logical layers.
- Run the encoder schedule forward (`0..8`) and the decoder schedule in reverse (`8..0`) so the second half reuses the same weights with mirrored skip structure.
- Serialize only tensor state, excluding the Python schedule lists that are part of the recurrent control flow.

Why this is interesting:
- The challenge explicitly invites parameter tying and recurrent depth.
- This variant adds logical depth and compute without adding a second set of block weights.
- The resulting compressed artifact stays comfortably under the 16 MB cap.

Configuration:
- Hardware: Apple `M5 Max`, MLX `0.31.1`
- Data: published `fineweb10B_sp1024` export, full validation split, `1/195` training shards
- Layout: `VOCAB_SIZE=1024 NUM_LAYERS=18 UNIQUE_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2`
- Tied embeddings: `TIE_EMBEDDINGS=1`
- Batching: `TRAIN_BATCH_TOKENS=8192 TRAIN_SEQ_LEN=1024 VAL_BATCH_SIZE=131072`
- Training length: `ITERATIONS=300`

Command:
```bash
RUN_ID=mirrorrec_18l_9u_300it_fix1 \
ITERATIONS=300 \
MAX_WALLCLOCK_SECONDS=0 \
TRAIN_BATCH_TOKENS=8192 \
VAL_LOSS_EVERY=0 \
VAL_BATCH_SIZE=131072 \
TRAIN_LOG_EVERY=50 \
NUM_LAYERS=18 \
UNIQUE_LAYERS=9 \
python train_gpt.py
```

Key metrics:
- Pre-quant eval: `val_loss:3.7694`, `val_bpb:2.2325`
- Post-quant roundtrip eval: `val_loss:3.77618886`, `val_bpb:2.23647175`
- Train time: `295399ms` (`step_avg:984.66ms`)
- Serialized model int8+zlib: `7990030 bytes`
- Code size: `50818 bytes`
- Total submission size int8+zlib: `8040848 bytes`

Notes:
- This is not a record-track claim. It is a local non-record experiment intended to test whether mirrored block reuse is a productive direction under the parameter cap.
- The final script includes the serialization fix needed for recurrent schedules: only tensor state is exported and quantized.
