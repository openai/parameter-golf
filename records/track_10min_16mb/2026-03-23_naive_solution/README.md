Test submission to verify the format and submission process. This is **not** a competitive entry — just the unmodified `train_gpt.py` baseline run for 100 steps on a single GPU.

Configuration (unchanged from baseline):
- Layout: `VOCAB_SIZE=1024 NUM_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2`
- Tied embeddings
- 100 training steps only

Key metrics (from `train.log`):
- Final post-quant eval: `val_loss:3.3216`, `val_bpb:1.9672`
- Exact: `final_int8_zlib_roundtrip_exact val_bpb:1.96723459`
- Train time: `33316ms` (`step_avg:333.16ms`)
- Peak memory: `10303 MiB allocated`, `10554 MiB reserved`
- Serialized model int8+zlib: `10,433,965 bytes`
- Code size: `47,686 bytes`
- Total submission size int8+zlib: `10,481,651 bytes`

Included files:
- `train_gpt.py` — unmodified baseline snapshot
- `train.log` — full training log
- `submission.json` — leaderboard metadata
