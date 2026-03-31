This record captures the `CorePromotion Experiment 13` run on Modal (8xH100 SXM).

Architecture discovered via autoresearch loop on 1xH100, then validated at full competition scale.

Configuration:
- Layout: `VOCAB_SIZE=1024 NUM_LAYERS=12 MODEL_DIM=448 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2`
- Tied output/input embeddings: `TIE_EMBEDDINGS=1`
- Optimizer: `matrix_lr=0.08 scalar_lr=0.04 embed_lr=0.05`
- Schedule: `warmdown_iters=500 warmup_steps=20`
- Batching: `TRAIN_BATCH_TOKENS=524288 TRAIN_SEQ_LEN=1024`

Command:
```bash
NCCL_IB_DISABLE=1 \
RUN_ID=competition_8xH100_run1 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=600 \
TRAIN_LOG_EVERY=50 \
VAL_LOSS_EVERY=200 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Platform: Modal (8xH100 SXM, basilias workspace)

Key metrics (from `train.log`):
- Training stopped at `7024/20000` steps due to wallclock cap.
- Pre-quant eval at stop: `val_loss:2.0879`, `val_bpb:1.2366`
- Post-quant roundtrip eval: `val_loss:2.0975`, `val_bpb:1.2422`
- Exact printed metric: `final_int8_zlib_roundtrip_exact val_bpb:1.24223589`
- Train time: `599784ms` (`step_avg:85.39ms`)
- Peak memory: `11921 MiB allocated`, `12300 MiB reserved`
- Serialized model int8+zlib: `16111650 bytes`
- Code size: `47686 bytes`
- Total submission size int8+zlib: `16159336 bytes`

NOTE: Artifact is 159KB OVER the 16MB cap (16,159,336 > 16,000,000). Not a valid submission.
NOTE: val_bpb 1.2422 is worse than the baseline (1.2244). Further optimization needed.

Model params: 17,342,176
Quantization gap: 0.0056 (1.2422 - 1.2366)

Included files:
- `train_gpt.py` (code snapshot used for the run)
- `train.log` (exact remote training log)
- `submission.json` (leaderboard metadata)
