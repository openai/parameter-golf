This record captures `DenseContextQuantTrim sp1024`, a dense long-context
`8xH100` run that fits the official `10 minute / 16,000,000 byte` track caps.

Trainer changes in this snapshot:
- dense baseline-family training path
- `TRAIN_SEQ_LEN=2048`
- sliding-window validation with `VAL_CONTEXT_LEN=2048` and `VAL_SLIDE_TOKENS=512`
- clip-search PTQ with `INT8_CLIP_CANDIDATES=1.0,0.95,0.9,0.85`
- hybrid export for `tok_emb.weight`
  - top-scoring rows kept in `fp16`
  - remaining rows quantized in per-row `int8`
  - exact embedding matrix reconstructed before final roundtrip evaluation

Configuration:
- Layout: `VOCAB_SIZE=1024 NUM_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2`
- Tied output/input embeddings: `TIE_EMBEDDINGS=1`
- Sequence length: `TRAIN_SEQ_LEN=2048`
- Validation window: `VAL_CONTEXT_LEN=2048 VAL_SLIDE_TOKENS=512`
- Batching: `TRAIN_BATCH_TOKENS=524288 VAL_BATCH_SIZE=524288`
- Optimization: `MATRIX_LR=0.020 SCALAR_LR=0.020 TIED_EMBED_LR=0.030 MUON_MOMENTUM=0.99`
- Schedule: `WARMUP_STEPS=20 WARMDOWN_ITERS=3000 GRAD_CLIP_NORM=0.3`
- RoPE: `ROPE_BASE=200000`
- Export trim: `INT8_EMBED_FP16_TOP_ROWS=304 INT8_EMBED_FP16_SCORE=row_rms`

Command (equivalent invocation using this record snapshot):
```bash
RUN_ID=dcqt1_8xh100_trim_runpod \
DATA_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
TRAIN_SEQ_LEN=2048 \
VAL_CONTEXT_LEN=2048 \
VAL_SLIDE_TOKENS=512 \
TRAIN_BATCH_TOKENS=524288 \
VAL_BATCH_SIZE=524288 \
ITERATIONS=20000 \
MAX_WALLCLOCK_SECONDS=594 \
VAL_LOSS_EVERY=20000 \
TRAIN_LOG_EVERY=100 \
MATRIX_LR=0.020 \
SCALAR_LR=0.020 \
TIED_EMBED_LR=0.030 \
MUON_MOMENTUM=0.99 \
WARMUP_STEPS=20 \
WARMDOWN_ITERS=3000 \
GRAD_CLIP_NORM=0.3 \
ROPE_BASE=200000 \
INT8_KEEP_FLOAT_FP16_NAME_PATTERNS= \
INT8_EMBED_FP16_TOP_ROWS=304 \
INT8_EMBED_FP16_SCORE=row_rms \
INT8_CLIP_CANDIDATES=1.0,0.95,0.9,0.85 \
python -m torch.distributed.run --standalone --nproc_per_node=8 \
  /workspace/parameter-golf/records/track_10min_16mb/2026-03-20_DenseContextQuantTrim_sp1024_8xH100/train_gpt.py
```

Key metrics (from `train.log`):
- Timed training stopped at `11424/20000` steps due to the wallclock cap.
- Pre-quant eval at stop: `val_loss:1.9834`, `val_bpb:1.1747`
- Post-quant roundtrip eval: `val_loss:1.9888`, `val_bpb:1.1779`
- Exact printed metric: `final_int8_zlib_roundtrip_exact val_bpb:1.17787170`
- Train time: `594025ms` (`step_avg:52.00ms`)
- Peak memory: `10247 MiB allocated`, `10744 MiB reserved`
- Serialized model int8+zlib: `15919719 bytes`
- Code size: `61389 bytes`
- Total submission size int8+zlib: `15981108 bytes`

Training volume:
- Global batch: `524288` tokens/step
- Total train tokens seen: `5999953920`

Included files:
- `train_gpt.py` (code snapshot used for the run)
- `train.log` (exact remote training log)
- `submission.json` (submission metadata)
