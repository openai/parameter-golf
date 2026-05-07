This record captures a cap-valid 10-minute 8xH100 run using a quality-first mixed-quant setup.

Trainer snapshot:
- `train_gpt_frontier.py` copied as `train_gpt.py` in this folder
- decimal cap enforced: `SUBMISSION_SIZE_CAP_BYTES=16000000`
- compression backend selection: `QUANT_COMPRESSOR=auto` (selected `zstd`)
- int6 packing disabled: `INT6_PACK=0`
- mixed quantization: `MIXED_QUANT_INT6_CATS=mlp,attn` (embeddings on int8 path)

Configuration:
- Layout: `VOCAB_SIZE=1024 NUM_LAYERS=11 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4`
- Capacity: `MLP_MULT=3.0 BIGRAM_VOCAB_SIZE=2048`
- Context/batch: `TRAIN_SEQ_LEN=2048 TRAIN_BATCH_TOKENS=786432`
- Eval: `EVAL_STRIDE=64`
- XSA: last 4 layers (`[7, 8, 9, 10]`)
- VE: enabled, layers `9,10`

Command (track-relevant params):
```bash
SUBMISSION_SIZE_CAP_BYTES=16000000 \
EVAL_STRIDE=64 \
QUANT_COMPRESSOR=auto \
INT6_PACK=0 \
MIXED_QUANT_KEEP_FLOAT_MAX_NUMEL=16384 \
MIXED_QUANT_INT6_CATS=mlp,attn \
MLP_MULT=3.0 \
BIGRAM_VOCAB_SIZE=2048 \
RUN_ID=00_quant_quality_tweak_mlp3_bg2048_no_embed_int6 \
torchrun --standalone --nproc_per_node=8 ./train_gpt.py
```

Key metrics (from `train.log`):
- Timed training stopped at `7020/20000` due to wallclock cap (`600071ms`)
- Pre-quant eval at stop: `val_loss:1.9239`, `val_bpb:1.1394`
- Post-EMA diagnostic: `val_loss:1.9221`, `val_bpb:1.1384`
- Final int6 roundtrip exact: `val_loss:1.93586004`, `val_bpb:1.14652536`
- Final sliding-window exact (stride 64): `val_loss:1.89573468`, `val_bpb:1.12276383`
- Code size: `74406 bytes`
- Serialized model int6+zstd: `15487334 bytes`
- Total submission size int6+zstd: `15561740 bytes`
- Cap check: `PASS` (`margin +438260`)

Included files:
- `train_gpt.py` (exact trainer snapshot used for this run)
- `train.log` (full raw run log)
- `submission.json` (leaderboard metadata)
- `run_manifest.tsv` and `results_summary.txt` (batch provenance)
