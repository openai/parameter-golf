This folder captures a non-record local MLX submission for adaptive eval-time context.

The idea in this snapshot is simple: do one coarse pass over the validation stream, mark the harder windows from that pass, then rescore only those windows with a finer stride. The training setup stays close to the baseline MLX path; the change is in how the final roundtrip evaluation spends extra context.

This is not a leaderboard claim. It is a local Apple Silicon result meant to document the idea, the code snapshot, and a same-setup comparison against standard final evaluation.

Configuration:
- Hardware: Apple M4 Pro, 48 GB unified memory
- Track: non-record, local Apple Silicon MLX
- Tokenizer/data: `fineweb10B_sp1024`, first train shard, first `32768` validation tokens
- Model: SP-1024, `9x512`, `KV4`, tied embeddings
- Training length: `200` iterations, `8192` train tokens/step
- Final eval mode: adaptive
- Adaptive eval settings: `coarse_stride=256`, `fine_stride=64`, `hard_fraction=0.25`

Command used for the included adaptive run:
```bash
cd records/track_non_record_16mb/2026-03-19_AdaptiveEvalContext_MLX_M4Pro_sp1024_200it
RUN_ID=cmp200_adapt_c256_f64_h025 \
SEED=1337 \
ITERATIONS=200 \
TRAIN_BATCH_TOKENS=8192 \
GRAD_ACCUM_STEPS=8 \
VAL_LOSS_EVERY=0 \
VAL_BATCH_SIZE=32768 \
VAL_MAX_TOKENS=32768 \
FINAL_ROUNDTRIP_EVAL=1 \
FINAL_EVAL_MODE=adaptive \
FINAL_EVAL_COARSE_STRIDE=256 \
FINAL_EVAL_FINE_STRIDE=64 \
FINAL_EVAL_HARD_FRACTION=0.25 \
FINAL_EVAL_BATCH_SEQS=16 \
DATA_PATH=../../../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../../../data/tokenizers/fineweb_1024_bpe.model \
../../../.venv/bin/python train_gpt.py > train.log 2>&1
```

Included result (`train.log`):
- Pre-quant eval at stop: `val_loss:4.1575`, `val_bpb:2.4070`
- Post-roundtrip eval: `val_loss:4.15029331`, `val_bpb:2.40284524`
- Eval time for final adaptive roundtrip pass: `2386ms`
- Selected windows: `hard_windows:31/124`, `fine_windows:124`
- Serialized model int8+zlib: `11239210 bytes`
- Code size: `58701 bytes`
- Total submission size int8+zlib: `11297911 bytes`

Same-setup reference (`compare_standard.log`):
- Standard final eval: `val_loss:4.16789573`, `val_bpb:2.41303630`
- Eval time: `321ms`

So in this local fixed-step proxy, the adaptive pass improves the final roundtrip score by about `0.01019 bpb` over the same setup with standard final evaluation, but it also increases final eval time. That tradeoff is the main reason this is being submitted as a non-record WIP rather than as a score claim.

Included files:
- `train_gpt.py` - exact MLX code snapshot used for the run
- `train.log` - adaptive local run log
- `compare_standard.log` - same-setup standard-eval comparison log
- `submission.json` - metadata for the run
