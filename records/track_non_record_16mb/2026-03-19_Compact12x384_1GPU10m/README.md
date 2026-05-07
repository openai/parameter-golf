This non-record submission explores a compact architecture under the same 10-minute wallclock budget on 1xH100.

Compute note:
- This run uses `1x H100` due to limited compute budget while waiting for grant-backed `8xH100` record attempts.

Idea:
- Keep the training/eval pipeline unchanged from `train_gpt.py`.
- Reduce model width while adding depth to target a better compression/quality tradeoff under the 16,000,000-byte artifact cap.

Configuration:
- Hardware: `1x H100 80GB`
- Wallclock cap: `600s`
- Layout: `VOCAB_SIZE=1024 NUM_LAYERS=12 MODEL_DIM=384 NUM_HEADS=6 NUM_KV_HEADS=3 MLP_MULT=2`
- Embeddings: tied (`TIE_EMBEDDINGS=1`)
- Batch/sequence: `TRAIN_BATCH_TOKENS=524288 TRAIN_SEQ_LEN=1024`

Command used:
```bash
RUN_ID=exp1_d12_w384 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
NUM_LAYERS=12 \
MODEL_DIM=384 \
NUM_HEADS=6 \
NUM_KV_HEADS=3 \
MAX_WALLCLOCK_SECONDS=600 \
VAL_LOSS_EVERY=200 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

Results:
- Stop point: `step 1029` at `600.027s`
- Pre-quant stop eval: `val_loss=2.3095`, `val_bpb=1.3678`
- Post-quant roundtrip exact: `val_loss=2.31202878`, `val_bpb=1.36931367`
- Serialized model int8+zlib: `9,620,416 bytes`
- Code size: `47,686 bytes`
- Total submission size int8+zlib: `9,668,102 bytes`

Why this is useful:
- Compared with the same-session 1xH100 baseline run (`val_bpb=1.34993042`, `12,768,125 bytes` total), this model is weaker in score but significantly smaller in artifact size.
- This run is a concrete negative-result datapoint for the width/depth tradeoff in the non-record track.

Included files:
- `train_gpt.py` (exact script snapshot used)
- `train.log` (full training/eval output)
- `submission.json` (metadata)
