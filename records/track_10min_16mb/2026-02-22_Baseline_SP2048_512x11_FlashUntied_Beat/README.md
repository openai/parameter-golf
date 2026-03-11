This records the `11x512` SP-2048 baseline shape rerun that beat the user-provided baseline on `final_int8_zlib_roundtrip val_bpb` after forcing Flash SDPA and using untied embeddings.

Command (with explicit track-relevant params):
```bash
RUN_ID=baseline_flash_u0_11x512x8_20260222_011400 \
TOKENIZER_KIND=sp \
TOKENIZER_PATH=data/matched_10B/tokenizers/fineweb_2048_bpe.model \
ENABLE_VAL_BPB=1 \
DATA_PATH=data/matched_10B/datasets/fineweb10B_sp2048 \
VOCAB_SIZE=2048 \
TIE_EMBEDDINGS=0 \
NUM_LAYERS=11 \
MODEL_DIM=512 \
NUM_HEADS=8 \
ITERATIONS=10000 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Flash-attention note:
- The saved `train_gpt.py` snapshot in this folder hard-forces Flash SDPA (`flash=True`, other SDPA backends disabled) to avoid slow fallback paths.

Key metrics (from `train.log`):
- End-of-training fp eval: `val_loss:2.2956`, `val_bpb:1.1442`
- Post-quant roundtrip eval: `val_loss:2.3150`, `val_bpb:1.1539`
- Submission size int8+zlib total: `28933077 bytes`
- Peak memory: `15595 MiB allocated`, `15984 MiB reserved`

Comparison to user baseline (same `11x512` shape):
- User baseline post-quant roundtrip `val_bpb`: `1.1556`
- This run post-quant roundtrip `val_bpb`: `1.1539` (better by `0.0017`)

Timing caveat (important):
- `train_time` in this saved run is `824415ms`, and `step_avg` is inflated.
- The pod was intermittently contaminated by unrelated background `train_gpt_simple_no_tied_embeddings.py` sweeps running concurrently on the same `speedruna1` box.
- Treat this record as a valid score/artifact capture, but **not** as a clean `<10 minute train_time` track-eligibility run.

Included files:
- `train_gpt.py` (training code snapshot used for the run)
- `train.log` (full training log copied from `speedruna1-0`)
- `params.env` (explicit env/command params)

