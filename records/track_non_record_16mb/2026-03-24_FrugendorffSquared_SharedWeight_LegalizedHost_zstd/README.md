# Frugendorff Squared Shared-Weight Legalized Host

This record captures Siddhant Gupta's unlimited-compute non-record submission built from the shared-weight Frugendorff host family introduced in PR `#579`.

This run is not intended to satisfy the 10-minute cutoff for the main leaderboard. It keeps the byte-first shared-weight layout, disables the imported replay/distillation extras, and hard-requires `zstandard` so export accounting cannot silently fall back to `zlib`.

Configuration:
- Track: `non-record`, unlimited compute, still under the `16,000,000` byte artifact cap
- Hardware used for this run: `1x H100 SXM 80GB`
- Layout: `NUM_LAYERS=6 NUM_LOOPS=2 MODEL_DIM=640 NUM_HEADS=10 NUM_KV_HEADS=5 MLP_MULT=4`
- XSA / RoPE / VE: `XSA_LAST_N=2 ROPE_DIMS=16 VE_ENABLED=1 VE_DIM=128 VE_LAYERS=2,3`
- Tied embeddings and LR split: `TIE_EMBEDDINGS=1 EMBED_LR=0.035 HEAD_LR=0.0 MATRIX_LR=0.025 SCALAR_LR=0.025`
- Batching: `TRAIN_BATCH_TOKENS=786432 TRAIN_SEQ_LEN=2048`
- Wallclock cap: `MAX_WALLCLOCK_SECONDS=4800`
- Audit-safety switches: `TTT_BURST_ENABLED=0 DISTILL_ENABLED=0`

Command (track-relevant params):
```bash
OMP_NUM_THREADS=1 \
TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
PYTHONUNBUFFERED=1 \
RUN_ID=approach5_nonrecord_1x_20260325_1532 \
DATA_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
NUM_LAYERS=6 \
NUM_LOOPS=2 \
MODEL_DIM=640 \
NUM_HEADS=10 \
NUM_KV_HEADS=5 \
MLP_MULT=4 \
BIGRAM_VOCAB_SIZE=2048 \
BIGRAM_DIM=128 \
XSA_LAST_N=2 \
ROPE_DIMS=16 \
VE_ENABLED=1 \
VE_DIM=128 \
VE_LAYERS=2,3 \
DTG_ENABLED=0 \
LOGIT_SOFTCAP=30.0 \
TTT_BURST_ENABLED=0 \
DISTILL_ENABLED=0 \
MAX_WALLCLOCK_SECONDS=4800 \
TRAIN_BATCH_TOKENS=786432 \
TRAIN_SEQ_LEN=2048 \
python -m torch.distributed.run --standalone --nproc_per_node=1 train_gpt.py
```

Key metrics (from `train.log`):
- Timed training stopped at `4121/20000` steps due to the `4800s` wallclock cap.
- Stop-checkpoint eval at cap: `val_loss:1.9639`, `val_bpb:1.1631`
- Post-EMA diagnostic: `val_loss:1.9635`, `val_bpb:1.1629`
- Final int6 roundtrip exact: `val_loss:1.98727403`, `val_bpb:1.17697562`
- Final sliding-window exact: `val_loss:1.94705614`, `val_bpb:1.15315937`
- Sliding eval time: `951907ms`
- Train time at stop: `4800619ms` (`step_avg:1164.92ms`)
- Peak memory: `30482 MiB allocated`, `30618 MiB reserved`

Artifact accounting:
- Serialized model: `110820611 bytes`
- Serialized model int6+zstd: `15848577 bytes`
- Code size: `75257 bytes`
- Total submission size int6+zstd: `15923834 bytes`
- Margin to the `16,000,000` byte cap: `76166 bytes`

Included files:
- `train_gpt.py` (exact code snapshot used for the run)
- `train.log` (exact training log for the submitted run)
- `launcher.log` (launcher-side stdout capture)
- `submission.json` (submission metadata)
- `requirements.txt` (Python package dependencies)
- `run_official_8x_h100.sh` (self-contained launcher; defaults to `NPROC_PER_NODE=1` to reproduce `train.log`)
- `setup_and_run_official_8x_h100.sh` (dependency/setup wrapper)
- `run_summary.md` (concise metric summary)
