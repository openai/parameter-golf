# Non-record submission: NativeBonsaiBinary

**Final candidate:** `4xh100_5x1024_rankoffset_time590_20260430_225727`

This is a **non-record submission**. Only 4xH100s were available for this run, so it is not a valid 8xH100 leaderboard record attempt, and the result is not competitive with current SOTA record-track submissions. It is included to document a native 1-bit grouped-binary direction inside Parameter Golf. I have spent a lot of time analyzing the PrismML Bonsai models (both Binary and Ternary), including some NanoGPT tests, so I adapted the techniques for this competition.

## Summary

NativeBonsaiBinary is a 5-layer, 1024-wide SP8192 transformer trained with PyTorch CUDA DDP on 4x NVIDIA H100 80GB. It uses grouped binary STE linears with learned scales and native packet accounting, then exports an LZMA-compressed packet plus counted code size.

The run is called **Bonsai** because it adopts the 1-bit structural idea from Bonsai's Qwen3 1-bit models: compact grouped binary weights with learned scales around the binary structure. This submission documents that direction; it does not claim a leaderboard win.

Development notes from earlier 1xH100 and DDP tests are in `EXPERIMENT_NOTES.md`. The main lesson was that grouped binary STE training is update-starved at very large batches: `TRAIN_BATCH_TOKENS=524288` only gave 373 updates in 10 minutes and scored `val_bpb=1.8968`, while the best confirmed 1xH100 32k run reached `val_bpb=1.4302`. The final 4xH100 run keeps the global batch at 32k to preserve update count.

## Final Result

| Metric | Value |
|--------|-------|
| Run ID | `4xh100_5x1024_rankoffset_time590_20260430_225727` |
| Hardware | 4x NVIDIA H100 80GB |
| Train mode | PyTorch CUDA DDP |
| DDP world size | 4 |
| Global train batch tokens | 32,768 |
| Local train batch tokens | 8,192 |
| Time cap | 590.0s |
| Actual stop | step 9,824 |
| Actual train time | 587.561s |
| val_loss | 3.5004 |
| val_bpb | 1.3551 |
| LZMA packet plus code | 15,087,895 bytes |
| Size limit | 16,000,000 bytes |
| TTT | disabled |

Status: under the 16MB size limit and under 600s train time on the available 4xH100 run.

## Model Config

| Component | Setting |
|-----------|---------|
| Tokenizer/data | SP8192 from `kevclark/parameter-golf` |
| Layers | 5 |
| Model dim | 1024 |
| Heads | 16 |
| KV heads | 4 |
| Embed dim | 254 |
| Vocab size | 8192 |
| MLP | 4x SwiGLU |
| Quantization | grouped binary STE linears |
| Quant group size | 128 |
| Embeddings/control tensors | high precision |
| Optimizer | split Muon |
| MATRIX_LR / SCALAR_LR / TIED_EMBED_LR | 0.006 / 0.006 / 0.009 |
| Sequence length | 1024 |
| TTT during final score | disabled |

## Important Fixes

Two fixes were applied in `train_gpt.py` before the final candidate run:

1. **DDP rank data offset.** Before this, each DDP rank started from the same train shard/position, so gradient averaging saw duplicated minibatches. The final candidate offsets `TokenStream.file_idx` by `RANK % len(files)` so ranks train on distinct shards.
2. **Synchronized wallclock stop.** Before this, rank 0 could decide to stop while other ranks kept training, causing a DDP hang before save. The final candidate all-reduces a `stop_now` flag across ranks so every rank exits the training loop on the same step.

## Reproduction

The submitted package uses `train_gpt.py` as the canonical entry point.

```bash
cd records/track_non_record_16mb/2026-04-30_NativeBonsaiBinary
bash run_4xh100_5x1024.sh
```

The final run command shape was:

```bash
RUN_ID=4xh100_5x1024_rankoffset_time590_rerun \
PYTHONUNBUFFERED=1 \
DATA_PATH="$PWD/data/datasets/fineweb10B_sp8192" \
TOKENIZER_PATH="$PWD/data/tokenizers/fineweb_8192_bpe.model" \
OUT_DIR="$PWD/logs_4xh100_5x1024" \
DDP=1 \
VOCAB_SIZE=8192 \
EMBED_DIM=254 \
NUM_LAYERS=5 \
MODEL_DIM=1024 \
NUM_HEADS=16 \
NUM_KV_HEADS=4 \
MLP_MULT=4.0 \
ITERATIONS=20000 \
WARMUP_STEPS=5 \
WARMDOWN_ITERS=8000 \
TRAIN_BATCH_TOKENS=32768 \
GRAD_ACCUM_STEPS=1 \
TRAIN_SEQ_LEN=1024 \
VAL_BATCH_SIZE=524288 \
TRAIN_LOG_EVERY=200 \
VAL_LOSS_EVERY=0 \
MAX_WALLCLOCK_SECONDS=590.0 \
TIE_EMBEDDINGS=1 \
QUANT_MODE=binary \
QUANT_GROUP_SIZE=128 \
QUANTIZE_EMBEDDINGS=0 \
BINARY_CENTER_MODE=none \
SKIP_ROUNDTRIP_EVAL=1 \
SKIP_FINAL_VAL=1 \
SAVE_DEBUG_ZLIB=0 \
TTT_ENABLED=0 \
OPTIMIZER_NAME=split_muon \
MATRIX_LR=0.006 \
SCALAR_LR=0.006 \
TIED_EMBED_LR=0.009 \
MUON_MOMENTUM=0.99 \
MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=500 \
MUON_BACKEND_STEPS=5 \
ROPE_DIM=16 \
LOGIT_SOFTCAP=30 \
SOFTCAP_MODE=tanh \
ROPE_BASE=1000000 \
PARALLEL_RESIDUAL_START_LAYER=10000 \
QK_GAIN=1.0 \
MLP_ACT=swiglu \
torchrun --standalone --nproc_per_node=4 train_gpt.py
```

After training, packet size is computed with:

```bash
python export_native_packet_size.py \
  "$OUT_DIR/${RUN_ID}_mlx_model.npz" \
  --mode binary \
  --group-size 128 \
  --code train_gpt.py
```

The checkpoint filename contains `mlx_model` only because the trainer uses a legacy filename suffix from Mac development. This run used PyTorch CUDA.


## Bad / Discarded Runs

- `4xh100_5x1024_32k_9800it_20260430_223735`: `val_bpb=1.4736`; legal size and time, but trained before the DDP rank data offset fix.
- 9L x 768 attempts: default 1xH100 learning rates were unstable on this 4x DDP path; loss spiked into double digits early and these runs were stopped.

Additional earlier diagnostics, size probes, and future 8xH100 recommendations are recorded in `EXPERIMENT_NOTES.md`.

## Compliance Notes

- This lives in `records/track_non_record_16mb/` because it is not a valid 8xH100 leaderboard attempt.
- The package includes the expected `README.md`, `submission.json`, `train_gpt.py`, launch script, dependency file, and helper dependencies.
- The package includes the final training log, score log, and packet-size log.
- The packet exporter reports code bytes using `train_gpt.py`, matching the challenge rule that counted code should live in the training script.
- The script downloads training data before the run; downloaded dataset shards and checkpoints should not be committed.
- No TTT was used for the final score.

## Files

- `train_gpt.py`: canonical submission entry point.
- `run_4xh100_5x1024.sh`: launch script for the 4xH100 handoff run.
- `export_native_packet_size.py`: native binary packet-size accounting script.
- `data/cached_challenge_fineweb.py`: SP8192 data/tokenizer downloader used by the run script.
- `requirements.txt`: Python dependencies.
- `EXPERIMENT_NOTES.md`: earlier batch-size, size-probe, and DDP handoff notes.
- `4xh100_5x1024_rankoffset_time590_20260430_225727.txt`: final training log.
- `4xh100_5x1024_rankoffset_time590_20260430_225727_packet_size.txt`: final packet-size log.
- `4xh100_5x1024_rankoffset_time590_20260430_225727_score.txt`: final score log.

## Credits

- Bonsai / Qwen3 1-bit model family: source inspiration for the grouped 1-bit structural direction.
- Parameter Golf community binary and ternary submissions: reference point for documenting 1-bit compression tradeoffs in the non-record track.
