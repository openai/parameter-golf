# Skinny looped RLM — 512d / 6L / loop 2×3 / seq2048 / Muon LR 0.02

## Architecture

**Looped Transformer (RLM):** prefix and suffix are **6 distinct** transformer blocks; the **middle** uses **2 weight-tied blocks** applied **3 times** (`LOOP_BLOCKS=2`, `LOOP_ITERS=3`). Standard **GQA + RoPE**, **ReLU² MLP** with **3×** expansion.

## Techniques

| Item | Setting |
|------|---------|
| Sequence length | `TRAIN_SEQ_LEN=2048` |
| Optimizer | Muon (matrix) + Adam (scalars), `MATRIX_LR=0.02`, `SCALAR_LR=0.02` |
| Weight decay | `MUON_WD=0.04` (decoupled, compression headroom) |
| Embeddings | `TIE_EMBEDDINGS=1`, `EMBED_BITS=8` (STE fake-quant) |
| Loss | `LOSS_POS_WARMUP=256` (position ramp) |
| Clock | `MAX_WALLCLOCK_SECONDS=590` |

## Reproduction

```bash
MODEL_DIM=512 NUM_LAYERS=6 LOOP_BLOCKS=2 LOOP_ITERS=3 \
MATRIX_LR=0.02 SCALAR_LR=0.02 TIED_EMBED_LR=0.05 \
NUM_HEADS=8 NUM_KV_HEADS=8 TIE_EMBEDDINGS=1 \
MLP_MULT=3 MUON_WD=0.04 EMBED_BITS=8 LOSS_POS_WARMUP=256 \
TRAIN_SEQ_LEN=2048 TRAIN_BATCH_TOKENS=524288 \
ITERATIONS=100000 WARMUP_STEPS=200 WARMDOWN_ITERS=3000 \
MAX_WALLCLOCK_SECONDS=590 VAL_LOSS_EVERY=0 TRAIN_LOG_EVERY=200 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 SEED=1337 RUN_ID=repro_submit_2048_lr02 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Metrics (this run)

| Metric | Value |
|--------|--------|
| Wallclock | `590080 ms` |
| Steps | `6106` |
| `step_avg` | `96.64 ms` (at final val line) |
| Pre-quant `val_bpb` | `1.1683` |
| **`final_int8_zlib_roundtrip_exact val_bpb`** | **`1.17503786`** |
| `Total submission size int8+zlib` | **`14903768`** bytes (≤ 16_000_000) |
| `Code size` | `66048` bytes |
| Peak VRAM | `19506 MiB` allocated |

## Included files

- `train_gpt.py` — snapshot used for training  
- `train.log` — full master log (includes embedded source + `nvidia-smi` + training)  
- `submission.json` — leaderboard metadata  

If the upstream PR template expects the weight blob, add **`final_model.int8.ptz`** from the same run next to these files (same directory as `train_gpt.py`).
