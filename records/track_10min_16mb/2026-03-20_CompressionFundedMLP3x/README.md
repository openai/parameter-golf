# Compression-Funded MLP3x

**Mean val_bpb: 1.1647** (3 seeds on 8xH100 SXM)

## Idea

Int6 block-weight compression frees up enough artifact budget to widen the MLP from 2x to 3x. The extra feed-forward capacity more than pays for itself. Pair that with the seq2048 long-context recipe and stride-256 sliding-window eval, and you get a clean win over the current SOTA.

The logic is simple: int6 quantization with zlib brings a 9-layer model well under 16MB, so spend those saved bytes on a fatter MLP instead of leaving headroom.

## What changed from baseline

- Int6-style per-row quantization + zlib for all large weight matrices
- `tok_emb.weight` kept in fp16 (quantization errors compound through both input and output paths when embeddings are tied)
- Last two `attn.c_k.weight` tensors kept in fp16
- `MLP_MULT` bumped from 2 to 3
- Seq2048 long-context training: `TRAIN_BATCH_TOKENS=786432`, `TRAIN_SEQ_LEN=2048`
- Lower learning rates: `MATRIX_LR=0.02`, `SCALAR_LR=0.02`, `TIED_EMBED_LR=0.03`
- Muon momentum warmup: `MUON_MOMENTUM=0.99`, warmup from 0.92 over 1500 steps
- `WARMDOWN_ITERS=3000`, `GRAD_CLIP_NORM=0.3`
- Sliding-window eval: `EVAL_SEQ_LEN=2048`, `EVAL_STRIDE=256`
- Full validation split scored (no tail truncation)

## Results

| Seed | val_loss | val_bpb | Steps | ms/step | Artifact bytes |
|------|----------|---------|-------|---------|----------------|
| 1337 | 1.98027 | 1.17283 | 5,358 | 112.0 | 15,933,155 |
| 1338 | 1.96103 | 1.16143 | 6,817 | 88.0 | 15,953,329 |
| 1339 | 1.95830 | 1.15982 | 7,170 | 83.7 | 15,958,024 |
| **Mean** | **1.96653** | **1.16469** | | | |

Seed 1337 ran slower (112ms/step vs ~82-88ms) because it was the first run in the session and paid the CUDA kernel compilation cost. Seeds 1338 and 1339 are more representative of steady-state performance.

All artifacts under 16MB. All training runs within 600s. Eval times ~36.5s each.

## vs. Current SOTA

| Run | val_bpb |
|-----|---------|
| This submission (best, seed 1339) | 1.1598 |
| This submission (mean) | 1.1647 |
| Muon WD + 10L (current SOTA) | 1.1748 |

Mean improvement: 0.0101 nats. Best seed improvement: 0.0150 nats. Both clear the 0.005 threshold.

## Configuration

- Layout: `VOCAB_SIZE=1024 NUM_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=3`
- Tied embeddings: `TIE_EMBEDDINGS=1`
- Learning rates: `MATRIX_LR=0.02 SCALAR_LR=0.02 TIED_EMBED_LR=0.03`
- Quantization: `WEIGHT_QUANT_BITS=6 KEEP_LAST_K_FP16=2`
- Batching: `TRAIN_BATCH_TOKENS=786432 TRAIN_SEQ_LEN=2048`

## Command

```bash
RUN_ID=compression_funded_mlp3x_seed1337 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 \
./records/track_10min_16mb/2026-03-20_CompressionFundedMLP3x/train_gpt.py
```

Seed sweep:

```bash
for SEED in 1337 1338 1339; do
  RUN_ID=compression_funded_mlp3x_seed${SEED} \
  DATA_PATH=./data/datasets/fineweb10B_sp1024 \
  TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
  VOCAB_SIZE=1024 \
  SEED=${SEED} \
  torchrun --standalone --nproc_per_node=8 \
  ./records/track_10min_16mb/2026-03-20_CompressionFundedMLP3x/train_gpt.py
done
```

## Hardware

8x NVIDIA H100 80GB HBM3 (SXM), RunPod community cloud.

## Included files

- `train_gpt.py` — record script
- `train.log` — aggregate summary of all 3 seeds
- `train_seed1337.log`, `train_seed1338.log`, `train_seed1339.log` — full logs
- `submission.json` — leaderboard metadata
