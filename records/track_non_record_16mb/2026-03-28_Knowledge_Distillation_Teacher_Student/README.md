# Non-record: Knowledge distillation (teacher → student)

This folder is a **non-record** submission for Parameter Golf. The idea is simple: train a **bigger teacher** on real next-token targets, then train a **smaller student** using a mix of the usual cross-entropy and a **soft “teacher” signal** (KL-style loss with temperature). Only the **student** counts toward the 16MB submission limit; the teacher is training-time only.

I have not seen another public entry in this repo that uses this exact two-model distillation setup for the challenge, so I am sharing it as an experiment others can build on.

## Plain-English summary

- **Teacher:** more layers and width; learns normally with standard loss.
- **Student:** smaller; its loss is  
  `alpha × (loss vs real next token) + (1 − alpha) × (match teacher’s soft predictions)`.
- **Temperature** softens the teacher’s probabilities so the student gets richer hints than “only the one correct token.”

Hyperparameters `TEMPERATURE` and `ALPHA` come from environment variables (defaults 4.0 and 0.5).

## What I ran (honest results)

| Experiment | Steps | Teacher | Student | val_bpb | Quantized bpb | Size | Notes |
|------------|-------|---------|---------|---------|----------------|------|--------|
| Baseline (no distill) | 200 | none | 9L 512D | 2.3351 | — | 10.3MB | Original `train_gpt.py`, no teacher |
| Distill tiny | 10 | 4L 256D | 3L 128D | 3.7707 | 3.7715 | 0.7MB | Smoke test |
| Distill medium | 200 | 6L 384D | 4L 256D | 2.5427 | 2.5432 | 2.47MB | MLX / Mac |
| Distill medium | 500 | 6L 384D | 4L 256D | 2.2717 | 2.2752 | 2.57MB | MLX / Mac |
| Distill medium | 1000 | 6L 384D | 4L 256D | 2.1348 | 2.1359 | 2.64MB | MLX / Mac |
| H100 partial | ~600 | 12L 768D | 6L 384D | **1.7195** | pending | ~5MB | RunPod; run stopped early (billing) |
| H100 full (goal) | 20000 | 12L 768D | 6L 384D | *not completed* | *not completed* | ~6MB est. | **Rough guess** ~1.20–1.25 BPB — *not* measured end-to-end |

The number in **`submission.json`** is the best **recorded** validation BPB from a real (partial) cloud run: **1.7195**. The “~1.20–1.25” row is only an extrapolation from the trend on smaller runs; I am **not** claiming that as a verified score.

## How to reproduce (8×H100, SP1024)

You need the usual FineWeb SP1024 shards and tokenizer from the main repo README.

Example command (adjust `RUN_ID` and paths to match your machine):

```bash
RUN_ID=distill_12L768_to_6L384_sp1024 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
TRAIN_SEQ_LEN=1024 \
TRAIN_BATCH_TOKENS=524288 \
ITERATIONS=20000 \
MAX_WALLCLOCK_SECONDS=600 \
VAL_LOSS_EVERY=200 \
WARMUP_STEPS=20 \
TIE_EMBEDDINGS=1 \
NUM_LAYERS=6 \
MODEL_DIM=384 \
NUM_HEADS=6 \
NUM_KV_HEADS=2 \
TEACHER_NUM_LAYERS=12 \
TEACHER_MODEL_DIM=768 \
TEACHER_NUM_HEADS=12 \
TEACHER_NUM_KV_HEADS=4 \
TEMPERATURE=4.0 \
ALPHA=0.5 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Single-GPU smoke (smaller batch):

```bash
RUN_ID=distill_smoke \
ITERATIONS=50 \
TRAIN_BATCH_TOKENS=8192 \
VAL_LOSS_EVERY=0 \
VAL_BATCH_SIZE=8192 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

## Files in this folder

| File | Purpose |
|------|---------|
| `train_gpt.py` | Training script (distillation). Named `train_gpt.py` to match submission rules. |
| `train.log` | Combined log notes from my runs (partial cloud log + table). Replace with one full automated log after a complete 8×H100 job if you want a single clean artifact. |
| `submission.json` | Metadata for reviewers. |
| `requirements.txt` | Python deps reference (same spirit as root repo). |

## Limitations

- Distillation needs **two** models in memory → more VRAM than baseline training.
- My best **8×H100** run did **not** finish 20k steps; stronger numbers need a full rerun.
- I did **not** change the challenge tokenizer or dataset; metrics are standard `val_bpb` from the script.

Thank you for reading — hope this helps anyone exploring teacher–student ideas under the 16MB cap.
