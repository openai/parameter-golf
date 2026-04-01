# Depth-Recurrent Test-Time Training

This record captures a non-record submission having the following particularities: depth recurrence, and chunk-causal test-time LoRA
adaptation at evaluation.

## Motivation

Under the 16MB artifact cap, the model size is tightly constrained. There is little room to add
layers or width. This submission explores a complementary axis: instead of making the model
bigger through depth, make it bigger through depth recurrence. Depth recurrence runs each encoder and decoder block multiple
times before propagating activations forward, effectively increasing representational depth without
adding parameters. At evaluation time, chunk-causal TTT LoRA adapters let the model specialize to
each document's distribution before predicting it, refining one chunk at a time.

## Configuration

- **Track:** `non-record-16mb`, 600s wallclock cap, under the `16,000,000` byte artifact cap
- **Layout:** `VOCAB_SIZE=1024 NUM_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2`
- **Tied embeddings:** `TIE_EMBEDDINGS=1`
- **Depth recurrence:** `ENCODER_LOOPS=2 DECODER_LOOPS=2`
- **Batching:** `TRAIN_BATCH_TOKENS=1572864 TRAIN_SEQ_LEN=1024`
- **TTT LoRA:** `TTT_LORA_RANK=8 TTT_CHUNK_SIZE=256`

## Running

```bash
MAX_WALLCLOCK_SECONDS=600 \
VAL_LOSS_EVERY=0 \
torchrun --standalone --nproc_per_node=8 records/track_non_record_16mb/2026-03-23_DepthRecurrent_TTT/train_gpt.py
```

## Key Metrics

Reported metrics are **averages across three seeds** (1337, 42, 404).

- Post-quant TTT LoRA eval: `val_loss: 2.0416`, `val_bpb: 1.2092`
- Pre-quant eval: `val_loss: 2.0950`, `val_bpb: 1.2408`
- Total submission size: `15,544,590 bytes`
- Training stopped at `2570` steps due to the 600s wallclock cap

## Included Files

- `train_gpt.py` (code snapshot used for the run)
- `submission.json` (leaderboard metadata)
- 3 run log files for each seed
