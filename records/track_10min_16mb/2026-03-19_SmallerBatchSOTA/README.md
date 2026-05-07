# Smaller Batch SOTA Stack

## Summary

Adopts the current SOTA stack (MLP 3x, int6, fp16 embed, late-K fp16, sliding eval) with one change: smaller batch size (524K vs 786K tokens). This yields ~50% more optimizer steps in the same 10-minute budget, improving final quality.

## Key Insight

The SOTA record uses TRAIN_BATCH_TOKENS=786432. Reducing to 524288 makes each step faster (55ms vs ~83ms) and produces 10,831 steps vs ~7,199. More frequent optimizer updates improve learning in this regime.

## Configuration

- Layout: `VOCAB_SIZE=1024 NUM_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4`
- Tied embeddings: `TIE_EMBEDDINGS=1`
- Training geometry: `TRAIN_SEQ_LEN=2048 TRAIN_BATCH_TOKENS=524288`
- MLP: `MLP_HIDDEN=1536` (3x expansion)
- Learning rates: `TIED_EMBED_LR=0.03 MATRIX_LR=0.02 SCALAR_LR=0.02`
- Muon: `MUON_BACKEND_STEPS=5`
- Export: `INTX_BITS=6 INT8_KEEP_TOK_EMB_FP16=1 INT8_KEEP_LAST_KV_LAYERS_FP16=2`
- Eval: `EVAL_SEQ_LEN=2048 EVAL_STRIDE=256 EVAL_BATCH_SEQS=256`

## Command

```bash
TRAIN_SEQ_LEN=2048 \
TRAIN_BATCH_TOKENS=524288 \
TIED_EMBED_LR=0.03 \
MATRIX_LR=0.02 \
SCALAR_LR=0.02 \
MUON_BACKEND_STEPS=5 \
MLP_HIDDEN=1536 \
INTX_BITS=6 \
INT8_KEEP_TOK_EMB_FP16=1 \
INT8_KEEP_LAST_KV_LAYERS_FP16=2 \
EVAL_SEQ_LEN=2048 \
EVAL_STRIDE=256 \
EVAL_BATCH_SEQS=256 \
MAX_WALLCLOCK_SECONDS=600 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Key Metrics

- Pre-quant eval: `val_bpb:1.1785` (step 10831)
- Post-quant (int6 + zlib, sliding eval stride=256): `val_loss:1.96392463 val_bpb:1.16314679`
- Quantization gap: ~0.015 bpb
- Train time: 599961ms (step_avg: 55.39ms)
- Steps: 10,831/20,000 (wallclock limited)
- Model params: 21,778,504
- Artifact: 15,859,538 bytes (code: 58,672 + model: 15,800,866)

## Hardware

Run on 8xH100 (Hyperbolic).

## Included Files

- `train_gpt.py` (code snapshot)
- `train.log` (training log)
- `submission.json` (metadata)
