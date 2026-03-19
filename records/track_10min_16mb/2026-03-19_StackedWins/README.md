# Stacked Wins: Sliding Window Eval + Seq2048 + FP16 Embed + LR Tuning

This submission combines three proven improvements that were previously submitted independently, leveraging the fact that they target orthogonal aspects of the pipeline (training architecture, quantization, and evaluation strategy).

## Merged Improvements

| Source Submission | Technique | Aspect | Expected Contribution |
|---|---|---|---|
| SlidingWindowEval | Stride-64 sliding window eval | Evaluation | -0.032 BPB |
| LongContextSeq2048 | 2048 sequence length | Training | -0.019 BPB (pre-quant) |
| FP16Embed_WD3600 | FP16 embedding passthrough | Quantization | -0.005 BPB (quant gap) |
| FP16Embed_WD3600 | Warmdown 3600, tuned LRs | Training | ~-0.003 BPB |

## Configuration

Architecture (from seq2048 + fp16embed):
- Layout: `VOCAB_SIZE=1024 NUM_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4`
- MLP hidden: `MLP_HIDDEN=992` (shrunk from 1024 to fit fp16 embedding in 16MB)
- Sequence length: `TRAIN_SEQ_LEN=2048`
- Tied embeddings: `TIE_EMBEDDINGS=1`

LR schedule (from fp16embed + seq2048):
- `TIED_EMBED_LR=0.04` (from seq2048)
- `MATRIX_LR=0.06` (from fp16embed)
- `SCALAR_LR=0.032` (from seq2048)
- `WARMDOWN_ITERS=3600` (from fp16embed)

Evaluation (from sliding window):
- `EVAL_STRIDE=64` (sliding window stride)
- `EVAL_BATCH_SEQS=32` (batched sliding window eval)

Quantization:
- Tied embedding (`tok_emb.weight`) kept in fp16 (from fp16embed)
- All other weights: standard int8 per-row quantization

## Command

```bash
RUN_ID=stacked_wins_v1 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
NUM_LOOPS=1 \
LORA_RANK=0 \
QAT=0 \
EVAL_STRIDE=64 \
EVAL_BATCH_SEQS=1024 \
MAX_WALLCLOCK_SECONDS=600 \
TRAIN_LOG_EVERY=200 \
VAL_LOSS_EVERY=1000 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Expected Results

Based on component-level improvements (gains may not be perfectly additive):
- Estimated post-quant BPB: **~1.16 - 1.17** (vs current SOTA 1.1925)
- Pending validation on 8xH100 SXM

## LR Tuning Note

The matrix_lr and scalar_lr combine values from two different submissions:
- FP16Embed used `MATRIX_LR=0.06` with seq_len=1024
- Seq2048 used `MATRIX_LR=0.032` with seq_len=2048

We start with `MATRIX_LR=0.06` (fp16embed's value) since the longer warmdown (3600) should help stabilize the higher LR. If this proves unstable, fall back to 0.04 or 0.032.

## Files

- `train_gpt.py` - Combined training script
- `README.md` - This file
- `submission.json` - Leaderboard metadata (to be updated with actual results)
