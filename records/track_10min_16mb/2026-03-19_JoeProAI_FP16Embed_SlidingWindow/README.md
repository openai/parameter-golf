This record combines two proven improvements over the naive baseline:

1. **FP16 embedding passthrough**: keep `tok_emb.weight` in fp16 during int8 quantization instead of quantizing it. The tied embedding/output head is the most sensitive tensor; this eliminates nearly all post-quant BPB degradation (~0.007 -> ~0.0005).

2. **Sliding window evaluation** (stride=64): score every validation token with near-maximum context (960+ tokens) instead of the baseline's average ~512 tokens. Each window advances by 64 tokens, scoring only the rightmost 64 positions.

3. **Warmdown/LR tuning**: `WARMDOWN_ITERS=3600` and `MATRIX_LR=0.06` to better match the actual step budget under the 10-minute wallclock cap.

## Configuration

```
VOCAB_SIZE=1024  NUM_LAYERS=9  MODEL_DIM=512  NUM_HEADS=8  NUM_KV_HEADS=4
MLP_HIDDEN=992  TIE_EMBEDDINGS=1  WARMDOWN_ITERS=3600  MATRIX_LR=0.06
EVAL_STRIDE=64  EVAL_BATCH_SEQS=1024
```

## Run Command

```bash
RUN_ID=joeproai_fp16_slide64 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MLP_HIDDEN=992 \
WARMDOWN_ITERS=3600 \
MATRIX_LR=0.06 \
EVAL_STRIDE=64 \
EVAL_BATCH_SEQS=1024 \
MAX_WALLCLOCK_SECONDS=600 \
TRAIN_LOG_EVERY=200 \
VAL_LOSS_EVERY=1000 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Expected Results

Combining FP16 embed (~0.005 BPB improvement from reduced quant gap) with sliding window eval (~0.032 BPB improvement from richer context) should yield approximately **val_bpb ~1.187** on 8xH100, pending verification run.

| Component | Estimated BPB Impact |
|---|---|
| Baseline post-quant | 1.2244 |
| + FP16 embed (quant gap reduction) | -0.005 |
| + Warmdown/LR tuning | -0.005 |
| + Sliding window eval (stride=64) | -0.027 |
| **Expected combined** | **~1.187** |

## Note

This is an initial submission. Training logs and exact metrics will be added after an 8xH100 verification run.

## Files

- `train_gpt.py` (combined code: FP16 embed passthrough + sliding window eval)
- `submission.json` (leaderboard metadata)
