This record implements sliding window evaluation, showing that eval strategies alone can provide significant improvements.

**Note on `train_gpt.py`:** The included script contains some unused experimental code paths (QAT, looped architectures) that are **all disabled by default** and were not active during this run. Only the sliding window evaluation code (`eval_val_sliding`, `forward_logits`, `EVAL_STRIDE`, `EVAL_BATCH_SEQS`) is used. The command below shows the exact invocation.

## Key Idea: Sliding Window Evaluation

The baseline evaluates by chopping the validation set into non-overlapping 1024-token chunks. The problem is that the first token in each chunk has zero context. On average, each token gets ~512 tokens of context.

Sliding window evaluation uses overlapping windows with a configurable stride. With `EVAL_STRIDE=64` and `TRAIN_SEQ_LEN=1024`, each window advances by 64 tokens, but only the rightmost 64 tokens (which have 960+ tokens of context) are scored. Every token in the validation set is scored exactly once, but with near-maximum context.

## Results

| Metric | Naive Baseline | This Submission |
|---|---|---|
| Pre-quant val_bpb | 1.2172 | 1.2196 |
| **Post-quant val_bpb** | **1.2244** | **1.1925** |
| **Improvement** | — | **-0.0319** |
| Training steps | 13,780 | 13,450 |
| Eval time (8xH100) | ~16s | 70s |
| Artifact size | 15,863,489 bytes | 15,874,829 bytes |

The pre-quant BPB is nearly identical (training is the same). The 0.032 improvement comes entirely from scoring tokens with richer context during evaluation.

## Configuration

Architecture and training are identical to the Naive Baseline:
- Layout: `VOCAB_SIZE=1024 NUM_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2`
- Tied output/input embeddings: `TIE_EMBEDDINGS=1`
- Tied embedding LR: `TIED_EMBED_LR=0.05`
- Batching: `TRAIN_BATCH_TOKENS=524288 TRAIN_SEQ_LEN=1024`

Evaluation-specific parameters:
- `EVAL_STRIDE=64` (sliding window stride; baseline uses non-overlapping = stride 1024)
- `EVAL_BATCH_SEQS=1024` (number of windows per forward pass for GPU utilization)

## Command

```bash
RUN_ID=8xh100_slide64_v2 \
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

The `NUM_LOOPS=1 LORA_RANK=0 QAT=0` flags explicitly disable all unused code paths (these are also the defaults).

## Key Metrics (from `train.log`)

- Timed training stopped at `13450/20000` steps due to the wallclock cap.
- Pre-quant eval at stop: `val_loss:2.0592`, `val_bpb:1.2196`
- Post-quant sliding window eval: `val_loss:2.0135`, `val_bpb:1.1925`
- Exact printed metric: `final_int8_zlib_roundtrip_exact val_bpb:1.19250007`
- Train time: `600028ms` (`step_avg:44.61ms`)
- Peak memory: `10119 MiB allocated`, `10294 MiB reserved`
- Eval time: `69881ms` (sliding window, stride=64, batch_seqs=1024)
- Serialized model int8+zlib: `15816489 bytes`
- Code size: `58340 bytes`
- Total submission size int8+zlib: `15874829 bytes`

## Training Volume

- Global batch: `524288` tokens/step
- Total train tokens seen: `7,055,769,600`

## Included Files

- `train_gpt.py` (code snapshot used for the run, includes `eval_val_sliding` function)
- `train.log` (exact remote training log)
- `submission.json` (leaderboard metadata)
