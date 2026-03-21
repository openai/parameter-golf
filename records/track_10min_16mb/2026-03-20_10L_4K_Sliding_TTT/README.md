# 10L + 4K Context Negative Result

## Status
This submission is intended as a **non-record / negative result** for the 10-minute, 16MB track.

The run combines several ideas that have worked well elsewhere in the challenge:

- 10 transformer layers
- 4K training context
- overlapping sliding-window evaluation
- rank-8 LoRA test-time training (TTT)
- QAT-style fake quantization during training
- selective FP16 passthrough for a few sensitive tensors

However, this particular combination performed substantially worse than the public baselines and existing 4K records.

## Main Takeaway
The main failure mode appears to be **coverage collapse under the 10-minute budget**.

This run used:

- `TRAIN_SEQ_LEN=4096`
- `TRAIN_BATCH_TOKENS=393216`
- `Training steps = 10200`

So the run processed:

- `10200 * 393216 = 4,010,803,200` training tokens
- approximately **4.01B tokens total**

The public default cached dataset for this tokenizer family is **8B training tokens**, so this run only covered roughly **half of the default training budget** within the 10-minute wallclock limit.

The result suggests that, for this configuration, the extra context length cost outweighed the long-context benefit.

## Configuration

- `VOCAB_SIZE=1024`
- `NUM_LAYERS=10`
- `MODEL_DIM=512`
- `NUM_HEADS=8`
- `NUM_KV_HEADS=4`
- `MLP_MULT=2`
- `TRAIN_SEQ_LEN=4096`
- `TRAIN_BATCH_TOKENS=393216`
- `EVAL_SEQ_LEN=4096`
- `EVAL_STRIDE=64`
- `TTT_EVAL_SEQ_LEN=4096`
- `WARMDOWN_ITERS=20000`
- `MATRIX_LR=0.06`
- `TIED_EMBED_LR=0.07`
- `SCALAR_LR=0.06`
- `QAT=1`

## Command

```bash
RUN_ID=10L_4K_sliding_ttt \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Results
All scores below are from the **post-quantization round-trip model**.

| Metric | Score |
|--------|-------:|
| Post-quant TTT val_loss | 3.1049 |
| Post-quant TTT val_bpb | 1.8389 |
| Post-quant sliding val_loss | 4.23623869 |
| Post-quant sliding val_bpb | 2.50893917 |
| Training steps | 10200 |
| Training tokens seen | 4.01B |
| TTT eval time | 208728 ms |
| Sliding eval time | 322323 ms |
| Artifact size | 13361078 bytes |

## Interpretation
This run should be read as a **negative result** rather than a competitive record.

The experiment tried to combine multiple promising ingredients at once, but in practice the 4K setup reduced total training coverage too much under the fixed 10-minute budget. The end result was worse than both the public baseline and the existing long-context records.

## Included Files
- `train_gpt.py`
- `train.log`
- `submission.json`
