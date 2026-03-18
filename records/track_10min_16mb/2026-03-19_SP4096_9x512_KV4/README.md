# SP4096: Larger Vocabulary for Better Bits-Per-Byte

## Summary

Switching from the baseline's 1024-token SentencePiece BPE vocabulary to a 4096-token vocabulary yields a significant BPB improvement. The larger vocabulary compresses text 26% more efficiently (0.306 vs 0.414 tokens/byte), and the additional embedding parameters (18.6M vs 17.1M) fit comfortably within the 16MB artifact budget.

No architectural or hyperparameter changes were made -- the improvement comes entirely from the tokenizer.

## Approach

The key insight is that BPB = (val_loss / ln2) * tokens_per_byte. A better tokenizer directly reduces the tokens_per_byte multiplier. While a larger vocabulary increases per-token cross-entropy (harder prediction task), the compression gain more than compensates.

**Tokenizer training:** SentencePiece BPE trained on 500K FineWeb documents with vocab_size=4096, byte_fallback=True, split_digits=True, nmt_nfkc normalization.

**Compression ratio:** 0.306 tokens/byte vs 0.414 for the baseline sp1024 (26% fewer tokens for the same text).

## Configuration

```
VOCAB_SIZE=4096
NUM_LAYERS=9
MODEL_DIM=512
NUM_HEADS=8
NUM_KV_HEADS=4
MLP_MULT=2
TIE_EMBEDDINGS=1
TRAIN_SEQ_LEN=1024
TRAIN_BATCH_TOKENS=524288
```

All other hyperparameters are default (same as baseline).

## Results

**Status: Work in progress. Awaiting 8xH100 validation run.**

### RTX 5090 (1xGPU, 10 min) -- Preliminary

| Config | Steps | val_bpb (post-quant) | Compressed |
|--------|-------|---------------------|------------|
| sp1024 baseline (control) | 457 | 1.5086 | 9.8MB |
| **sp4096 (this submission)** | **938** | **1.3217** | **13.6MB** |

Improvement: -0.187 BPB (12.4% better) on single GPU.

Note: Single GPU results are not directly comparable to the 8xH100 leaderboard due to fewer training steps. The 8xH100 run will complete ~13K+ steps vs 938 on 1xGPU.

### Tokenizer Compression Analysis

| Tokenizer | Vocab | tokens_per_byte |
|-----------|-------|----------------|
| sp512 BPE | 512 | 0.544 |
| sp1024 BPE (baseline) | 1024 | 0.414 |
| sp2048 BPE | 2048 | 0.351 |
| **sp4096 BPE (ours)** | **4096** | **0.306** |
| sp8192 BPE | 8192 | 0.272 |

## Command

```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

With environment variables:
```bash
DATA_PATH=./data/datasets/fineweb10B_sp4096
TOKENIZER_PATH=./data/tokenizers/fineweb_4096_bpe.model
VOCAB_SIZE=4096
MAX_WALLCLOCK_SECONDS=600
```

## Included Files

- `train_gpt.py` -- training script (unmodified from baseline)
- `submission.json` -- leaderboard metadata
- `README.md` -- this file
- `train.log` -- (pending 8xH100 run)
