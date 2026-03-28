# Adaptive Precision Embedding Quantization

**val_bpb: 1.1217** (4-seed mean) | **15.8 MB** | 8×H100 SXM

## The Idea

Analysis of the FineWeb training data revealed that token frequency follows a heavy-tailed distribution:

- **Top 100 tokens** cover **53.2%** of all text
- These include: `.` `,` `the` `s` `to` `and` `ing` `of` `a` `in`...

Instead of uniform quantization across all embedding weights, this submission applies **adaptive precision quantization**:

- **Top 100 tokens → int8** (higher precision for 53% of text)
- **Remaining 924 tokens → int6** (standard precision)

The intuition: errors in frequent tokens compound across the entire dataset, so they deserve more precision.

## Results (4 seeds, 8xH100 SXM)

| Seed | val_bpb |
|------|---------|
| 1 | **1.121** |
| 2 | 1.122 |
| 3 | 1.1217 |
| 4 | 1.1222 |

**Mean: 1.1217 | Std: 0.0005**

## Files

- `train_16MBQTo.py` - Training script with adaptive precision quantization
- `top_tokens.py` - Set of top 100 most frequent token IDs
- `submission.json` - Submission metadata
- `train_seed1.log` - Training log seed 1
- `train_seed2.log` - Training log seed 2
- `train_seed3.log` - Training log seed 3
- `train_seed4.log` - Training log seed 4

## Run Command

```bash
SEED=1337 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=8 train_gpt.py

## Credits
∙ Base model: PR #549 stack by @abaybektursun
