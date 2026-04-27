# Hierarchical Quantized Embedding (HQE) — Non-record Submission

**val_bpb: 1.3592** | 1×H100 SXM | 595s | 14.78MB

## Motivation

Token frequency in natural language follows Zipf's law. In the SP1024 vocabulary,
the top 32 tokens (3.1%) account for ~29% of all occurrences. Standard uniform
quantization (e.g. int6) treats all tokens equally, wasting precision on rare tokens
and under-serving frequent ones.

## Method: Hierarchical Quantized Embedding (HQE)

Assign different bit-widths to embedding rows based on token frequency rank:

| Tier | Token Range | Bits | Tokens | Freq Coverage |
|------|-------------|------|--------|---------------|
| 1    | 0–31        | fp16 | 32     | ~29%          |
| 2    | 32–127      | int8 | 96     | ~40%          |
| 3    | 128–511     | int6 | 384    | ~25%          |
| 4    | 512–1023    | int4 | 512    | ~6%           |

Gradients flow through all tiers via Straight-Through Estimator (STE).
Scale parameters per tier are learned jointly with weights.

## Memory Analysis

- Uniform int6 baseline: 394,240 bytes
- HQE proposed: 364,544 bytes
- **Savings: 29,696 bytes (7.5%)**

## Results

| Metric | Value |
|--------|-------|
| val_bpb | 1.3592 |
| val_loss | 2.2949 |
| Model size (int8+zlib) | 14,784,934 bytes (14.78MB) |
| Training time | 595s (1×H100 SXM) |
| Steps completed | 1774/20000 |

Baseline val_bpb: 1.2244

## Discussion

This submission demonstrates that frequency-aware mixed precision embedding is
trainable via STE and fits within the 16MB artifact limit. The performance gap
vs baseline is partly due to running on 1×H100 (vs 8×H100) and hitting the
wallclock cap at step 1774. With 8×H100 and full training, further improvement
is expected.

The approach is complementary to existing SOTA techniques (depth recurrence,
SP8192 tokenizer, MuonEq-R) and could be stacked on top of them.

## Reproduction

```bash
git clone https://github.com/anjing00monyet-arch/parameter-golf.git
cd parameter-golf
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1
RUN_ID=hqe_sp1024 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

## Files

- `train_gpt.py` — training script with HQE integrated
- `hierarchical_embed.py` — HQE module
- `run_8xh100_20260425_201122.log` — training log
- `submission.json` — metadata
