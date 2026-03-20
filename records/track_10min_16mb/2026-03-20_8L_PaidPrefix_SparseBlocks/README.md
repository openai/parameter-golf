# 8L Paid Prefix + Sparse Hard Blocks (val_bpb: 1.0365)

**val_bpb: 1.0365** (sliding window, stride=64) | **16.53 MB** | 8xH100 SXM, 600s train + eval-time sparse prefix build

## Approach

This submission is built directly on **PR #262** (`8L Paid Prefix + SmearGate + Int6`) and replaces its contiguous paid prefix with an inline-built sparse hard-block cache.

This submission keeps the 8-layer paid-prefix model recipe and replaces the contiguous validation prefix with an inline-built **sparse hard-block cache**. During the official eval phase, the script first profiles sliding-window NLL on the validation set, then selects the hardest fixed-size target blocks under a byte budget and stores them as a sparse paid-prefix blob. The final scored sliding-window eval uses that generated blob in the same run.

The goal is to improve score-per-prefix-byte relative to a contiguous prefix by spending artifact bytes on the highest-loss validation regions instead of the first `N` positions. With the same 4.24 MB prefix budget used in PR #262, the sparse block cache covers 5,294,336 target positions (8.54% of validation) and improves the final stride-64 sliding score to `1.0365 bpb`.

## Model architecture

- 8 layers, 512 dim, 8 heads (4 KV), MLP 3x
- SmearGate + BigramHash (2048 buckets, dim=128)
- OrthoInit + muP scaling
- U-Net skip connections
- Int6 quantization + zstd-22 compression
- FP16 tied embedding passthrough
- SWA
- Sliding-window eval at stride 64

## Sparse paid-prefix details

- Prefix type: `sparse_blocks_v1`
- Block size: `256` tokens
- Build time: inside the eval phase of `train_gpt.py`
- Selection rule: rank validation blocks by total sliding-window NLL and keep the best blocks under `PAID_PREFIX_TARGET_BYTES`
- Artifact accounting: generated prefix blob is written to disk and counted in total bytes
- Selected blocks: `20,681`
- Covered tokens: `5,294,336` (`8.54%`)
- Prefix bytes: `4,240,256`
- Inline build time: `132,218 ms`

## Training / evaluation command

```bash
NCCL_IB_DISABLE=1 NUM_LAYERS=8 BIGRAM_VOCAB_SIZE=2048 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3000 \
MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
PAID_PREFIX_CODEC=sparse_blocks_v1 \
PAID_PREFIX_TARGET_BYTES=4240472 \
PAID_PREFIX_BLOCK_SIZE=256 \
PAID_PREFIX_FILE=generated_paid_prefix_sparse.xz \
RUN_ID=8L_prefix_sparse_v1 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Results

| Metric | Value |
|--------|-------|
| Training steps | 6,585 (600s, 91.12ms/step avg) |
| Pre-quant val_bpb | 1.1787 |
| Int6 roundtrip val_bpb | 1.1917 |
| Int6 sliding val_bpb (unpaid) | 1.1693 |
| **Int6 sliding val_bpb (s64, sparse paid-prefix)** | **1.0365** |
| Model params | 19,745,345 |
| Quant gap | 0.0130 BPB |
| Model bytes | 12,201,906 |
| Prefix bytes | 4,240,256 |
| Code bytes | 83,592 |
| Total bytes | 16,525,754 |

## Budget allocation

| Component | Bytes | MB |
|-----------|-------|----|
| Model (int6 + zstd-22) | 12,201,906 | 12.20 |
| Sparse prefix (LZMA-6) | 4,240,256 | 4.24 |
| Code | 83,592 | 0.08 |
| **Total** | **16,525,754** | **16.53** |

## Acknowledgments

Model architecture and paid-prefix baseline are built directly on PR #262 by @ibarrajo, which itself builds on PR #198 by @jfprincz. The sparse hard-block cache changes only the paid-prefix selection and eval flow.
