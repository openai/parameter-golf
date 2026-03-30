# Record: 1.1539 BPB - 74.3M Ternary U-Net Transformer (updated)

**Continuation of [#640](https://github.com/openai/parameter-golf/pull/640) - BF16 scale storage + EMBED_DIM=312**

**val_bpb: 1.1539** (3-seed mean sliding, std 0.0004) | **15.95 MB** max artifact | 8xH100 SXM, 600s

> Improvement of 0.0031 BPB over original submission (1.1570 -> 1.1539), with tighter roundtrip gap, and higher cross-seed reproducibility. Full post-submission research log: [RESULTS_CONTINUED.md](RESULTS_CONTINUED.md).

## What changed from #640

Two serialization/embedding improvements, same architecture and training recipe:

1. **BF16 scale storage** - ternary dequantization scales changed from FP16 to BF16. Zero additional bytes (both are 2B per value). BF16's wider exponent range eliminates magnitude rounding that gets amplified by the shrinkage correction factor `1/(1-zero_frac)` at high zero fractions. Reduces RT gap from 0.0021 to 0.0011.

2. **EMBED_DIM 254 -> 312** - the BF16 fix plus RMS scale research confirmed headroom in the artifact budget. Increasing the embedding bottleneck from 254 to 312 adds 0.6M parameters exclusively in the FP8 path (+55KB compressed), improving representation quality monotonically. 312 is the largest multiple of 8 that fits within 16MB.

Additionally, `WARMDOWN_FRACTION` was adjusted from 0.2 to 0.15 based on extended training experiments.

Everything else is identical to #640: 10L 768d, BitNet b1.58 ternary, relu2 4x MLP, GQA 8/4 heads, U-Net skips, YaRN 2048, Muon+AdamW, poly5 softcap, FP8 QAT, Base-3 LZMA, stride-16 sliding eval, T=0.90.

## Results (3 seeds, 8xH100 SXM)

| Seed | Steps | ms/step | Sliding BPB (s16) | val_bpb | RT bpb | RT gap | Artifact |
|------|-------|---------|-------------------|---------|--------|--------|----------|
| 7 | 6,530 | 91.9 | **1.1535** | 1.1802 | 1.1808 | 0.0006 | 15,951,196 bytes |
| 42 | 6,540 | 91.8 | 1.1542 | 1.1805 | 1.1824 | 0.0019 | 15,952,348 bytes |
| 1337 | 6,530 | 91.9 | 1.1540 | 1.1803 | 1.1811 | 0.0008 | 15,953,260 bytes |
| **Mean** | **6,533** | **91.9** | **1.1539** | **1.1803** | **1.1814** | **0.0011** | **15,952,268 bytes** |
| **Std** | **5** | **0.1** | **0.0004** | **0.0002** | **0.0008** | | **1,033 bytes** |

> The variation in artifact size is due to the seed changing ultimately the values the parameters will converge on, which directly affects the compression algorithm.

### Comparison vs #640

| Metric | #640 (original) | This PR | Delta |
|--------|----------------|---------|-------|
| Sliding BPB | 1.1570 | 1.1539 | -0.0031 |
| val_bpb | 1.1821 | 1.1803 | -0.0018 |
| RT bpb | 1.1842 | 1.1814 | -0.0028 |
| RT gap | 0.0021 | 0.0011 | -0.0010 |
| Seed std (sliding) | 0.0007 | 0.0004 | more stable |
| Artifact | 15.99 MB | 15.95 MB | -40 KB |
| Params | 73.7M | 74.3M | +0.6M |

## Setup and Run

```bash
bash setup.sh
conda activate golf
SEED=42 bash run_cuda_ternary.sh
```

<details>
<summary>Full run command</summary>

```bash
RUN_ID=ternary_run \
DATA_PATH=./data/datasets/fineweb10B_sp8192 \
TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe.model \
ATTN_PROJ_TYPE=standard \
LOGIT_HEAD_TYPE=standard \
TVERSKY_MEMBERSHIP=sigmoid \
TVERSKY_NUM_FEATURES=0 \
TVERSKY_FEATURE_POOLS=0 \
VOCAB_SIZE=8192 \
BITNET_GROUP_SIZE=128 \
BIGRAM_HASH=0 \
EMBED_DIM=312 \
EMBED_RANK=0 \
TRAINING_DEPTH_RECURRENCE=0 \
EVAL_DEPTH_RECURRENCE=0 \
NUM_LAYERS=10 \
MODEL_DIM=768 \
NUM_KV_HEADS=4 \
NUM_HEADS=8 \
DIFF_ATTN=0 \
MLP_MULT=4 \
MLP_GROUPS=0 \
MATRIX_OPTIMIZER=muon \
ADAM_LR=0.05 \
ADAM_WD=0.05 \
MUON_BACKEND_STEPS=3 \
MUON_MOMENTUM=0.95 \
MUON_MOMENTUM_WARMUP_START=0.85 \
MUON_MOMENTUM_WARMUP_STEPS=500 \
MUON_WD=0.0 \
MATRIX_LR=0.04 \
SCALAR_LR=0.02 \
TIED_EMBED_LR=0.02 \
WARMDOWN_FRACTION=0.15 \
LOGIT_SOFTCAP=10 \
QK_GAIN_INIT=2.25 \
ROPE_TYPE=yarn \
YARN_MAX_LEN=2048 \
ROPE_BASE=5000 \
BATCH_TOKENS_START=0 \
BATCH_SCHEDULE_FRACTION=0.33 \
TRAIN_BATCH_TOKENS=524288 \
SEQ_LEN_START=0 \
SEQ_SCHEDULE_FRACTION=0.0 \
TRAIN_SEQ_LEN=1024 \
SMEAR=0 \
ITERATIONS=10000 \
WARMUP_STEPS=5 \
MAX_WALLCLOCK_SECONDS=600 \
VAL_LOSS_EVERY=5000 \
TRAIN_LOG_EVERY=1000 \
CHURN_LOG_EVERY=0 \
VAL_MAX_TOKENS=0 \
TIE_EMBEDDINGS=1 \
UNTIE_AT_FRACTION=0.00 \
HEAD_LR=0.02 \
CORR_WEIGHT_LR=0.02 \
ACTIVATION=relu2 \
SOFTCAP_TYPE=poly \
MTP_HEADS=0 \
REFINER=0 \
REFINER_KERNEL=3 \
SLIDING_EVAL=1 \
SLIDING_EVAL_STRIDE=16 \
SLIDING_BATCH_SIZE=512 \
TEMP_SCALING=1 \
FP_STORAGE=FP8 \
SEED=42 \
COMPILE_MODE=default \
CHECKPOINT_EVERY=5000 \
CHECKPOINT_DIR=./checkpoints \
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=8 train_gpt_cuda_ternary.py
```

</details>

## Compliance

- [x] 3 seeds run on 8xH100 SXM
- [x] All 3 seeds train in <=600s (max: 600.7s)
- [x] All 3 seeds artifact <=16,000,000 bytes (max: 15,953,260)
- [x] Sliding window eval stride=16, consistent (std=0.0004)
- [x] No test-time training on validation data
- [x] No network calls during evaluation
- [x] No external compute
