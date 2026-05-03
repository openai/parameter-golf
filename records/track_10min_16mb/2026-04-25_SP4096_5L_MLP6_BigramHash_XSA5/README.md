# SP4096 5L MLP6 BigramHash XSA5

**val_bpb: 1.16362299**

Five improvements over the naive baseline, found via ~200 automated experiments using [Karpathy's autoresearch framework](https://github.com/karpathy/autoresearch) on 1×H100.

### Changes from Baseline

**1. SP4096 Tokenizer**

Switched from the default 1024-vocab SentencePiece tokenizer to 4096-vocab. Larger vocabulary improves tokenization efficiency, reducing BPB by encoding text more compactly. Data downloaded from `kevclark/parameter-golf` HuggingFace repo.

**2. 5-Layer Architecture with MLP_MULT=6**

On a 5-minute 1×H100 budget, shallower but wider models outperform deeper ones because they get more training steps. The optimal architecture found was 5 layers with 6× MLP expansion (hidden=3072 from base 512), confirmed via sweep across 5L/6L/7L/8L and MLP_MULT 4/5/6.

**3. BigramHash Embeddings (dim=256, kaiming init)**

Augments token embeddings with bigram hash features. Critical implementation detail: kaiming initialization is required — zero or random init causes dead neurons. BIGRAM_DIM=256 confirmed optimal (320 exceeds 15.8MB safety limit).

**4. Full XSA — Cross-Sequence Attention on all 5 layers**

Applies cross-sequence attention on all 5 layers. Removes self-value bias from attention output via orthogonal projection. XSA on all layers beats partial XSA (3 or 4 layers) for this shallow model.

**5. Brotli-11 Compression**

Replaced zlib with Brotli-11 for weight compression. Provides significantly better compression ratio for int8 weights at the cost of slightly longer serialization time.

### Configuration

```
NUM_LAYERS=5
MODEL_DIM=512
NUM_HEADS=8
NUM_KV_HEADS=4
MLP_MULT=6
VOCAB_SIZE=4096
TRAIN_SEQ_LEN=2048
XSA_LAYERS=5
BIGRAM_VOCAB_SIZE=4096
BIGRAM_DIM=256
PARTIAL_ROPE_DIMS=32
ROPE_BASE=3000
LOGIT_SOFTCAP=30
QK_GAIN_INIT=7.0
```

Optimizer:

```
MATRIX_LR=0.04
SCALAR_LR=0.03
TIED_EMBED_LR=0.03
MUON_WD=0.055
BETA2=0.98
WARMUP_STEPS=250
WARMDOWN_ITERS=1400
```

### Run Command

```bash
RUN_ID=official_submission \
DATA_PATH=./data/datasets/fineweb10B_sp4096 \
TOKENIZER_PATH=./data/tokenizers/fineweb_4096_bpe.model \
VOCAB_SIZE=4096 \
MAX_WALLCLOCK_SECONDS=600 \
VAL_LOSS_EVERY=200 \
torchrun --standalone --nproc_per_node=8 records/track_10min_16mb/2026-04-25_SP4096_5L_MLP6_BigramHash_XSA5/train_gpt.py
```

Note: SP4096 data must be downloaded from the alternate repo:
```bash
rm -f data/manifest.json
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \
python3 data/cached_challenge_fineweb.py --variant sp4096 --train-shards 10
```

### Key Metrics

| Metric | Value |
|---|---|
| Pre-quant val_bpb (last step) | 1.1760 |
| **Post-quant val_bpb (int8+brotli roundtrip)** | **1.16362299** |
| Quantization penalty | +0.0124 BPB |
| Artifact size (int8+brotli) | 15,753,699 bytes |
| Code size | 69,973 bytes |
| Total submission size | 15,753,699 bytes |
| Steps completed | 14,227 / 20,000 |
| Step time | 42.18ms average on 8×H100 SXM |
| Parameters | 22,949,929 |

### Included Files

- `train_gpt.py` — full training + quantization + evaluation script
- `train.log` — complete training log from the official 8×H100 run
- `submission.json` — leaderboard metadata
- `README.md` — this file
