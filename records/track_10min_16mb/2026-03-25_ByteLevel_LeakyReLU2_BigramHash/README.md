# Byte-Level Tokenizer-Free Transformer

**First tokenizer-free byte-level model to beat the sp1024 baseline in Parameter Golf.**

This submission operates directly on raw UTF-8 bytes (vocab=256) with no tokenizer, no BPE, and no SentencePiece. It demonstrates that a well-tuned byte-level transformer can match and exceed the compression quality of the sp1024 token-level baseline on the FineWeb validation set, while using a fundamentally simpler input representation.

## Architecture

- **Input**: Raw UTF-8 bytes, vocab\_size=256
- **Layers**: 13 pure self-attention layers (`BLOCK_PATTERN=AAAAAAAAAAAAA`)
- **Model dim**: 512, **Heads**: 8/8 MHA (no GQA)
- **MLP**: 3× hidden (1536), LeakyReLU² activation (`F.leaky_relu(x, 0.5).square()`)
- **Features**: SmearGate + ByteBigramHash (4096 buckets, 32 dim)
- **Skip connections**: U-Net style encoder-decoder with learned skip weights
- **Tied embeddings**: Yes (byte embedding table shared with output projection)
- **Logit softcap**: 30.0
- **Parameters**: 27.6M (27,571,816)

### Key Design Choices

1. **No tokenizer**: The model predicts one byte at a time from raw UTF-8 input. BPB is measured directly (nats/byte / ln(2)) with no tokenizer-dependent conversion.

2. **Pure attention at seq\_len=4096**: Byte-level sequences are ~2.44× longer than sp1024 token sequences. Despite the quadratic attention cost, pure attention at 4096 positions outperforms SSM/attention hybrids because FlashAttention is highly optimized on H100, while SSM kernels (even compilable pure-PyTorch implementations) are 2-7× slower per layer.

3. **LeakyReLU²**: Replaces ReLU² with `F.leaky_relu(x, negative_slope=0.5).square()`, allowing negative pre-activations to contribute small gradient signal. Used by PR #549 (merged SOTA).

4. **ByteBigramHash**: Hashed byte-bigram embeddings capture local byte-pair statistics (e.g., common UTF-8 multi-byte sequences, ASCII digrams). Maps `(prev_byte * 256 + curr_byte) % 4096` to a 32-dim embedding, projected to model dim via a linear layer. Added after SmearGate.

5. **Sliding window evaluation**: stride=512, seq\_len=4096. Each byte is scored with up to 4096 bytes of context. This is the standard evaluation method used by merged SOTA submissions.

## Data Preparation

The byte-level dataset is created by decoding the sp1024 tokenized shards back to raw UTF-8 bytes.
A standalone conversion script is included:

```bash
# First download the sp1024 dataset
python data/cached_challenge_fineweb.py --variant sp1024

# Convert to byte-level shards
python convert_to_bytes.py \
    --src data/datasets/fineweb10B_sp1024 \
    --dst data/datasets/fineweb10B_bytes \
    --tokenizer data/tokenizers/fineweb_1024_bpe.model
```

The conversion produces 81 shards (80 train + 1 val) with ~2.44× more positions than sp1024 (bytes vs tokens).

## Training Configuration

```bash
BLOCK_PATTERN="AAAAAAAAAAAAA" \
TRAIN_BATCH_TOKENS=393216 TRAIN_SEQ_LEN=4096 \
VOCAB_SIZE=256 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=8 \
MLP_MULT=3 MLP_HIDDEN=1536 \
WARMDOWN_ITERS=3500 \
MATRIX_LR=0.035 TIED_EMBED_LR=0.05 SCALAR_LR=0.04 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=2500 \
ITERATIONS=20000 MAX_WALLCLOCK_SECONDS=600 \
VAL_LOSS_EVERY=0 TRAIN_LOG_EVERY=1000 WARMUP_STEPS=10 \
USE_COMPILE=1 SEED=1337 \
SMEAR_GATE=1 SWA_EVERY=50 SWA_LAST_FRAC=0.5 \
GRAD_CLIP_NORM=0.3 \
BIGRAM_HASH_BUCKETS=4096 BIGRAM_HASH_DIM=32 \
VAL_SLIDING_STRIDE=512 VAL_SLIDING_MAX_TOKENS=10000000 \
DATA_PATH=./data/datasets/fineweb10B_bytes \
torchrun --standalone --nproc_per_node=8 train_byte_model.py
```

## Results (4-seed significance test)

| Seed | Sliding BPB | Non-overlap BPB | Artifact | Under 16 MiB |
|------|------------|----------------|----------|---------------|
| 1337 | **1.2146** | 1.2306 | 15.53 MB | Yes |
| 42   | **1.2120** | 1.2278 | 15.80 MB | Yes |
| 2025 | **1.2174** | 1.2327 | 16.45 MB | Yes |
| 7    | **1.2166** | 1.2319 | 15.46 MB | Yes |

### Statistical Significance

| Comparison | Mean BPB | Δ BPB | Δ nats | t-stat | p (one-sided) |
|-----------|---------|-------|--------|--------|---------------|
| vs Official baseline (1.2244) | 1.2151 | 0.0093 | 0.0064 | -7.60 | **0.0024** |
| vs Post-quant baseline (1.2269) | 1.2151 | 0.0118 | 0.0081 | -9.65 | **0.0012** |

- **99% CI**: [1.2080, 1.2223] — official baseline 1.2244 is outside the CI
- **All 4 seeds individually beat the official baseline**
- **All artifacts under 16 MiB** (16,777,216 bytes)

### JEPA Auxiliary Loss Study

We also tested adding a JEPA-style latent prediction auxiliary loss (predict future byte embeddings from hidden states):

| Config | Sliding BPB | Δ vs no-JEPA |
|--------|------------|-------------|
| No JEPA (best) | **1.2146** | — |
| JEPA K=4, weight=0.10 | 1.2390 | +0.024 (worse) |
| JEPA K=4, weight=0.01 | 1.2206 | +0.006 (worse) |

The JEPA auxiliary loss hurts BPB at this scale due to gradient competition with the primary cross-entropy objective and the small byte embedding space (256 entries).

## Key Metrics (seed 42 — best sliding BPB)

- Training stopped at step 7196/20000 (600s wallclock cap)
- Step average: 83.4 ms/step
- Peak memory: 12,069 MiB allocated, 12,546 MiB reserved
- EMA selected as final weights (decay=0.997)
- Pre-quant EMA: val\_bpb=1.2249, sliding\_bpb=1.2090
- Post-quant int6+zstd22: val\_bpb=1.2278, sliding\_bpb=1.2120
- Serialized model int6+zstd22: 15,721,735 bytes
- Code size: 73,320 bytes
- Total submission: 15,795,055 bytes

## Requirements

```
torch>=2.11.0
sentencepiece
zstandard
```

FlashAttention 3 (Hopper) is used when available via `flash_attn_interface`.

## Included Files

- `train_byte_model.py` — Complete training script (model + training loop + eval + serialization)
- `convert_to_bytes.py` — Standalone data conversion script (sp1024 tokens to raw bytes)
- `requirements.txt` — Python dependencies
- `submission.json` — Leaderboard metadata
- `README.md` — This file