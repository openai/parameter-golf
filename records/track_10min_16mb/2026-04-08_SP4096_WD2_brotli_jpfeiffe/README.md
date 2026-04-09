# Record: SP4096 + Compressibility Regularization

**val_bpb: 1.11349** (6-seed mean, std 0.00053) | **~15.68 MB** | 8xH100 SXM, 600s | No TTT

**Improvement over current SOTA ([PR #1019](https://github.com/openai/parameter-golf/pull/1019), 1.11474 BPB):** -0.00125 BPB (Welch t=-4.19, df=6.6, p=0.00289)

No TTT, no n-gram cache, no eval-time logit bias. All gains are from training-side changes.

## Results

| Seed | Steps | ms/step | Pre-quant BPB | **Sliding BPB** | Artifact | Pruning |
|------|-------|---------|---------------|-----------------|----------|---------|
| 314 | 6,699 | 89 | 1.1260 | **1.11410** | 15,665,083 | 0% |
| 42 | 6,664 | 90 | 1.1261 | **1.11418** | 15,667,940 | 0% |
| 999 | 6,659 | 90 | 1.1255 | **1.11348** | 15,697,830 | 0% |
| 1337 | 6,658 | 90 | 1.1253 | **1.11307** | 15,660,616 | 0% |
| 2024 | 6,664 | 90 | 1.1261 | **1.11306** | 15,693,397 | 0% |
| 7 | 6,659 | 90 | 1.1255 | **1.11305** | 15,686,495 | 0% |
| **Mean** | | | | **1.11349** | | |

Current SOTA (PR #1019, exact 3-seed mean): **1.11473509 BPB**. This run's exact 6-seed mean is **1.11348911 BPB**. Delta: **-0.00124599 BPB**.

Using the exact per-seed scores from PR #1019 (`1.11508120`, `1.11437394`, `1.11475014`) and this run, Welch's t-test gives **t = -4.19**, **df = 6.6**, **p = 0.00289** (one-sided).

## Main Changes

Two training-side changes on the [PR #1019](https://github.com/openai/parameter-golf/pull/1019) base:

### 1. SP4096 Tokenizer (up from SP1024)

Larger vocabulary reduces tokens-per-byte from ~0.59 to ~0.30, allowing the model to see more context per training step. The tied embedding grows from 1024x512 to 4096x512, adding ~1.1MB to the artifact. SP4096 data from [sproos/parameter-golf-tokenizers](https://huggingface.co/sproos/parameter-golf-tokenizers) on HuggingFace, tokenized from the same FineWeb documents (identical `docs_sha256`, see `data_lineage.md`).

### 2. Compressibility Regularization (WARMDOWN_WD_MULT=2.0)

During the LR warmdown phase, weight decay ramps from 1x to 2x base WD. This pushes weights toward zero during the final training steps, reducing post-quantization entropy (4.72 -> 4.58 bits) and compressed artifact size. The ~1.5MB compression savings from WD=2.0 absorbs the ~1.1MB embedding cost of SP4096, with ~300KB headroom. Selective pruning never triggers (0% on all 6 seeds).

### 3. Brotli-11 Compression (replacing LZMA-9)

Brotli-11 consistently beats LZMA-9 by 200-400KB on quantized weight streams. The artifact is saved as whichever compressor produces the smaller output (brotli wins on all runs). Load path auto-detects format.

## Why It Works

The key insight: WD compression frees enough artifact budget to absorb SP4096's larger embedding without exceeding 16MB. Without WD=2.0, SP4096 requires aggressive selective pruning (57% of +/-1 values) which destroys quality. With WD=2.0, the weights are naturally more compressible and pruning is never needed.

| Config | SW BPB | Artifact | Pruning |
|--------|--------|----------|---------|
| SP1024 + WD=1.0 (PR #1019) | 1.1151 | 15.9MB | mild |
| SP4096 + WD=1.0 | 1.1359 | 16.1MB* | 57.5% |
| SP4096 + WD=2.0 | **1.1135** | 15.7MB | 0% |

*Over 16MB budget

## Architecture

Unchanged from PR #1019:

| Component | Setting |
|-----------|---------|
| Layers | 11 (512d, 8 GQA heads, 4 KV heads) |
| MLP | 3x (1536) with LeakyReLU(0.5)^2 |
| Attention | XSA on all 11 layers |
| BigramHash | 3072 x dim=112 |
| Tokenizer | **SP4096** (was SP1024) |
| Quantization | INT6 per-row, GPTQ with AR self-gen calibration |
| Compression | **Brotli-11** (was LZMA-9) |
| WD | **WARMDOWN_WD_MULT=2.0** (ramps during warmdown) |
| RoPE | Partial (16/64 dims) |
| WARMDOWN_ITERS | 4000 |

## Reproduction

```bash
# Download SP4096 data from sproos
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('sproos/parameter-golf-tokenizers',
    allow_patterns=['datasets/fineweb10B_sp4096/*', 'tokenizers/fineweb_4096_bpe.*'],
    local_dir='./data')
"

# Run (8xH100 SXM)
VOCAB_SIZE=4096 \
DATA_PATH=./data/datasets/fineweb10B_sp4096 \
TOKENIZER_PATH=./data/tokenizers/fineweb_4096_bpe.model \
BIGRAM_VOCAB_SIZE=3072 BIGRAM_DIM=112 \
WARMDOWN_ITERS=4000 WARMDOWN_WD_MULT=2.0 \
SEED=314 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Verification

- **Data lineage**: `docs_sha256` matches official repo byte-for-byte (see `data_lineage.md`)
- **BPB sanity check**: Manual byte counting matches reported BPB within float64 precision (see `bpb_verification.md`)
- **Pruning**: 0% on all 6 seeds
- **Artifact size**: 15.66-15.70MB (all under 16MB)
