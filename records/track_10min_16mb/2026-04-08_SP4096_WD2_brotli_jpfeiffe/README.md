# Record: SP4096 + Compressibility Regularization

**val_bpb: 1.11349** (6-seed mean, std 0.00053) | **~15.68 MB** | 8xH100 SXM, 600s | No TTT

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

Exact 6-seed mean: **1.11348911 BPB**. Current merged SOTA (PR #1019) exact 3-seed mean: **1.11473509 BPB**. Welch's t-test: **t = -4.19**, **df = 6.6**, **p = 0.00289** (one-sided).

No TTT, no n-gram cache, no eval-time logit bias. All gains are from training-side changes.

---

## Changes

Three changes to the PR #1019 base:

### 1. SP4096 Tokenizer

Vocabulary increased from SP1024 to SP4096. Tokens-per-byte drops from ~0.59 to ~0.30, allowing the model to see more context per training step. The tied embedding grows from 1024x512 to 4096x512, adding ~1.1MB to the artifact.

SP4096 data from [sproos/parameter-golf-tokenizers](https://huggingface.co/sproos/parameter-golf-tokenizers), tokenized from the same FineWeb documents as the official SP1024 data (identical `docs_sha256`; see `data_lineage.md`).

### 2. WARMDOWN_WD_MULT=2.0

During LR warmdown, effective weight decay increases from 1x to 2x base WD. The mechanism: `group["weight_decay"] = base_wd * (1 + (mult - 1) * (1 - lr_scale))`, applied to all optimizer param groups before each step. Muon and AdamW both consume the updated WD via their standard `p.data.mul_(1.0 - lr * wd)` path.

This produces a more peaked post-quantization weight distribution (entropy 4.72 → 4.58 bits, zeros 8.3% → 11.4%), reducing brotli-compressed artifact size by ~1.5MB.

### 3. Brotli-11 Compression

Both lzma-9 and brotli-11 are computed; the smaller result is saved as the artifact. Brotli-11 was smaller on all 6 seeds. The load path auto-detects format (try lzma first, fall back to brotli).

### Why These Three Stack

WARMDOWN_WD_MULT=2.0 frees ~1.5MB of artifact budget through compression. This headroom absorbs SP4096's +1.1MB embedding cost. All 6 seeds fit under 16MB without selective pruning (0% on all seeds).

Without WARMDOWN_WD_MULT, SP4096 requires aggressive selective pruning (57.5% of +/-1 values zeroed) which destroys quality (SW BPB degrades from 1.113 to 1.136).

---

## Architecture

| Component | Setting |
|-----------|---------|
| Layers | 11 (512d, 8 GQA heads, 4 KV heads) |
| MLP | 3x (1536) with LeakyReLU(0.5)^2 |
| Attention | XSA on all 11 layers |
| BigramHash | 3072 x dim=112 |
| Tokenizer | **SP4096** |
| Quantization | INT6 per-row, GPTQ with AR self-gen calibration |
| Compression | **Brotli-11 selected when smaller than LZMA-9** |
| Weight Decay | **WARMDOWN_WD_MULT=2.0** (ramps from 1x to 2x during warmdown) |
| WARMDOWN_ITERS | 4000 |

---

## Verification

- Manual BPB recompute matches logged value to 4e-6 (`bpb_verification.md`)
- SP4096 tokenized from same FineWeb documents as SP1024 baseline; `docs_sha256` identical (`data_lineage.md`)

---

## Reproduction

```bash
# Download SP4096 data
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
