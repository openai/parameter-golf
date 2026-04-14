# Record: 4096-Vocab + Larger Model + High WD + Simplifications — val_bpb 1.09785

**val bpb: 1.09785** (3-seed mean, std=0.0004)

| Seed | Steps | Pre-quant BPB | Post-quant BPB | **Sliding BPB** | Artifact |
|-|-|-|-|-|-|
| 42   | 5967  | 1.10411 | 1.11588 | **1.09744** | 15,915,268 |
| 1337 | 5962  | 1.10482 | 1.11631 | **1.09795** | 15,905,460 |
| 2025 | 5961  | 1.10507 | 1.11641 | **1.09816** | 15,927,782 |
| **Mean** | | 1.10467 | 1.11620 | **1.09785** | 15,916,170 |

## Overview

This script builds on the 03-23 leaderboard [record](https://github.com/openai/parameter-golf/blob/main/records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md). The main changes are:

### Fixes
* Fixed a small bug in the sliding window evaluation causing it to score tokens at the end of the val dataset multiple times. This bug didn't significantly affect results: it added roughly 2k duplicate contributions to the total loss and byte counts over a validation set of about 6M tokens. The faulty line was:
 `window_starts = [ws for ws in range(0, total_tokens, stride) if min(ws + seq_len, total_tokens) - ws >= 1]`, and it should be:
 `window_starts = [ws for ws in range(0, total_tokens, stride) if  ws + seq_len - stride  <  total_tokens]`

### Simplifications
* Use XSA in all layers instead of only the last 4.
* Removed parameter banking and distributed muon implementation and instead just used Muon + DDP.
* Removed test time training. I doubt that 0.1% additional tokens will improve the model
  generally, and for long docs I think it makes more sense to work on extending the sequence length.
* Removed quantization-aware training, since it appeared to provide little or no benefit.
* Removed gated attention.
* Removed value residuals.
* Removed hash embeddings, which are probably less necessary after increasing the vocab size.
* Removed the smear gate, for the same reason.

### Additions
* Increased the vocabulary size from 1024 to 4096. I used the existing `data/download_hf_docs_and_tokenize.py` to build the sentencepiece tokenizer and pre-tokenized data. The tokenizer model grew by ~50kb, but even with that added, the final artifacts are below the 16MB cap. A larger vocab means the model sees more context for the same sequence length and more train data per step.
* Use a bigger but more strongly regularized model. I discovered that the compressibility of a weight matrix (i.e., quantized-and-compressed-mb / raw-mb) correlates extremely well with the matrix's root-mean-square (`torch.sqrt(torch.mean(x**2))`) with an R^2 near 0.99. This suggests that the weight decay is a good lever for reducing the compressed size, which can let us add more parameters to the model. In particular this script uses:
  * Higher weight decays: muon weight decay increased 0.04 -> 0.085, and added an embeddings weight decay of 0.085. Additionally, decreased the adam weight decay 0.04 -> 0.02, as scalar parameters shouldn't need to be low-magnitude.
  * Wider MLPs, increasing `mlp_mult` 3 -> 4.
  * A decreased learning rate 0.025 -> 0.02, as larger models generally benefit from smaller LRs.
* Added the coprime-stride data loader from [#726](https://github.com/openai/parameter-golf/pull/726). The benefit is that it avoids showing the model sequences from the same document in the same/nearby minibatches by jumping around the data files.
* Added GPTQ Hessian-aware quantization. My implementation is based on [#1060](https://github.com/openai/parameter-golf/pull/1060) and reserves some time from training for Hessian computation.
* Use more efficient byte shuffle + brotli compression from [#1089](https://github.com/openai/parameter-golf/pull/1089).
* Added sigmoid-gated skip connections to the unet, also from [#1089](https://github.com/openai/parameter-golf/pull/1089).
* Increased `qk_gain_init` 1.5 -> 4 following [#1125](https://github.com/openai/parameter-golf/pull/1125).

## Requirements

Flash Attention 3 (Hopper) is required. The script imports `flash_attn_interface` directly and was run with PyTorch 2.11.0+cu130. Install commands:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu130
pip install --no-cache-dir \
  "https://download.pytorch.org/whl/cu130/flash_attn_3-3.0.0-cp39-abi3-manylinux_2_28_x86_64.whl"
pip install -r requirements.txt
```

The tokenizer and pre-tokenized data (sp4096) is available on my [HuggingFace](https://huggingface.co/datasets/kevclark/parameter-golf). You can download it with:

```bash
rm -f data/manifest.json
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \
  python3 data/cached_challenge_fineweb.py --variant sp4096 --train-shards 143
```

Note this first deletes any existing `data/manifest.json` because the download script caches the manifest locally, and a stale one from the default repo won't include sp4096. Alternatively, to regenerate the tokenizer and dataset from scratch:

```bash
cat > data/tokenizer_specs_4096.json << 'EOF'
[
  {
    "name": "sp_bpe_4096",
    "kind": "sentencepiece_bpe",
    "vocab_size": 4096,
    "tokenizer_train_docs": 5000000
  }
]
EOF
python3 data/download_hf_docs_and_tokenize.py \
  --output-root data \
  --tokenizer-config data/tokenizer_specs_4096.json \
  --skip-byte
```

## Run Command

```bash
RUN_ID=1337 SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```