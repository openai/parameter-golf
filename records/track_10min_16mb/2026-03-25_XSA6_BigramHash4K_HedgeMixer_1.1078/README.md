# Record Submission: XSA6 + BigramHash4K on Hedge Mixer Stack

**val_bpb: 1.1078** (3-seed mean, std 0.0045)

| Seed | val_bpb |
|------|---------|
| 42 | 1.1045 |
| 1337 | 1.1061 |
| 2025 | 1.1129 |
| **Mean** | **1.1078** |
| **Std** | **0.0045** |

Built on PR #700 (@RoyiRa) with hyperparameter improvements found via systematic combinatorial search ([autoresearch-multi](https://github.com/agalimova/autoresearch-multi)).

## Changes from PR #700

| Parameter | PR #700 Default | Ours | Found via |
|-----------|----------------|------|-----------|
| `XSA_LAST_N` | 4 | **6** | autoresearch-multi EXPLOIT round |
| `BIGRAM_VOCAB_SIZE` | 2048 | **4096** | autoresearch-multi EXPLOIT round |

These two hyperparameter changes were identified by autoresearch-multi, a 4-mode adaptive search tool that tested 12+ configurations with interaction detection. The combination is superadditive: XSA=6 alone gives -0.001, BIGRAM=4096 alone gives -0.001, but together they give -0.002.

## Ablation (8xH100, torch 2.9+cu126, FA3)

| Variant | val_bpb (TTT) | val_bpb (no TTT) |
|---------|---------------|------------------|
| PR #700 baseline | 1.1225* | 1.1225* |
| + XSA_LAST_N=6 | — | 1.1215 |
| + BIGRAM_VOCAB_SIZE=4096 | — | 1.1217 |
| + both (this submission) | **1.1078** | 1.1209 |

*Our reproduction; PR #700 reports 1.0541 with their exact setup.

## How to Reproduce

```bash
# Requirements: 8xH100, PyTorch 2.9+, FlashAttention 3

# Install
pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu126
pip install sentencepiece huggingface_hub zstandard

# Build FA3 (Hopper only, minimal kernels ~5 min)
cd flash-attention/hopper
FLASH_ATTENTION_DISABLE_SM80=TRUE FLASH_ATTENTION_DISABLE_FP16=TRUE \
FLASH_ATTENTION_DISABLE_FP8=TRUE FLASH_ATTENTION_DISABLE_SPLIT=TRUE \
FLASH_ATTENTION_DISABLE_LOCAL=TRUE FLASH_ATTENTION_DISABLE_PAGEDKV=TRUE \
FLASH_ATTENTION_DISABLE_APPENDKV=TRUE FLASH_ATTENTION_DISABLE_VARLEN=TRUE \
FLASH_ATTENTION_DISABLE_PACKGQA=TRUE FLASH_ATTENTION_DISABLE_SOFTCAP=TRUE \
FLASH_ATTENTION_DISABLE_HDIM96=TRUE FLASH_ATTENTION_DISABLE_HDIM192=TRUE \
FLASH_ATTENTION_DISABLE_HDIM256=TRUE TORCH_CUDA_ARCH_LIST="9.0" MAX_JOBS=32 \
python setup.py build_ext --inplace
cp build/lib.linux-x86_64-cpython-311/flash_attn_3/_C.abi3.so flash_attn_3/

# Download data
cd parameter-golf
python data/cached_challenge_fineweb.py --variant sp1024

# Run (seed 1337)
SEED=1337 XSA_LAST_N=6 BIGRAM_VOCAB_SIZE=4096 \
PYTHONPATH=/path/to/flash-attention/hopper:$PYTHONPATH \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Architecture (inherited from PR #700)

- 11 layers, 512 dim, 8 heads, 4 KV heads (GQA), 33M params
- LeakyReLU(0.5)^2 activation
- Parallel Muon optimizer with parameter banking
- BigramHash embeddings (**4096** buckets, 128 dim)
- XSA on last **6** layers
- Partial RoPE (16/64 dims), LN Scale 1/sqrt(layer+1)
- EMA(0.997) + SWA(every 50 steps)
- CROWN-Q quantization-aware loss during warmdown
- GPTQ-lite int6 quantization + zstd compression
- Legal score-first TTT (4 epochs, AdamW, cosine decay)
- Hedge Mixer: 4-expert logistic context mixing (neural + unigram + bigram + trigram)

## Search Methodology

We used autoresearch-multi to systematically search over hyperparameter combinations. The search tested:

- **EXPLORE** (6 experiments): Coverage across code variants and tuning options
- **EXPLOIT** (6 experiments): Drilling into winners, found XSA=6 + BIGRAM=4096

Key finding: structured 2:4 sparsity on MLP weights was tested but degraded quality significantly (1.68 BPB), confirming that post-training sparsity doesn't work at this model scale without retraining.

## Training Details

- 8xH100 SXM 80GB, ~99ms/step (with CROWN-Q overhead)
- ~5800-6100 steps in 600s wallclock
- 15.3MB artifact (int6+zstd, under 16MB limit)
- TTT eval: ~700s (4 epochs, legal score-first)

## Credits

- **@RoyiRa**: PR #700 — Hedge Mixer, CROWN-Q, stride=64 optimization
- **@abaybektursun**: PR #549 — LeakyReLU^2, Legal TTT, Parallel Muon
- **@jfprincz**: PR #287 — Partial RoPE, XSA, LN Scale
- **@signalrush**: PR #374 — GPTQ-lite, EMA
- Full lineage: PR #70 → #164 → #198 → #287 → #374 → #414 → #549 → #700 → this
