# Record: Scylla Tokenizer + Full GPTQ + XSA-all + FA3

**val_bpb: 0.9485** (3-seed mean, std 0.0008) | **~15.6 MB** | 8×H100 SXM | No TTT

## Results (8×H100 SXM)

| Seed | Steps | ms/step | Sliding BPB (s64) |
|------|-------|---------|--------------------|
| 1337 | 6,716 | 87.9 | 0.9491 |
| 42 | — | ~88 | **0.9476** |
| 2025 | — | ~88 | 0.9489 |
| **Mean ± Std** | | | **0.9485 ± 0.0008** |

## vs Competition

| Submission | BPB | Improvement |
|-----------|-----|-------------|
| PR #1143 (Scylla, old stack) | 1.0806 | **-0.1321 (-12.2%)** |
| PR #1089 (Turbo-Muon) | 1.1086 | -0.1601 |
| Merged SOTA (PR #549) | 1.1194 | -0.1709 |

## What's New

This submission combines the Scylla tokenizer (PR #1143) with the modern training stack (PR #1060), achieving a result far better than either alone.

### Tokenizer: Scylla (998 tokens)
- TokenMonster-derived vocabulary, pruned from `english-1024-clean-v1`
- Created by @simon-marcus through iterative autoresearch (PR #1143)
- 998 active tokens (vs 1024 for SentencePiece)
- Better byte-per-token efficiency via ungreedy multi-branch tokenization
- Retokenized FineWeb: 194 train shards (~19.4B tokens) + 1 val shard

### Training Stack (PR #1060 base)
- **Full Hessian GPTQ** — Cholesky error compensation, 64-batch calibration in 6.7s
- **XSA on all 11 layers** — exclusive self-attention everywhere
- **Coprime-stride multi-shard loader** — diverse batches across 194 shards
- **FlashAttention 3** — Hopper native kernels (pre-built wheel)
- **Parallel Muon** + Parameter Banking — 3-phase overlapped optimizer

### Why It's Better Than #1143 Alone
PR #1143 used the old SOTA stack (PR #549 base) which lacks:
- Full GPTQ (used GPTQ-lite → worse quantization)
- XSA on all layers (used last 4 → less cross-position mixing)
- Coprime data loader (sequential loading → less batch diversity)
- More training data (we used 194 shards vs their 79)

### No TTT Needed
TTT was tested and found neutral (0.9491 with TTT, 0.9491 without). Full GPTQ eliminates the need for test-time adaptation.

## Architecture

- 11L, 512d, 8H/4KV (GQA), MLP 3× LeakyReLU(0.5)²
- XSA on all 11 layers, BigramHash(2816×112), SmearGate
- Partial RoPE (16d), LN Scale 1/√(l+1)
- Shared ValueEmbedding (dim=128, layers 9-10)
- EMA (decay=0.997) + Tight SWA (every 50 steps)
- Full Hessian GPTQ int6 + LZMA compression

## Timing

| Phase | Time |
|-------|------|
| Training (6,716 steps @ 88ms) | 591s |
| GPTQ calibration | 6.7s |
| Sliding window eval (stride=64) | 92s |

## Reproduction

```bash
# Install
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
pip install "https://download.pytorch.org/whl/cu128/flash_attn_3-3.0.0-cp39-abi3-manylinux_2_28_x86_64.whl"
pip install sentencepiece huggingface-hub datasets numpy tqdm tokenmonster

# Retokenize FineWeb with Scylla vocab (once, ~90 min)
python3 retokenize.py --vocab candidate.vocab --output-dir data/datasets/fineweb10B_scylla

# Train
SEED=1337 DATA_PATH=./data/datasets/fineweb10B_scylla \
TOKENIZER_PATH=./candidate.vocab TOKENIZER_META_PATH=./candidate.meta.npz \
VOCAB_SIZE=998 XSA_LAST_N=11 USE_GPTQ=1 GPTQ_RESERVE_MS=9000 TTT_ENABLED=0 \
BIGRAM_VOCAB_SIZE=2816 BIGRAM_DIM=112 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Included Files
- `train_gpt.py` — Training script (PR #1060 + tokenizer metadata loading)
- `candidate.vocab` — Scylla tokenizer (998 tokens)
- `candidate.meta.npz` — Per-token byte accounting metadata
- `train_seed{1337,42,2025}.log` — Training logs for all 3 seeds

## Credits
- **Scylla tokenizer**: @simon-marcus (PR #1143)
- **Training stack**: @resouer (PR #1060), @abaybektursun (PR #549)
- **Retokenization pipeline**: Built for this submission
