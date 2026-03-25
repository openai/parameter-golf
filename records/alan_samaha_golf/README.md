# GolfStudent -- 16MB Hybrid LM for OpenAI Parameter Golf

**Score:** TBD | **Author:** Alan Samaha | **Params:** ~16.4M

## Architecture

| Component | Value |
|---|---|
| d_model | 288 |
| Layers | 14 (10 LinearRecurrence + 4 Attention) |
| Vocab | 1024 (sp1024, weight-tied) |
| Attention | Flash via SDPA + RoPE, every 3rd layer |
| Recurrence | Gated linear recurrence, O(L) time |
| FFN | SwiGLU (3x expansion) |
| Quantization | Per-row INT8 + zlib (contest format) |

## Key Techniques

- **Hybrid architecture**: Linear recurrence handles long-range state at O(L); attention provides global context at attention layers. Weight tying of embedding/output-projection recovers ~885KB vs 4096 vocab.
- **Muon optimizer**: Newton-Schulz orthogonalization for matrix gradients (same technique as leaderboard #1-3).
- **EMA checkpoint**: decay=0.999, updated every step. Export uses EMA weights (consistently better BPB than raw checkpoint).
- **Warmdown LR**: last 15% of wallclock budget, LR decays linearly to 0. Matches warmdown_iters=1200 in baseline.

## Running

```bash
# From openai/parameter-golf root, after downloading data:
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10

# Smoke test (1 GPU, 60s):
RUN_ID=smoke ITERATIONS=50 MAX_WALLCLOCK_SECONDS=60 VAL_LOSS_EVERY=0 \
  DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
  TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
  VOCAB_SIZE=1024 \
  torchrun --standalone --nproc_per_node=1 records/alan_samaha_golf/train_gpt.py

# Full 8xH100 run (10-min cap):
RUN_ID=run1 \
  DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
  TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
  VOCAB_SIZE=1024 \
  torchrun --standalone --nproc_per_node=8 records/alan_samaha_golf/train_gpt.py
```

## Requirements

```
sentencepiece>=0.1.99
torch>=2.2.0
numpy>=1.24.0
```
