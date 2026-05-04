# Two-Level Dirichlet Posterior + Per-Order OBCL + Phrase Cache

**val_bpb: 0.11556** (3-seed mean, std 0.0000057) | **~15.1 MB** | 8xH100 SXM

## Results (8xH100 80GB SXM, Rancho Cordova CA)

| Seed | Val BPB | Eval Time |
|------|---------|-----------|
| 1337 | 0.11555061 | 419s |
| 42 | 0.11556435 | 370s |
| 2025 | 0.11555875 | 359s |
| **Mean** | **0.11556 (std 0.0000057)** | |

## Architecture

EBLS: 3 shared transformer blocks looped 3x + 2 unique = 11 effective layers.
512d, 8 heads, 4 KV heads (GQA), MLP 3x with LeakyReLU(0.5)², per-virtual-layer LoRA rank 8.

## Key Techniques

- **Two-level Dirichlet smoothing** with per-order OBCL concentrations (50.0 for bigrams → 1.86 for 14-grams)
- **Phrase suffix matching** at probe lengths [20, 16] with Dirichlet concentration 1.0
- **15-gram backoff** (orders 2-15, 4M hash buckets)
- **Complementary training** (alpha=0.50, orders 2-5)
- **GPTQ int6 + LZMA** compression
- **EMA 0.997 + SWA** weight averaging
- **XSA** on all 11 layers

## Credits

Built on the shoulders of the community:
- @signalrush (PR #414) — GPTQ + EMA + warmdown foundation
- @Robby955 (PR #900) — Dirichlet smoothing, OBCL, phrase cache
- @himanshudongre (PR #846) — two-pass rescoring concept
- @deanbrr (PR #659) — original N-gram cache concept
- @newjordan (PR #674) — first legal implementation
- @pentxayc (PR #803) — complementary training

## Run Command

```bash
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 MAX_WALLCLOCK_SECONDS=560 XSA_LAST_N=11 \
WARMDOWN_ITERS=4000 CLIP_RANGE=31 COMPRESSOR=lzma \
NUM_KV_HEADS=4 EVAL_STRIDE=64 \
GPTQ_ENABLED=1 GPTQ_CALIB_BATCHES=64 GPTQ_CALIB_SOURCE=val \
GPTQ_BLOCK_SIZE=128 SWA_ENABLED=1 LATE_QAT_THRESHOLD=0.15 \
COMP_ENABLED=1 COMP_ALPHA=0.50 COMP_ORDER=5 COMP_WARMUP=200 COMP_MIN_COUNT=3 \
NGRAM_CACHE=1 NGRAM_ORDER=15 NGRAM_MIN_ORDER=2 \
NGRAM_BUCKETS=4194304 NGRAM_DIRICHLET=1 NGRAM_CONCENTRATION=5.0 \
NGRAM_TEMPERATURE=1.0 \
NGRAM_PER_ORDER_CONC="50.0,50.0,6.95,2.98,2.05,2.05,2.05,1.86,1.86,1.86,1.86,1.86,1.86,1.86" \
PHRASE_CACHE=1 PHRASE_BUCKETS=1048576 PHRASE_PROBE_LENGTHS=20,16 \
PHRASE_DIRICHLET=1 PHRASE_CONCENTRATION=1.0 PHRASE_MIN_COUNT=1 \
NCCL_TIMEOUT=3600 SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```
