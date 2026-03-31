# Learned Routing + Two-Pass N-gram Rescoring + Extended Orders

**val_bpb: TBD** | **~15.9 MB** | 8xH100 SXM

## Key Innovation: Combining Learned Routing with Two-Pass Rescoring

PR #834 introduced a learned `Linear(512->7)` routing head trained end-to-end against the mixer objective, but uses single-pass eval. PR #846 introduced two-pass rescoring (rescore cold-cache early chunks with the full cache) but uses a heuristic entropy-sigmoid alpha. This submission combines both: a learned routing head with two-pass rescoring, plus extended n-gram orders (2-12) and larger hash tables.

## Techniques

### Learned Multi-Expert Routing Head
- `Linear(512, 12)` head reads transformer hidden state
- Routes between 1 neural expert + 11 n-gram orders (2-12)
- Trained end-to-end with frozen n-gram oracle during training
- Masked softmax: invalid orders (insufficient context) masked to -inf
- Neural floor: 5% minimum weight on neural expert

### Two-Pass N-gram Rescoring
- Pass 1: Standard sequential chunk eval with causal cache building
- Pass 2: Rescore first 15 chunks with the full cache (no cache updates)
- Early chunks improve dramatically (from ~1.15 BPB to ~0.12 BPB)
- Adds ~50-60s to eval time

### Extended N-gram Orders (2-12)
- 11 n-gram expert orders vs 6 (PR #834) or 8 (PR #846)
- 8M bucket hash tables (vs 1M or 4M) for fewer collisions
- Per-order min_count thresholds

### TTT -> N-gram Pipeline
- TTT adapts model weights on already-scored chunks
- N-gram eval uses TTT-adapted weights (not base model)
- Better neural expert contribution in the mixture

## Architecture

PR #834/414 stack:
- 11 layers, 512d, 8H, 8KV
- LeakyReLU(0.5)^2 MLP (3.5x)
- U-Net skip connections, SmearGate, BigramHash(6144)
- Partial RoPE (16/64), LN Scale, XSA on all layers
- VE128 on layers 9-10
- EMA(0.997) + Tight SWA
- GPTQ int5 + zstd-22, 3% pruning
- Late QAT with Soft-Round STE + CROWN-Q

## Run Command

```bash
TWO_PASS_ENABLED=1 TWO_PASS_RESCORE_CHUNKS=15 \
NGRAM_MAX_ORDER=12 NGRAM_BUCKETS=8388608 \
TTT_TO_NGRAM=1 \
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=6144 XSA_LAST_N=11 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 LATE_QAT_THRESHOLD=0.5 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=1 TTT_LR=0.0005 TTT_EPOCHS=4 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=2 TTT_MOMENTUM=0.9 TTT_GRAD_CLIP=1.0 \
MIXER_LOSS_WEIGHT=0.1 MIXER_NEURAL_FLOOR=0.05 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Credits

- **Learned routing head + frozen oracle**: PR #834 by @AnirudhRahul
- **Two-pass rescoring**: PR #846 by @himanshudongre
- **Base architecture**: PR #414 by @signalrush, PR #549 by @abaybektursun
- **N-gram cache concept**: PR #659/#779 by @deanbrr
- **TTT recipe**: PR #461 by @Christopher-Lee-McClendon
- **LeakyReLU activation**: PR #493/#518 by @parinzee/@sofiabod
