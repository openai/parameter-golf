11L 512d int6+zstd, 7-gram cache, 100ep cosine TTT, GPTQ.

## setup

```bash
pip install -r requirements.txt
pip install flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291
python3 data/cached_challenge_fineweb.py --variant sp1024
```

## run

```bash
SEED=1337 NUM_LAYERS=11 MODEL_DIM=512 MLP_MULT=3.0 \
LEAKY_RELU=0.5 XSA_LAST_N=11 VRL_ENABLED=1 GATED_ATTN=1 \
BIGRAM_VOCAB_SIZE=4096 BIGRAM_DIM=128 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=0 \
ROPE_DIMS=16 LN_SCALE=1 \
LATE_QAT=1 QAT_ENABLED=0 \
TTT_ENABLED=1 TTT_LR=0.001 TTT_EPOCHS=100 TTT_COSINE=1 TTT_ADAMW=1 TTT_PER_LAYER_LR=1 TTT_ETA_MIN_RATIO=0.01 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3000 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
PRUNE_PCT=0.07 GPTQ_ENABLED=1 GPTQ_BATCHES=256 \
NGRAM_ALPHA=0.40 NGRAM_ORDER=7 NGRAM_BUCKETS=4000000 NGRAM_ADAPTIVE=1 NGRAM_CONF_SCALE=1 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## toggles

- `GPTQ_ENABLED=0` - skip hessian-aware quant, use naive per-row int6
- `NGRAM_ALPHA=0` - disable n-gram eval cache
- `NGRAM_CONF_SCALE=0` - disable count-confidence weighting
- `LEAKY_RELU=0` - standard ReLU (before squaring)
- `XSA_LAST_N=0` - no exclusive self-attention
- `VRL_ENABLED=0` - no value residual
- `GATED_ATTN=0` - no per-head attention gate
- `TTT_ENABLED=0` - skip test-time training
