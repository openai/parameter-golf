# Int8 Bigram+Cache Prototype (16MB / 10min)

## Summary
- 12 logical layers built from 2 shared Transformer blocks (d_model 640, 8 MQA/GQA heads, 4× MLP, RMSNorm, GELU, RoPE/ALiBi, tied embeddings).
- LSQ-lite QAT on all linears (per-row scales) to align training with int8 export.
- Inference fusion: KN-smoothed bigram prior (~4MB uint32) + short-context cache mixture; logits = model + λ_bigram·logP_bigram + λ_cache·logP_cache.
- Regularization + stability: label smoothing 0.05, EMA 0.999 tail, optional SWA tail.
- Target: ≤16,000,000 bytes artifact (code + int8 weights + priors) and <10 min train on 8×H100.

## Training recipe
```
RUN_ID=int8_bigram_proto \
NUM_LAYERS=12 SHARED_BLOCKS=2 MODEL_DIM=640 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=4 \
TRAIN_SEQ_LEN=2048 TRAIN_BATCH_TOKENS=1048576 ITERATIONS=9000 WARMUP_STEPS=2000 \
LSQ_QAT=1 LSQ_PER_ROW=1 LABEL_SMOOTHING=0.05 EMA_DECAY=0.999 SWA_STEPS=300 \
ENABLE_BIGRAM_PRIOR=1 BIGRAM_LAMBDA=0.3 BIGRAM_SMOOTHING=0.1 CACHE_LAMBDA=0.05 CACHE_SIZE=256 \
MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```
- Optimizer: AdamW β1=0.9 β2=0.95 wd=0.05; grad_clip=1.0; warmup to LR 3e-3 then cosine.
- FlashAttention2 + compiled model enabled; ~1M tokens/step target.

## Export
- Per-row int8 quantization + zlib, plus embedded extras: bigram counts, λs, shared/QAT flags.
- Loader dequantizes to bf16; bigram counts reduced across ranks and rehydrated at eval.

## Status
- Code path implemented and compiled locally; full 8×H100 10-minute run not yet executed (seeking compute grant). Placeholder log included.

## Files
- `train_gpt.py`: training + export + priors.
- `train.log`: placeholder for future 10-min run.
- `submission.json`: metadata; val_bpb TBD pending run.
