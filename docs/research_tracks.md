# Parameter Golf Research Tracks

Priority order is dictated by the challenge rules:

1. stay under the `16,000,000` byte artifact cap
2. stay within the `10 minute / 8xH100` training budget for record attempts
3. optimize post-roundtrip `val_bpb`, not pre-quant loss

## Integrated now

- Post-compression-aware training:
  - sampled int8 reconstruction regularizer
  - optional ternary-weight regularizer
  - optional outlier suppression penalty
- Weight sharing / recurrence:
  - shared-block transformer via `NUM_UNIQUE_BLOCKS`
- Factorized embeddings:
  - optional `EMBED_DIM < MODEL_DIM`
- Hybrid eval-time compute:
  - optional recent-token cache bias during validation / roundtrip eval
- Local proxy iteration:
  - capped validation
  - optional skip of expensive final roundtrip eval
  - proxy sweep launcher

## Current knobs

- `NUM_UNIQUE_BLOCKS`
- `EMBED_DIM`
- `COMPRESSION_REG_WEIGHT`
- `TERNARY_REG_WEIGHT`
- `OUTLIER_REG_WEIGHT`
- `EVAL_CACHE_MIX_WEIGHT`
- `EVAL_CACHE_SIZE`
- `FINAL_ROUNDTRIP_EVAL`
- `ROUNDTRIP_VAL_MAX_TOKENS`

## Local proxy reference point

All local comparisons below use the same quick 3090 proxy envelope:

- `MAX_WALLCLOCK_SECONDS=180`
- `TRAIN_BATCH_TOKENS=32768`
- `VAL_MAX_TOKENS=1048576`
- `FINAL_ROUNDTRIP_EVAL=0`
- baseline architecture:
  - `NUM_LAYERS=12`
  - `NUM_UNIQUE_BLOCKS=12`
  - `MODEL_DIM=384`
  - `EMBED_DIM=0`
  - `NUM_HEADS=6`
  - `NUM_KV_HEADS=3`

## Roundtrip proxy track

Use this when ranking experiments on a more faithful local objective:

- keep the same baseline architecture unless explicitly testing architecture
- enable `FINAL_ROUNDTRIP_EVAL=1`
- keep `ROUNDTRIP_VAL_MAX_TOKENS` capped so the run stays practical on a 3090
- treat this as the local approximation to the actual challenge metric

## Latest findings

- Quick local baseline:
  - run: `baseline3090_20260318_170251`
  - result: `val_bpb=2.0916`, `val_loss=3.4910`
  - total artifact: `6,831,983` bytes
  - interpretation: current local number to beat
- Hybrid eval sidecar, recent-token + bigram continuation bias:
  - run: `sidecar3090_20260318_172524`
  - knobs: `EVAL_CACHE_MIX_WEIGHT=0.03`, `EVAL_BIGRAM_MIX_WEIGHT=0.05`, `EVAL_CACHE_SIZE=16`
  - result: `val_bpb=2.0970`, `val_loss=3.5000`
  - total artifact: `6,810,819` bytes
  - delta vs baseline: `+0.0054 bpb` worse, `21,164` bytes smaller
  - interpretation: close enough to keep around for later tuning, not good enough to become the default path
- Compression-aware baseline, reconstruction regularization `0.01`:
  - run: `compress3090_20260318_174132`
  - result: `val_bpb=2.0943`, `val_loss=3.4954`
  - total artifact: `6,812,935` bytes
  - delta vs baseline: `+0.0027 bpb` worse, `19,048` bytes smaller
  - interpretation: strongest experimental branch so far
- Compression-aware baseline, reconstruction regularization `0.005`:
  - run: `compress3090_half_20260318_1750`
  - result: `val_bpb=2.0928`, `val_loss=3.4930`
  - total artifact: `6,829,073` bytes
  - delta vs baseline: `+0.0012 bpb` worse, `2,910` bytes smaller
  - interpretation: best pre-roundtrip proxy result outside the plain baseline
- Matched roundtrip-proxy baseline:
  - run: `baselinert3090_20260318_181344`
  - exact final roundtrip result: `val_bpb=2.11089617`, `val_loss=3.56464830`
  - total artifact: `6,705,058` bytes
- Matched roundtrip-proxy compression baseline:
  - run: `compressrt3090_20260318_175828`
  - knobs: `COMPRESSION_REG_WEIGHT=0.005`
  - exact final roundtrip result: `val_bpb=2.06085837`, `val_loss=3.48014999`
  - total artifact: `6,839,798` bytes
  - delta vs matched roundtrip baseline: `-0.05003780 bpb`, about `2.37%` better
  - interpretation: compression-aware training is now the leading local research branch when measured on a more faithful objective

## Immediate next step

- Sweep around `COMPRESSION_REG_WEIGHT=0.005` on the roundtrip proxy track
- add tiny `OUTLIER_REG_WEIGHT` values and keep the rest of the setup fixed
- rank experiments by `final_int8_zlib_roundtrip_exact val_bpb`

## Next experiments

- Zlib-aware QAT baseline:
  - rank on capped post-roundtrip proxy first
  - then sweep reconstruction / outlier penalties around the best roundtrip result
- Recurrent shared-block transformer:
  - vary `NUM_LAYERS`, `NUM_UNIQUE_BLOCKS`, `EMBED_DIM`
  - test whether smaller unique depth plus more effective depth improves proxy `val_bpb`
- Tiny hybrid sidecar model:
  - revisit only if a small weight sweep can push it under baseline
  - longer-term version should replace the current recency bias with a real adaptive mixer over:
    - recent-token cache
    - tiny n-gram model
    - neural logits

## Medium-term work

- Global/shared codebook quantization across layers
- Basis-generated per-layer weights or hypernetwork-style weight generation
- Test-time adaptation with strict reset semantics
- Token-adaptive recurrent depth / halting policy

## Deferred until the model is stronger

- Tokenizer redesign
- aggressive code-size golf
- heavy hyperparameter brute force
