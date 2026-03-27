## Title

Record: 0.0214 bpb - Low Eval-Time Memory Regime: Packed Training N-gram Artifact + Learned Gate (No Phrase Cache)

## Body

**3-seed mean val_bpb = 0.02137047 +/- 0.00002830** | **15.85 MB max total size**

All within budget: training < 600s, eval < 600s, artifact < 16MB.

## Summary

- Keep the packed order-2..9 training n-gram artifact and learned weighting gate over the neural model plus n-gram experts.
- Remove the logistic context mixer and long phrase cache from the final eval path, leaving a simpler low eval-time memory regime built around the packed cache, learned gate, and online logit calibration.
- Keep the compliant causal path: context-only gate validity, cached-batch GPTQ calibration, packed cache loaded from the artifact itself, and `TTT_EPOCHS=0`.

## Results

Current completed runs:

| Seed | Final val_bpb | Artifact bytes | Total bytes | Eval time | Notes |
|------|---------------|----------------|-------------|-----------|-------|
| 1337 | 0.02140207 | 14,868,762 | 15,029,658 | 391s | `USE_MIXER=0`, `USE_PHRASE_CACHE=0`, `TTT_EPOCHS=0` |
| 42 | 0.02134745 | 15,688,602 | 15,849,498 | 391s | `USE_MIXER=0`, `USE_PHRASE_CACHE=0`, `TTT_EPOCHS=0` |
| 7 | 0.02136190 | 15,201,862 | 15,362,758 | 390s | `USE_MIXER=0`, `USE_PHRASE_CACHE=0`, `TTT_EPOCHS=0` |

Final 3-seed mean final val_bpb: `0.02137047` with sample std `0.00002830`.

## Low Eval-Time Memory Regime

- No logistic context mixer at eval time.
- No long phrase cache at eval time.
- The remaining eval-time adaptation path is the packed order-2..9 n-gram cache from the artifact, causal online n-gram updates, and online logit calibration.
- This removes the large auxiliary GPU mixer tables from the previous variant while preserving the packed-cache scoring path.
- On the final seed-7 no-mixer artifact, disabling only the long phrase cache already improved eval BPB from `0.04881917` to `0.02134985`, which motivated the 3-seed rerun.

## Causal Inference Scheme

1. Start eval by deserializing the packed order-2..9 n-gram cache from the submitted artifact itself.
2. For each validation chunk, run the model once using only left context and the current packed-cache state.
3. Query n-gram experts from the current cache using left context only; expert availability depends only on context evidence, not on the true next token.
4. Blend neural + n-gram experts and score the chunk before any mutation of cache or model state.
5. After scoring, append the chunk tokens to the streaming n-gram cache for future chunks.
6. The reported final path uses `TTT_EPOCHS=0`, so there is no backward adaptation step in the submission path.

## Key Changes

- Packed order-2..9 training n-gram cache embedded into the artifact itself.
- Learned weighting gate over neural + order-2..9 n-gram experts.
- Bigram hash embedding removed to create artifact headroom for the packed cache.
- Logistic context mixer removed from the final eval path.
- Long phrase cache removed from the final eval path.
- Context-only gate validity retained.
- GPTQ calibration still uses cached training batches from the same timed run.

## Compliance

- This is **not a 2-pass method**.
- Validation is scored in a **single causal pass**: each chunk is scored before that chunk is used for cache updates.
- The warm-start n-gram cache used at eval step 0 is **part of the artifact itself**, not a separate runtime input.
- The packed n-gram cache in the artifact is derived from **training data only** and is produced within the 600 second training budget.
- The learned gate does **not** use the true next token to decide which experts are available.
- GPTQ calibration runs inside the reserved pre-export budget using cached training batches from the same timed run.
- The current reported numbers use `TTT_EPOCHS=0`.

## Reproduction

```bash
pip install -r requirements.txt

SEED=1337 \
DATA_PATH=/root/parameter-golf/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/root/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
ARTIFACT_NGRAM_EXPORT=1 \
MAX_WALLCLOCK_SECONDS=600 \
VAL_LOSS_EVERY=0 \
USE_MIXER=0 USE_PHRASE_CACHE=0 MIXER_HEAD=multi \
USE_NGRAM_CACHE=1 NGRAM_EVAL_ORDER=9 \
TRAIN_ORACLE_BUCKETS=32768 NGRAM_EVAL_BUCKETS=32768 \
USE_REGIME_TRACKER=0 USE_LOGIT_CAL=1 \
TTT_EPOCHS=0 TTT_FREEZE_BLOCKS=2 TTT_LR=0.0001 \
TTT_CHUNK_TOKENS=131072 SKIP_SLIDING=1 EVAL_STRIDE=64 TTT_TEMPERATURE=0.85 \
CROWN_Q_LAMBDA=0.01 PRUNE_PCT=0.05 BIGRAM_VOCAB_SIZE=0 \
GPTQ_CALIBRATION_SEQS=128 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```
