# Record: 0.0214 bpb - Low Eval-Time Memory Regime: Packed Training N-gram Artifact + Learned Gate (No Phrase Cache)

**Status:** finalized compliant 3-seed record folder.

**3-seed mean final val_bpb:** `0.02137047` (std `0.00002830`)

## Included Files

- `train_gpt.py`
- `requirements.txt`
- `submission.json`
- `PR_DRAFT.md`
- `logs/train_seed1337.log`
- `logs/train_seed42.log`
- `logs/train_seed7.log`

This folder intentionally does **not** bundle copied model weights. Artifact sizes are documented from the train logs.

## Verified Results

All numbers below are the final causal `final_int6_ttt_exact` result with the packed order-2..9 training cache loaded from the artifact at eval start and then updated online.

| Seed | Final val_bpb | Artifact bytes | Total bytes | Eval time | Notes |
|------|---------------|----------------|-------------|-----------|-------|
| 1337 | **0.02140207** | 14,868,762 | 15,029,658 | 391s | `USE_MIXER=0`, `USE_PHRASE_CACHE=0`, `TTT_EPOCHS=0` |
| 42 | **0.02134745** | 15,688,602 | 15,849,498 | 391s | `USE_MIXER=0`, `USE_PHRASE_CACHE=0`, `TTT_EPOCHS=0` |
| 7 | **0.02136190** | 15,201,862 | 15,362,758 | 390s | `USE_MIXER=0`, `USE_PHRASE_CACHE=0`, `TTT_EPOCHS=0` |

Final 3-seed mean final val_bpb: `0.02137047` with sample std `0.00002830`.

## Low Eval-Time Memory Regime

This variant keeps the packed order-2..9 training n-gram artifact and learned gate, but removes the two extra eval overlays that had been sitting on top:

1. **No logistic context mixer.**
2. **No long phrase cache.**

The remaining eval-time adaptation path is:

1. load the packed order-2..9 cache from the artifact,
2. score with the learned neural + n-gram gate,
3. apply online logit calibration,
4. update the streaming n-gram cache only after scoring.

The motivating ablation was immediate: on the final seed-7 no-mixer artifact, turning off only the long phrase cache dropped eval BPB from `0.04881917` to `0.02134985`, which then held up in the full 3-seed reruns above.

## Main Submission Shape

This submission keeps:

- packed order-2..9 training n-gram cache stored inside the artifact
- learned multi-expert gate over neural + order-2..9 n-gram experts
- online logit calibration
- cached-batch GPTQ export path

Compared with the earlier packed-cache submission, the final path removes:

- logistic context mixer
- long phrase cache
- bigram hash embedding
- heuristic / hybrid switching logic
- cache-maturity decay

## Why It Works

The packed training cache already gives the learned gate a strong warm-start low-order signal at eval step 0. In this setting, the extra eval-time overlays were not helping:

- the mixer overlapped heavily with the packed low-order n-gram signal
- the long phrase cache overrode the already-strong packed-cache probabilities in a way that significantly hurt final BPB

Removing both left a simpler, more memory-efficient eval path that also scored much better.

## Causal Evaluation Path

1. Load the packed training n-gram cache from the artifact itself.
2. Score the next validation chunk with only left context and the current cache state.
3. Query n-gram experts using only left context; expert availability depends only on context evidence.
4. Blend neural + n-gram experts and score the chunk before any mutation.
5. Update the streaming n-gram cache after scoring the chunk.
6. The reported runs use `TTT_EPOCHS=0`, so there is no backward adaptation step in the final path.

## Compliance

- **Single-pass eval:** this is not a 2-pass or rescoring method.
- **No future-token leakage:** validation chunks are scored before their tokens are added to the streaming cache.
- **Artifact-bundled warm start:** the cache loaded at eval step 0 is part of the artifact itself.
- **Packed cache is training-only:** the serialized n-gram payload comes from training data produced inside the 600 second training budget.
- **Context-only gate mask:** the learned gate does not use the true next token to decide which experts are available.
- **Cached GPTQ calibration:** quantization calibration uses batches already seen during training.
- **No backward TTT in final path:** the current reported numbers use `TTT_EPOCHS=0`.
- **Artifact under 16MB:** all three runs remain below the limit.

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

## Notes

- `logs/train_seed1337.log`, `logs/train_seed42.log`, and `logs/train_seed7.log` correspond to the final no-mixer / no-phrase compliant reruns.
- `submission.json` reflects the 3-seed mean and worst-case total size from this final path.
