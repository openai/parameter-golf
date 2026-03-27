# Record: 0.0498 bpb - Packed Training N-gram Artifact + Learned Weighting Gate

**Status:** finalized 3-seed record folder.

**3-seed mean final val_bpb:** `0.04983971` (std `0.00009313`)

## Included Files

- `train_gpt.py`
- `requirements.txt`
- `submission.json`
- `logs/train_seed1337.log`
- `logs/train_seed42.log`
- `logs/train_seed7.log`

This folder intentionally does **not** bundle copied model weights. The artifact sizes are documented from the train logs, which is what the submission README requirements ask for.

## Results So Far

All BPB numbers below are the final causal `final_int6_ttt_exact` result with the packed training n-gram artifact loaded at eval start and then updated online.

| Seed | Final val_bpb | Model bytes | Artifact bytes | Eval time | Notes |
|------|---------------|-------------|----------------|-----------|-------|
| 1337 | **0.04980772** | 15,506,861 | 15,667,164 | 604s | packed 32K training cache |
| 42 | **0.04994462** | 15,697,568 | 15,857,871 | 603s | packed 32K training cache |
| 7 | **0.04976679** | 15,413,928 | 15,574,231 | 602s | packed 32K training cache |

Final 3-seed mean final val_bpb: `0.04983971` with sample std `0.00009313`.

Each artifact packs the order-2..9 training n-gram cache into the submission itself using 32K buckets with 32-bit count tables (`2,097,152` raw bytes), so eval starts with a pre-warmed n-gram cache instead of an empty one. This matters because the learned weighting gate is trained against a fully warm oracle and relies heavily on strong low-order n-gram coverage, especially the order-2 / bigram-level signal inside the packed cache.

## Key Changes

This submission starts from PR `#880`'s single-pass stack and keeps:

- score-first TTT
- backward-looking order-2..9 n-gram cache
- long phrase cache overlay
- online logit calibration
- GPTQ/CROWN-Q export path

The main modifications are:

1. Replace the hand-written n-gram blend with a learned multi-expert gate (`alpha_head`) over the neural model probability plus n-gram experts for orders 2 through 9.
2. Prefill a frozen order-2..9 oracle from the full 8B-token training stream inside the 600 second training budget.
3. Serialize the compact 32K-bucket training cache into the artifact as 32-bit count tables so evaluation starts pre-warmed.
4. Keep only one eval path: packed cache loaded at step 0, causal online cache updates after scoring, and score-first TTT.
5. Remove the bigram hash embedding to recover artifact headroom for the packed cache, since the packed n-gram artifact now provides the warm low-order / bigram-level signal the learned gate wants at eval start.
6. Remove the earlier cache-maturity decay and hybrid/heuristic switching logic so the cleaned-up submission matches the intended behavior directly.

## Bucket-Size Ablation

In our ablations, smaller n-gram hash tables tended to do better than larger ones for this learned-gate setup. In particular, the 32K bucket setting consistently beat the larger bucket sizes we tested and also made it feasible to pack the training cache into the artifact while staying under the 16MB limit.

That made 32K the best practical operating point:

- better BPB than the larger bucket variants we tried
- small enough packed payload to fit in the artifact
- warm order-2..9 cache available immediately at eval start

## Training

During training, each batch queries the frozen training oracle for per-order probabilities and validity masks. The model predicts gate logits from hidden states, mixes the neural and n-gram experts with a masked softmax plus a neural floor, and adds `-log(p_mix)` alongside the standard language-model CE loss.

The frozen oracle is built once from the entire training stream and then kept read-only during optimization.

## Evaluation

Evaluation uses one simplified path only:

1. Load the packed training n-gram cache from the artifact.
2. Score the next validation chunk with the current model and current cache state.
3. Update the cache only after the chunk has been scored.
4. Run TTT only after that chunk has already been scored.

There is no heuristic/learned switch and no cache-maturity decay in this cleaned-up version.

## Compliance

- **Single-pass eval:** this is not a 2-pass or rescoring method.
- **No future-token leakage:** validation chunks are scored before their tokens are added to the streaming cache.
- **Packed cache is training-only:** the serialized n-gram payload comes from training data produced inside the 600 second training budget.
- **Score-first TTT:** each chunk is evaluated before any adaptation on that chunk.
- **Artifact under 16MB:** both completed seeds are below the limit.

## Reproduction

```bash
pip install -r requirements.txt

SEED=1337 \
ARTIFACT_NGRAM_EXPORT=1 \
MAX_WALLCLOCK_SECONDS=600 \
USE_MIXER=1 MIXER_ETA=0.1 MIXER_HEAD=multi \
USE_NGRAM_CACHE=1 NGRAM_EVAL_ORDER=9 \
TRAIN_ORACLE_BUCKETS=32768 NGRAM_EVAL_BUCKETS=32768 \
USE_PHRASE_CACHE=1 USE_REGIME_TRACKER=0 USE_LOGIT_CAL=1 \
TTT_EPOCHS=2 TTT_FREEZE_BLOCKS=2 TTT_LR=0.0001 \
TTT_CHUNK_TOKENS=131072 EVAL_STRIDE=64 TTT_TEMPERATURE=0.85 \
CROWN_Q_LAMBDA=0.01 PRUNE_PCT=0.05 BIGRAM_VOCAB_SIZE=0 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Notes

- `logs/train_seed1337.log`, `logs/train_seed42.log`, and `logs/train_seed7.log` contain the full training histories for the 3 completed seeds.
- `submission.json` now reflects the completed 3-seed result.
