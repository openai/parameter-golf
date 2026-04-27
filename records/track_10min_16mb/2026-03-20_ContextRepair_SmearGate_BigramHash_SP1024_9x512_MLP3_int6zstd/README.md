# Context-Repair SmearGate+BigramHash MLP3 int6+zstd

## Summary

Dense 9-layer 512-dim GQA transformer with SmearGate + BigramHash context features, trained on 8xH100 SXM for 600s (12,511 steps). Uses MLP_MULT=3 for dense capacity reinvestment, int6+zstd export with fp16 embedding passthrough. Total artifact: 14,999,691 bytes.

This submission was developed entirely through AI agent coordination by a surgeon with zero software engineering or ML research background, as an experiment in what frontier AI models can achieve when given proper tools and context.

## Approach

### Context-Feature Repair

During development, we identified a specific model failure: under sliding-window evaluation, tokens scored at shifted positions with carried context from prior windows produce dramatically worse predictions. We diagnosed this through a custom sliding diagnostic that splits evaluation into "carried-context" and "shared-context" halves and measures per-region BPB independently.

We found that adding SmearGate (learned gating over shifted input features) and BigramHash (hashed bigram embeddings providing explicit local context) repairs this failure surface. In controlled 1xH100 canary experiments with confound isolation (same 1-shard training substrate, features on vs off), the carried-context penalty shrank substantially, with the features-on canary showing a near-healthy carried-context gap versus a large penalty in the features-off control.

A full 8xH100 serious retrain produced a 14,999,691-byte artifact with corrected post-export exact 1.2006 BPB. Subsequent saved-checkpoint local sliding replay on the raw checkpoint scored 1.1646 BPB and showed carried-context improvement rather than collapse, suggesting the repair holds at scale.

### Evaluation Methodology

During development we found that the starter script's exact evaluation path currently scores exact BPB through compiled `model(x, y)`. On our serious checkpoint, that path reported an optimistic exact score, while fresh eager and fresh compiled logits-only replay agreed with each other at a materially different value. We therefore switched exact evaluation to `forward_logits` + manual `F.cross_entropy` and treat saved-checkpoint replay as the authoritative gate. All scores in this submission use this corrected evaluation path.

## Configuration

```
NUM_LAYERS=9
MODEL_DIM=512
NUM_HEADS=8
NUM_KV_HEADS=4
MLP_MULT=3
USE_SMEAR_GATE=1
BIGRAM_VOCAB_SIZE=4096
BIGRAM_DIM=128
QUANT_BITS=6
SERIAL_COMPRESSOR=zstd
EMBED_FP16=1
EVAL_STRIDE=0 (during training; sliding eval done post-hoc)
```

## Key Metrics

| Metric | Value |
|--------|-------|
| Pre-quant exact val_bpb | 1.1956 |
| Post-export int6+zstd roundtrip exact val_bpb | 1.20059549 |
| Saved-checkpoint exact replay (Apple MPS, float32) | 1.19503051 |
| Saved-checkpoint sliding replay (stride 512, Apple MPS) | 1.16457631 |
| Total submission bytes (int6+zstd) | 14,999,691 |
| Code bytes | 68,815 |
| Training steps | 12,511 |
| Training wallclock | 600.038s |
| Hardware | 8xH100 80GB SXM |

## Sliding Diagnostic (Carried-Context Repair)

| Region | Exact BPB | Sliding BPB | Delta |
|--------|-----------|-------------|-------|
| Carried-context half | 1.2253 | 1.1645 | -0.0609 |
| Shared-context half | 1.1647 | 1.1647 | +0.0000 |
| Aggregate | 1.1950 | 1.1646 | -0.0305 |

The model now benefits from carried context under sliding evaluation, confirming the context-feature repair.

## Known Limitations

- Sliding-window replay was performed locally on Apple MPS with float32 dtype, not on H100 with bfloat16.
- Post-export sliding replay has not yet been completed (raw sliding only).

## Included Files

- `train_gpt.py` — code snapshot used for the run
- `train.log` — training log from 8xH100 run
- `submission.json` — leaderboard metadata
