# Cubric N-gram Accumulator

## Concept

The neural model's n-gram interpolation uses a fixed alpha range adapted by model
entropy. The Cubric accumulator makes this **temporally adaptive** — it tracks how
well the n-gram is predicting on the current document and shifts the alpha bounds
accordingly.

## Mechanism

After each scored segment (score-first legal):
1. Measure: did the n-gram blend improve or hurt NLL vs pure model?
2. EMA-update a `cubric_reliability` signal (positive = helping, negative = hurting)
3. Shift `alpha_min` and `alpha_max` by `reliability * boost_scale`

Early in eval: few n-gram counts, reliability ≈ 0, alpha = base settings.
As eval progresses: n-gram tables fill, reliability signal grows, alpha adapts
to the document's actual n-gram predictability.

## Parameters

| Param | Default | Meaning |
|-------|---------|---------|
| CUBRIC_ENABLED | 0 | Turn accumulator on/off |
| CUBRIC_DECAY | 0.95 | EMA decay for reliability (higher = more memory) |
| CUBRIC_BOOST_SCALE | 0.15 | Max alpha shift from accumulator |

## Legality

Score-first compliant. The accumulator only reads from already-scored segments.
Alpha adjustment depends on model output + past n-gram performance, never on
future targets.

## Running

```bash
bash concepts/cubric_ngram/run_ab.sh
```
