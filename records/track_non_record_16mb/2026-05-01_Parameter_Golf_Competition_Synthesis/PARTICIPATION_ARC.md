# Participation Arc: How the Research View Changed

This file is not meant as a personal victory lap.  It is included because the
conclusions in this package did not appear at once.  They came from a sequence
of record attempts, non-record studies, invalid/closed early ideas, and public
legality debates.

The useful signal is the evolution of the research method: from "find a clever
trick" toward "separate mechanism, legality, denominator, quantization, and
hardware-budget evidence."

## Chronological PR Map

| Date | PR | Type | What it tested | Lesson carried forward |
|---|---:|---|---|---|
| 2026-03-26 | #826 | closed record attempt | Order-9 n-gram backoff + score-first TTT + GPTQ-Int5 | Very large eval-time memory gains are easy to overstate; legality and scoring semantics matter first. |
| 2026-03-26 | #846 | closed record attempt | Two-pass n-gram rescoring | Two-pass / rescoring-style methods can produce spectacular numbers while violating the spirit or letter of single-pass evaluation. |
| 2026-03-28 | #1012 | non-record | JEPA-LM transfer from synthetic success to real language | Synthetic/local success does not imply FineWeb BPB transfer. |
| 2026-03-28 | #1013 | non-record | S4D-Lin / SSM hybrid after Mamba-style failures | SSM hybrids are interesting, but the 10min/16MB regime strongly favors well-optimized transformer lineages. |
| 2026-04-01 | #1227 | non-record | 28 experiments in 5 days | Small-scale tests lie unless they preserve the same bottleneck as the real run. |
| 2026-04-02 | #1259 | non-record | KNN hidden-state retrieval | Retrieval that helps weak models can disappear or reverse on strong contextual models. |
| 2026-04-03 | #1301 | non-record | Adapters on random linear maps, freeze schedules, 7 architecture variants | Mechanical novelty is cheap; frontier transfer is the hard part. |
| 2026-04-04 | #1341 | non-record | TTT/GPTQ interaction | Quantized structure can defeat naive adaptation; adaptation and quantization must be designed together. |
| 2026-04-15 | #1642 | non-record | Causal n-gram logit blend | Legal, bug-free eval-time memory can still be a null result at scale. |
| 2026-04-18 | #1716 | record attempt | SP8192 + BigramHash d=32 + Path-A-v3 passthrough quantization | Small causal input features can help when integrated into the right base; compression/code packing matters. |
| 2026-04-18 | #1718 | non-record | Eval-time lever ablations companion to #1716 | Negative ablations are necessary to avoid cargo-culting record ingredients. |
| 2026-04-30 | #1965 | record candidate | Long-context no-QV rank56/prefix3000 TTT | Tiny HP wins on some seeds can be wiped out by a tail seed; fixed seed policy matters. |
| 2026-05-01 | this PR | non-record synthesis | Final-day frontier transfer autopsy + competition-wide synthesis | The strongest final contribution is a mechanism-level map, not a marginal or disputed record claim. |

## The Through-line

### Phase 1: Fast cleverness was not enough

The earliest n-gram attempts produced dramatic-looking scores, but they also
made clear that a compression contest is mostly a measurement contest once
eval-time state is allowed.  That pushed my later work toward explicit C1-C4
checks and away from "the number looks good, therefore it is good."

### Phase 2: Local proxies had to earn trust

The JEPA, SSM, KNN, and random-adapter studies all had the same meta-lesson:
small or synthetic tests are useful only when they reproduce the limiting
factor of the real 8xH100, 600-second, 16 MB setting.  Otherwise they are idea
generators, not evidence.

### Phase 3: Compression and quantization became part of the model

PR #1716 was the point where the deployed artifact became the object of study:
BigramHash features, passthrough quantization choices, code packing, and byte
limits had to work together.

### Phase 4: Legality became a first-class research variable

By the final week, the main question was often not "does this reduce BPB?" but:

```text
What alphabet is being scored?
Was the distribution normalized?
Was the state strictly prefix-only?
Were bytes counted exactly once?
Was eval-time preprocessing inside the timer?
Were validation tokens adapted before being scored?
```

That is the research frame this closing synthesis uses.

### Phase 5: The final frontier taught a stop rule

The last 8xH100 attempts did not fail subtly.  They failed at pre-quant.  That
made the conclusion cleaner: in a mature compressed stack, downstream
quantization and legal TTT polish a strong model; they do not rescue a model
that is already much worse before quantization.

## Why this arc matters beyond one participant

This sequence mirrors the competition's own evolution:

1. clever eval-time hacks,
2. architecture exploration,
3. compression and quantization craft,
4. tokenizer/denominator disputes,
5. legality-aware systems thinking.

That systems view is the main thing I would carry forward into future small
model research.
