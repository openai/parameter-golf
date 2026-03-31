# bandit_wagon_battery — Crawler as Sparse Attention Battery

## Background

The crawler loops 3× over the same bottleneck with identical causal attention in each
loop. All three loops compete for the same local context window. This forces the loops
into double duty: refining representations AND propagating information across distance
via multi-hop neighborhood aggregation. The distance propagation job is what causes
inter-loop activation distribution divergence — and that divergence is the quantization gap.

**Hypothesis:** By giving each loop a different RoPE frequency scale, the 3 loops
specialize into a multi-scale attention battery — each operating at a different temporal
resolution with a single shared weight budget. Distance propagation is handled directly
rather than emergently across hops.

## Mechanism: Per-Loop RoPE Scaling

Standard RoPE: `freqs = outer(positions, inv_freq)` where `inv_freq = base^(-2j/dim)`

Battery: `freqs = outer(positions, inv_freq / scale)` per loop

Dividing `inv_freq` by `scale` → **lower frequencies** → slower angular rotation between
positions → **wider effective attention range**.

```
scale=1: standard local attention   (loop 0 — high frequency, dense local)
scale=3: inv_freq/3 → 3× wider     (loop 1 — medium frequency, phrase/clause level)
scale=9: inv_freq/9 → 9× wider     (loop 2 — low frequency, sentence/paragraph level)
```

**Zero additional parameters.** Just a change to cos/sin computation per loop.

## Why This Attacks The Quantization Gap

With standard attention, inter-loop distributions diverge chaotically because loop N
is processing the accumulated errors of loops 0..N-1 while also doing distance propagation.
The distributions are unpredictably different.

With the battery, inter-loop distributions are **structurally different by design**:
- Loop 0 always carries high-frequency local signal
- Loop 2 always carries low-frequency long-range signal

A single int8 scale covering "local texture" vs "global structure" is far more tractable
than covering "progressively corrupted same-scale features."

**Additionally:** if this works, raw val_bpb should ALSO improve — not just quant gap.
This is the tell. If both metrics move, the battery is improving learning efficiency,
not just quantization robustness.

## Arms (also covered in run_all_ablations.sh)

| ID | Loop 0 | Loop 1 | Loop 2 | Purpose |
|----|:------:|:------:|:------:|---------|
| BWB-00/CTRL | 1 | 1 | 1 | **Control** |
| BWB-01 | 1 | 2 | 4 | Gentle ascending — powers of 2 |
| BWB-02 | 1 | 3 | 9 | **Core hypothesis** — moderate ascending |
| BWB-03 | 1 | 5 | 25 | Aggressive ascending |
| BWB-04 | 9 | 3 | 1 | Descending — global→local order |
| BWB-05 | 1 | 9 | 1 | Middle loop wide only |
| BWB-06 | 1 | 1 | 9 | Final loop wide only |
| BWB-07 | 9 | 1 | 1 | First loop wide only |

## Results — Mega Ablation (seed=444, 500 steps, 80 shards)

CTRL-00 = 1.44184974. Threshold to qualify: beat control by ≥0.005.

### BWC — Flat Choke Sweep

| ID | Choke dim | Step avg | Raw BPB | INT6_SW_BPB | Quant gap | Delta |
|----|:---------:|:--------:|:-------:|:-----------:|:---------:|:-----:|
| CTRL-00 | — | 545ms | 1.4414 | 1.44185 | +0.0004 | control |
| BWC-01 | 32 | 527ms | 1.4501 | 1.45004 | -0.0001 | +0.00819 |
| BWC-02 | 128 | 524ms | 1.4358 | 1.43674 | +0.0009 | **-0.00511** |
| BWC-03 | 256 | 540ms | 1.4398 | 1.44071 | +0.0009 | -0.00114 |
| BWC-04 | 512 | 582ms | 1.4298 | **1.42887** | -0.0009 | **-0.01298** |

### BWS — Loop Smear

| ID | Config | Step avg | Raw BPB | INT6_SW_BPB | Quant gap | Delta |
|----|--------|:--------:|:-------:|:-----------:|:---------:|:-----:|
| BWS-01 | smear=1 | 585ms | 1.4440 | 1.44628 | +0.0023 | +0.00443 |

Dead. Smear gate hurts.

### BWT — Encoder Tap Sweep

| ID | Config | Step avg | Raw BPB | INT6_SW_BPB | Quant gap | Delta |
|----|--------|:--------:|:-------:|:-----------:|:---------:|:-----:|
| BWT-01 | tap=32 shared all | 534ms | 1.4336 | 1.43227 | -0.0013 | **-0.00958** |
| BWT-02 | tap=32 per-loop all | 530ms | 1.4404 | 1.44133 | +0.0009 | -0.00052 |
| BWT-03 | tap=16 per-loop all | 532ms | 1.4348 | 1.43268 | -0.0021 | **-0.00917** |
| BWT-04 | tap=64 per-loop all | 532ms | 1.4434 | 1.44346 | +0.0001 | +0.00161 |
| BWT-05 | tap=32 per-loop **deep** | 531ms | 1.4317 | **1.43004** | -0.0017 | **-0.01181** |
| BWT-06 | tap=32 per-loop shallow | 533ms | 1.4343 | 1.43322 | -0.0011 | **-0.00863** |

Best tap: BWT-05 (deep, per-loop, tap=32) → -0.01181. Deep encoder layers beat shallow.
Tap=64 hurts (+0.00161). Sweet spot: tap=16–32. Shared tap (BWT-01) competitive with per-loop.

### BWB — Battery (Per-Loop RoPE Scale)

| ID | Scales | Step avg | Raw BPB | INT6_SW_BPB | Quant gap | Delta |
|----|--------|:--------:|:-------:|:-----------:|:---------:|:-----:|
| BWB-01 | **1,2,4** | 524ms | 1.4387 | **1.43769** | **-0.0010** | -0.00416 |
| BWB-02 | 1,3,9 | 524ms | 1.4419 | 1.44470 | +0.0028 | +0.00285 |
| BWB-03 | 1,5,25 | 517ms | 1.4424 | 1.44283 | +0.0004 | +0.00098 |
| BWB-04 | **9,3,1** | 527ms | 1.4415 | 1.44156 | **+0.0001** | -0.00029 |
| BWB-05 | 1,9,1 | 515ms | 1.4419 | 1.44237 | +0.0005 | +0.00052 |
| BWB-06 | 1,1,9 | 516ms | 1.4453 | 1.44797 | +0.0027 | +0.00612 |
| BWB-07 | 9,1,1 | 521ms | 1.4433 | 1.44355 | +0.0003 | +0.00170 |

No battery arm clears the 0.005 threshold standalone on flat MLP.
Best raw BPB: BWB-01 (1,2,4). Best quant gap: BWB-01 (-0.0010), BWB-04 (9,3,1, +0.0001).

**Key finding — wide-LAST is poisonous:** 1,1,9 (+0.0027) and 1,3,9 (+0.0028) both blow
the quant gap. Wide-FIRST (9,3,1, 9,1,1) keeps distributions convergent. Descending order
is architecturally more compatible with flat MLP quantization than ascending.

**Interpretation:** Battery alone cannot override flat MLP's single-scale quantization
constraint. 1,3,9 needs pyramid's per-loop routing to absorb distribution divergence.
BWCB + BWCD series test battery on pyramid-512 to validate this coupling.

Reference: BW2-00 (XSA=11, no battery) → 1.52365

## Phase 2 — Loop-Matched Skipgram Features (not yet built)

BWB Phase 1 (this series) tests only the attention-side temporal specialization (per-loop
RoPE scaling). The input feature side is still unspecialized: the bigram hash table feeds
distance-1 features equally to all three loops regardless of their causal horizon.

**The mismatch:**
- Loop 0: RoPE scale=1 (local) + bigrams at distance 1 → aligned
- Loop 1: RoPE scale=3 (medium) + bigrams at distance 1 → MISMATCHED
- Loop 2: RoPE scale=9 (distant) + bigrams at distance 1 → MISMATCHED

**Phase 2 hypothesis:** Pair each loop's RoPE scale with skipgram features at the matching
skip distance. Loop 1 gets skip-3 features. Loop 2 gets skip-9 features. Both the attention
mechanism AND the input representation are tuned to the same temporal resolution per loop.

**Prerequisite:** BWB Phase 1 must confirm that RoPE-only specialization helps before
adding the feature-side component. Phase 2 is a follow-on series (BWB-P2), not part of
the current mega ablation.
