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

## Results

| ID | Scales | Step avg (ms) | Raw val_bpb | INT6_SW_BPB | Quant gap | Delta |
|----|--------|:-------------:|:-----------:|:-----------:|:---------:|:-----:|
| CTRL | 1,1,1 | TBD | TBD | TBD | TBD | control |
| BWB-01 | 1,2,4 | TBD | TBD | TBD | TBD | TBD |
| BWB-02 | 1,3,9 | TBD | TBD | TBD | TBD | TBD |
| BWB-03 | 1,5,25 | TBD | TBD | TBD | TBD | TBD |
| BWB-04 | 9,3,1 | TBD | TBD | TBD | TBD | TBD |
| BWB-05 | 1,9,1 | TBD | TBD | TBD | TBD | TBD |
| BWB-06 | 1,1,9 | TBD | TBD | TBD | TBD | TBD |
| BWB-07 | 9,1,1 | TBD | TBD | TBD | TBD | TBD |

Reference: BW2-00 (XSA=11, no battery) → 1.52365
