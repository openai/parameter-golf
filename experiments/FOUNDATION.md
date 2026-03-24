# Foundation Hypothesis — Recursive Compressed Transformers

## Core Claim

Cadence is a primary control variable in recursive compressed transformers.
BPB is strongly shaped by the ratio of crawler-heavy steps (C) to normalization/clean steps (N).
The optimal cadence is likely architecture-dependent.

## Why This Matters

A recursive system (weight-shared crawler blocks looping multiple times) is fundamentally
different from a flat transformer. The C/N ratio controls:

- **Gradient interference** — C steps fire the crawler twice with bidirectional PD.
  High C-step ratio means gradients from both firings compete every step.
- **Refinement behavior** — More C steps = more double-firing consensus events.
  But diminishing returns if the ref can't absorb updates fast enough.
- **Quantization sensitivity** — GPTQ Hessian sees different activation distributions
  depending on whether the final step was C or N.
- **Convergence rate per wall-second** — C steps cost ~2x compute. Cadence 1
  (all C) gets ~1,200 steps in 150s. Cadence 4 gets ~1,700.

A 3cx2 system (6 effective recursive depth) may need a different rhythm than a 2cx2
(4 effective recursive depth) because more layers means more opportunities for
gradient interference AND more refinement capacity per firing.

## Decision Rule

No major new mechanism work until cadence/BPB laws are mapped.
Every near-term run must contribute to:
1. Defining cadence behavior, OR
2. Testing cadence portability across architectures

## Notation

- **4x2** = 2 crawler blocks x 2 loops = 4 effective recursive depth (RC-0: 4f+2cx2)
- **6x2** = 3 crawler blocks x 2 loops = 6 effective recursive depth (3f+3cx2)
- **Cadence N** = 1 C-step per N total steps (cadence 2 = C/N alternating)
- **C-step** = crawler double-fires, consensus blending, PD gradient flows both directions
- **N-step** = crawler single-fires, ref provides outbound gradient only

## Active Fronts

| Front | Question | Status |
|-------|----------|--------|
| **H1** | What does cadence do to BPB on a balanced 4x2 system? | **COMPLETE — recursion is overhead** |
| **H2** | Does optimal cadence change on a 6x2 system? | **COMPLETE — yes, 6x2 more sensitive** |
| **H3** | Should each crawler block have its own cadence (shape of recursive pressure)? | **DEPRIORITIZED — recursion itself is net negative** |
| **H4** | Does a crawler bank at the U-Net bottleneck improve GS v7? | READY |

## Measurement Protocol

Compare at **matched wall-clock**, not matched step count.

Diagnostics per arm:
- `fast_val_bpb` at steps 500, 1000, 1500
- `delib_scale` trajectory (is PD alive or dying?)
- `train_loss` on `is_crawl=1` vs `is_crawl=0` rows
- Final `sliding_window_bpb`, `post_ema_bpb`, `quant_gap`
- Total steps achieved in budget

Verdict thresholds:
- Delta >= 0.001 BPB = **significant**
- Delta 0.0005-0.001 = **marginal** (needs 0.50 confirmation)
- Delta < 0.0005 = **noise floor** (NEUTRAL)
