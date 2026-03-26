# H3: Per-Block Cadence Gradient (Shape of Recursive Pressure)

## Question
Should each crawler block in a recursive stack have its own cadence, and does
the optimal cadence *shape* differ by architecture?

## Motivation
H1 and H2 treat cadence as a global uniform value — every block in the crawler
stack gets the same C/N ratio. But the gradient pressure on each block is NOT
uniform. In a 3-block crawler stack:
- Block 0 receives the freshest representation from the flat layers
- Block 1 operates on a partially refined intermediate
- Block 2 produces the final output that feeds into the loss

These blocks face different optimization landscapes. A uniform cadence may be
leaving performance on the table by over-firing blocks that need rest or
under-firing blocks that need more refinement.

## Shapes to Test

**Funnel** (high cadence early, low late):
```
Block 0: cadence 1 (all C)    — aggressive front-loading
Block 1: cadence 2 (C/N)      — moderate
Block 2: cadence 4 (C/N/N/N)  — coast to output
```
Rationale: early blocks do heavy lifting to set up representations,
later blocks stabilize. Gradient interference decreases toward output.

**Pregnant / Diamond** (low edges, high middle):
```
Block 0: cadence 3             — light touch on input
Block 1: cadence 1 (all C)    — deliberation engine
Block 2: cadence 3             — light touch on output
```
Rationale: the middle of the stack is where recurrence has the most
room to explore. Edge blocks act as adapters.

**Inverse Funnel** (low early, high late):
```
Block 0: cadence 4             — let representations form
Block 1: cadence 2             — moderate
Block 2: cadence 1 (all C)    — hammer the final output
```
Rationale: let the input representation crystallize before applying
heavy recursive refinement. Final block closest to loss needs most
gradient signal.

**Uniform** (control — same as H1/H2 winner):
```
Block 0-N: cadence K           — whatever H1/H2 determines is best
```

## Architecture Dependence

The critical question: does the optimal shape change with stack depth?

- **2-block crawler (4x2)**: Only 2 blocks. Shape is essentially just
  "front vs back." Limited expressiveness.
- **3-block crawler (6x2)**: 3 blocks can express funnel, diamond, inverse.
  This is the minimum viable depth for shape experiments.
- **4+ blocks**: If we ever go here, the shape space explodes.

Prediction: 6x2 will show a measurable shape effect because 3 blocks is
enough internal structure for differentiated pressure. 4x2 may show nothing
because 2 blocks is too coarse.

## Prerequisites
- H1 + H2 results (establishes that cadence matters at all)
- Code change: per-block cadence support (`CRAWLER_CADENCE_PER_BLOCK=2,1,3`)
- Only run on 3f+3cx2 initially (3 blocks = minimum for shape)

## Status
BLOCKED — waiting on H1/H2 results and code change.

## Implications If Confirmed
- Cadence is not a scalar knob — it's a *vector* over the recursive stack
- Architecture design must co-optimize depth AND cadence shape
- The "pregnant shape" finding would suggest recursive transformers have
  an internal specialization structure analogous to encoder/bottleneck/decoder
- Opens research direction: can the model learn its own cadence schedule?
  (adaptive per-block gating of C vs N steps)

## Verdict
PENDING — blocked until per-block cadence support is implemented.
