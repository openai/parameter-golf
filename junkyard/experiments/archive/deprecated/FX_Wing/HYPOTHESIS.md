# FX-Wing — Instructed Recurrence: Hypothesis & Ablation Plan

## Core Hypothesis

**H0 (main):** Content-derived loop instructions allow shared crawler weights to behave
differently across iterations, resolving the gradient conflict that killed Frugendorff.

The flat encoder runs once and generates a per-token instruction vector for each loop
iteration. The crawler receives `x + inst[k]` where `inst[k]` is derived from the actual
token context — not a fixed learned scalar. This lets the model learn:
- Loop 1: "extract local syntactic signal for this token type"
- Loop 2: "integrate longer-range semantic context"

**Expected result:** FX-Wing (USE_CRAWLER=1, INST_DIM=32) beats the control
(USE_CRAWLER=0, same architecture otherwise) by ≥0.002 int6 BPB.

---

## Ablation Ladder (run in order)

### A1 — Control: no crawler, no instructions
```
USE_CRAWLER=0
```
Baseline GPT (flat blocks only). Establishes the floor.

### A2 — Frugendorff baseline: crawler + old fixed offsets
```
USE_CRAWLER=1  INST_DIM=0  CRAWLER_LOOPS=2
```
Equivalent to the original CrawlerGPT with orthogonal `loop_pos` vectors.
Tests whether the fix (content-derived) actually helps vs. the legacy approach.

### A3 — FX-Wing: crawler + content-derived instructions (main hypothesis)
```
USE_CRAWLER=1  INST_DIM=32  CRAWLER_LOOPS=2
```
The new architecture. Should beat A1 and A2.

### A4 — Instruction bottleneck width
```
INST_DIM=16  (narrow)
INST_DIM=64  (wide)
```
How much information needs to flow from encoder to crawler per iteration?
If 16 matches 32 → the signal is low-dimensional, instructions are simple.
If 64 > 32 → richer instructions help, consider going wider.

### A5 — More loops
```
CRAWLER_LOOPS=3  INST_DIM=32
```
With instructions, can we get more out of additional recurrence?
(Was useless in Frugendorff — the conflict got worse with more loops.)

### A6 — More crawler blocks
```
NUM_CRAWLER_LAYERS=2  CRAWLER_LOOPS=2  INST_DIM=32
```
Deeper shared section vs. more loops of a single block.

---

## Further Research Directions (if A3 confirms)

### FX-1: Gated Instructions
Add a learned sigmoid gate on the instruction offset:
```
g = sigmoid(gate_proj(inst[k]))
x_loop = x + g * offset
```
Gate learns to suppress instructions for tokens where the encoder is confident
and the crawler should just pass through.

### FX-2: Asymmetric Instruction Depth
Generate instructions not just from the final encoder state but from each
flat encoder layer separately. Loop k uses the output of encoder layer k.
```
inst[k] = proj(flat_encoder_layer[k].output)
```
Forces a direct correspondence between encoder depth and crawler iteration.

### FX-3: Bidirectional Instruction Flow
After each crawler loop, let the crawler's output modulate the *decoder* flat blocks
via a symmetric instruction channel. The encoder plans → crawler acts → decoder refines.

### FX-4: Instruction Diversity Regularization
Add a cosine similarity penalty between `inst[0]` and `inst[1]` to encourage
the two loop instructions to be genuinely different (not collapse to same behavior).
Prevents the model from learning trivial near-identical instructions.

### FX-5: Scale Up
If FX-Wing works at 5-min validation scale, run a full 10-min 8xH100 training run
with the best A-series config. This becomes the new submission candidate.

---

## Decision Criteria

| Result | Next Step |
|--------|-----------|
| A3 > A1 AND A3 > A2 | Confirmed. Run A4/A5/A6 for optimization. |
| A3 > A1 but A3 ≈ A2 | Instructions help but fixed offsets are good enough. Keep FX-Wing for novelty. |
| A3 ≈ A1 | Architecture neutral. Recurrence gives nothing at this scale. Park FX-Wing. |
| A3 < A1 | Regression. Debug: check instructions aren't collapsing to zero (init issue). |
