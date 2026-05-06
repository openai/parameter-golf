# Parameter Golf — Fractal Transformer Research Plan
**DGX Spark · GB10 · March 2026**

---

## Challenge Summary

| Constraint | Value |
|------------|-------|
| Artifact size | ≤16MB (code + int8 quantized + zlib compressed weights) |
| Training time | ≤10 minutes on 8×H100 |
| Metric | bits-per-byte (BPB) on FineWeb validation set |
| Baseline | 1.2244 BPB |
| Record threshold | ≤1.2194 BPB (must beat by ≥0.005) |
| 4-hour unlimited baseline | 1.2074 BPB |
| Challenge window | March 18 → April 30, 2026 |
| Repo | https://github.com/newjordan/parameter-golf |

---

## Our Approach: Fractal Transformer + Gravity + AttnRes

### Core Thesis

Weight-shared transformer layers with learned gravitational auxiliary losses
and attention residuals will achieve lower BPB than the baseline's 9-unique-layer
architecture within the same 16MB parameter budget.

### Three Innovations Combined

**1. Fractal Architecture (Weight Sharing / Depth Recurrence)**

Instead of 9 unique layers, use 3 unique layers repeated in 3 loops.

```
CURRENT BASELINE:
  9 unique layers × 512 dim = ~14M params

OUR APPROACH:
  3 unique layers × 3 loops = 9 effective layers
  Wider layers (~700 dim) with same total param count
  Loop position embedding tells shared weights which pass they're on
```

Why this helps:
- Fewer unique parameters → more room in 16MB budget → wider layers
- Wider layers = richer features per layer
- Weight sharing compresses extremely well under int8+zlib
- Depth recurrence explicitly encouraged by the challenge README

**2. Gravity (Learned Auxiliary Losses)**

At the end of each loop, peek at the output using the shared lm_head and
compute an auxiliary cross-entropy loss. The weights are LEARNED parameters.

```python
self.gravity_weights = nn.Parameter(torch.tensor([0.1, 0.3, 1.0]))

total_loss = 0
for loop in range(3):
    x = run_shared_layers(x, loop_pos=loop)
    loop_logits = lm_head(rms_norm(x))
    loop_loss = cross_entropy(loop_logits, targets)
    total_loss += softplus(self.gravity_weights[loop]) * loop_loss
```

Why this helps:
- 3× gradient signal — every layer gets direct supervision, not diluted backprop
- Model discovers optimal loop weighting during training
- Especially powerful with weight sharing: same weights receive gradient from 3 depths
- Zero new parameters (3 scalars for weights, reuses existing lm_head)
- ~1.2% compute overhead (2 extra lm_head calls)

The "gravity" analogy:
- Loop 1 output is far from the target → strong pull, large updates
- Loop 2 is closer → medium pull, refinement
- Loop 3 is nearest → full weight, precision
- Each loop starts from a better position because the previous loop was already pulled toward the answer

**3. AttnRes (Attention Residuals)**

Replace fixed skip connections with learned, input-dependent attention over depth.
From Moonshot's paper (arxiv:2603.15031).

```
Standard residuals:  x = x + layer_output  (fixed, uniform weight)
AttnRes:             x = softmax(query · [prev_outputs]) · [prev_outputs]
```

Each layer has a single learned query vector w_l ∈ R^d that attends over all
previous loop outputs. The softmax produces content-aware, input-dependent
weights instead of fixed uniform accumulation.

Why this helps:
- Paper shows 1.25× compute equivalent for near-zero parameter cost
- Replaces BOTH the baseline's U-Net skips AND resid_mix
- Only 9 × dim ≈ 4,608 new parameters
- Critical for weight sharing: lets later loops selectively reference earlier loops

### What We Remove From Baseline

| Component | Parameters | Replaced By |
|-----------|-----------|-------------|
| U-Net encoder/decoder split | structural | Fractal loops |
| skip_weights (9 × 512) | 4,608 | AttnRes queries |
| resid_mix (9 × 2 × 512) | 9,216 | AttnRes |
| **Total removed** | **~13,824** | |

### What We Add

| Component | Parameters | Purpose |
|-----------|-----------|---------|
| AttnRes queries (9 layers) | 4,608 | Selective depth attention |
| Loop position embeddings (3 loops) | ~2,100 | Tell weights which loop they're in |
| Gravity weights (3 scalars) | 3 | Learned auxiliary loss weighting |
| **Total added** | **~6,711** | |

**Net: ~7,113 parameters saved → reinvested into wider layers.**

---

## Architecture Diagram

```
INPUT TOKENS (1024 vocab)
    │
    ▼
EMBEDDING (1024 × ~700 dim)
    │
    ▼
LOOP 1 (broad strokes):
    ├── Layer A (attention + MLP, loop_pos=0)
    ├── Layer B (attention + MLP, loop_pos=0)
    ├── Layer C (attention + MLP, loop_pos=0)
    ├── GRAVITY: peek → compute loss₁ (learned weight ~0.1)
    └── Store loop 1 output for AttnRes
    │
    ▼
LOOP 2 (refinement):
    ├── AttnRes: attend over [embedding, loop1_output]
    ├── Layer A (attention + MLP, loop_pos=1)  ← same weights as loop 1
    ├── Layer B (attention + MLP, loop_pos=1)
    ├── Layer C (attention + MLP, loop_pos=1)
    ├── GRAVITY: peek → compute loss₂ (learned weight ~0.3)
    └── Store loop 2 output for AttnRes
    │
    ▼
LOOP 3 (precision):
    ├── AttnRes: attend over [embedding, loop1_output, loop2_output]
    ├── Layer A (attention + MLP, loop_pos=2)  ← same weights again
    ├── Layer B (attention + MLP, loop_pos=2)
    ├── Layer C (attention + MLP, loop_pos=2)
    └── FINAL LOSS: full cross-entropy (weight = 1.0)
    │
    ▼
OUTPUT: logits → BPB
```

Each loop tightens the representation:
- Loop 1: rough sketch (only sees embedding)
- Loop 2: refinement (sees embedding + loop 1 output via AttnRes)
- Loop 3: precision (sees full history, committed to answer)

---

## Information Tightening Mechanisms

### Gravity (primary — Frosty's intuition)
Each loop is pulled toward the final answer by its own loss signal. Later loops
start from better positions because earlier loops were already course-correcting.
The model learns how hard each loop should pull (learned gravity weights).

### AttnRes (secondary — from Moonshot paper)
Selective attention over previous loop outputs. Later loops can choose which
earlier representations are useful for each specific token, not a fixed blend.

### Future: Ring Buffer + Temperature Cooling (Phase 4)
- Ring buffer: bounded memory with eviction of unhelpful previous states
- Temperature: AttnRes attention sharpens with depth (soft early, committed late)
- Only add if Phase 1-3 show signal

---

## Experiment Sequence

### Phase 1: Establish Weight Sharing Baselines
1. Run baseline as-is → establish local BPB reference
2. 3 shared layers × 3 loops, same total params, ~512 dim → does sharing work?
3. 3 shared layers × 3 loops, wider ~700 dim → does width help?
4. 2 shared layers × 4 loops, widest ~850 dim → more loops?
5. 4 shared layers × 2 loops, ~620 dim → fewer loops?

### Phase 2: Add Gravity
6. Best config from Phase 1 + gravity with learned weights
7. Compare: gravity learned vs gravity fixed [0.1, 0.3, 1.0] vs no gravity

### Phase 3: Add AttnRes
8. Best from Phase 2 + full AttnRes
9. Test: AttnRes before attention only / before MLP only / both
10. Test: AttnRes with vs without gravity

### Phase 4: Advanced Mechanisms
11. Add ring buffer (bounded memory with eviction)
12. Add temperature cooling on AttnRes
13. Try combining all mechanisms

### Phase 5: Optimize for Submission
14. Verify int8+zlib artifact ≤16MB
15. Tune width to maximize quality within size budget
16. Port winning config to official train_gpt.py style
17. Run on cloud 8×H100, verify 10-minute timing
18. Prepare submission folder for /records

---

## Workflow

### Local (DGX Spark, free, unlimited)
- Adapted research fork without Triton/torch.compile dependency
- Shorter training budget (2 min per experiment)
- Smaller batch size
- Same model, data, tokenizer, BPB metric
- Results won't match H100 numbers but relative ordering transfers
- Run 50-100 experiments to find winning configuration
- Autoresearch agent runs overnight (Phase 1-4)

### Cloud (H100s, paid, limited)
- Take best configuration from local experiments
- Run at full scale: 8×H100, 10 minutes, full batch
- Verify BPB, artifact size, timing
- Prepare official submission

---

## Source Material

### Attention Residuals (Moonshot)
- Paper: arxiv:2603.15031
- Repo: https://github.com/MoonshotAI/Attention-Residuals
- Core: replace fixed residual connections with softmax attention over depth
- Result: matches 1.25× compute baseline at near-zero parameter cost

### Autoresearch (Karpathy)
- Repo: https://github.com/karpathy/autoresearch
- Core: AI agent modifies train.py, trains 5 min, keeps/discards, loops forever
- Adapted as our outer optimization loop

### Parameter Golf Baseline
- Repo: https://github.com/openai/parameter-golf
- Architecture: 9-layer GPT, 512 dim, 1024 vocab, GQA, Muon optimizer
- Key features: U-Net skip connections, resid_mix, ReLU², logit softcapping
- BPB: 1.2244 (10 min), 1.2074 (4 hour)

---

## Key Insight

The competition rewards compression quality per parameter. Weight sharing is
the ultimate compression — the same function applied repeatedly. AttnRes gives
that repeated function the ability to selectively reference its earlier outputs.
Gravity ensures every repetition is actively pulled toward the correct answer.

The fractal structure means each loop genuinely tightens the representation:
same weights, progressively richer input, direct loss supervision at every
stage. The model isn't just repeating — it's refining.

---

*Plan authored by Octavian + Frosty · Spark-2949 · 2026-03-18*
