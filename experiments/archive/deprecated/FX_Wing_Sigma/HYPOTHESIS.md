# FX_Wing_Sigma — N-gram as Smoothing Reference

## Predecessor
FX_Wing_Delta established flow instructions: each loop's instruction is recomputed
from the current activation state rather than pre-planned from the encoder. This
reduced gradient conflict and (hypothesis) narrows the quantization gap.

FX_Wing_Sigma asks: can we make n-gram statistics a first-class architectural
component — not a post-hoc eval trick, but a conditioning signal that guides how
much each crawler loop invests in each token?

---

## The Core Insight

N-gram entropy at token position t is a direct readout of how much information
the neural model needs to provide. When n-gram entropy is low, the token is
locally predictable — the neural signal is a small correction to a well-calibrated
base rate. When n-gram entropy is high, the neural model must carry the full load.

**Current architecture (FX_Wing_Delta):**
The crawler loops treat every token identically — same instruction magnitude,
same compute depth, regardless of whether the token is predictable or novel.

**FX_Wing_Sigma:**
Gate the instruction amplitude by n-gram entropy. Predictable tokens get a weak
instruction (crawler barely fires). Unpredictable tokens get full instruction
(all loops at full depth). The crawler loops become **adaptive compute** guided
by what n-grams can't explain.

---

## Hypotheses

### H0 — Entropy-gated instructions (main)

```python
# ngram_entropy: [B, T, 1]  — per-token entropy of the training n-gram oracle
entropy_gate = torch.sigmoid(gate_proj(ngram_entropy))   # learned threshold
inst_k = loop_inst_up[k](loop_inst_proj(x)) * entropy_gate
x_loop = x + inst_k
```

**Claim:** The model learns to route compute via the gate:
- Low entropy tokens: gate → 0, instruction ≈ 0, crawler is near identity
- High entropy tokens: gate → 1, full instruction, all 4 loops at max depth

**Why this helps BPB:** Predictable tokens stop consuming loop capacity. That
capacity is reallocated to hard tokens where it matters. Effective depth of
processing on hard tokens increases without adding parameters or compute.

**Why this helps quantization:** Predictable tokens → small neural activations
(just tiny corrections to n-gram baseline) → tight, consistent activation
distribution across all 4 loops for the easy tokens → quantization range needed
is narrow → less multi-context scale mismatch → quant gap shrinks.

---

### H1 — N-gram residual training

**Claim:** Instead of training the model to predict the full token distribution,
train it to predict the RESIDUAL over the n-gram baseline:

```
L_residual = CE(neural_logits + log_p_ngram, target)
```

The neural model never wastes capacity re-deriving what n-grams already know.
It learns purely the difference — the part n-grams can't capture.

This is structurally related to how boosting works: each component learns what
the previous component failed on. Here the n-gram is the first component and
the neural network learns the error signal.

**Implementation:** Add `log_p_ngram` as a logit bias during the training forward
pass. This is already partially in the codebase via the mixer head — Sigma
generalizes it to the primary loss rather than an auxiliary head.

---

### H2 — Per-loop entropy routing

**Claim:** Each crawler loop should be sensitive to a DIFFERENT n-gram order.
N-gram order captures different "scales" of predictability:

- **Loop 0**: gate on bigram entropy (2-token local predictability)
- **Loop 1**: gate on trigram entropy (3-token context)
- **Loop 2**: gate on 5-gram entropy (medium range)
- **Loop 3**: gate on full-order entropy (whatever the oracle uses)

Each loop specializes to resolving uncertainty at its own scale. Loop 0 handles
locally predictable tokens fast; loop 3 only fires for tokens that remain
uncertain even after long context.

This is loop specialization achieved via training signal rather than architectural
constraint — the loops learn WHEN to fire, not just HOW to transform.

---

### H3 — DeltaNet seeded from n-gram distribution

**Claim:** Instead of initializing DeltaNet state S to zeros, seed it with a
soft representation of the current context's n-gram distribution:

```python
ngram_seed = seed_proj(ngram_dist_embedding)  # [B, H, Dh, Dh]
delta_state = ngram_seed                       # start from n-gram prior
# then delta rule updates refine from this baseline
```

The delta rule `S += β*(v - S@k)` is a correction rule — it corrects S toward
the data. If S starts at zeros, early loop iterations are wasted learning the
base rate that n-grams already capture. If S starts at the n-gram distribution,
every delta rule update is immediately refining beyond what n-grams know.

**Expected result:** DeltaNet converges faster and to a better final state
when seeded from the n-gram prior than from zeros.

---

## Ablation Ladder

### S0 — FX_Wing_Delta (control)
Flow instructions, DeltaNet from zeros, no n-gram gating.
Reference point for all Sigma ablations.

### S1 — Entropy gate, no DeltaNet
```
ENTROPY_GATE=1  DELTA_NET_HEADS=0
```
Isolates H0. Does gating the instruction amplitude by n-gram entropy improve
val_bpb or quant gap over FX_Wing_Delta?

### S2 — Entropy gate + DeltaNet (full Sigma)
```
ENTROPY_GATE=1  DELTA_NET_HEADS=2
```
Main hypothesis. Gate + DeltaNet together. Does the gate give DeltaNet's
state more useful content to accumulate?

### S3 — Per-loop entropy routing (H2)
```
ENTROPY_GATE=1  ENTROPY_GATE_PER_LOOP=1  DELTA_NET_HEADS=2
```
Each loop gates on a different n-gram order. Requires multi-order entropy
available during training (already computed by ngram_eval oracle).

### S4 — Residual training (H1)
```
NGRAM_RESIDUAL_TRAINING=1  ENTROPY_GATE=1
```
Add log_p_ngram as logit bias in primary loss. Most radical change — different
training objective. Run ONLY if S1/S2 confirm the entropy gate concept works.

### S5 — DeltaNet seeding (H3)
```
ENTROPY_GATE=1  DELTA_NET_HEADS=2  DELTA_NET_NGRAM_SEED=1
```
Seed S from n-gram distribution representation. Tests whether DeltaNet benefits
from a warm start vs cold (zero) start.

---

## What Changes vs FX_Wing_Delta

### New hyperparameter:
```python
entropy_gate_enabled = bool(int(os.environ.get("ENTROPY_GATE", "0")))
entropy_gate_per_loop = bool(int(os.environ.get("ENTROPY_GATE_PER_LOOP", "0")))
```

### New parameter in CrawlerGPT:
```python
# Learned threshold on n-gram entropy → instruction amplitude
self.entropy_gate_proj = nn.Linear(1, 1, bias=True)  # per-token scalar gate
nn.init.zeros_(self.entropy_gate_proj.weight)         # start open (gate=0.5)
```

### Change in _run_crawler:
```python
for loop in range(self.crawler_loops):
    inst_k = self.loop_inst_up[loop](self.loop_inst_proj(x))
    if self.entropy_gate_proj is not None and ngram_entropy is not None:
        gate = torch.sigmoid(self.entropy_gate_proj(ngram_entropy))
        inst_k = inst_k * gate
    x_loop = x + inst_k
    ...
```

The n-gram entropy is already available during training via the `TrainNgramOracle`.
It needs to be passed through `CrawlerGPT.forward()` to `_run_crawler`. One extra
tensor through the forward pass — no new computation, just routing.

---

## Decision Criteria

| Outcome | Interpretation | Next Step |
|---------|---------------|-----------|
| int6 gap < 0.2 AND val_bpb ≤ 1.12 | Sigma solves the regression AND matches SOTA | Submit |
| int6 gap < 0.5, val_bpb competitive | N-gram gating helps but quant needs per-loop scales | Add per-loop scales |
| val_bpb improves but int6 unchanged | Training benefit only, not quant benefit | Dig into why gate doesn't help quant |
| No improvement over FX_Wing_Delta | N-gram entropy not a useful gate signal | Park Sigma, try H1 (residual training) separately |

---

## Why This Could Be Significant

Every transformer architecture currently treats n-gram statistics as an external
oracle — something you add at eval time, not something the architecture is aware of
during training. FX_Wing_Sigma makes n-gram statistics a first-class training signal
that shapes WHERE the neural computation happens.

If the gate works, you have a model that:
1. **Knows when to trust itself** — high entropy → invest loops, low entropy → defer to n-gram
2. **Learns residuals by construction** — small activations on easy tokens = neural is learning the hard part
3. **Quantizes cleanly** — tight activation distributions on the majority of tokens (the easy ones)
4. **Fits in 4.5 MB** — compression advantage of FX_Wing_Delta preserved

The n-gram oracle stops being a post-processing trick and becomes part of the
architecture's training signal. That is the step change.

---

## Prerequisites

- FX_Wing_Delta results confirming flow instructions reduce quant gap
- `TrainNgramOracle` entropy values accessible during CrawlerGPT forward pass
- n-gram entropy passed as `ngram_entropy: Tensor | None` to `_run_crawler`

**Do not implement until FX_Wing_Delta confirms:**
1. Flow instructions improve int6 roundtrip BPB vs FX_Wing static instructions
2. The architecture trains stably with flow + DeltaNet at 8×H100 scale
