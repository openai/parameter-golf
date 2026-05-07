# Adaptive Importance Routing Memory Model (AIR-MM)

Internal architecture document for the Parameter Golf challenge.

---

## 1. Working Name

**Adaptive Importance Routing Memory Model (AIR-MM).**

The name captures the three core ideas: tokens are scored for importance (Adaptive Importance), compute is routed accordingly (Routing), and the mechanism is designed to evolve toward a compact, memory-centric running state (Memory Model).

## 2. Core Idea

Standard transformers treat every token the same: each position gets the same attention computation, the same MLP pass, and the same residual update magnitude. Under tight parameter and compute budgets, this is wasteful. Most tokens in a sequence are predictable filler; a handful carry the real information.

AIR-MM introduces a lightweight mechanism that estimates which tokens matter more, then selectively scales their residual updates. Tokens that carry more signal get full-strength updates. Tokens that are predictable get softer updates, freeing capacity for what matters.

The shift is from **uniform compute allocation** to **importance-based compute allocation**.

## 3. Primary Objectives

- **Reduce wasted computation.** Stop spending equal effort on every token when some tokens carry far more signal.
- **Preserve accuracy under tight constraints.** The Parameter Golf challenge has hard size and time caps. Every parameter and every FLOP must earn its keep.
- **Maximize intelligence per byte and per second.** The goal is not just a smaller model but a smarter one for the same budget.
- **Introduce real mechanism innovation.** Not just shrinking a known architecture, but adding a genuinely novel routing mechanism that could generalize beyond this challenge.

## 4. Architectural Thesis

The long-term direction for AIR-MM is a **memory-centric architecture** with four properties:

1. **Compact running state.** Instead of relying entirely on the full residual stream, the model maintains lightweight internal signals that summarize what has been seen.
2. **Lightweight importance signals.** Token importance is estimated cheaply from the hidden state, not from expensive auxiliary models or heuristics.
3. **Tiny learned controller.** A small projection network (tens to hundreds of parameters per layer, not thousands) decides how much each token's update matters.
4. **Importance-weighted updates.** Residual additions from attention and MLP are scaled by the controller's output, so the model dynamically adjusts its own update strength.

This is fundamentally different from pruning, distillation, or mixture-of-experts. There is no discrete routing, no dropped tokens, no expert selection. Every token still gets processed; the question is how much weight its update carries.

## 5. Novel Components

### 5.1 Compact Running State

The importance gate reads from the hidden state at each layer and computes a per-token scalar. This scalar is the simplest possible "running state" about token importance. In v2+, this could evolve into a multi-dimensional running signal (recency, surprise, context-shift).

### 5.2 Importance Signals

In v1, the only signal is a learned projection from the token's hidden representation. The model learns, during training, which hidden-state patterns correspond to tokens that benefit from stronger updates.

Future signals (not yet implemented):
- **Recency:** How recently the token was attended to.
- **Surprise:** How much the token's representation deviated from the model's prediction.
- **Context shift:** Whether the token marks a boundary between topics or structures.

### 5.3 Routing Controller

A tiny two-layer network per block:

```
hidden_state -> Linear(dim, hidden) -> ReLU -> Linear(hidden, 1) -> Sigmoid
```

With `hidden = 32` and `dim = 512`, this adds ~16K parameters per layer, or ~150K total for 9 layers. This is small relative to the millions of parameters in attention and MLP weights.

The output is a scalar in (0, 1), which is then mapped to [min_scale, 1] to ensure no token is ever fully silenced.

### 5.4 Importance-Weighted Update Rule

The gating is applied multiplicatively to both the attention and MLP residual updates:

```
importance = min_scale + (1 - min_scale) * sigmoid(controller(x))
x = x + importance * scaled_attn_out
x = x + importance * scaled_mlp_out
```

The `min_scale` floor (default 0.25) is a safety mechanism. Even the lowest-importance token still gets 25% of its normal update. This prevents degenerate solutions where the model learns to shut off entire positions.

## 6. How This Differs From a Hybrid Model

The previous experimental direction was a hybrid architecture: a smaller transformer with some structural changes. That approach was about shrinking a known architecture and hoping the pieces fit.

AIR-MM is different in a specific way: **the novelty is in dynamic routing, not in static architecture choices.** The transformer blocks themselves are unchanged. The attention mechanism is unchanged. The MLP is unchanged. What changes is that the model has a new degree of freedom: it can decide, per token and per layer, how strongly to apply its own updates.

This is a mechanism that does not exist in standard transformers. It is not a variant of dropout (which is random), not mixture-of-experts (which is discrete and token-dropping), and not early exit (which stops processing entirely). It is a continuous, learned, per-token scaling of residual updates.

## 7. Conservative v1 Implementation Strategy

The v1 implementation is deliberately minimal:

- **Do not rewrite the model.** Work inside the existing `Block` class.
- **Add one small module.** `TokenImportanceGate` is a self-contained two-layer network.
- **Wire it into the forward pass.** The gate's output multiplies the attention and MLP residual updates before addition.
- **Preserve the baseline path.** When `USE_IMPORTANCE_ROUTING=0`, the gate is not created and the forward pass is identical to the original.
- **Make it ablatable.** The feature is toggled by a single hyperparameter. Comparing gated vs. ungated is a one-line change.
- **Keep it debuggable.** The gate logs its average importance score every N steps, so you can watch the distribution during training.
- **No new dependencies.** Everything is standard PyTorch.

### What v1 does NOT do:

- No multi-signal routing (only learned importance from hidden states).
- No per-head or per-channel gating (scalar per token only).
- No learned min_scale (it is a fixed hyperparameter).
- No recency, surprise, or context-shift signals.

## 8. Success Criteria for v1

1. **Baseline preserved.** With `USE_IMPORTANCE_ROUTING=0`, the model trains identically to the original.
2. **No training instability.** With routing enabled, training loss should converge smoothly without spikes or divergence.
3. **Importance distribution is non-trivial.** The model should learn to assign different importance scores to different tokens, not collapse to a uniform distribution.
4. **Competitive val_bpb.** With routing enabled, val_bpb should be at least as good as the baseline, ideally better.
5. **Minimal parameter overhead.** The gate parameters should be <5% of total model parameters.
6. **No wallclock regression.** The gate computation should add negligible overhead to training time.

## 9. Risks and Risk Controls

| Risk | Likelihood | Impact | Control |
|------|-----------|--------|---------|
| Gate collapses to uniform (all tokens ~0.5) | Medium | Low | Monitor per-block importance logs; if flat, increase hidden dim or remove floor temporarily |
| Gate saturates to 1.0 (no effect) | Medium | Low | Sigmoid + floor design makes this the safe failure mode; model degrades gracefully to baseline |
| Training instability from gating gradients | Low | High | min_scale floor prevents zero gradients; gate reads from hidden state (not gated output) to avoid feedback loops |
| Parameter overhead exceeds budget | Low | Medium | Gate is ~16K params/layer; monitor total model size |
| Incompatibility with torch.compile | Low | High | Gate uses only standard ops (Linear, ReLU, Sigmoid); test with compile early |
| Interference with Muon optimizer | Low | Medium | Gate params are routed to Adam via CONTROL_TENSOR_NAME_PATTERNS, not Muon |

## 10. v2 Direction

Once v1 is validated and stable, the next extensions:

- **Recency signal.** Track an exponential moving average of attention weights per position. Tokens that have been heavily attended to recently get a recency boost.
- **Surprise signal.** Compare each token's representation to the model's running prediction. Large deviations indicate surprising content that deserves stronger updates.
- **Context-shift signal.** Detect boundaries between topics or structures (e.g., paragraph breaks, code/prose transitions) and boost importance at transition points.
- **Adaptive controller.** Replace the fixed two-layer MLP with a small controller that takes multiple signals as input (learned importance + recency + surprise + context-shift) and produces a combined routing decision.
- **Per-head gating.** Instead of a single scalar per token, produce one gate per attention head, allowing the model to selectively amplify or dampen individual heads.
- **Learned min_scale.** Make the floor a learned parameter (with a constraint to stay positive) so the model can decide its own safety margin.

The key design constraint for v2: each new signal should be cheap to compute, easy to ablate, and independently testable.
