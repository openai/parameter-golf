# Strategy: PolyGLU in Parameter Golf

## The Core Hypothesis

The Parameter Golf challenge optimizes `L(N)` — lowest loss given a fixed parameter budget (16MB artifact). Every parameter must earn its keep. The current SOTA approaches all use a **single fixed activation function** (ReLU² in the baseline, LeakyReLU² in the leader). Our hypothesis:

> **PolyGLU's per-neuron activation routing can extract more expressive power per parameter than a fixed activation, giving a measurable BPB improvement within the same parameter budget.**

## Why This Should Work

### 1. Information Density Per Connection
PolyGLU enriches each neuron's computational repertoire. Instead of every FFN neuron computing the same nonlinearity, each neuron selects the best activation for its role. This is like giving each neuron a specialized tool rather than a one-size-fits-all hammer.

### 2. Emergent Depth-Dependent Specialization
The paper's key finding: when trained freely, **early layers prefer GELU (probabilistic gating) while deep layers prefer Tanh (bounded compression)**. In a tiny model with only 9-11 layers, this depth-dependent specialization could squeeze more representation capacity out of fewer layers.

### 3. Near-Zero Cost Convergence
The routing converges to near-deterministic selections (0.030% of max entropy) without any auxiliary loss. This means:
- During training, the model discovers optimal per-neuron activations via Gumbel-Softmax
- At inference time (eval), routing is nearly one-hot — minimal computational overhead
- The routing parameters themselves are tiny (a few thousand per layer)

### 4. The 16MB Constraint Is Actually Favorable
With int8 quantization + compression, routing parameters (α, β, gate_net) are negligible:
- Per layer: ~49K routing params → ~49KB in int8 → compressed even further
- For 11 layers: ~540K params → well under 1% of the 16MB budget
- These extra params may "pay for themselves" by making the MLP more expressive

## What We're NOT Doing

- We are NOT building a 600M model. We're adapting PolyGLU for a ~4M parameter model.
- We are NOT using K=4 activations necessarily. We should experiment with K=2 or K=3 to save params.
- We are NOT expecting the same neurotransmitter maps. At this scale, we just want the activation diversity to improve loss.
- We are NOT adding a separate training loop. PolyGLU integrates into the existing training loop with tau annealing.

## Competitive Advantage

Looking at the leaderboard, every single submission uses the same activation: `relu²` (from the baseline) or `LeakyReLU(0.5)²`. Nobody has tried adaptive activations. PolyGLU is a completely orthogonal optimization that **stacks on top of** all existing tricks (quantization, EMA, XSA, TTT, etc.).

## Risk Assessment

**Low risk**: PolyGLU adds minimal parameters and the Gumbel-Softmax is well-understood. If it doesn't help, the routing will collapse to a single activation (recovering the baseline) and we lose only the small routing overhead.

**Medium opportunity**: The depth-dependent specialization finding suggests that a 10-11 layer model with different activations per layer could be strictly better than one with a uniform activation.

**High ceiling**: If PolyGLU discovers that different neurons genuinely benefit from different nonlinearities at this scale, we get free representational capacity that no other submission has.

## Integration Philosophy

1. **Start from the current SOTA stack** — don't reinvent the wheel. Take the best existing submission as a base.
2. **Replace the MLP activation** with a PolyGLU-style mechanism adapted for the small model.
3. **Keep all existing optimizations** — quantization (int8/int6), EMA/SWA, Muon optimizer, sliding window eval, etc.
4. **Tune the activation palette** — the 600M model used {ReLU, Tanh, SiLU, GELU}. For the small model, consider using squared versions or LeakyReLU variants that the community has found effective.
5. **Anneal tau** — use the same Gumbel-Softmax temperature schedule (1.0 → 0.1) within the 10-minute training window.
