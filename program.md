# Parameter Golf Research Program

## Objective
Minimize validation bits-per-byte (BPB) on FineWeb, subject to:
- 16MB total artifact size (code + compressed int8+zlib weights)
- 10 minutes training on 8×H100s
- Scored on post-quantization roundtrip BPB

Current baseline: 1.2244 BPB (9-layer, 512-dim, 4 KV heads, tied embeddings)
Baseline uses ~15.8MB of 16MB budget. Step time ~43ms on 8×H100. Reaches step ~13780 in 10 min.

## Key Insight
This is a compression challenge. The question is: how much language understanding can you pack
into 16MB of int8+zlib compressed weights, trained in 10 minutes?

Three levers: (1) better architecture per parameter, (2) better training per step,
(3) better compression per parameter.

## Research Priorities

## What We've Learned On The 1xH100 Proxy
- Baseline (`9x512`, `mlp_mult=2`) reached `1.6419` BPB in 180s with lots of artifact headroom (~8.9MB).
- A deeper/narrower shape (`11x448`) improved to `1.6201` and still compressed well.
- The current local best uses more of the budget: `12x512`, `mlp_mult=3`, `matrix_lr=0.05`, reaching `1.5956` at ~14.8MB.
- Pushing that model larger or more aggressive is risky. A simple LR increase to `matrix_lr=0.08` got `1.5698`, but only by going over the 16MB cap (~16.6MB), so it is not a valid keeper.
- Two short-horizon SwiGLU variants were worse than the current ReLU² setup on this proxy, so generic "modern LLM" substitutions are not enough by themselves.

## Immediate Implications
- The proxy reward is not "best architecture in the abstract"; it is "best post-quantization BPB after ~300 steps on 1xH100 under the artifact cap".
- The search should stay near the current best size regime and avoid large parameter-count jumps unless paired with an explicit compression story.
- Near-term work should emphasize shape, optimizer, schedule, attention-head layout, and quantization-friendliness before larger architecture overhauls.

### Phase 1: Hyperparameter Tuning (low risk, quick wins)
- Learning rate sweeps near the current best, especially `matrix_lr`, `scalar_lr`, and tied embedding LR/init
- Model shape around the current frontier: small changes in depth/width rather than big parameter jumps
- MLP expansion ratio near the current best budget usage
- Number of KV heads and attention head layout
- Sequence length and batch-size adjustments that trade context for step count
- Warmup/warmdown schedule tuning for short-horizon convergence

### Phase 2: Architecture Modifications (medium risk, higher potential)
- Weight sharing across layers (e.g., share every other layer's weights → more effective depth for free)
- Depth recurrence: run the same set of layers multiple times per forward pass
- Different MLP activations only if they are paired with a concrete short-horizon or compression hypothesis
- Mixture of experts with top-k routing
- Different attention variants
- Adjust the U-Net skip connection strategy

### Phase 3: Training Tricks (medium risk)
- Sequence length warmup (start short, increase during training → more steps early)
- Gradient accumulation schedule
- Better learning rate schedule shapes
- Stochastic depth / layer dropout during training

### Phase 4: Compression (high potential, careful testing needed)
- Quantization-aware training (simulate int8 noise during training)
- Lower-bit quantization schemes (int4 with learned codebooks)
- Structured pruning before quantization
- Better compression-friendly weight distributions

## Experiment Guidelines
- Make ONE change per experiment so we can isolate effects
- Always explain the hypothesis: why should this help?
- On the 1xH100 proxy, prefer changes that preserve or improve steps/sec
- Monitor artifact size — current best is already using most of the useful budget
- If over 16MB, the experiment only matters if it reveals a path to recover compression
- Track both pre-quantization and post-quantization BPB; some changes learn better but quantize worse
- Failed experiments are valuable — note what didn't work and why
- When a hyperparameter sweep finds an optimum, combine it with the best known config
- Prioritize changes that are likely to help given what we've learned so far
