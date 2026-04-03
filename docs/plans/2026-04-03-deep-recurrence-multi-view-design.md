# Deep Recurrence with Adaptive Floor Attention and Multi-View Parallelism

**Date:** 2026-04-03
**Target:** 10-minute 8×H100 leaderboard (16 MB artifact, 600s training)
**Codename:** SFW v3 — DeepFloor

## Core Idea

Replace the standard transformer stack (attention-then-FFN × L layers) with a single recurrent block applied thousands of times. The 16 MB artifact stores a tiny recurrent block. Depth comes from recurrence (free in artifact cost, paid in compute). Width comes from running multiple independent "views" across GPUs (representation parallelism, not data parallelism).

Cross-token communication is the key design variable. Two parallel architectures are tested head-to-head:

**Architecture A (Floor):** The recurrent block is purely per-token. Cross-token mixing happens via discrete softmax attention passes ("floors") that fire adaptively when hidden states signal uncertainty.

**Architecture B (Fused):** Linear/kernel attention is baked into the recurrent block. Every step includes cheap cross-token mixing through a running accumulator. No floors. This is structurally a state-space model.

## Architecture

### The Recurrent Block

A single learned block applied repeatedly to each token's hidden state:

```
h_{t+1} = RecurrentBlock(h_t)          # Architecture A (per-token, no cross-token mixing)
h_{t+1} = RecurrentBlock(h_t, S_t)     # Architecture B (S is a running cross-token accumulator)
```

At d=64 with attention-only structure (Q, K, V, O projections + layernorm):
- ~12,288 parameters per block
- At ternary quantization: ~2.4 KB per block

In Architecture A, the block is purely per-token between floors — trivially parallelizable across tokens and GPUs.

In Architecture B, the block includes a running accumulator S that carries cross-token information. S is updated incrementally each step: `S_{t+1} = S_t + phi(k_t) v_t^T`. The per-step cost of maintaining S is O(d^2) — same order as the rest of the block. Tokens are still processed independently; they read from and write to a shared S.

Design choices within the block (residual connections, gating, nonlinearity) are evolvable but should be tested empirically before being added to the genome.

### Cross-Token Communication

#### Architecture A: Floor Attention

Periodically, all tokens synchronize via a single softmax attention pass. This is the only cross-token operation in the model. Between floors, the recurrence is purely per-token.

The floor fires adaptively based on zero-parameter signals from the hidden state itself:
- Rate of change (how fast h is still moving between steps)
- Norm dynamics (sudden jumps suggest surprise/uncertainty)
- Convergence detection (state has settled, needs new information)

Guardrails prevent degenerate behavior:
- `floor_min_interval`: minimum recurrent steps between floors (don't fire constantly)
- `floor_max_interval`: maximum steps before forcing a floor (don't drift forever)

The attention block itself is a standard single-pass attention (Q, K, V, O + layernorm) at the same d=64 dimension. Same parameter cost as the recurrent block.

**Training vs eval:** During training, floors fire on a fixed schedule (every N steps) for stable gradient flow. At eval time, the adaptive trigger takes over. The fixed schedule is a hyperparameter; the adaptive threshold can be calibrated against it.

#### Architecture B: Fused Linear Attention

No floors. Every recurrence step includes cross-token mixing via a running accumulator:

```
S_{t+1} = S_t + phi(k_t) v_t^T     # accumulate key-value state
o_t = phi(q_t) S_t                   # read cross-token info
```

Where `phi` is a kernel feature map (candidates: ELU+1, random features / FAVOR+, identity).

The accumulator S is a d×d matrix shared across tokens within a view, updated incrementally at O(d^2) per step. This is structurally a linear transformer / state-space model.

**Input-dependent selection (from Mamba):** The accumulator should not blindly accumulate. The recurrent block gates what enters S based on the current hidden state — a learned selection of what information is worth compressing into the running state vs what can be discarded. This is the key insight from Mamba/S6 that separates modern SSMs from vanilla linear RNNs. Cost: one additional d×d gate matrix in the block (~4K params at d=64).

**Advantages over floors:**
- No discrete synchronization events — simpler control flow
- Cross-token info available at every step, not just at floor boundaries
- No adaptive trigger logic needed

**Disadvantages:**
- Linear attention is less expressive than softmax per application
- The accumulator S grows noisy over thousands of steps (may need periodic normalization or decay)
- Less studied at extreme recurrence depths

### Attention Kernel

Floor attention fires rarely — every 50-500 recurrence steps. This fundamentally changes the softmax vs linear attention trade-off. In standard transformers, softmax wins because it's maximally expressive and attention runs at every layer. Here, the recurrence does the heavy lifting. The floor just needs to share information across tokens quickly and get out of the way.

This means:
- **Expressiveness is secondary** — the recurrent block handles per-token refinement
- **Latency is primary** — a slow floor blocks thousands of recurrence steps behind it
- **Fusibility matters** — a single-kernel operation keeps the recurrence loop tight

There is a real, well-studied landscape of kernel and linear attention alternatives:

**Candidates for head-to-head comparison (pre-evolution, not in the genome):**

1. **Softmax** (baseline): Standard `softmax(QK^T) @ V`. O(n^2), three passes (max, exp, normalize), well-understood. The quality benchmark.

2. **ReLU attention**: `relu(QK^T) @ V`. One pass, trivially fusable. Unnormalized — output scale drifts, but the recurrence absorbs scale changes at the next step.

3. **Linear / kernel attention**: `phi(Q)(phi(K)^T V)` where phi is a feature map (e.g., ELU+1, random features as in Performer/FAVOR+). O(n) in sequence length, associative reordering allows streaming. Well-proven for long sequences. At our sequence lengths (256-512) the O(n) vs O(n^2) gap is small, but the constant-factor kernel fusion benefit may still matter.

4. **Performer (FAVOR+)**: Random feature map approximation of softmax. Mathematically grounded, widely cited, proven in research and some production systems. Gives softmax-like behavior at linear cost.

5. **Squared attention**: `(QK^T)^2 @ V`. Simple, no normalization, single kernel. Less studied but cheap.

The winner becomes a fixed architectural choice. The key evaluation criteria for our use case, in priority order:
1. Wall-clock time for a single floor pass (including kernel launch overhead)
2. Whether quality degrades at the sparse firing rates we use (every 50-500 steps)
3. Stability over thousands of recurrence steps between floors

Note: Hybrid approaches (Longformer-style local+global, BigBird) are designed for long-sequence scaling and are less relevant here. Our sequences are short (256-512 tokens) and the scaling concern is depth of recurrence, not breadth of attention.

### Multi-View Representation Parallelism

Each GPU runs the full recurrence depth on the same input sequence, but with a different learned projection from the embedding space. Eight GPUs produce eight independent d=64 latent representations of the same tokens.

At the output, views are combined to produce logits. Combination method is evolvable:
- Average logits (zero extra parameters)
- Learned weighted vote (8 scalar weights)
- Concatenate-and-project (512 → vocab, costs 512 × vocab parameters)

Each view shares the same recurrent block and attention block weights. The only per-view unique parameters are the embedding projections (vocab × 64 each).

### Parameter Budget (Ternary, 16 MB)

Available parameters: ~84.7M ternary values

| Component | Parameters | Notes |
|-----------|-----------|-------|
| Shared recurrent block | ~12K | Applied thousands of times |
| Shared attention block | ~12K | Fires adaptively |
| Per-view embedding projections (×8) | 8 × vocab × 64 | For 8192 BPE: 4.2M |
| Output combination | 0 – 512 × vocab | Depends on method |
| Remaining budget | ~80M+ | Available for deeper/wider blocks, memory tables, etc. |

The architecture is *radically* parameter-efficient. Almost the entire 16 MB budget is unspent by the core model. This surplus can be used for:
- Multiple distinct recurrent blocks applied in a repeating sequence (e.g., blocks A-B-C-A-B-C...)
- Larger embedding tables / richer tokenizers
- Persistent hidden memory tables (as in SFW v2b)
- Wider recurrent dimension

## Evolutionary Genome

### RecipeGenome Fields

**Architecture:**
- `recurrent_dim`: hidden dimension (32, 48, 64, 96)
- `num_distinct_blocks`: how many unique recurrent blocks cycle (1, 2, 4, 8)
- `block_has_residual`: whether the recurrent step adds or replaces (bool)
- `block_nonlinearity`: activation function in the block (relu, gelu, swish)
- `quantization`: artifact precision (ternary, int4, int6)
- `view_count`: number of independent views (2, 4, 8)
- `view_combination`: how views merge at output (average, weighted, project)
- `tokenizer_gene`: tokenizer choice (bytes, sp_bpe_1024, sp_bpe_8192)

**Cross-token mode:**
- `cross_token_mode`: which architecture (floor, fused)

**Floor policy (Architecture A only):**
- `floor_metric`: what signal triggers attention (norm_velocity, step_delta, entropy)
- `floor_threshold`: sensitivity of the trigger
- `floor_min_interval`: minimum recurrence steps between floors
- `floor_max_interval`: maximum steps before forced floor

**Fused attention (Architecture B only):**
- `kernel_feature_map`: feature map for linear attention (elu_plus_1, favor_plus, identity)
- `accumulator_decay`: decay factor on S to prevent noise buildup over thousands of steps (0.99, 0.999, 1.0)

**Recurrence:**
- `recurrence_step_size`: damping factor for state updates
- `max_eval_depth`: total recurrent steps at inference

**Training:**
- `base_lr`: learning rate
- `weight_decay`: regularization
- `train_floor_interval`: fixed floor interval used during training

### Evolutionary Schedule

Using the existing staged approach:

- **2×120s**: Two views, trained 120 seconds each. Find two good recurrent blocks and floor policies.
- **4×30s**: Crossover and mutate into four views. Shorter training to refine.
- **8×15s**: Fill all eight GPU views. Brief fine-tuning to differentiate views.

Crossover operates naturally:
- Swap a view's embedding projection between parents
- Swap the recurrent block between parents
- Swap floor policy genes between parents

## Training Strategy

### Forward Pass (Training)

1. Embed input tokens via view-specific projection
2. Apply recurrent block N times with floor attention every `train_floor_interval` steps
3. Combine view outputs
4. Compute loss, backprop

Backprop does not go through all recurrence steps. Use truncated BPTT:
- Forward through K recurrence steps (e.g., K=20-40)
- Backprop through those K steps
- Detach and continue forward for the next K steps

This bounds training memory and time while still learning the recurrent dynamics.

### Forward Pass (Eval)

1. Embed input tokens via view-specific projection
2. Apply recurrent block up to `max_eval_depth` times
3. Floor attention fires adaptively based on hidden state dynamics
4. Combine view outputs at the end
5. Compute loss (BPB)

Eval depth can be much larger than training depth because there's no backward pass.

### Hardware Layout (Training)

Each GPU trains one or more views. With 8 GPUs and 2 initial views:
- GPUs 0-3: view A (data parallel across 4 GPUs for faster training)
- GPUs 4-7: view B

As the population grows to 8 views:
- One GPU per view, no data parallelism
- Each GPU trains independently — no cross-GPU communication during recurrence

### Hardware Layout (Eval)

- 8 GPUs, one view each
- Each GPU runs full recurrence depth independently
- Combine logits at the end (one allreduce of vocab-sized vectors)

## Key Experiments (Pre-Evolution)

These should be run as fixed controlled comparisons before adding choices to the genome:

1. **Architecture A vs B**: Floor (softmax) vs Fused (linear attention). Same recurrent block, same seed, same depth. The fundamental question: does discrete burst attention or continuous accumulated attention work better at extreme recurrence depths?

2. **Depth vs expressiveness**: d=32 × ~27K steps vs d=64 × ~6.8K steps vs d=128 × ~1.7K steps. Which depth/width trade-off produces the best BPB?

3. **Truncated BPTT depth**: K=10 vs K=20 vs K=40 training unroll. How much training depth does the recurrent block need to learn?

4. **Floor interval**: fixed floors at N=10, 50, 100, 500. Where does attention actually help?

5. **Number of distinct blocks**: 1 block repeated vs 2-4 blocks in a cycle. Does block diversity help?

## Relationship to Prior Work

- **SFW v0-v2b**: Proved that runtime memory augmentation works for this competition. The persistent hidden memory system can potentially be integrated as an additional component if the parameter budget allows.
- **Evolutionary benchmark**: Showed that the population wants depth over width. The recipe evolution hit the 16 MB wall at d=512, layers=9. This architecture removes that wall.
- **Competition landscape**: The leader (1.1194 BPB) uses TTT. Several entries use depth recurrence. The competition explicitly requests Universal Transformers and state-space models. Architecture A is in the Universal Transformer family. Architecture B is in the SSM family. Both are on the wish list.

## Success Criteria

- Beat the naive baseline (1.2244 BPB) with the recurrent architecture alone
- Show that increasing eval depth improves BPB (depth recurrence is working)
- Show that multiple views improve over a single view (representation parallelism is working)
- Show that adaptive floor attention outperforms fixed-interval floors
- Competitive with leaderboard entries (target: sub-1.15 BPB)

## Open Questions

- How much of the 80M+ surplus ternary parameter budget should go to wider recurrent blocks vs more distinct blocks vs memory tables?
- Does the persistent hidden memory from v2b compose well with this architecture, or does the deep recurrence make it redundant?
- Should different views have different floor policies, or share one?
- Can the output combination method itself be adaptive (weight views differently per token)?
- **The fundamental boundary question:** What information cannot survive compression into a d=64 state over thousands of recurrence steps? Architecture A bets the answer is "some things, and the model can detect when." Architecture B bets "nothing, if the selection gate is good enough." The A vs B experiment is an empirical answer to this question on FineWeb.
