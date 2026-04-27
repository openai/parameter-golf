# Anticipatory Transformer V2

## Approach

Builds on proven SOTA techniques (10L, Int5-MLP/Int6-Attn mixed quantization, BigramHash 10240, SmearGate, OrthoInit, Muon WD=0.04, SWA, sliding window eval at stride=64, zstd-22) and adds three innovations from anticipation geometry, validated across multiple domains (language modeling, ASR, behavioral coding agents):

### Innovation 1: MoE Trajectory Routing (10 experts, ~15K params)

10 specialist expert networks, each mapping 7 hidden-state-derived scalars to per-head attention modulation. Routing is via learned centroids (no gating network overhead).

**Architecture**:
```
hidden_states -> Conv1d(dim, 8, k=3) -> GELU -> Linear(8, 7) -> Sigmoid -> 7 scalars
7 scalars -> nearest centroid -> expert[i](7 -> 16 -> n_heads) -> per-head scale [0.8, 1.2]
```

Each expert specializes on a behavioral regime derived from the trajectory dynamics:
- Stabilization experts learn narrow, focused attention
- Transition experts learn gradient-boosted attention at boundaries
- Oscillation experts learn alternating dispersion patterns

**Validated results**:
- Wikitext-2 (4-way controlled): MoE inscription -7.4% perplexity vs control
- Hard routing beats soft embedding (-6.8%) and scalars-only (-4.4%)
- 291K N'Ko ASR: without trajectory scalars, baseline collapses to 100% CER

**Budget impact**: ~15K params = ~11KB int6 = 0.07% of 16MB.

### Innovation 2: Entropy-Weighted Training Loss (parameter-free)

Per-token entropy from the model's own distribution upweights hard tokens:

```
weights = 1.0 + alpha * normalized_entropy
loss = mean(per_token_CE * weights / mean(weights))
```

Alpha=0.15 gives mild upweighting (easy tokens ~1.0, hardest ~1.15). Only active during training. Eval uses standard CE for fair comparison.

### Innovation 3: 7-Scalar Trajectory Extraction (from hidden state dynamics)

A lightweight Conv1d + Linear network extracts 7 anticipation scalars per position:
1. **commitment** — convergence/lock-in of the representation
2. **uncertainty** — ambiguity of futures
3. **transition_pressure** — rate of convergence (dC/dt - dU/dt)
4. **recovery_margin** — distance from constraints
5. **phase_stiffness** — prosodic/structural consistency
6. **novelty** — unexplored territory
7. **stability** — smoothness/jerk

Scalars are recomputed at the encoder/decoder boundary (not just from initial embeddings) since hidden states evolve significantly through the network.

### Base Techniques (from SOTA)

1. 10 layers, 512 dim, 3x MLP (hidden=1536)
2. Mixed Int5/Int6 quantization with per-row scaling + zstd-22
3. BigramHash embedding (10240 buckets, dim=128, projected to 512)
4. SmearGate (learned adjacent token blending)
5. Orthogonal initialization with muP output scaling
6. Muon optimizer with WD=0.04, momentum warmup 0.92->0.99 over 1500 steps
7. SWA every 50 steps over last 40% of training
8. Sliding window eval at stride=64
9. 3% magnitude pruning before quantization
10. Gradient clipping at 0.3

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| num_layers | 10 |
| model_dim | 512 |
| mlp_mult | 3.0 (hidden=1536) |
| train_seq_len | 2048 |
| train_batch_tokens | 786,432 |
| warmdown_iters | 3000 |
| matrix_lr | 0.02 |
| scalar_lr | 0.02 |
| tied_embed_lr | 0.03 |
| muon_weight_decay | 0.04 |
| grad_clip_norm | 0.3 |
| eval_stride | 64 |
| swa_every | 50 |
| swa_start_frac | 0.4 |
| bigram_vocab_size | 10240 |
| bigram_dim | 128 |
| entropy_loss_alpha | 0.15 |
| moe_n_experts | 10 |
| moe_n_scalars | 7 |
| moe_ortho_weight | 0.01 |
| compressor | zstd (level 22) |

## Design Decisions

1. **MoE modulation is multiplicative on attention output, not additive bias on scores.** This maintains Flash Attention compatibility (SDPA kernel), avoiding the O(seq^2) materialization of a bias matrix. The experts produce per-head, per-position scale factors in [0.8, 1.2], applied post-SDPA.

2. **Shared experts across all layers.** One set of 10 experts is reused across all attention layers, reducing parameter count. Scalars are recomputed at the encoder/decoder boundary to capture evolved representations.

3. **Orthogonality penalty on centroids.** A small penalty (0.01) encourages the 10 routing centroids to specialize on different regions of scalar space, preventing expert collapse.

4. **Entropy-weighted loss only during training.** Eval uses standard CE for fair BPB comparison.

## Usage

```bash
# Full V2 (MoE + entropy loss)
torchrun --standalone --nproc_per_node=8 train_gpt.py

# Ablation: MoE only
ENTROPY_LOSS_ENABLED=0 TRAJECTORY_ATTN_ENABLED=0 MOE_TRAJECTORY_ENABLED=1 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py

# Ablation: Entropy only
MOE_TRAJECTORY_ENABLED=0 TRAJECTORY_ATTN_ENABLED=0 ENTROPY_LOSS_ENABLED=1 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py

# Ablation: V1 trajectory gate only
MOE_TRAJECTORY_ENABLED=0 ENTROPY_LOSS_ENABLED=0 TRAJECTORY_ATTN_ENABLED=1 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py

# Baseline (all innovations off)
MOE_TRAJECTORY_ENABLED=0 ENTROPY_LOSS_ENABLED=0 TRAJECTORY_ATTN_ENABLED=0 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Lineage

This submission connects three validated research threads:

1. **Anticipation Geometry** (Diomande, 2026): 7-scalar framework for characterizing hidden state dynamics. Originally developed for coding agent trajectory analysis (71.8% convergence prediction accuracy, p<0.007). MoE inscription routing validated on wikitext-2.

2. **N'Ko ASR** (Papers 1-6): Trajectory scalars proven necessary and sufficient for CTC ASR convergence on low-resource tonal languages. Without them, attention collapses. With them: 29.95% CER on 291K Bambara data. TAR (depth routing) was a negative result; trajectory alone is sufficient.

3. **Behavioral Motif Routing** (KARL): 10 behavioral categories derived from coding agent sessions. Each category (stabilization, transition, oscillation, recovery, etc.) maps to a distinct attention pattern. Applied here as hard MoE routing for LM attention heads.
