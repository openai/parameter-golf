# Model 6: "The Hive" — Build Spec

**Classification:** PRIVATE — DO NOT SUBMIT UNTIL ENDGAME
**Target bpb:** Unknown (experimental)
**Approach:** Insect brain architecture + ant colony optimization
**Nature Analog:** Bee brain (90% hardwired, 10% plastic) + ant pheromone trails

---

## Core Concept

A bee has 1M neurons but performs complex navigation, pattern recognition, and communication. How? Most of its brain is hardwired pattern detectors — only a small fraction adapts through experience.

We do the same: create a LARGE model (~160M params) where 90% of weights are frozen (random but structured initialization, never trained) and only 10% are trainable low-rank adapters. The frozen weights act as a "random projection" — a fixed feature extraction backbone. The small trainable portion learns to READ those features for language prediction.

## Architecture

### Frozen Backbone (14.4 MB, ~90% of budget)
- 16 transformer layers, 512 dim, 8 heads
- ALL weights frozen after initialization (orthogonal init for good random projections)
- These are NOT trained — they serve as fixed feature extractors
- Stored as int4 (4 bits/param) since exact values don't matter much — just the projection structure
- At int4: 14.4MB stores ~28.8M parameters

### Trainable Adapters (1.1 MB, ~10% of budget)
- Low-rank adapter (LoRA-style) at each layer: rank 8, applied to Q, K, V, O projections
- Per layer: 4 × (512 × 8 + 8 × 512) × 2 bytes = ~65KB
- 16 layers × 65KB = ~1.04 MB
- Plus trainable layer norms, biases, output head
- These are the ONLY weights that get gradient updates

### Stigmergic Residual Connections
- Residual connections accumulate a running "pheromone" score per position
- High-traffic positions (frequently attended) get amplified residuals
- Low-traffic positions get dampened
- Implemented as a learnable decay factor per layer: `residual = x + alpha * pheromone + (1-alpha) * hidden`
- Pheromone updates: `pheromone = beta * pheromone + (1-beta) * attention_weights.mean(dim=head)`

## Training Strategy

1. Initialize all 16 layers with orthogonal weights (structured random)
2. Freeze backbone immediately
3. Train ONLY the adapters + norms + pheromone parameters
4. Training is 10-50x faster because 90% of params have no gradients
5. More training steps in 10 minutes = better convergence for the adapters

## Parameter Budget

| Component | Size | Trainable? |
|-----------|------|-----------|
| Frozen backbone (16L, int4) | ~14.4 MB | No |
| LoRA adapters (rank 8, all layers) | ~1.0 MB | Yes |
| Layer norms + biases | ~0.1 MB | Yes |
| Pheromone parameters | ~0.05 MB | Yes |
| Embeddings (tied, fp16) | ~0.5 MB | Yes |
| **Total** | **~16.0 MB** | 10% |

## Why This Could Work

- Lottery Ticket Hypothesis proves that random networks contain winning sub-networks
- Random projections are provably good feature extractors (Johnson-Lindenstrauss lemma)
- LoRA has proven that tiny adapters can steer large frozen models effectively
- Training only 10% of params = 10x more effective use of the 10-minute window
- Nobody in the competition is using this approach

## Key Risks
- Random backbone may not provide useful features for language modeling
- Int4 frozen weights may be too noisy for coherent representations
- LoRA rank 8 may not have enough capacity to steer the model
- Pheromone mechanism adds complexity without proven benefit

## Output
- `train_gpt_model6.py`
