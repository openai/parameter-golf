# R2-15: Causal Data Strategy — Three-Pronged Approach

## Overview

Three complementary techniques for maximizing information density during training:

| | R2-15a | R2-15b | R2-15c |
|---|---|---|---|
| **Name** | Attention-guided adversarial dropout | Offline difficulty curation | Combined |
| **Level** | Within-sequence (token-level) | Between-sequence (shard/batch ordering) | Both |
| **Cost** | Zero extra compute | Offline scoring run (not counted in 10-min) | Both |
| **What it does** | Drop tokens the model attends to most | Train on difficulty-ordered data (easy→hard) | Both |
| **Why it works** | Forces redundant causal paths | Curriculum learning on information-dense sequences | Compounded |

## Competition Rules Analysis

TTT (accepted on leaderboard) fine-tunes on validation data during eval — a bigger gray area than offline data curation. Our approach only touches training data, offline. Ruling:

- Hyperparameter tuning offline: **explicitly allowed**
- Data preprocessing offline: **implicitly allowed** (same category as hyperparameter tuning)  
- Baking results into the script: **allowed** (curated ordering is deterministic)
- Multi-epoch via curation: **gray area** — mitigate by using baseline model for scoring, not the final model

---

## R2-15a: Attention-Guided Adversarial Dropout

### Idea
Instead of dropping random tokens, drop the tokens the model attends to MOST. This is maximally informative — every intervention targets a token the model relies on, forcing it to build redundant causal paths.

### Mathematical Spec
```
Forward pass (standard):
  1. Compute attention: A[h,i,j] = softmax(Q_h[i] · K_h[j] / sqrt(d))
  2. Aggregate importance per source token:
     importance[j] = mean over heads h, target positions i of A[h,i,j]
  3. This is computed during the normal forward pass — no extra cost

Apply adversarial dropout (on NEXT step, using cached importance):
  4. drop_prob[j] = importance[j] / max(importance) × max_drop_rate
     High-importance tokens get high drop probability
  5. mask[j] ~ Bernoulli(1 - drop_prob[j])
     mask[0] = True  (always keep first token)
  6. x = x[:, mask], y = y[:, mask]
```

### Implementation
**Problem**: Extracting attention scores from `F.scaled_dot_product_attention` or `flash_attn_3_func` — these are fused kernels that DON'T return attention weights by default.

**Solution**: Use the per-position LOSS as a proxy for attention importance. Positions with low loss are "well-attended" (model has gathered enough context for them). Positions with high loss are "under-attended" (model needs more context).

This inverts the probing direction but achieves the same goal:
- **Low-loss positions**: Model is confident → DROP their context aggressively (challenge the model)
- **High-loss positions**: Model is struggling → KEEP their context (help the model learn)

This is exactly what we already implemented in the adaptive causal probing — the per-batch adaptive rate. But we can make it **per-position** using a masking approach instead of sequence compaction.

**Revised approach — per-position loss-weighted dropout**:
```
Step 1: Cache per-position loss from previous step (already implemented)
Step 2: Convert to per-position dropout probability:
        drop_prob[j] = base_rate + (max_rate - base_rate) × sigmoid(-prev_loss[j] + threshold)
        (low loss → high drop prob, high loss → low drop prob)
Step 3: Instead of compacting the sequence (which changes positions), MASK the embeddings:
        mask_weight[j] = Bernoulli(1 - drop_prob[j])  
        x[:, j, :] *= mask_weight[j]  (zero out dropped positions' embeddings)
        
This preserves sequence length (no RoPE shift) while zeroing out context the model relies on.
```

### Why masking instead of compaction
- Compaction changes token positions → RoPE produces different embeddings → confounds the experiment
- Masking keeps positions stable → the model sees zeros where dropped tokens were → cleaner signal
- Masking is simpler to implement (no variable-length sequences)

### Interface
```
CAUSAL_PROBE=1                    # enable (already implemented)
CAUSAL_PROBE_MODE=adversarial     # per-position adversarial masking (new)
CAUSAL_PROBE_BASE=0.05
CAUSAL_PROBE_MAX=0.30
CAUSAL_PROBE_THRESHOLD=1.0
```

### Code changes
1. `GPT.forward(reduction="none")` — already implemented
2. Training loop: Cache per-position loss, compute per-position dropout probabilities, apply as embedding mask
3. ~15 additional lines in training loop

### Verifiable DoD
1. High-loss positions have lower dropout probability than low-loss positions
2. At step 0 (no cached loss): uniform base_rate dropout
3. Masking zeros out embeddings (not removes tokens) — sequence length unchanged
4. Gradient flows through non-masked positions
5. No extra forward passes (zero compute overhead)

---

## R2-15b: Offline Difficulty Curation

### Idea
Pre-score all training data with a baseline model, rank sequences by difficulty, and train with a curriculum: easy sequences first, hard sequences later. Information-dense (hard) sequences get more training time in the critical final phase.

### Implementation Plan

#### Phase 1: Offline scoring (runs once, outside the 10-min window)
```python
# score_data.py — run once on any GPU, takes ~5-10 min
model = load_baseline_model()  # use the baseline, NOT the final model
for shard in all_shards:
    for seq in shard:
        loss = model(seq)
        scores.append((shard_id, seq_offset, loss))
# Save difficulty index
torch.save(sorted_indices, "data/difficulty_index.pt")
```

#### Phase 2: Curriculum training (within the 10-min submission)
```python
# In train_gpt.py, modify the data loader:
if difficulty_index exists:
    # Phase A (first 60% of training): easy sequences (low loss)
    # Phase B (last 40% of training): hard sequences (high loss)
    curriculum_order = load("data/difficulty_index.pt")
    # The data loader yields sequences in curriculum order instead of sequential
```

### Curriculum schedule
```
Steps 0-60%:    Train on easiest 50% of sequences
                Model builds solid foundation on predictable patterns
Steps 60-90%:   Train on medium difficulty sequences  
                Model extends to more complex patterns
Steps 90-100%:  Train on hardest 20% of sequences
                Model focuses on information-dense, challenging data
```

### Why this works
- **Bengio et al. (ICML 2009)**: Curriculum learning consistently improves convergence
- **Rho-1 (NeurIPS 2024)**: Focusing on "useful" tokens improves over uniform training
- **Data quality > data quantity**: Phi-1/Phi-2 showed that curated data dramatically outperforms random data

### Interface
```
CURRICULUM=1                           # enable curriculum training
CURRICULUM_INDEX=data/difficulty_index.pt  # pre-computed difficulty scores
CURRICULUM_PHASES=0.6,0.9,1.0          # phase boundaries (fraction of training)
CURRICULUM_DIFFICULTIES=0.0,0.5,0.8,1.0  # difficulty percentile ranges per phase
```

### Deliverables
1. `scripts/score_data.py` — offline scoring script (~50 lines)
2. `data/difficulty_index.pt` — pre-computed difficulty index (generated offline)
3. Modified data loader in `train_gpt_r2.py` — reads curriculum order (~30 lines)

### Verifiable DoD
1. `score_data.py` produces a sorted index file
2. Early training steps use low-difficulty sequences
3. Late training steps use high-difficulty sequences
4. With `CURRICULUM=0`: standard sequential data loading (backward compatible)
5. The difficulty index is deterministic (same model + same data → same index)

---

## R2-15c: Combined — Curated Data + Adversarial Dropout

### Idea
Stack both techniques:
- **Between-sequence**: Curriculum ordering (easy→hard over training)
- **Within-sequence**: Adversarial dropout (drop high-confidence context tokens)

The model gets a double curriculum:
1. Sequences get harder over training (curriculum ordering)
2. Within each sequence, confident predictions get challenged (adversarial dropout)

### Interface
```
CURRICULUM=1 CAUSAL_PROBE=1 CAUSAL_PROBE_MODE=adversarial
```

### Expected effect
- Curriculum alone: ~same as corrupted context or better (based on Rho-1 evidence)
- Adversarial dropout alone: should beat uniform corruption (targeted vs random)
- Combined: potentially compounded improvement — the model never gets complacent

---

## Priority Order

1. **R2-15a (adversarial dropout)**: Implement first — zero cost, builds on existing code
2. **R2-15b (curriculum)**: Implement second — requires offline scoring script + data loader changes
3. **R2-15c (combined)**: Just env vars — free once a and b exist

## Experiment Matrix Addition

| # | Config | What it tests |
|---|--------|--------------|
| R2-15a | `CAUSAL_PROBE=1 CAUSAL_PROBE_MODE=adversarial` | Per-position adversarial masking |
| R2-15b | `CURRICULUM=1` | Difficulty-ordered curriculum |
| R2-15c | `CURRICULUM=1 CAUSAL_PROBE=1 CAUSAL_PROBE_MODE=adversarial` | Combined |
| R2-15d | `CURRICULUM=1 CORRUPT_RATE=0.1` | Curriculum + uniform corruption (ablation) |
