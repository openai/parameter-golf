# Record: Learned Multi-Expert Gate + Frozen Oracle + Backoff TTT (3-seed mean val_bpb=0.1663)

**val_bpb: 0.1663** (3-seed mean, std 0.0003) | **<16 MB** | 8xH100 SXM, 600s

## Results (8xH100 80GB SXM)

| Seed | Pre-TTT bpb | Post-TTT bpb | Eval time | Artifact |
|------|-------------|--------------|-----------|----------|
| 1337 | 1.1265 | **0.1661** | 308s | 15.74 MB |
| 42 | 1.1320 | **0.1663** | 305s | 15.76 MB |
| 2024 | 1.1352 | **0.1666** | 303s | 15.25 MB |
| **Mean** | 1.1312 | **0.1663** | 305s | |
| **Std** | | **0.0003** | | |

## Background

PR #779 (deanbrr) introduced the BackoffNgramMixer with entropy-adaptive alpha and drift-free TTT, achieving 0.6683 BPB. The entropy-adaptive alpha uses a hand-crafted heuristic capped at 0.60, which significantly underweights the n-gram cache when it becomes mature during later eval chunks.

This submission replaces the fixed heuristic with a **learned multi-expert gate** trained end-to-end during the main training loop, and introduces a **frozen n-gram oracle** pre-computed from training data for efficient gradient-based gate training.

## Technique

### 1. Learned Multi-Expert Gate (Transformer Head)

Instead of a fixed entropy-based alpha, we add a small `nn.Linear(model_dim, 7)` head to the GPT model that outputs per-token logits over 7 experts:
- Expert 0: Neural model prediction
- Experts 1-6: N-gram orders 2 through 7

The gate is trained end-to-end alongside the main language modeling objective. During the forward pass:

1. Compute standard cross-entropy loss from neural logits
2. Compute per-expert probabilities: `[p_neural, p_2gram, p_3gram, ..., p_7gram]`
3. Apply masked softmax over valid experts (masking orders with insufficient context)
4. Enforce a 5% minimum floor on the neural expert weight for stability
5. Compute mixed probability: `p_mixed = sum(weights * expert_p)`
6. Add mixer loss: `L_mixer = -log(p_mixed)` weighted by 0.1

The gate learns from the model's hidden state which expert to trust for each token, enabling per-token routing that a fixed heuristic cannot match.

### 2. Frozen N-gram Oracle (Pre-computed from Training Data)

To provide the n-gram probabilities needed for the mixer loss during training, we pre-fill the `BackoffNgramMixer` hash tables from all 80 training shards (8B tokens) at the start of training. This takes ~19 seconds and is counted within the 10-minute wallclock budget.

After pre-filling, the tables are frozen — no `update()` calls during training. The alpha head sees mature n-gram statistics from step 1, enabling effective gradient-based learning throughout training.

The "future token leakage" from using full-corpus statistics is negligible: any single token contributes ~1/8B = 0.000000000125 to the aggregate counts.

### 3. GPU-Native BackoffNgramMixer

The entire n-gram mixer operates on GPU using PyTorch tensor operations:
- Count tables: `torch.int32` tensors on device (1M buckets × 2 tables × 6 orders = 48MB)
- Updates via `torch.scatter_add_` (no CPU-GPU transfers)
- Hash lookups via direct tensor indexing

This eliminates the CPU bottleneck from the original numpy implementation.

### 4. Pre-compilation of Mixer Loss Path

The mixer forward+backward path is pre-compiled via `torch.compile` using dummy data before the wallclock timer starts. This avoids a ~12s JIT compilation penalty during training. The pre-compilation uses zero tensors and does not touch training data.

## Order of Operations (Legality Proof)

### Training Phase (within 600s wallclock)

```
1. Model init, warmup steps, torch.compile           [OUTSIDE wallclock]
   - Standard model warmup (20 steps) + state reset
   - torch.compile of mixer path with DUMMY ZEROS     ← no training tokens

2. ──── WALLCLOCK STARTS (t0 = time.perf_counter()) ────

3. N-gram pre-fill (~19s)                            [INSIDE wallclock]
   - Stream all 80 training shards through BackoffNgramMixer.update()
   - Hash tables populated with full-corpus n-gram counts
   - Tables FROZEN after this point — no more update() calls during training

4. Training loop (~562s, ~5400 steps)                [INSIDE wallclock]
   For each step:
     a. Load mini-batch (x, y) from training data
     b. Query FROZEN n-gram tables:
        train_mixer._ngram_backoff_p(x, y) → per-order probabilities
        (lookup only, no update — tables unchanged since step 3)
     c. Forward pass through GPT model:
        - Compute neural logits from transformer
        - Cross-entropy loss on neural logits
        - alpha_head(hidden_state) → 7 expert gate logits
        - Masked softmax over valid experts (neural + n-gram orders 2-7)
        - 5% floor on neural expert weight
        - mixed_p = weighted sum of expert probabilities
        - mixer_loss = -log(mixed_p), added to CE with weight 0.1
     d. Backward pass + optimizer step (Muon + Adam)
     e. EMA weight update (decay=0.997)

5. ──── WALLCLOCK ENDS (~581s of 600s budget) ────
```

### Evaluation Phase (after training, ~305s)

```
6. Serialize model: EMA weights → int6+zstd (15.7 MB)

7. Load quantized model into fresh eval_model

8. TTT eval (eval_val_sliding_ttt):
   - Create FRESH BackoffNgramMixer (empty, no training data)
   - 60 chunks × 1M tokens each, stride=64

   For each chunk ci:
     ┌─ Phase 1: SCORE (torch.inference_mode, no gradient) ─┐
     │  For each batch of windows in this chunk:             │
     │    a. Forward pass → neural logits + gate logits      │
     │    b. Query eval mixer for n-gram probabilities       │
     │       (only tokens ALREADY in the cache from          │
     │        previously scored chunks 0..ci-1)              │
     │    c. Multi-expert mixing with learned gate           │
     │    d. Record NLL for scored positions                 │
     └──────────────────────────────────────────────────────┘
     │
     │  dist.barrier() — all ranks finish scoring chunk ci
     │
     ├─ Cache update: mixer.update(val_tokens[ci_start:ci_end])
     │  (tokens from chunk ci added to cache AFTER scoring)
     │
     ┌─ Phase 2: TRAIN on chunk ci (already scored = legal) ┐
     │  Standard cross-entropy TTT on Q projections only     │
     │  (no mixer loss — just CE on neural logits)           │
     │  Cosine LR decay across chunks                        │
     └──────────────────────────────────────────────────────┘
```

Key invariants:
- **Training**: N-gram tables frozen after pre-fill. Only lookups during gradient steps — never updated from training batches.
- **Eval**: Fresh cache. Each chunk scored BEFORE its tokens are added to the cache. No future token information can leak.
- **TTT training**: Uses standard cross-entropy loss only (not mixer loss). Unfreezes Q projections + norms + alpha_head.

## What the Gate Learned

The expert logit statistics reveal a clear hierarchy (seed 1337):

| Expert | Mean Logit | Interpretation |
|--------|-----------|----------------|
| Neural | -5.52 | Rarely trusted |
| 2-gram | -16.78 | Almost never used |
| 3-gram | -12.13 | Rarely used |
| 4-gram | -8.94 | Rarely used |
| 5-gram | -6.21 | Sometimes used |
| 6-gram | -3.48 | Moderately used |
| **7-gram** | **+8.09** | **Dominant expert** |

The 7-gram expert is the only one with a positive mean logit, confirming it as the dominant predictor when the cache is mature. The gate automatically falls back to lower-order n-grams or the neural model when higher orders lack coverage.

## Wallclock Budget Breakdown

| Phase | Time | Inside wallclock? |
|-------|------|-------------------|
| Model init + warmup steps | ~25s | No |
| torch.compile (standard path) | ~8s | No |
| torch.compile (mixer path, dummy zeros) | ~12s | No |
| **N-gram pre-fill (8B tokens)** | **~19s** | **Yes** |
| **Training (~5400 steps)** | **~562s** | **Yes** |
| Eval (sliding window + TTT) | ~305s | After training |

Total training wallclock: ~581s of 600s budget.

## Compliance

- **Score-first TTT:** Each chunk scored under `torch.inference_mode()` before any training on that chunk
- **Backward-looking n-gram:** Eval-time cache built from scratch; counts only from already-scored chunks, updated strictly after scoring
- **N-gram pre-fill counted in wallclock:** The 19s pre-fill from training data is inside the 10-minute budget
- **Frozen oracle during training:** After pre-fill, n-gram tables are read-only — no `update()` calls during the training loop
- **torch.compile outside wallclock:** Pre-compilation uses dummy zeros, no training tokens accessed
- **No oracle selection:** Gate depends on model hidden state, never compares mixed vs original NLL
- **No training data at eval:** Eval mixer is created fresh, built causally from validation data only
- **TTT uses CE loss only:** TTT training step uses standard cross-entropy, not the mixer loss
- **Token count verified:** ratio_scored = 1.000000
- **Artifact under 16MB:** Max 15.76 MB across seeds

## Reproduction

```bash
pip install zstandard
SEED=1337 MAX_WALLCLOCK_SECONDS=600 \
USE_MIXER=1 MIXER_ETA=0.02 MIXER_HEAD=multi \
QTTT=1 TTT_EPOCHS=1 TTT_FREEZE_BLOCKS=1 TTT_LR=0.00003 \
TTT_CHUNK_TOKENS=1048576 EVAL_STRIDE=64 \
CROWN_Q_LAMBDA=0.01 PRUNE_PCT=0.08 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## TTT Configuration

| Parameter | Setting |
|-----------|---------|
| Unfrozen params | Q projections + norms + alpha_head (QTTT=1) |
| TTT LR | 0.00003 with cosine decay across chunks |
| Chunk size | 1M tokens (60 chunks) |
| Epochs per chunk | 1 |
| Optimizer | AdamW |
| Loss | Standard cross-entropy (byte-weighted) |
| Mixer eta | 0.02 |

## Architecture

11L, 512d, GQA 8H/8KV, MLP 3x, LeakyReLU(0.5)^2, XSA all 11 layers, Value Residual, Gated Attention, SmearGate, BigramHash(4096), Partial RoPE(16/64), LN Scale, EMA(0.997). Tied embeddings. Muon optimizer. Multi-expert gate head (Linear 512→7). ~5400 steps in 581s (19s pre-fill + 562s training).

## Credits

- **PR #779 deanbrr** - BackoffNgramMixer, entropy-adaptive alpha, drift-free TTT, base architecture
- **PR #700 RoyiRa** - Base architecture, TTT framework, stride=64 eval
- **PR #606 gowtham0992** - int5 + Soft-Round QAT model
- **PR #727 Asukabot0** - Multi-order backoff concept, entropy-adaptive alpha formula
- **PR #461 Christopher-Lee-McClendon** - TTT recipe foundations
- **PR #518 sofiabod** - LeakyReLU(0.5)^2, cosine TTT scheduling
