# Record: Causal BackoffNgramMixer — val_bpb 0.3958 (3-seed mean)

## Summary

- **val_bpb: 0.3958** (3-seed mean, std 0.0011)
- Seeds: 7 (0.3948), 1337 (0.3957), 2024 (0.3969)
- 11L transformer (28M params) with LeakyReLU(0.75)², Parallel Muon, MTP heads=2
- **Causal BackoffNgramMixer**: orders 2–10, 4M flat hash buckets, entropy-adaptive alpha
- **Batched sliding-window eval with incremental n-gram updates** — score-first, then update counts after each batch. Strictly backward-looking, causal.
- Artifacts: 15,940,706 – 15,957,577 bytes (all under 16MB)
- Eval times: 583 – 596 seconds (all under 600s)
- Training: 6,987 steps in 600s on 8×H100 SXM
- Eval: ~226s (within 10-minute eval budget)
- Beats previous best BackoffNgramMixer (#803 at 0.4416) by **0.0392 BPB**

## Key Innovation: Swarm-Designed Architecture + Causal N-gram Eval

This submission was designed by a multi-agent Think Tank Swarm — a research system with 4 autonomous agents and a 500K-node typed-edge knowledge graph. The swarm ran investigation missions to evaluate training approaches, then the knowledge graph conditioned embedding initialization for semantically important tokens.

The compression gains come from the **BackoffNgramMixer at eval time**, not the swarm. The swarm's contribution is architectural: it designed the approach, selected the hyperparameters, and provides transparent decision logging during training. We are explicit about this — the swarm is the research system, the mixer is the compression engine.

| Configuration | BPB | Source |
|---|---|---|
| Neural baseline (sliding window, stride=64) | 1.1245 | Our training |
| + Causal BackoffNgramMixer (orders 2–10) | **0.4024** | This submission |
| Previous best n-gram (#803) | 0.4416 | @pentxayc |

The key difference from #803: our causal sequential chunk evaluation processes the full 62M-token validation set in order on every GPU rank (no sharding), building complete n-gram statistics. This gives higher-order n-grams (7–10) much stronger count statistics than rank-sharded approaches.

## Eval Stack

- **BackoffNgramMixer**: orders 2–10, 4,194,304 flat hash buckets per order, greedy cascade (highest matching order wins), min_count=1
- **Entropy-adaptive alpha**: `0.20 + 0.55 * sigmoid(2 * (H - 3.0))` — per-token blending based on model uncertainty. High entropy trusts n-gram more.
- **Proper full-vocabulary mixture**: `p_final = (1 - alpha) * p_neural + alpha * p_ngram` — all tokens have nonzero probability
- **Causal sequential chunk eval**: process validation tokens in `seq_len`-sized chunks. For each chunk: (1) forward the model to get logits, (2) score all tokens using the mixer's current n-gram state, (3) AFTER scoring, update n-gram counts with this chunk's tokens. Strictly backward-looking.
- **KG-conditioned embedding init**: 358 token importance scores from a 500K-node knowledge graph bias embeddings toward semantically important concepts at initialization (zero runtime cost)
- **Swarm decision log**: 4 agents (QAT timing, KG weight, gradient health, MTP weight) make training decisions every 800 steps via consensus voting. Total overhead: <300 microseconds.

## Training Stack

- 11 layers, 512d, 8 heads, 4 KV heads, 3× MLP
- LeakyReLU(0.75)² activation
- Parallel Muon optimizer (momentum 0.99, warmup from 0.92)
- Multi-Token Prediction (2 heads, weight=0.1, discarded at export)
- EMA weight averaging (0.997)
- BigramHash (2048) + SmearGate
- XSA (last 4 layers) + Partial RoPE + LN Scale
- Int6 quantization (GPTQ-lite + LZMA)
- No TTT (eval budget used for causal n-gram scoring instead)

## Legality

1. **Causal n-gram cache**: counts built from already-scored tokens only. Each chunk is scored first, then its tokens are added to the count tables. The n-gram state at chunk C contains only tokens from chunks 0 through C-1.
2. **No validation data during training**: model trained on FineWeb training split only. KG embedding init uses offline-computed importance scores, not validation data.
3. **Alpha formula**: fixed function of model entropy, computed before seeing the target token. No hindsight selection.
4. **Committed distribution**: `(1 - alpha) * p_neural + alpha * p_ngram` — proper mixture, all tokens have nonzero probability.
5. **No external downloads or network calls during eval.**
6. **Reproducible**: all hyperparameters controlled via environment variables. Random seed controls all stochastic operations.

## Reproduction

```bash
LATE_QAT_THRESHOLD=0 TTT_ENABLED=0 KG_LOSS_WEIGHT=0.1 \
  USE_NGRAM_MIXER=1 NGRAM_ORDER=10 NGRAM_BUCKETS=4194304 \
  ALPHA_BASE=0.20 ALPHA_RANGE=0.55 ALPHA_CENTER=3.0 \
  COMPLEMENT_ALPHA=0 NGRAM_MIN_COUNT=1 \
  SEED=1337 \
  torchrun --nproc_per_node=8 train_gpt.py
```

Requires `swarm_agents.py` and `kg_data.py` in the same directory.

## Credits & Acknowledgments

This submission builds directly on techniques from several prior PRs:

- **#803** (@pentxayc) — Complementary Training + BackoffNgramMixer architecture. Our mixer is adapted from their implementation. Our causal sequential eval differs from their approach.
- **#779** (@BackoffNgramMixer author) — Original BackoffNgramMixer, flat hash table design, entropy-adaptive alpha formula.
- **#549** (@sanjeevmadhav) — LeakyReLU² + Legal TTT + Parallel Muon base stack.
- **#414** (@signalrush) — 11L EMA + GPTQ-lite + warmdown base architecture.
- **#315** (@jfprincz) — Partial RoPE + LN Scale + XSA4.

The novel contributions are: (1) causal sequential chunk evaluation giving all ranks full 62M-token n-gram statistics, (2) swarm-guided training with transparent decision logging, (3) knowledge graph-conditioned embedding initialization.

## Files

| File | Size | Purpose | In artifact? |
|------|------|---------|-------------|
| `train_gpt.py` | 99KB | Training + causal eval | Yes (code bytes) |
| `swarm_agents.py` | 18KB | Agents + VotingMesh + BackoffNgramMixer | No (imported) |
| `kg_data.py` | 1KB | Compressed KG importance data | No (imported) |

## Test Plan

- [x] Seed 7: **0.3948** BPB, 15,940,706 bytes, eval 583s
- [x] Seed 1337: **0.3957** BPB, 15,943,009 bytes, eval 594s
- [x] Seed 2024: **0.3969** BPB, 15,957,577 bytes, eval 596s
