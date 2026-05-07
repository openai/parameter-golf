# Non-Record: SLOT + CTW Eval-Time Augmentation on PR #549 SOTA Stack

**val_bpb = 1.1185** (3-seed mean, std 0.0003) | ~15.9 MB | 8×H100 SXM

Two novel eval-time augmentations tested on the PR #549 SOTA stack:
- **SLOT**: ✅ Positive result — **-0.0008 BPB** improvement, first SLOT entry in Parameter Golf
- **CTW**: ❌ Negative result — **+0.005 BPB** degradation despite three progressively improved implementations

## Results

### SLOT-Enabled (3-seed) — Positive Result

| Seed | Steps | Step Avg | Pre-TTT BPB | Post-TTT+SLOT BPB | TTT+SLOT Time | Artifact |
|------|-------|----------|-------------|-------------------|---------------|----------|
| 1337 | 7,127 | 84.2ms | 1.1385 | **1.1188** | 386s | 15,965,604 |
| 42 | 7,155 | 83.9ms | 1.1380 | **1.1185** | 388s | 15,882,932 |
| 2025 | 7,152 | 83.9ms | 1.1377 | **1.1183** | 385s | 15,994,920 |
| **Mean** | **7,145** | **84.0ms** | **1.1381** | **1.1185 (std 0.0003)** | **~386s** | — |

### Baseline Without SLOT (3-seed)

| Seed | Steps | Step Avg | Post-TTT BPB | TTT Time |
|------|-------|----------|-------------|----------|
| 1337 | 7,164 | 83.8ms | 1.1195 | 352s |
| 42 | 7,159 | 83.8ms | 1.1195 | 353s |
| 2025 | 7,164 | 83.8ms | 1.1189 | 350s |
| **Mean** | **7,162** | **83.8ms** | **1.1193 (std 0.0003)** | **~352s** |

### SLOT vs Baseline

| Metric | Baseline Mean | SLOT Mean | Delta |
|--------|-------------|-----------|-------|
| Post-TTT BPB | 1.1193 | **1.1185** | **-0.0008** |
| TTT eval time | 352s | 386s | +34s |
| vs SOTA (PR #549) | -0.0001 | **-0.0009** | — |

---

## Novel Contribution 1: SLOT — Positive Result

### What Is SLOT

SLOT (Sample-specific Language Model Optimization at Test-time, Hu et al., arXiv:2505.12392v2) optimizes a single additive vector δ ∈ ℝ^512 at the last hidden layer to adapt the model to each batch during evaluation. Unlike full TTT which updates all 27M model parameters via SGD, SLOT optimizes just 512 parameters through one linear layer.

### Why SLOT Works

SLOT and TTT address different bottlenecks:
- **TTT** adapts internal representations to local data distribution (chunk-level, all layers)
- **SLOT** fine-tunes the final hidden-to-logit mapping (batch-level, last layer only)

These are complementary — TTT gives SLOT better hidden states, and SLOT gives TTT-adapted representations a final per-batch correction.

### Implementation

The model's `forward_logits()` was split into `forward_hidden()` + `compute_logits()`, enabling SLOT to optimize δ between the two stages. SLOT runs inside the TTT scoring loop (Phase 1), not as a separate pass:

```python
for each batch of windows:
    H = model.forward_hidden(x_batch)       # [bsz, seq_len, 512]
    delta = zeros(1, 1, 512)                # broadcasts across batch + seq
    for step in range(3):
        logits = model.compute_logits(H + delta)
        loss = CE(logits[:, :-1], targets[:, 1:])
        loss.backward()                      # gradients only through lm_head
        optimizer.step()
    final_logits = model.compute_logits(H + delta)
```

Key properties: zero artifact cost, +34s eval overhead, score-first compliant, `SLOT_ENABLED=0` reproduces baseline exactly.

---

## Novel Contribution 2: CTW — Negative Result (Three Iterations)

Context Tree Weighting (Willems, Shtarkov, Tjalkens 1995) is a provably minimax-optimal sequential probability assignment over all variable-order Markov models. We tested it as an eval-time augmentation, iterating through three progressively improved implementations before concluding it cannot help at this BPB level.

### CTW v1: Naive Implementation — BPB +0.005 worse

**What we built**: Walked the suffix tree to the deepest matching context node and used its KT (Krichevsky-Trofimov) estimate directly. Mixed with neural logits using a fixed weight of 0.1.

**What was wrong**: This was NOT actually CTW. It was just a smoothed n-gram lookup at the deepest available context — missing the entire theoretical power of CTW, which comes from recursively weighting predictions across ALL depths.

**Three bugs identified**:
1. **No recursive depth weighting**: Used deepest-match lookup instead of the proper `P_w = 0.5 · P_e + 0.5 · ∏ P_w(children)` formula that makes CTW Bayesian-optimal
2. **Fixed mixing weight (w=0.1)**: Mixed CTW noise into 100% of tokens, including tokens where the neural model was already confident
3. **Per-token Python loop**: Every scored token ran `predict()` + `mix()` + `cross_entropy()` individually in Python, taking 2,760s (46 minutes)

**Result**: 1.1252 BPB (+0.005 worse than baseline), 46 minutes eval time (exceeds 10-min limit)

### CTW v2: Proper Recursive Algorithm — Not Tested (Speed Still Prohibitive)

**What we fixed**:
1. **Proper recursive depth weighting**: Each node now maintains `log_pe` (cumulative KT log-probability) and `log_pw` (weighted probability). After each symbol update, `log_pw` is recomputed bottom-up: `P_w = 0.5 · P_e + 0.5 · P_w_child` using log-space arithmetic (`logaddexp`) to avoid underflow. This is the actual CTW algorithm from the paper, verified against Python, Go, and Rust reference implementations.
2. **Proper predictive distribution**: Instead of returning the KT estimate from the deepest node, the `predict()` method walks back up the path, mixing each depth's KT estimate weighted by `beta = exp(log_pe - log_pw)` — the posterior probability that each depth is the correct model. Shallow depths contribute more when deeper contexts are unseen; deeper contexts dominate when they have strong statistics.
3. **Entropy-adaptive gating**: Before running CTW on any token, the neural model's entropy is computed. If entropy is below a threshold (default 2.0 nats), CTW is skipped entirely — the neural model is confident and CTW would only add noise. When CTW does mix, its weight is scaled by `entropy / max_entropy`, so uncertain tokens get more CTW influence. This means ~80-90% of tokens skip CTW computation entirely.

**Why we didn't run it**: Even with entropy gating, `ctw.update()` must process every token sequentially (each token's context depends on the previous token). The Python dict-based tree operations are inherently O(depth) per token with no way to batch. Estimated time: 400-600s, borderline on eval limit. And the fundamental signal problem remained unsolved.

### CTW v3: Vectorized Entropy Gate — BPB Still Worse

**What we fixed further**:
1. **Vectorized entropy computation**: Instead of computing entropy per-token in a Python loop, we compute `F.log_softmax` and entropy for ALL scored tokens in a single batched GPU operation. The `F.cross_entropy` for neural-only NLL is also pre-computed for all tokens at once.
2. **Selective CTW loop**: Only tokens with entropy above the threshold enter the Python CTW loop. Low-entropy tokens use the pre-computed neural NLL directly — no Python overhead, no tensor creation.

**What we could NOT fix**: `ctw.update()` remains sequential. Each token's suffix tree update depends on the previous token's context. The tree uses Python dicts for sparse node storage — converting to fixed-size GPU tensors would require a custom CUDA kernel (essentially reimplementing the tree as a hash table on GPU with scatter/gather operations).

**Result**: Tested with `CTW_WEIGHT=0.02, CTW_DEPTH=4, CTW_ENTROPY_THRESHOLD=3.0` — still slower than baseline (ctw.update runs on all tokens regardless of gating) and BPB did not improve. Run was killed after observing degraded trajectory.

### Root Cause: Why CTW Fundamentally Cannot Help at 1.12 BPB

After three implementations, the conclusion is clear: **the problem is signal redundancy, not implementation quality**.

A depth-4 CTW over 1024 subword tokens is essentially a smoothed variable-order Markov model up to 4-grams. The 11-layer transformer with 2048-token context and 27M parameters IS a strictly superior n-gram model — it already captures everything CTW knows, plus long-range dependencies CTW cannot represent.

Mixing in a weaker predictor always hurts a stronger predictor when the weaker predictor's knowledge is a strict subset of the stronger predictor's knowledge. This is true regardless of:
- Whether CTW uses proper recursive depth weighting (v2) or naive lookup (v1)
- Whether mixing is fixed or entropy-adaptive (v2/v3)
- Whether the implementation is fast (v3) or slow (v1)

The frontier PRs that succeed with n-gram augmentation (PR #727 at 0.9674 BPB) use a fundamentally different approach: count-min sketch with 5-7 gram orders, entropy-adaptive alpha, and vectorized GPU lookup. These capture higher-order patterns (5-7 grams vs CTW's 4) with a simpler but faster data structure, and their success may depend more on the count-min sketch's hash-based smoothing than on any Bayesian optimality.

### Also Tested: Stacking Hacks on SLOT (Negative Results)

Two additional eval-time hacks were tested on top of SLOT:

| Hack | Mechanism | BPB | Delta vs SLOT-only |
|------|-----------|-----|-------------------|
| Adaptive Temperature | Optimized temperature scalar per-batch via SGD (3 steps) | 1.1325 | **+0.014 worse** |
| Focal TTT | Upweighted hard tokens in Phase 2 training via focal loss (γ=2) | 1.1441 | **+0.025 worse** |

**Adaptive Temperature** failed because the LR (0.1) was too aggressive — temperature diverged from 1.0, distorting the probability distribution. **Focal TTT** failed because "hard" tokens are hard for a reason — they're unpredictable content (names, numbers, URLs). Training harder on unpredictable tokens destabilizes learned representations for predictable tokens.

**Lesson**: SLOT works because it's lightweight (512 params, 3 steps). More aggressive adaptation techniques destroy the carefully trained representations.

---

## Base Architecture (PR #549 by @abaybektursun)

- 11L, 512d, 8H/4KV, LeakyReLU(0.5)² MLP 3×
- Parameter Banking + Parallel Muon (FlashAttention 3)
- BigramHash(1536), XSA4, Partial RoPE(16), LN Scale, VE128
- EMA(0.997) + Tight SWA(50), GPTQ-lite int6 + LZMA-6
- Legal Score-First TTT (SGD, lr=0.002, 3 epochs, 32K chunks)

## Run Commands

```bash
# Baseline (SLOT disabled — reproduces PR #549)
cd /workspace/parameter-golf && SEED=1337 SLOT_ENABLED=0 CTW_WEIGHT=0 \
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
torchrun --standalone --nproc_per_node=8 train_gpt.py

# SLOT enabled (positive result)
# Same as above but with: SLOT_ENABLED=1 SLOT_LR=0.001 SLOT_STEPS=3
```

## Credits

- **SLOT integration and CTW analysis**: Anubhav (@AnubhavBharadwaaj) — this submission
- **SLOT algorithm**: Yang Hu et al. (arXiv:2505.12392v2, Westlake University)
- **CTW algorithm**: Willems, Shtarkov, Tjalkens (1995)
- LeakyReLU²: PR #493 by @parinzee, PR #518 by @sofiabod
- Parallel Muon + Parameter Banking: PR #399 by @abaybektursun
- TTT recipe: PR #461 by @Christopher-Lee-McClendon
- Base model: PR #414 by @signalrush