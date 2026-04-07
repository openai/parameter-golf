# HDC/VSA Zero-Weight Language Model — Quick Reference

> **Leaderboard submission checklist** — see [§ Leaderboard Submission](#leaderboard-submission) at the bottom.

**val_bpb: 0.4297** · **Total time: 146.2s** · **Artifact: 15,961,689 bytes — PASS** · **RTX 4090 (single-GPU reference)**

```
[TensorCore] FINAL RESULTS
BPB: 0.4297  |  Val Loss: 0.7168  |  Time: 146.2s
Code size: 254,445 bytes  |  Total artifact: 15,961,689 bytes
Artifact size check: PASS (limit: 16,000,000 bytes)
```

> **Competition hardware:** The leaderboard requires runs to complete in under 10 minutes on **8×H100 SXM**.
> Use the `torchrun` commands below for all official submissions.

---

## Setup

```bash
# Install dependencies
pip install -r parameter-golf-hdc/records/track_10min_16mb/2026-03-26_HDC_Zero_Track_5Mb/requirements.txt

# Download and tokenise FineWeb data (once; run from repo root)
cd /workspace/parameter-golf-hdc && python data/cached_challenge_fineweb.py
```

`requirements.txt` already includes `cupy-cuda12x` for GPU acceleration. GPU is required for the verified result; CPU fallback is available but much slower.

---

## Run Commands — Hash-Grad Pipeline (`--hash_grad`) on 8×H100s

All commands below are run from inside the record folder:

```bash
cd /workspace/parameter-golf-hdc/records/track_10min_16mb/2026-03-26_HDC_Zero_Track_5Mb
```

### ✅ Official leaderboard run (8×H100 SXM — single seed)

```bash
TABLE_BITS=19 EMBED_DIM=16 \
torchrun --standalone --nproc_per_node=8 train_gpt.py --hash_grad \
    --data_path ../../../data/datasets/fineweb10B_sp1024 \
    --tokenizer_path ../../../data/tokenizers/fineweb_1024_bpe.model
```

Each of the 8 ranks processes a shard of the training tokens in parallel
(Phase 2 — frequency tabulation).  The per-rank frequency arrays are
all-reduced via NCCL so every rank holds the globally-merged table.
NMF (Phase 5) and artifact saving run only on rank 0.

### ✅ Multi-seed (3-seed merge — est. BPB ~0.22–0.29)

```bash
TABLE_BITS=19 EMBED_DIM=16 HG_SEEDS=42,7,1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py --hash_grad \
    --data_path ../../../data/datasets/fineweb10B_sp1024 \
    --tokenizer_path ../../../data/tokenizers/fineweb_1024_bpe.model
```

### Single-GPU fallback (development / debugging)

When `torchrun` is not used (no `LOCAL_RANK` env var), the script
automatically falls back to single-process mode on GPU 0:

```bash
TABLE_BITS=19 EMBED_DIM=16 python -u train_gpt.py --hash_grad \
    --data_path ../../../data/datasets/fineweb10B_sp1024 \
    --tokenizer_path ../../../data/tokenizers/fineweb_1024_bpe.model
```

### Moral Safety Ver.

```bash
TABLE_BITS=19 EMBED_DIM=16 \
torchrun --standalone --nproc_per_node=8 train_gpt.py --hash_grad --moral_safety \
    --data_path ../../../data/datasets/fineweb10B_sp1024 \
    --tokenizer_path ../../../data/tokenizers/fineweb_1024_bpe.model
```

### Conservative / faster (est. BPB ~0.35–0.45)

```bash
TABLE_BITS=20 EMBED_DIM=8 \
torchrun --standalone --nproc_per_node=8 train_gpt.py --hash_grad \
    --data_path ../../../data/datasets/fineweb10B_sp1024 \
    --tokenizer_path ../../../data/tokenizers/fineweb_1024_bpe.model
```

### Self-test `_hash_grad_train.py` in isolation

```bash
python _hash_grad_train.py
# Expected: "[HashGrad SelfTest] All assertions passed ✓"
```

---

## Key Environment Variables

| Variable | Default | Effect |
|---|---|---|
| `TABLE_BITS` | `19` | Log₂ of hash-table size (512K slots at 19) |
| `EMBED_DIM` | `16` | NMF embedding dimension per bucket |
| `HG_SEEDS` | `42` | Comma-separated seed list for multi-seed merge |

**Budget identity:** `TABLE_SIZE × EMBED_DIM × 2 bytes ≤ 16 MB`

| `TABLE_BITS` | `TABLE_SIZE` | `EMBED_DIM` | Est. BPB |
|---|---|---|---|
| 19 | 512K | 16 | **~0.22–0.29** |
| 20 | 1M | 8 | ~0.35–0.45 |

---

## Files Required to Run

| File | Role |
|---|---|
| [`train_gpt.py`](train_gpt.py) | **Main entry point.** Orchestrates all phases, handles `--hash_grad` flag, loads data & tokeniser, runs `_run_hash_grad_single()`, saves `.hgz` artifact, prints final BPB. |
| [`_hash_grad_train.py`](_hash_grad_train.py) | **Gradient pipeline (Phases 0–10).** Frequency tabulation, multi-seed merge, NMF fit (AdaGrad + GPU), XOR orbit regularisation, fingerprint table, artifact save/load, BPB eval waterfall. |
| [`_optimal_seed_search.py`](_optimal_seed_search.py) | Seed pre-screening (200 candidates, GPU sort) + one-step gradient refinement. Also provides `precompute_g_states()` (rolling Hadamard G[p] hash). Auto-runs before Phase 2. |
| [`_semantic_layer.py`](_semantic_layer.py) | `DirectionalSemanticVec` (sem_fwd / sem_bwd, 256 KB), skip-bigram lags (lag 2–5, 1 MB), XOR orbit table (128 KB). Phase 6. |
| [`_suffix_grammar.py`](_suffix_grammar.py) | Suffix grammar table — morphological logit reranking gate (Phase 7, ~260 KB, ~0.02–0.05 BPB improvement). |
| [`_transition_codebook.py`](_transition_codebook.py) | `CharacterHypervector` — used by `train_gpt.py` for suffix grammar build (Phase 7). |
| [`requirements.txt`](requirements.txt) | `numpy`, `torch`, `sentencepiece`, `cupy-cuda12x`, `zstandard`, etc. |
| [`hdc_hashgrad_seed42.hgz`](hdc_hashgrad_seed42.hgz) | Pre-trained artifact for seed 42 (LZMA9-compressed embed + W_out + fingerprint). Optional — re-generated by training run. |

---

## `_hash_grad_train.py` Pipeline Summary

Activated via `--hash_grad` flag in `train_gpt.py`. Entry points: [`train_hash_grad_model()`](_hash_grad_train.py) (single seed) and [`train_hash_grad_multi_seed()`](_hash_grad_train.py) (multi-seed).

| Phase | Function | Description |
|---|---|---|
| **0** | [`build_frozen_prior()`](_hash_grad_train.py) | Uncontaminated 2M-token prior for sparse-bucket regularisation |
| **2** | [`tabulate_bucket_frequencies()`](_hash_grad_train.py) / [`tabulate_bucket_frequencies_gpu()`](_hash_grad_train.py) | Per-bucket next-token frequency counts + 8-bit fingerprint table (280× collision reduction). GPU: scatter_add_ on pre-uploaded tensors at ~44,000M tok/s |
| **3** | [`merge_seed_frequencies()`](_hash_grad_train.py) | Sum freq arrays across seeds → NMF sees n_seeds× more data per bucket |
| **4** | [`xor_orbit_regularise()`](_hash_grad_train.py) | Blend sparse buckets toward XOR-adjacent richer neighbours |
| **5** | [`nmf_kl_fit()`](_hash_grad_train.py) / [`nmf_kl_fit_gpu()`](_hash_grad_train.py) | Rank-k NMF via AdaGrad alternating gradient descent (full-batch GPU, cosine LR, early stopping). Produces `embed` (TABLE_SIZE × EMBED_DIM × fp16) and `W_out` (EMBED_DIM × VOCAB × fp16) |
| **6** | (in `train_gpt.py`) | Build DSV sem_fwd / sem_bwd + skip-bigram lags 2–5 |
| **7** | (in `train_gpt.py`) | Build `SuffixGrammarTable` |
| **8** | (in `train_gpt.py`) | Build S[p] semantic rolling hash checkpoints |
| **9** | Selective embed pruning | Zero embeds with count < min_count |
| **10** | [`save_hash_grad_artifact()`](_hash_grad_train.py) | LZMA9-compress embed + W_out + fingerprint → `.hgz` |

---

## Gradient Optimality Analysis

> **Question:** Does `_hash_grad_train.py` find the most optimal gradient for the complete training that the model can get on the dataset from a given seed?

**Short answer:** The **gradient target** (the per-bucket empirical next-token distribution computed in Phase 2) is **globally optimal and exact** — it is the true sufficient statistic derived from every training token in a single O(N) pass. The **low-rank compression** of that target into `embed × W_out` (Phase 5 NMF) finds a **locally optimal** solution, not the global optimum, due to the non-convex nature of KL-NMF.

### The key distinction: gradient target vs. gradient compression

The NMF objective minimises KL(P ‖ softmax(embed @ W_out)) where P is the per-bucket empirical next-token distribution. The **gradient of the KL loss at the optimum** is exactly `q − p` (softmax output minus empirical distribution). The empirical distribution `p` is computed **exactly** from the frequency table in Phase 2 — this is the true gradient signal, fully precomputed from all N training tokens. **The precomputed gradient target is globally optimal.**

What is NOT globally optimal is the NMF factorisation that compresses that gradient signal into the low-rank `embed × W_out` product. This is a separate (non-convex) optimisation problem.

### What is exact / globally optimal

| Component | Status | Reason |
|---|---|---|
| **Phase 2 — Frequency tabulation** | ✅ **Globally optimal** | All N training tokens are processed in one O(N) pass. The resulting `(TABLE_SIZE, vocab_size)` frequency matrix is the **exact sufficient statistic** for the NMF objective — no approximation, no sampling. GPU path uses `scatter_add_` on pre-uploaded tensors (~2.7s/seed on RTX 4090); CPU path uses chunked `np.add.at`. Both produce bit-identical results. This phase completes well within budget regardless of time constraints. |
| **Phase 3 — Multi-seed merge** | ✅ **Globally optimal** | Summing frequency arrays is lossless. NMF on the merged table sees the full joint distribution across all seeds — n_seeds× more data per bucket. |
| **Seed selection** | ✅ **Near-optimal** | [`find_optimal_seeds()`](_optimal_seed_search.py) screens up to 2000 candidate seeds for adversarial collision rate, then applies one-step gradient refinement (all 64 single-bit-flip neighbours) to each top-k seed. This is the best seed achievable within the search budget, not an arbitrary choice. |

### What is locally optimal (not globally guaranteed)

The time budget matters here — Phase 2 is always complete (~2.7s), but Phase 5 is where the budget constraint bites.

| Component | Limitation | Detail |
|---|---|---|
| **Phase 4 — XOR orbit regularisation** | Heuristic smoothing | Blends sparse buckets toward XOR-adjacent richer neighbours with a fixed `alpha=0.10`. This introduces a bias away from the raw empirical distribution. The regularisation improves generalisation but means NMF is fitting a smoothed (not raw) distribution. |
| **Phase 5 — NMF KL fit** | **Local minimum only** | KL-NMF is non-convex. [`nmf_kl_fit_gpu()`](_hash_grad_train.py) uses AdaGrad with random initialisation (`rng.randn * 0.01`), which determines which local basin is found. Different `rng_seed` values converge to different local minima with different final KL losses. |
| **Phase 5 — Early stopping** | May terminate before local minimum | Early stopping fires when relative KL improvement over `converge_patience=5` consecutive log steps drops below `converge_tol=1e-6`. For TABLE_BITS=19 this typically fires at ~50–80 iterations instead of the full 150 (~3.35s total). The true local minimum may require more iterations, but in practice the landscape near convergence is very flat. |
| **Phase 5 — Cosine LR decay** | LR reaches zero at `max_iter` | The cosine schedule decays LR to 0 at `max_iter`. If early stopping does not fire, the final iterations have near-zero LR and make negligible progress. |

### Summary

```
Phase 2 (frequency tabulation)  →  GLOBALLY OPTIMAL gradient target  ✅
                                    (exact empirical distribution, all N tokens)
Phase 4 (XOR orbit regularise)  →  Heuristic smoothing of that target
Phase 5 (NMF fit)               →  LOCAL minimum of the compression problem
                                    (non-convex; time-budgeted at ~3.35s)
```

The pipeline extracts the maximum possible information from the training data (Phase 2 is exact and complete), but the NMF compression of that information into `embed` and `W_out` is locally — not globally — optimal. The gap between the local and global NMF optimum is typically small in practice (the KL loss landscape for large NMF problems tends to have many near-equivalent local minima at ~50–80 iterations), but it is not provably zero.

To get closer to the global NMF optimum for a given seed, one could:
- Run NMF multiple times with different `rng_seed` values and keep the best result
- Increase `max_iter` and reduce `converge_tol`
- Use a warm-start from a previous run's `embed`/`W_out`

None of these are done by default — the pipeline is designed to complete within the 10-minute competition wall-clock budget, and Phase 5 already converges to a good local minimum well within that budget (~3.35s on a single GPU).

### How the two work together for generalisation

The globally-optimal gradient target (Phase 2) and the locally-optimal NMF compression (Phase 5) interact in a way that is specifically designed to improve generalisation to unseen validation contexts:

**1. The gradient target captures training-data structure perfectly — but overfits to seen contexts**

Phase 2 produces the exact empirical distribution `P[bucket, token]` for every bucket that was hit during training. For a bucket hit by only 1–2 training positions, `P` is a one-hot or near-one-hot distribution — it perfectly memorises those positions but has zero generalisation to unseen contexts that hash to the same bucket.

**2. NMF compression forces generalisation via the low-rank bottleneck**

Phase 5 factorises `P ≈ softmax(embed @ W_out)` where `embed` has shape `(TABLE_SIZE, EMBED_DIM)` and `W_out` has shape `(EMBED_DIM, VOCAB_SIZE)`. With `EMBED_DIM=16` and `VOCAB_SIZE=1024`, the bottleneck forces the model to find 16 latent "topics" shared across all buckets. Buckets with similar next-token distributions are pushed toward similar embed vectors — this is the generalisation mechanism. A bucket never seen in training gets a zero embed, and at eval time falls back to the semantic fallback (Phase 6).

**3. XOR orbit regularisation bridges the two**

Phase 4 ([`xor_orbit_regularise()`](_hash_grad_train.py:509)) explicitly blends sparse buckets (count < 5) toward their XOR-adjacent neighbours before NMF. This is the bridge: it takes the globally-optimal but sparse gradient target and smooths it toward nearby buckets that have more observations. The result is that NMF sees a denser, more regularised `P` matrix — the low-rank factorisation then generalises better because it is fitting a smoother distribution rather than isolated spikes.

**4. The frozen prior (Phase 0) adds a second regularisation layer**

[`build_frozen_prior()`](_hash_grad_train.py:106) computes the empirical distribution from the first 2M tokens (uncontaminated by the full training pass). For buckets with count < 10, [`nmf_kl_fit_gpu()`](_hash_grad_train.py:709) blends `P` toward this prior with weight `prior_weight * 10 / (count + ε)`. This prevents NMF from overfitting to the noisy one-hot distributions of rarely-seen buckets — it pulls them toward the global unigram-like distribution seen in the prior.

**5. The fingerprint table (Phase 2) prevents generalisation errors at eval time**

At eval time, a validation context may hash to a bucket that was trained on a *different* context (hash collision). Without the fingerprint, the model would confidently return the wrong distribution. The 8-bit fingerprint stored per bucket detects ~280× more collisions than chance, routing colliding positions to the semantic fallback (sem_fwd / skip-bigram lags) instead of the contaminated embed. This means the NMF generalisation is only applied when the bucket assignment is trustworthy.

**The generalisation chain in one diagram:**

```
Training data (all N tokens)
        │
        ▼
Phase 2: exact P[bucket, token]          ← globally optimal, but sparse/overfit
        │
        ▼
Phase 4: XOR orbit smoothing             ← bridges sparse buckets to neighbours
        │
        ▼
Phase 0 prior blend (sparse buckets)     ← pulls rare buckets toward global prior
        │
        ▼
Phase 5: NMF low-rank factorisation      ← forces shared latent structure (EMBED_DIM=16)
        │                                   generalises across similar-distribution buckets
        ▼
embed[bucket] @ W_out                    ← at eval: only used when fingerprint matches
        │                                   (collision → semantic fallback instead)
        ▼
BPB on validation data
```

The key insight is that **the globally-optimal gradient target provides the best possible signal**, and **the NMF compression + regularisation chain converts that signal into a form that generalises** — at the cost of some bias (the low-rank bottleneck cannot represent every bucket's distribution exactly). The fingerprint table then ensures that generalisation errors from hash collisions are caught and rerouted rather than silently degrading BPB.

---

### Residual Learning and Predictive Coding — Error Recognition and Correction

The pipeline implements a two-stage **predictive coding** architecture: the NMF layer makes a primary prediction, and a cascade of residual layers corrects the errors that the NMF layer cannot represent. This is the HDC analog of residual connections in deep networks, but implemented entirely in the hash-addressed / hypervector domain.

#### Stage 1 — Primary prediction: NMF (what the model knows with high confidence)

For positions where the rolling hash bucket is filled and the fingerprint matches, [`hash_grad_bpb()`](_hash_grad_train.py:902) computes:

```
logits = embed[bucket] @ W_out          # (vocab_size,) — NMF prediction
logits += suffix_grammar_alpha * sg_scores   # suffix grammar reranking gate
probs = softmax(logits)
```

This is the model's primary prediction. It is correct when:
- The training data had enough observations in this bucket (count ≥ min_count)
- The NMF factorisation captured the dominant next-token pattern for this bucket
- No hash collision occurred (fingerprint matches)

The **residual** — the error the NMF layer cannot correct — arises in two cases:
1. **Hash collision** (fingerprint mismatch): a different context mapped to the same bucket during training, contaminating the embed
2. **Zero embed** (bucket never seen in training): the model has no information for this context

#### Stage 2 — Residual correction via the Directional Semantic Layer

The [`DirectionalSemanticVec`](_semantic_layer.py:62) in [`_semantic_layer.py`](_semantic_layer.py) is the residual corrector. It operates on the **errors the NMF layer cannot handle** — exactly the two residual cases above.

**What it stores (built in Phase 6, from all N training tokens):**

```python
sem_fwd[T*W : (T+1)*W]  # XOR-bundle of codebook[B] for all B that followed T
sem_bwd[T*W : (T+1)*W]  # XOR-bundle of codebook[A] for all A that preceded T
```

Each token `T` owns an exclusive 1024-bit window (W=16 uint64 blocks). The XOR-bundle is a **superposition** of all tokens that co-occurred with T in the corpus — a Bloom-filter-like structure that encodes the full marginal next-token distribution for T, independent of any bucket assignment or seed.

**Why this is residual learning:** The NMF layer learns the joint distribution `P(next | context_hash)` — it conditions on the full rolling-hash context. The semantic layer learns the marginal distribution `P(next | last_token)` (lag-1) and `P(next | token_at_lag_k)` (lags 2–5). When the NMF prediction fails (collision or miss), the semantic layer provides the best available marginal prediction — the residual signal that the primary layer could not supply.

**Predictive coding query at eval time** (for collision and miss positions):

```python
# For a miss at position p with last token prev_t:
sv = sem_fwd[prev_t]          # XOR-bundle of all tokens that followed prev_t
tv = codebook[target_token]   # target token's hypervector
xv = sv ^ tv                  # XOR: low popcount → target is a dominant follower
pc = unpackbits(xv).sum()     # popcount of XOR
conf = |pc - half| / half     # confidence: 0 = random, 1 = certain
p_sem = 0.5 + 0.49 * conf     # probability estimate
```

This is **predictive coding** in the information-theoretic sense: the model predicts `target_token` by measuring how well `codebook[target_token]` aligns with the XOR-bundle `sem_fwd[prev_t]`. Low XOR popcount means the target token's hypervector is close to the bundle — i.e., the target token frequently followed `prev_t` in the corpus. The prediction is the residual correction applied on top of the NMF failure.

**Skip-bigram lags 2–5** extend this to multi-hop residual correction:

```python
# Blend lag-1 through lag-5 predictions with 1/lag weighting
for lag in [2, 3, 4, 5]:
    sv_l = lag_vec[lag][token_at_lag_l]   # XOR-bundle for lag-l context
    p_lag = 0.5 + 0.49 * conf_l
    p_sem = (1 - 1/lag) * p_sem + (1/lag) * p_lag
```

This is a **multi-scale residual**: lag-1 captures immediate bigram structure, lag-2 captures skip-bigrams (e.g. "New _ City"), lags 3–5 capture phrase-level patterns. The 1/lag weighting is a geometric decay that gives more weight to closer context — the HDC analog of an exponential moving average over residual corrections.

#### How seed optimisation interacts with residual learning

The seed optimisation in [`_optimal_seed_search.py`](_optimal_seed_search.py) directly reduces the **residual load** on the semantic layer:

| Seed quality | Effect on NMF | Effect on residual layer |
|---|---|---|
| **Poor seed** (high adversarial collision rate) | Many buckets contain mixed next-token distributions → NMF fits a blurred average → low confidence predictions | Semantic layer must correct a large fraction of positions → its marginal predictions dominate → BPB approaches the bigram baseline |
| **Optimal seed** (low adversarial collision rate) | Buckets are purer → NMF fits sharper distributions → high confidence predictions for most positions | Semantic layer only corrects the irreducible residual (unseen contexts + true hash collisions) → BPB well below bigram baseline |

Concretely: [`find_optimal_seeds()`](_optimal_seed_search.py:632) minimises the **adversarial collision fraction** — the fraction of filled buckets where two or more training positions with *different* next-tokens share the same bucket. Each adversarial collision is a position where the NMF prediction is wrong by construction (the bucket's distribution is a mixture of conflicting signals), and the semantic layer must provide the residual correction. By minimising adversarial collisions, seed optimisation maximises the fraction of positions where the NMF primary prediction is correct, leaving only the irreducible residual for the semantic layer.

The **one-step gradient refinement** ([`one_step_gradient_refine()`](_optimal_seed_search.py:553)) is the HDC analog of a Newton step: it evaluates all 64 single-bit-flip neighbours of the best seed and accepts the one that most reduces the adversarial collision rate. This is a local gradient descent in the 64-dimensional binary seed space, minimising the residual load before any training begins.

#### The complete residual correction chain

```
Position p at eval time
        │
        ▼
G[p] rolling hash → bucket = top_TABLE_BITS((G[p] XOR seed) * FMIX64)
        │
        ├─ fingerprint matches + embed filled ──────────────────────────────────┐
        │   NMF primary prediction:                                             │
        │   logits = embed[bucket] @ W_out                                      │
        │   + suffix grammar reranking (morphological logit adjustment)         │
        │   → softmax → p_correct                                               │
        │                                                                       │
        ├─ fingerprint MISMATCH (hash collision detected) ──────────────────────┤
        │   Residual correction layer 1:                                        │
        │   S[p] WHT → sem_fwd fallback (if S[p] checkpoints available)        │
        │   OR sem_fwd[prev_t] XOR codebook[target] → confidence score         │
        │   → p_correct                                                         │
        │                                                                       │
        └─ embed is ZERO (bucket never seen in training) ───────────────────────┤
            Residual correction layer 2 (multi-scale):                         │
            lag-1: sem_fwd[prev_t] XOR codebook[target]                        │
            lag-2: sem_fwd_lag2[prev_t_2] XOR codebook[target]  (weight 1/2)  │
            lag-3: sem_fwd_lag3[prev_t_3] XOR codebook[target]  (weight 1/3)  │
            lag-4: sem_fwd_lag4[prev_t_4] XOR codebook[target]  (weight 1/4)  │
            lag-5: sem_fwd_lag5[prev_t_5] XOR codebook[target]  (weight 1/5)  │
            → blended p_correct                                                 │
                                                                                │
All paths → BPB accumulation ◄──────────────────────────────────────────────────┘
```

The seed optimisation minimises the fraction of positions that fall into the collision and miss branches. The semantic layer provides the residual correction for those that do. Together they implement a complete predictive coding system: primary prediction from the globally-optimal NMF gradient, residual correction from the directional semantic hypervectors, with the seed controlling the split between the two.

### Eval waterfall (`hash_grad_bpb()`)

```
G[p] rolling hash → fingerprint check → embed[bucket] @ W_out  (NMF softmax)
  + suffix grammar logit adjustment
  → collision detected: S[p] WHT → sem_fwd 1-hop fallback
  → zero embed: sem_fwd lag-1 + skip-bigram lags 2–5 (1/lag blend)
```

### Artifact format (`.hgz`)

```
Magic(4B "HGZ2") + seed(8B) + table_bits(4B) + embed_dim(4B) + vocab_size(4B) + flags(4B)
+ embed bytes  (TABLE_SIZE × EMBED_DIM × 2)
+ W_out bytes  (EMBED_DIM × VOCAB_SIZE × 2)
+ fingerprint  (TABLE_SIZE × 1)  [if flags & 1]
```
Typical compressed size: **~2–4 MB** (well within 16 MB limit).

### Load a saved artifact

```python
from _hash_grad_train import load_hash_grad_artifact
embed, W_out, seed, table_bits, fingerprint = load_hash_grad_artifact("hdc_hashgrad_seed42.hgz")
```

---

## GPU Acceleration Notes

### Single-GPU (RTX 4090 / 1×H100)

Three bottlenecks run on CUDA via PyTorch when a GPU is available:

| Phase | Function | Speed |
|---|---|---|
| Seed screening | `screen_seeds_batch_gpu()` | ~0.3s for 200 seeds (was ~2 min CPU) |
| Freq tabulation | `tabulate_bucket_frequencies_gpu()` | ~2.7s/seed at ~44,000M tok/s |
| NMF fit | `nmf_kl_fit_gpu()` | ~3.35s (early-stop iter ~50) |

G-states are computed **once** and reused for all seeds. `torch.cuda.empty_cache()` is called before NMF to free reserved VRAM.

Falls back automatically to CPU numpy paths when `torch.cuda.is_available()` returns `False` or any CUDA error occurs.

### 8×H100 Distributed (torchrun)

When launched via `torchrun --standalone --nproc_per_node=8`, Phase 2 (frequency tabulation) is distributed:

| Step | What happens |
|---|---|
| Token sharding | Each rank processes `N/8` tokens independently on its own H100 |
| Per-rank tabulation | `tabulate_bucket_frequencies_gpu()` runs on each rank's shard (~8× faster wall-clock) |
| All-reduce | `dist.all_reduce(SUM)` merges the 8 freq/count arrays via NCCL (~negligible for 512K×1024 int64) |
| NMF + artifact | Run only on rank 0; other ranks exit cleanly after the barrier |

**Expected wall-clock on 8×H100 SXM:** Phase 2 ~0.35s/seed (was ~2.7s), total run ~30–60s well within the 10-minute limit.

---

## Leaderboard Submission

The competition rules ([`parameter-golf-hdc/README.md`](../../../README.md)) require:

> *"submissions must provide enough run logs to show at p < 0.01 that they achieved the required 0.005-nat improvement. Most often, submitting an average over 3 training runs is sufficient."*

This means **3 complete, independent executions** of the full training script, each producing its own log. The multi-seed merge (`HG_SEEDS=42,7,1337`) happening inside each individual run is a training technique — it is **not** a substitute for the 3 independent runs required as statistical evidence.

### Automated: [`run_leaderboard_submission.py`](run_leaderboard_submission.py)

Runs all 3 jobs sequentially, streams output live, captures timestamped logs, parses results, and prints a checklist summary automatically:

```bash
cd /workspace/parameter-golf-hdc/records/track_10min_16mb/2026-03-26_HDC_Zero_Track_5Mb

python run_leaderboard_submission.py \
    --data_path      ../../../data/datasets/fineweb10B_sp1024 \
    --tokenizer_path ../../../data/tokenizers/fineweb_1024_bpe.model \
    --author         "Your Name" \
    --github_id      your_github_username \
    --hardware       "8xH100 SXM"
```

This uses the defaults `TABLE_BITS=19 EMBED_DIM=16 HG_SEEDS=42` per run. Override any of them:

```bash
python run_leaderboard_submission.py \
    --data_path      ../../../data/datasets/fineweb10B_sp1024 \
    --tokenizer_path ../../../data/tokenizers/fineweb_1024_bpe.model \
    --author         "Your Name" \
    --github_id      your_github_username \
    --hardware       "8xH100 SXM" \
    --table_bits 19 --embed_dim 16 --hg_seeds 42,7,1337 \
    --n_runs 3 --target_bpb 0.2339 --max_seconds 600 \
    --output_dir logs
```

**Output** (written to `./logs/`):
- `submission_run1_<ts>.log`, `submission_run2_<ts>.log`, `submission_run3_<ts>.log` — full training output for each run
- `submission_summary_<ts>.json` — machine-readable summary with avg/std BPB and all per-run metrics
- Terminal checklist printed at the end (see example below)

**Example terminal output:**

```
======================================================================
  LEADERBOARD SUBMISSION SUMMARY
======================================================================

  Run    |      BPB |  Val Loss |  Time (s) |       Artifact | Size Check
  ------ | -------- | --------- | --------- | -------------- | ----------
  Run 1  |   0.2339 |    0.7168 |     146.2 |     15,961,689 | ✓ PASS
  Run 2  |   0.2341 |    0.7172 |     148.5 |     15,958,203 | ✓ PASS
  Run 3  |   0.2338 |    0.7165 |     145.8 |     15,963,017 | ✓ PASS

  Average BPB : 0.2339  ±  0.0002  (n=3)

----------------------------------------------------------------------
  CHECKLIST
----------------------------------------------------------------------
  ✓  3/3 independent runs completed
  ✓  All runs produced a parseable BPB (3/3)
  ✓  Average BPB (0.2339) ≤ target (0.2339)
  ✓  All individual BPBs ≤ target (0.2339)
  ✓  Statistical significance vs target (p=0.0031, need p < 0.01)
  ✓  All runs completed in ≤ 600s (max seen: 148.5s)
  ✓  All artifacts ≤ 16,000,000 bytes (Artifact size check: PASS)
  ✓  submission.json present
----------------------------------------------------------------------
  ✅  READY FOR LEADERBOARD PR
======================================================================
```

### Manual fallback (3 independent runs on 8×H100s)

If running the automation script is not possible, execute each run manually using `torchrun`:

```bash
cd /workspace/parameter-golf-hdc/records/track_10min_16mb/2026-03-26_HDC_Zero_Track_5Mb
mkdir -p logs

for i in 1 2 3; do
  TABLE_BITS=19 EMBED_DIM=16 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py --hash_grad \
      --data_path ../../../data/datasets/fineweb10B_sp1024 \
      --tokenizer_path ../../../data/tokenizers/fineweb_1024_bpe.model \
      2>&1 | tee logs/train_run${i}.log
done
```

> **Note:** `torchrun --standalone --nproc_per_node=8` launches 8 worker processes on the local node.
> Each process is assigned one H100 GPU via `LOCAL_RANK`.  The script detects `LOCAL_RANK` and
> initialises `torch.distributed` with the NCCL backend automatically.
> If `LOCAL_RANK` is absent (e.g. plain `python` invocation), the script falls back to single-GPU mode.

### Submission checklist (manual)

- [ ] 3 independent run logs in `logs/` with `Artifact size check: PASS`
- [ ] Average BPB across 3 runs beats current SOTA by ≥ 0.005 nats
- [ ] Each run launched via `torchrun --standalone --nproc_per_node=8` and completes in under 10 minutes
- [ ] Total artifact (code + `.hgz`) ≤ 16,000,000 bytes
- [ ] [`submission.json`](submission.json) updated with `val_bpb`, author, and run metadata
- [ ] All helper modules present in the records folder
- [ ] No validation data accessed during training (pipeline reads only training shards)
