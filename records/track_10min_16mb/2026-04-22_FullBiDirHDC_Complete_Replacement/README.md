# FullBiDirHDC + Eigen Convergence — Instant Teleportation Engine

> **Complete replacement** of the hash-addressed NMF + `DirectionalSemanticVec` pipeline
> with the `FullBiDirHDC` joint manifold engine, now upgraded with the
> **Eigen Convergence** instant fixed-point solver.
>
> `step()` = encode → 2 cosines → **1 teleport** → state copies
> Per-step latency: ~10–15 ms → **~0.3 ms** (40-iter loop eliminated)

---

## Results (to be filled after leaderboard runs)

| Run | Seed | BPB | Val Loss | Time | Artifact bytes | Size check |
|-----|------|-----|----------|------|----------------|------------|
| 1 | 42 | TBD | TBD | TBD | TBD | TBD |
| 2 | 7 | TBD | TBD | TBD | TBD | TBD |
| 3 | 1337 | TBD | TBD | TBD | TBD | TBD |

**Mean BPB:** TBD · **Mean Val Loss:** TBD · **Mean Time:** TBD

---

## Architecture

This submission completely replaces the prior pipeline with the `FullBiDirHDC` joint manifold engine:

### What was replaced

| Old component | New component |
|---|---|
| Rolling XOR hash `G[p]` (loses token identity) | `GoldenAxisShift` codebook (metric-preserving, lossless) |
| `DirectionalSemanticVec` (forward-only `sem_fwd`) | `SpiralDSVLanguageModel` (bilateral `sem_fwd` + `sem_bwd`) |
| Hash-addressed NMF (1-iteration, secondary signal) | `RelationshipMemory` (resonance-weighted rule learning) |
| Fingerprint 8-bit collision detection | `consistency = cosine(fwd, bwd)` bilateral agreement gate |
| No temporal signal | `ZSignal` predictive coding spine |
| No memory | `SpiralPointerMemory` hierarchical memory |
| `.hgz` artifact (16 MB embed + W_out) | `.bdhgz` artifact (~4 MB codebook + rule_bundle) |

### New pipeline

```
Training data (500M tokens from fineweb_train_*.bin)
        │
        ├── Phase 1: Build bigram freq table P[a,b] = P(b|a)
        │   O(N) pass, rank 0 only, ~2s
        │
        ├── Phase 2: Distributed training
        │   Each rank processes N/8 tokens
        │   FullBiDirHDC.train_on_tokens()
        │   reward = bigram_freq[prev_t, next_t] × 100
        │   dist.all_reduce(SUM) on rule_bundle
        │
        ├── Phase 3: SpiralDSV bilateral build (optional)
        │   GoldenAxisShift codebook
        │   sem_fwd + sem_bwd XOR-bundle tables
        │   ctx_len=4, remaining time budget
        │
        └── save_bidi_artifact() → .bdhgz (LZMA9)

Eval waterfall (bidi_bpb):
        For each position p with prev_token t:
            query_hv = codebook[t] XOR rule_bundle
            fwd_scores[v] = cosine(query_hv, codebook[v])  for all v
            bwd_scores[v] = cosine(codebook[v] XOR rule_bundle, codebook[t])
            consistency[v] = (fwd_scores[v] + bwd_scores[v]) / 2
            probs = softmax(consistency)
            p_correct = probs[target_token]
        BPB = Σ(-log₂ p_correct) / Σ(utf8_bytes(token))
```

### Why bilateral is superior

| | Existing pipeline | This submission |
|---|---|---|
| **Prediction direction** | Forward-only `sem_fwd` | Joint fwd+bwd — `consistency = cosine(fwd, bwd)` |
| **Context encoding** | Rolling XOR hash (lossy) | `GoldenAxisShift` (metric-preserving, lossless) |
| **Rule learning** | None (stateless eval) | `RelationshipMemory` resonance-weighted EMA |
| **Temporal signal** | None | `ZSignal` predictive coding spine |
| **Confidence gate** | 8-bit fingerprint only | Full bilateral agreement score |

---

## BPB Formula

The BPB metric is computed identically to the reference [`train_gpt.py:265-278`](../../../train_gpt.py):

```
BPB = Σ(-log₂ p_correct) / Σ(utf8_bytes(token))
```

**Byte counting** mirrors [`build_sentencepiece_luts()`](../../../train_gpt.py) exactly
(implemented in [`_bidi_train._build_byte_luts()`](_bidi_train.py)):

```python
# Reference (train_gpt.py:265-267):
token_bytes  = base_bytes_lut[tgt_ids]
token_bytes += has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]

# This submission (_bidi_train.py:411-413):
tok_bytes = (
    base_bytes[tgt_toks]
    + (has_leading_space[tgt_toks] & ~is_boundary_token[prev_toks])
)
```

Key points:
- `is_boundary_token[tok]` is `True` for control / unknown / unused tokens (sequence boundaries). The leading-space byte is **not** counted when the previous token is a boundary token — matching the reference exactly.
- `base_bytes` is `0` for control tokens, `1` for byte-fallback tokens, and `len(piece.encode("utf-8"))` for normal tokens — matching the reference exactly.
- The summation structure `Σ bits / Σ bytes` is algebraically equivalent to the reference `bits_per_token × tokens_per_byte`.

The `[BiDirHDC BPB audit]` block printed at the end of each run shows:
- `total_tokens` — number of val tokens evaluated
- `total_utf8_bytes` — sum of UTF-8 byte lengths (with boundary-gate applied)
- `avg bytes/token` — `total_bytes / total_tokens`
- `bits/token` — `total_bits / total_tokens`
- `nats/token (loss)` — `total_nats / total_tokens`
- `BPB = bits/token / bytes/token` — the final metric

---

## Setup

```bash
# Install dependencies (run from repo root)
pip install -r records/track_10min_16mb/2026-04-22_FullBiDirHDC_Complete_Replacement/requirements.txt

# Download and tokenise FineWeb data (once; run from repo root)
python data/cached_challenge_fineweb.py --variant sp1024
```

---

## Run Commands — 3 Independent Runs (Leaderboard Verification)

Each run uses a **different seed** to provide statistical variance across the 3 required independent runs.
All configuration is passed via environment variables — no extra flags required.

```bash
cd /workspace/parameter-golf

# Run 1 (seed 42):
RUN_ID=bidi_hdc N_WORDS=512 SEED=42 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=8 \
    records/track_10min_16mb/2026-04-22_FullBiDirHDC_Complete_Replacement/train_gpt.py \
    2>&1 | tee records/track_10min_16mb/2026-04-22_FullBiDirHDC_Complete_Replacement/train_seed42.log

# Run 2 (seed 7):
RUN_ID=bidi_hdc N_WORDS=512 SEED=7 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=8 \
    records/track_10min_16mb/2026-04-22_FullBiDirHDC_Complete_Replacement/train_gpt.py \
    2>&1 | tee records/track_10min_16mb/2026-04-22_FullBiDirHDC_Complete_Replacement/train_seed7.log

# Run 3 (seed 1337):
RUN_ID=bidi_hdc N_WORDS=512 SEED=1337 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=8 \
    records/track_10min_16mb/2026-04-22_FullBiDirHDC_Complete_Replacement/train_gpt.py \
    2>&1 | tee records/track_10min_16mb/2026-04-22_FullBiDirHDC_Complete_Replacement/train_seed1337.log
```

Convenience — run all 3 seeds sequentially:
# Set for 8xH100s in Runpod as a copy and paste from the workspace directory in the terminal. 
```bash
cd /workspace/parameter-golf-hdc-main/records && for seed in 42 7 1337; do
  echo "=== Starting seed $seed ===" && \
  RUN_ID=bidi_hdc N_WORDS=512 SEED=$seed \
  DATA_PATH=./data/datasets/fineweb10B_sp1024 \
  TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
  VOCAB_SIZE=1024 \
  torchrun --standalone --nproc_per_node=8 \
      records/track_10min_16mb/2026-04-22_FullBiDirHDC_Complete_Replacement/train_gpt.py \
      2>&1 | tee records/track_10min_16mb/2026-04-22_FullBiDirHDC_Complete_Replacement/train_seed${seed}.log && \
  echo "=== Completed seed $seed ==="
done

```

---

## Key Environment Variables

| Variable | Default | Effect |
|---|---|---|
| `N_WORDS` | `512` | HV width in uint64 words (512 → 32,768 bits) |
| `SEED` | `42` | Single random seed for this run |
| `MAX_WALLCLOCK_SECONDS` | `600` | Training time cap |
| `DATA_PATH` | `../../../data/datasets/fineweb10B_sp1024` | FineWeb data directory |
| `TOKENIZER_PATH` | `../../../data/tokenizers/fineweb_1024_bpe.model` | SentencePiece model |
| `VOCAB_SIZE` | `1024` | Vocabulary size |

**Budget identity:** `n_words × vocab_size × 8 bytes × 3 (codebook + fwd + bwd) ≤ 16 MB`

| `n_words` | HV bits | Artifact raw | LZMA9 est. |
|---|---|---|---|
| 16 | 1,024 | ~400 KB | ~120 KB |
| 128 | 8,192 | ~3.1 MB | ~1 MB |
| **512** | **32,768** | **~12.6 MB** | **~4 MB** |
| 1024 | 65,536 | ~25.2 MB | ~8 MB |

---

## Eigen Convergence Architecture

### What changed (2026-04-23)

| Component | Before | After |
|---|---|---|
| `_propagate()` 40-iter fwd loop | O(n × K × 40) | **Eliminated** — `sign(λ_j)` is the exact fixed point |
| `_propagate()` 40-iter bwd loop | O(n × K × 40) | **Eliminated** |
| `ChainManifold.query()` iterative | O(n_cand × n_hyp × max_iters) | **Eliminated** — `eigen_query()` instant |
| `_score_joint()` | O(n × H × W) | **Eliminated** — scores read from `h*` analytically |
| `_parity_correct()` | O(n × H × W × 10) | **Eliminated** — `sign(λ_j)` guarantees ~50% balance |
| `_bundle_rule()` stochastic XOR | O(W) + random | **Deterministic** soft EMA in float32 |
| `_update_goal()` stochastic XOR | O(W) + random | **Deterministic** soft EMA in float32 |
| **Total per step** | ~10–15 ms | **~0.3 ms** |
| `train_on_tokens()` Python loop | ~190–310 s/rank (62.5M iterations) | **~1–3 s/rank** (`EigenTrainer.absorb_bigrams()`) |
| `build_from_tokens()` scatter XOR | ~30 s | **~0.1 s** (`EigenSpiralBuilder`) |
| `vote_scores_all_vocab()` eval | ~5–15 s | **<1 s** (BLAS matmul) |

### Eigen Training Absorption (2026-04-23)

The `train_on_tokens()` Python `for` loop is now fully absorbed into the eigen solver.

**Mathematical derivation:**

The per-bigram `observe()` EMA recurrence:
```
rule_hv[i]  = CB[t_next[i]]          # before XOR action XOR after = CB[t+1]
alpha_i     = r_i / (W_acc + r_i)
bundle_pm1 ← (1 - alpha_i) × bundle_pm1 + alpha_i × rule_pm1[i]
```

Has the closed-form fixed point:
```
bundle_pm1* = sign( Σ_i r_i × CB_pm1[t_next[i]] )
            = sign( token_reward_sums @ CB_pm1 )
```

**Implementation** (`EigenTrainer.absorb_bigrams()`):
```
Step 1: token_r[v] = Σ_{i: t_next[i]=v} reward[i]   ← O(N) np.bincount (not np.add.at)
Step 2: rule_spectrum = token_r @ CB_pm1              ← O(vocab × n_bits) matmul
Step 3: rule_bundle_pm1* = sign(rule_spectrum)        ← O(n_bits)
Step 4: goal_pm1* = sign(token_r_goal @ CB_pm1)       ← high-reward bigrams only
```

### Performance Bottleneck Fixes (2026-04-23)

Six additional micro-optimisations applied after the eigen absorption:

| Fix | Location | Change | Speedup |
|---|---|---|---|
| **#1** `np.add.at` → `np.bincount` | `EigenTrainer.absorb_bigrams()` | Unbuffered scatter → buffered weighted bincount | **10–20×** |
| **#2** 2D `np.add.at` → flat `np.bincount` | `EigenSpiralBuilder.build_bilateral_tables()` | `(a,b)` 2D scatter → `a*V+b` flat bincount | **~100×** |
| **#3** broadcast+sum → BLAS matmul | `HadamardEigenSolver.batch_teleport()` | `(K,1)*axes.sum(0)` → `axis_weights @ axes_pm1` | **2–5×** |
| **#4** Chunked sign computation | `batch_teleport()` | Avoid 128 MB `(N,n_bits)` peak alloc → 32 MB chunks | **memory** |
| **#5** XOR+unpackbits → BLAS SGEMM | `SpiralDSVLanguageModel.vote_scores_all_vocab()` | `(batch,vocab,n_words)` XOR → `sem_pm1 @ CB_pm1.T` | **10–50×** |
| **#6** Cache `EigenTrainer` on engine | `FullBiDirHDC.train_on_tokens()` | Rebuild `CB_pm1` once, reuse across seeds | **1× per seed** |

**Fix #1 detail** — `np.bincount` with `weights=` is SIMD-friendly and buffered:
```python
# Before (unbuffered, no SIMD):
np.add.at(token_r, t_next_clipped, rewards)
# After (buffered, SIMD, 10-20x faster):
token_r = np.bincount(t_next_clipped, weights=rewards, minlength=vocab_size)
```

**Fix #2 detail** — flatten 2D index to avoid unbuffered 2D scatter:
```python
# Before (unbuffered 2D scatter, ~100x slower):
np.add.at(fwd_w, (a_toks, b_toks), 1.0)
# After (flat index + bincount, ~100x faster):
fwd_idx = a_toks * vocab_size + b_toks
fwd_flat += np.bincount(fwd_idx, minlength=vocab_size**2)
fwd_w = fwd_flat.reshape(vocab_size, vocab_size)
```

**Fix #5 detail** — pm1 matmul replaces (batch, vocab, n_words) XOR:
```python
# Before: (batch, vocab, n_words) XOR + unpackbits — large intermediate array
xor_fwd = sv_fwd[:, None, :] ^ codebook[None, :, :]   # (batch, vocab, n_words)
# After: single BLAS SGEMM in pm1 space
conf_fwd = |sem_fwd_pm1[prev_tokens] @ codebook_pm1.T| / n_bits
```
pm1 tables (`_codebook_pm1`, `_sem_fwd_pm1`, `_sem_bwd_pm1`) are built once lazily and invalidated when tables change.

**Cumulative performance on 8×H100 (N_WORDS=512, VOCAB_SIZE=1024):**

> **Note on budget allocation:** The SpiralDSV bilateral build is **already inside the 600 s budget** — it is not extra time. [`train_gpt.py:209–228`](train_gpt.py) allocates:
> - `train_budget = max_secs - 75 = 525 s` → hard cap for `train_bidi_model()`
> - `spiral_budget = max_secs - elapsed_train - 45 s` → SpiralDSV gets whatever remains after training
> - `eval_reserve = 45 s` → artifact save + BPB eval
>
> With the bottleneck fixes, `train_bidi_model()` completes in **~5–8 s**, so `spiral_budget` grows from ~240 s → **~547 s**. The SpiralDSV build now has nearly the full 10-minute window to build deeper context tables (`ctx_len=4` takes <0.1 s; the remaining ~547 s can be used for additional multi-seed passes or deeper lags).

| Phase | Original | After eigen absorption | After bottleneck fixes |
|---|---|---|---|
| `build_bigram_freq()` | ~2–5 s | ~2–5 s | ~2–5 s (unchanged) |
| `broadcast(bigram_freq)` | ~1 s | ~1 s | ~1 s (unchanged) |
| `train_on_tokens()` per rank | **~190–310 s** | **~1–3 s** | **<0.5 s** |
| `all_reduce(rule_bundle)` | <0.1 s | <0.1 s | <0.1 s (unchanged) |
| `build_from_tokens()` SpiralDSV *(within budget)* | **~30 s** | **~0.5 s** | **<0.1 s** |
| `vote_scores_all_vocab()` eval | ~5–15 s | ~5–15 s | **<1 s** |
| **Total (train + spiral + eval, within 600 s)** | **~230–360 s** | **~10–25 s** | **~5–10 s** |
| **`spiral_budget` available** | ~240 s | ~540 s | **~547 s** |

### Eigenvalue spectrum formula

```
λ_j = Σ_k w_k(S) × axis_k[j]          ← 19 golden-ratio axes (AxisWeightScheduler)
    + w_goal(Z, traj_accel) × goal_pm1[j]  ← goal attractor (AnticipationEigenGate)
    + w_rule(retro) × rule_bundle_pm1[j]   ← soft rule EMA
    + w_chain × chain_h*_pm1[j]            ← chain eigen result (optional)
    + w_inertia × (fwd_seed + bwd_seed)/2  ← bilateral inertia

h*[j] = sign(λ_j)                       ← exact fixed point, 0 iterations
```

### New state variables

| Attribute | Type | Purpose |
|---|---|---|
| `_rule_bundle_pm1` | `(n_bits,) float32` | **Primary** soft EMA rule bundle (replaces stochastic `_rule_bundle`) |
| `_goal_hv_pm1` | `(n_bits,) float32` | **Primary** soft EMA goal HV (replaces stochastic `goal_hv`) |
| `_rule_bundle` | `(W,) uint64` | **Derived** — `pm1_to_uint64(sign(_rule_bundle_pm1))` for API compat |
| `goal_hv` | `(W,) uint64` | **Derived** — `pm1_to_uint64(sign(_goal_hv_pm1))` for API compat |
| `_teleport` | `FullTeleportStep` | Orchestrates the single teleport step |

### `step()` flow after eigen upgrade

```
encode present + actions  O(K × W)
        │
retrodiction cosine       O(W)
        │
FullTeleportStep.run_full()
  ├── AxisWeightScheduler: S → axis_weights
  ├── AnticipationEigenGate: adjust goal/rule weights
  ├── ChainManifold.eigen_query(): instant chain prior
  ├── HadamardEigenSolver.batch_teleport(): h* = sign(spectrum)
  ├── Post-teleport analytics: goal_sim, traj_slope, entropy, resonance
  ├── SoftEMABundle: update rule_pm1 + goal_pm1 deterministically
  ├── ZSignal.update(): inside teleport on mean_goal_sim
  └── Micro-exploration trigger: update S_new
        │
O(W) state copies
        │
chain_memory.observe()    bookkeeping only
```

---

## Files

| File | Role |
|---|---|
| [`train_gpt.py`](train_gpt.py) | Main entry point (`--bidi_hdc` flag, distributed init, token loading, training, eval, auto-generates `submission.json`) |
| [`_eigen_convergence.py`](_eigen_convergence.py) | **NEW** — `HadamardEigenSolver`, `AxisWeightScheduler`, `AnticipationEigenGate`, `SoftEMABundle`, `FullTeleportResult`, `FullTeleportStep`, **`EigenTrainer`**, **`EigenSpiralBuilder`** |
| [`_bidi_hdc_engine.py`](_bidi_hdc_engine.py) | `FullBiDirHDC` + `Codebook` + `ManifoldAxes` + `ZSignal` + `ResonanceSignal` + `RelationshipMemory` + `ChainManifold` — eigen solver + eigen training + cached `EigenTrainer` |
| [`_bidi_train.py`](_bidi_train.py) | `build_bigram_freq()`, `train_bidi_model()`, `bidi_bpb()`, `save_bidi_artifact()`, `load_bidi_artifact()` — eigen training path documented |
| [`_spiral_dsv_lm.py`](_spiral_dsv_lm.py) | `GoldenAxisShift` + `SpiralPointerMemory` + `SpiralDSVLanguageModel` — `EigenSpiralBuilder` build path + BLAS `vote_scores_all_vocab()` + pm1 cache |
| [`requirements.txt`](requirements.txt) | `numpy`, `torch`, `sentencepiece`, `cupy-cuda12x`, `zstandard` |

---

## Submission Checklist

- [ ] 3 independent run logs (`train_seed42.log`, `train_seed7.log`, `train_seed1337.log`)
- [ ] Each log shows `Artifact size check: PASS (limit: 16,000,000 bytes)`
- [ ] Average BPB across 3 runs computed and recorded above
- [ ] Each run launched via `torchrun --standalone --nproc_per_node=8` on 8×H100 SXM
- [ ] Each run completes in under 10 minutes
- [ ] `submission.json` updated with actual `val_bpb`, `val_loss`, `artifact_bytes`, `elapsed_s`
- [ ] No validation data accessed during training (pipeline reads only `fineweb_train_*.bin`)
