# Implementation Plan: Spiral DSV + Eigen Frequency Method — DSV-Only Architecture

> **On goal/danger/oxytocin trajectory steering:** The `HadamardEigenSolver` steering machinery (goal HV, danger repulsion, oxytocin attraction) from `_eigen_convergence.py` is designed for agents navigating structured environments with explicit reward signals. For next-token prediction on FineWeb, the DSV bilateral signal already encodes the relevant attractors and repulsors implicitly through PMI-centered co-occurrence statistics — `sem_fwd[a]` actively opposes `CB[b]` for anti-correlated pairs (danger repulsion) and attracts toward `CB[b]` for correlated pairs (goal/oxytocin). Adding explicit steering vectors would duplicate what the DSV already does. The one genuine addition is **coherence gating** — a running document centroid that biases toward on-topic tokens (see §Coherence Gating Extension below).

**Target:** Replace the entire 2026-04-07 pipeline with a **DSV-only** system that:
1. Removes the NMF codebook (Hadamard codebook used only by DSV XOR queries)
2. Removes the NMF embed/W_out ("random gradient" — 1-iter NMF, near-random signal)
3. Keeps only the DSV + Spiral + Eigen frequency method as the sole prediction mechanism
4. Saves the artifact (sem_fwd + sem_bwd uint64 tables) in the same `.hgz` format

**Why this is correct:**

The current pipeline has two prediction paths:
- **NMF path** (`embed[bucket] @ W_out`): fires for filled + fingerprint-matched buckets. With `nmf_max_iter=1` the KL loss stays near `ln(vocab_size)` — the embed/W_out are a single AdaGrad step from random initialisation. The README explicitly states this is a "secondary signal" and "lightweight." The Hadamard codebook is **not used** by this path.
- **DSV path** (`sem_fwd[prev_t] ^ codebook[tgt]`): fires for collisions and misses. The README states this is "the dominant signal path." The Hadamard codebook is used **only** here.

The NMF codebook (Hadamard rows) and the NMF embed/W_out are therefore two separate things that can be independently removed:
- **Remove the Hadamard codebook**: replace the XOR-popcount query with the `SpiralDSVLanguageModel` bilateral matmul query, which uses its own internal codebook (GoldenAxisShift-derived, stored inside the model).
- **Remove the NMF embed/W_out**: route all positions through the DSV path. The NMF path's contribution is marginal (1-iter NMF ≈ random) and its 16 MB budget can be reallocated to a larger DSV.

**Budget reallocation:**

| Component | Current size | New size |
|---|---|---|
| `embed` (TABLE_SIZE × EMBED_DIM × 2B) | 16 MB | **0 MB** — removed |
| `W_out` (EMBED_DIM × VOCAB_SIZE × 2B) | ~32 KB | **0 MB** — removed |
| `sem_fwd` (vocab × n_words × 8B) | 128 KB | **up to 8 MB** — expanded |
| `sem_bwd` (vocab × n_words × 8B) | 128 KB | **up to 8 MB** — expanded |
| **Total** | ~16.3 MB | **≤ 16 MB** ✅ |

With the freed 16 MB, `n_words` (uint64 words per token) can be increased from 16 to **1024**, giving each token a 65,536-bit hypervector instead of 1,024 bits. This dramatically increases the XOR-bundle capacity and confidence resolution.

---

## Architecture: Before and After

### Before (current 2026-04-07 pipeline)

```
tokens (500M)
    │
    ├─ Phase 2: freq tabulation → (TABLE_SIZE, vocab) freq table
    ├─ Phase 4: XOR orbit regularisation
    ├─ Phase 5: NMF 1-iter → embed (TABLE_SIZE, 16) fp16
    │                       W_out (16, 1024) fp16
    │           [16 MB — primary budget consumer]
    │
    └─ Phase 6: DirectionalSemanticVec scatter-XOR
                → sem_fwd (1024, 16) uint64  [128 KB]
                → sem_bwd (1024, 16) uint64  [128 KB]
                → skip_bigram_lags[2..5]     [4 × 128 KB]

Eval:
    bucket filled + fp match → embed @ W_out  (NMF, ~random)
    collision               → sem_fwd XOR codebook  (DSV)
    miss                    → sem_fwd lag blend      (DSV)
```

### After (DSV-only + Spiral + Eigen)

```
tokens (500M)
    │
    └─ Phase 6 only: EigenTrainer.build_bilateral_from_tokens()
                     with GoldenAxisShift per-lag codebook rotation
                     + PMI centering
                     → sem_fwd (1024, 1024) uint64  [8 MB]
                     → sem_bwd (1024, 1024) uint64  [8 MB]
                     [Total: 16 MB — full budget used by DSV]

Eval (all positions → single DSV path):
    ALL positions → SpiralDSVLanguageModel.vote_scores_all_vocab()
                    bilateral: sign(sem_fwd_pm1[prev] + sem_bwd_pm1[next])
                    OR prev-only: sem_fwd_pm1[prev] @ codebook_pm1.T
```

---

## What Is Removed

| Component | Removed? | Reason |
|---|---|---|
| `precompute_g_states()` | ✅ Yes | Only needed for NMF bucket hashing |
| `tabulate_bucket_frequencies()` | ✅ Yes | Only needed for NMF freq table |
| `xor_orbit_regularise()` | ✅ Yes | Only needed for NMF freq smoothing |
| `nmf_kl_fit()` | ✅ Yes | 1-iter NMF ≈ random, marginal contribution |
| `build_frozen_prior()` | ✅ Yes | Only needed for NMF regularisation |
| `embed` array (TABLE_SIZE × EMBED_DIM) | ✅ Yes | 16 MB freed for DSV |
| `W_out` array (EMBED_DIM × VOCAB_SIZE) | ✅ Yes | Freed for DSV |
| `fingerprint_packed` array | ✅ Yes | Only needed for NMF collision detection |
| Hadamard codebook (vocab × 16 uint64) | ✅ Yes | Replaced by SpiralDSV internal codebook |
| `_transition_codebook.py` | ✅ Yes | Only used by suffix grammar + old codebook |
| Phases 0, 1, 2, 3, 4, 5, 8, 9 | ✅ Yes | All NMF phases |

## What Is Kept

| Component | Kept? | Reason |
|---|---|---|
| `_spiral_dsv_lm.py` | ✅ Yes | Core DSV + GoldenAxisShift |
| `_eigen_convergence.py` | ✅ Yes | `EigenTrainer.build_bilateral_from_tokens()` |
| `_gpu.py` | ✅ Yes | GPU matmul acceleration |
| Phase 6 (DSV build) | ✅ Yes | Primary prediction mechanism |
| Phase 7 (suffix grammar) | ✅ Optional | Small additional signal (~260 KB) |
| `.hgz` artifact format | ✅ Yes | Modified to store sem_fwd + sem_bwd |
| BPB formula | ✅ Yes | Unchanged |
| Distributed token loading | ✅ Yes | All 8 ranks load tokens for DSV scan |

---

## New n_words Calculation

With 16 MB total budget split equally between `sem_fwd` and `sem_bwd`:

```
8 MB per table = 8 × 1024 × 1024 bytes
vocab_size = 1024 tokens
n_words = 8 × 1024 × 1024 / (1024 × 8) = 1024 uint64 words per token
n_bits = 1024 × 64 = 65,536 bits per token
```

This is a **64× increase** in hypervector dimensionality (from 1,024 to 65,536 bits). The XOR-bundle confidence resolution scales as `O(sqrt(n_bits))` — a 64× increase in bits gives an 8× improvement in signal-to-noise ratio for the popcount confidence estimate.

---

## New Artifact Format

The `.hgz` artifact stores only the DSV tables:

```
Magic(4B "HGZ3") + vocab_size(4B) + n_words(4B) + flags(4B)
+ sem_fwd bytes  (vocab_size × n_words × 8)   [8 MB]
+ sem_bwd bytes  (vocab_size × n_words × 8)   [8 MB]
[Total uncompressed: 16 MB]
[LZMA9 compressed: ~2–4 MB — DSV tables compress well due to structure]
```

The `SpiralDSVLanguageModel` internal codebook (vocab × n_words uint64, ~8 MB) is **not** stored in the artifact — it is regenerated deterministically from `seed=42` at eval time. This keeps the artifact within the 16 MB limit.

---

## Step-by-Step Implementation

### Step 1 — Create new record folder

Create a new record folder (e.g. `2026-04-25_SpiralDSV_Eigen_DSVOnly`) as a clean copy of the 2026-04-07 folder. All changes are made in the new folder.

### Step 2 — Copy source files

Copy from `records/track_10min_16mb/2026-04-22_FullBiDirHDC_Complete_Replacement/`:
- `_spiral_dsv_lm.py`
- `_eigen_convergence.py`
- `_gpu.py`

### Step 3 — Write new `_semantic_layer.py`

This file is the only new code needed. It wraps `SpiralDSVLanguageModel` + `EigenTrainer` and exposes a simple `build_and_eval()` function.

**Key parameters:**

```python
VOCAB_SIZE = 1024
N_WORDS    = 1024   # 65,536 bits per token — uses full 16 MB budget
CTX_LEN    = 4      # lags 1..4 with GoldenAxisShift per lag
```

**Build path:**

```python
def build_spiral_dsv(
    tokens: np.ndarray,          # (N,) int32 — 500M training tokens
    vocab_size: int = 1024,
    n_words: int = 1024,
    ctx_len: int = 4,
    seed: int = 42,
    time_budget_s: float = 300.0,
    dist_rank: int = 0,
    dist_world_size: int = 1,
    verbose: bool = True,
) -> SpiralDSVLanguageModel:
    model = SpiralDSVLanguageModel(vocab_size=vocab_size, n_words=n_words, seed=seed)
    
    # Build GoldenAxisShift offsets for lags 1..ctx_len
    for c in range(1, ctx_len + 1):
        GOLDEN_AXES.offset(c)
    axis_word_shifts = [
        (GOLDEN_AXES._word_shifts[c], GOLDEN_AXES._bit_shifts[c])
        for c in range(1, ctx_len + 1)
    ]
    
    # EigenTrainer: frequency-weighted bilateral build with PMI centering
    trainer = EigenTrainer.from_codebook_uint64(
        codebook_vecs=model.codebook,   # (vocab, n_words) uint64
        goal_threshold=10.0,
    )
    result = trainer.build_bilateral_from_tokens(
        tokens=tokens,
        ctx_len=ctx_len,
        axis_word_shifts=axis_word_shifts,
        chunk_size=2_000_000,
        verbose=verbose,
        time_budget_s=time_budget_s,
        dist_rank=dist_rank,
        dist_world_size=dist_world_size,
    )
    
    if result['sem_fwd_u64'] is not None:
        model.sem_fwd = result['sem_fwd_u64']   # (vocab, n_words) uint64
        model.sem_bwd = result['sem_bwd_u64']
        model._built = True
        model._invalidate_pm1_cache()
    
    return model
```

**Eval path:**

```python
def eval_spiral_dsv_bpb(
    val_tokens: np.ndarray,
    model: SpiralDSVLanguageModel,
    base_bytes: np.ndarray,
    has_leading_space: np.ndarray,
    is_boundary_token: Optional[np.ndarray] = None,
    batch_size: int = 500_000,
) -> Tuple[float, float]:
    """Compute BPB using bilateral DSV scores for all positions."""
    N = len(val_tokens)
    total_bits = 0.0
    total_bytes = 0
    total_nats = 0.0
    total_toks = 0
    
    model._ensure_pm1_cache()  # build pm1 cache once
    
    for chunk_start in range(1, N, batch_size):
        chunk_end = min(chunk_start + batch_size, N)
        
        prev_toks = np.clip(val_tokens[chunk_start - 1 : chunk_end - 1].astype(np.int32),
                            0, model.vocab_size - 1)
        tgt_toks  = val_tokens[chunk_start:chunk_end].astype(np.int32)
        
        # Bilateral all-vocab scores: (batch, vocab_size) float32
        # Uses GPU-accelerated BLAS matmul via _ensure_pm1_cache
        scores = model.vote_scores_all_vocab(
            prev_tokens=prev_toks,
            next_tokens=None,   # prev-only mode (next token unknown at eval time)
        )
        
        # Extract probability of correct token
        p_correct = scores[np.arange(len(tgt_toks)), tgt_toks]
        p_correct = np.clip(p_correct, 1e-30, 1.0)
        
        # Byte count (same formula as reference train_gpt.py)
        prev_t = np.clip(val_tokens[chunk_start - 1 : chunk_end - 1].astype(np.int32),
                         0, base_bytes.shape[0] - 1)
        space_guard = has_leading_space[tgt_toks]
        if is_boundary_token is not None:
            space_guard = space_guard & ~is_boundary_token[prev_t]
        tok_bytes = np.maximum(
            np.where(space_guard,
                     base_bytes[tgt_toks].astype(np.float64) + 1,
                     base_bytes[tgt_toks].astype(np.float64)), 1)
        
        total_bits  += float(-np.log2(p_correct).sum())
        total_bytes += int(tok_bytes.sum())
        total_nats  += float(-np.log(p_correct).sum())
        total_toks  += len(tgt_toks)
    
    bpb = total_bits / total_bytes
    val_loss = total_nats / total_toks
    return float(bpb), float(val_loss)
```

### Step 4 — Write new `train_gpt.py` entry point

The new `train_gpt.py` is dramatically simpler than the current one. It only needs to:

1. Parse `--hash_grad` flag (keep for compatibility)
2. Load 500M training tokens from `fineweb_train_*.bin`
3. Call `build_spiral_dsv()` (Phase 6 only — no Phases 0–5)
4. Optionally build suffix grammar (Phase 7, ~260 KB, unchanged)
5. Save artifact (new `.hgz` format with sem_fwd + sem_bwd)
6. Load 5M val tokens and call `eval_spiral_dsv_bpb()`
7. Print BPB audit block

**Distributed execution:** All 8 ranks participate in the `EigenTrainer` histogram scan (Step 2 of `build_bilateral_from_tokens`). Rank 0 runs the matmul and eval. Non-zero ranks exit after the all-reduce.

### Step 5 — Write new artifact save/load

```python
def save_spiral_dsv_artifact(
    model: SpiralDSVLanguageModel,
    path: str,
) -> int:
    """Save sem_fwd + sem_bwd to LZMA9-compressed .hgz artifact.
    
    Returns artifact size in bytes.
    """
    import lzma, struct
    header = struct.pack('<4sIII',
        b'HGZ3',
        model.vocab_size,
        model.n_words,
        0,  # flags
    )
    payload = (
        header
        + model.sem_fwd.tobytes()   # (vocab, n_words) uint64
        + model.sem_bwd.tobytes()
    )
    compressed = lzma.compress(payload, preset=9)
    with open(path, 'wb') as f:
        f.write(compressed)
    return len(compressed)

def load_spiral_dsv_artifact(
    path: str,
    seed: int = 42,
) -> SpiralDSVLanguageModel:
    """Load sem_fwd + sem_bwd from .hgz artifact.
    
    Reconstructs the SpiralDSVLanguageModel with the saved tables.
    The internal codebook is regenerated from seed (deterministic).
    """
    import lzma, struct
    with open(path, 'rb') as f:
        data = lzma.decompress(f.read())
    magic, vocab_size, n_words, flags = struct.unpack_from('<4sIII', data, 0)
    assert magic == b'HGZ3', f"Bad magic: {magic}"
    offset = struct.calcsize('<4sIII')
    table_bytes = vocab_size * n_words * 8
    sem_fwd = np.frombuffer(data, dtype=np.uint64, count=vocab_size * n_words,
                             offset=offset).reshape(vocab_size, n_words).copy()
    sem_bwd = np.frombuffer(data, dtype=np.uint64, count=vocab_size * n_words,
                             offset=offset + table_bytes).reshape(vocab_size, n_words).copy()
    model = SpiralDSVLanguageModel(vocab_size=vocab_size, n_words=n_words, seed=seed)
    model.sem_fwd = sem_fwd
    model.sem_bwd = sem_bwd
    model._built = True
    model._invalidate_pm1_cache()
    return model
```

---

## Memory Budget Verification

| Array | Shape | dtype | Size |
|---|---|---|---|
| `sem_fwd` (stored) | (1024, 1024) | uint64 | 8 MB |
| `sem_bwd` (stored) | (1024, 1024) | uint64 | 8 MB |
| `sem_fwd_pm1` (eval cache) | (1024, 65536) | float32 | 256 MB |
| `sem_bwd_pm1` (eval cache) | (1024, 65536) | float32 | 256 MB |
| `codebook_pm1` (eval cache) | (1024, 65536) | float32 | 256 MB |
| `CB_composite_pm1` (build-time) | (4096, 65536) | float32 | 1 GB |
| `fwd_hist` (build-time) | (1024, 4096) | float32 | 16 MB |
| `bwd_hist` (build-time) | (1024, 4096) | float32 | 16 MB |

**Artifact size:** 16 MB uncompressed → ~2–4 MB LZMA9 compressed ✅

**Build-time RAM:** The `CB_composite_pm1` (1 GB) and pm1 caches (3 × 256 MB = 768 MB) are the dominant consumers. On 8×H100 SXM (80 GB VRAM each), this is well within budget. On a single RTX 4090 (24 GB), `CB_composite_pm1` alone may be too large — reduce `n_words` to 256 (16 MB total, 256 MB CB_composite) if needed.

**n_words scaling table:**

| n_words | n_bits | sem_fwd+bwd | CB_composite (ctx=4) | pm1 caches (×3) | Suitable for |
|---|---|---|---|---|---|
| 16 | 1,024 | 256 KB | 64 MB | 12 MB | Any GPU |
| 128 | 8,192 | 2 MB | 512 MB | 96 MB | RTX 4090 |
| 256 | 16,384 | 4 MB | 1 GB | 192 MB | RTX 4090 (tight) |
| **1024** | **65,536** | **16 MB** | **4 GB** | **768 MB** | **H100 SXM** |

For the 8×H100 leaderboard run, use `n_words=1024`. For local testing, use `n_words=128` or `n_words=256`.

---

## What the Suffix Grammar Contributes (Phase 7)

The suffix grammar table (`_suffix_grammar.py`, ~260 KB) provides morphological logit reranking. It uses `g_states` as a pseudo-context — but `g_states` is computed from the rolling hash of the NMF pipeline, which is being removed.

**Resolution:** The suffix grammar can be rebuilt to use the DSV `sem_fwd` scores as its context signal instead of `g_states`. Alternatively, it can be dropped entirely (saving ~260 KB and simplifying the pipeline). The suffix grammar's contribution to BPB is small relative to the DSV.

**Recommendation:** Drop Phase 7 in the initial implementation. Add it back if BPB improvement is needed.

---

## Data Flow Diagram (New Architecture)

```
tokens (500M int32)
        │
        │  All 8 ranks participate in histogram scan
        ▼
EigenTrainer.build_bilateral_from_tokens()
        │
        ├─ Step 1 (rank 0): CB_composite_pm1 (4×1024, 65536) fp32
        │   Each lag c gets GoldenAxisShift rotation:
        │   CB_c = roll(CB_uint64, word_shift_c) << bit_shift_c → pm1
        │
        ├─ Step 2 (all ranks, sharded): composite histogram scan
        │   fwd_hist[a, (c-1)*V + b] = count(b follows a at lag c)
        │   bwd_hist[b, (c-1)*V + a] = count(a precedes b at lag c)
        │   GPU scatter_add_ into on-device accumulators
        │   Fused NCCL all-reduce (fwd+bwd in one collective)
        │
        ├─ Step 3 (rank 0): PMI centering
        │   fwd_hist -= row_marginals × col_marginals / total
        │   (encodes anti-correlations as negative entries)
        │
        ├─ Step 4 (rank 0): dual GPU fp16 matmul
        │   sem_fwd_pm1 = sign(fwd_hist @ CB_composite_pm1)
        │   sem_bwd_pm1 = sign(bwd_hist @ CB_composite_pm1)
        │
        └─ Step 5 (rank 0): pack to uint64
           sem_fwd (1024, 1024) uint64  [8 MB]
           sem_bwd (1024, 1024) uint64  [8 MB]
                    │
                    ▼
           save_spiral_dsv_artifact() → .hgz (~2–4 MB compressed)
                    │
                    ▼
val_tokens (5M) → eval_spiral_dsv_bpb()
                    │
                    ├─ _ensure_pm1_cache(): unpack uint64 → float32 pm1 (once)
                    │
                    └─ vote_scores_all_vocab(prev_tokens)
                       GPU BLAS: sem_fwd_pm1[prev] @ codebook_pm1.T
                       → (batch, 1024) float32 scores
                       → p_correct = scores[range(B), tgt_toks]
                       → BPB = Σ(-log2 p) / Σ(utf8_bytes)
```

---

## Implementer Todo List

```
[ ] Create new record folder 2026-04-25_SpiralDSV_Eigen_DSVOnly/
[ ] Copy _spiral_dsv_lm.py from 2026-04-22 into new folder
[ ] Copy _eigen_convergence.py from 2026-04-22 into new folder
[ ] Copy _gpu.py from 2026-04-22 into new folder
[ ] Write _semantic_layer.py with build_spiral_dsv() and eval_spiral_dsv_bpb()
    [ ] build_spiral_dsv(): SpiralDSVLanguageModel + EigenTrainer.build_bilateral_from_tokens()
    [ ] Pass GOLDEN_AXES axis_word_shifts for lags 1..ctx_len
    [ ] eval_spiral_dsv_bpb(): vote_scores_all_vocab() for all positions
    [ ] BPB formula: total_bits / total_bytes (same as reference)
[ ] Write save_spiral_dsv_artifact() and load_spiral_dsv_artifact()
    [ ] Magic: b'HGZ3', header: vocab_size + n_words + flags
    [ ] Payload: sem_fwd.tobytes() + sem_bwd.tobytes()
    [ ] LZMA9 compression
    [ ] Codebook regenerated from seed at load time (not stored)
[ ] Write new train_gpt.py (simplified — DSV-only)
    [ ] --hash_grad flag routes to DSV-only pipeline
    [ ] Load 500M training tokens from fineweb_train_*.bin
    [ ] Call build_spiral_dsv() with dist_rank / dist_world_size
    [ ] Non-zero ranks exit after EigenTrainer all-reduce
    [ ] Rank 0: save artifact, load val tokens, call eval_spiral_dsv_bpb()
    [ ] Print [HashGrad BPB audit] block (same format as before)
    [ ] Print [TensorCore] FINAL RESULTS block
[ ] Set n_words=1024 for H100 leaderboard run
[ ] Set n_words=128 for local testing / RTX 4090
[ ] (Optional) Add coherence gating to eval_spiral_dsv_bpb()
    [ ] Maintain coherence_pm1 running mean per document
    [ ] Reset on is_boundary_token
    [ ] Augment h_star: sign(sem_fwd_pm1[prev] + W_COHERENCE * coh_norm)
    [ ] Tune W_COHERENCE on held-out val subset (start at 0.1)
[ ] Run 3-seed verification (seeds 42, 7, 1337) on 8×H100
    [ ] Confirm BPB ≤ 0.4118 (target: improvement over baseline)
    [ ] Confirm artifact size ≤ 16,000,000 bytes
    [ ] Confirm run time ≤ 600s
[ ] Update README.md with new BPB result and DSV-only architecture description
[ ] Update submission.json with new val_bpb
```

---

## Coherence Gating Extension (Optional — Small BPB Gain)

This is the one genuine trajectory-steering signal that has no analogue in the current DSV: a **running document centroid** that biases predictions toward tokens coherent with the document seen so far. It is the language-model analogue of the "goal HV" in ARC-AGI-3 — but dynamically updated from the document itself rather than fixed.

### Why the Other Steering Signals Do Not Help

| ARC-AGI-3 steering signal | Language model analogue | Already handled by DSV? |
|---|---|---|
| **Goal HV** (Z-axis attractor) | Fixed document topic | Partially — coherence gating below |
| **Danger repulsion** | Anti-correlated token avoidance | ✅ Yes — PMI centering makes `sem_fwd_pm1[a]` negative for anti-correlated pairs |
| **Oxytocin attraction** | Contextually appropriate tokens | ✅ Yes — `sem_fwd_pm1[a]` is positive for correlated pairs |
| **Trajectory steering** | Semantic drift direction | ✅ Yes — GoldenAxisShift per-lag encodes directional drift |
| **Coherence gating** | Running document centroid | ❌ Not yet — this is the one new signal |

### How Coherence Gating Works

During eval, maintain a running mean of the codebook vectors for all tokens seen in the current document:

```python
# Initialise per-document state
coherence_pm1 = np.zeros(n_bits, dtype=np.float32)
doc_token_count = 0
W_COHERENCE = 0.3   # tune on held-out subset; suggested range 0.1–0.5

for each position p in val_tokens:
    if is_boundary_token[prev_tok]:   # document boundary — reset
        coherence_pm1[:] = 0.0
        doc_token_count = 0

    # Update coherence with the previous token's codebook vector
    coherence_pm1 += codebook_pm1[prev_tok]
    doc_token_count += 1

    # Coherence-augmented bilateral query:
    # h*(b) = sign(sem_fwd_pm1[prev] + W_COHERENCE × coherence_pm1 / doc_token_count)
    coh_norm = coherence_pm1 / max(doc_token_count, 1)
    h_star = np.sign(sem_fwd_pm1[prev_tok] + W_COHERENCE * coh_norm)
    h_star[h_star == 0.0] = 1.0
    scores = (h_star @ codebook_pm1.T) / n_bits   # (vocab_size,) float32
    p_correct = float(scores[tgt_tok])
```

### Cost

- **Artifact storage:** Zero additional bytes — `coherence_pm1` is a runtime variable
- **Compute:** One `(n_bits,)` float32 vector addition per token during eval — negligible vs. the matmul
- **Hyperparameter:** `W_COHERENCE` — tune on a held-out subset of FineWeb val

### Implementation Note

The `is_boundary_token` signal is already available in the eval loop (passed to `hash_grad_bpb()` in the current pipeline). Reset `coherence_pm1` whenever `is_boundary_token[prev_tok]` is True.

For batched eval, maintain a per-sequence coherence vector and update it token-by-token. The batch dimension makes this slightly more complex but still O(batch × n_bits) per step.

### Expected Gain

Small but nonzero — estimated < 0.01 BPB improvement on FineWeb. FineWeb documents have topical coherence (filtered web pages), so the coherence attractor provides a genuine signal beyond bigram statistics. The gain is larger for longer documents.

---

## Risk Assessment

| Risk | Likelihood | Mitigation |
|---|---|---|
| `CB_composite_pm1` (4 GB) OOM on H100 | Low (H100 has 80 GB) | Reduce ctx_len to 2 if needed |
| pm1 cache (768 MB) OOM during eval | Low | Eval uses CPU BLAS fallback |
| BPB worse than 0.4118 without NMF | Medium | NMF was 1-iter ≈ random; DSV was dominant signal. Test with n_words=128 first |
| Suffix grammar incompatible without g_states | High | Drop Phase 7 initially |
| LZMA9 compression of 16 MB DSV tables | Low | DSV tables have structure → compress to ~2–4 MB |
| Coherence gating W_COHERENCE too high | Low | Start at 0.1; increase only if val BPB improves |

---

## Future Direction — Multi-Agent Ensemble via GoldenAxisShift Orthogonal Agents

> **Note for future work — not part of the current implementation.**

The [`GoldenAxisShift`](_spiral_dsv_lm.py:71) architecture naturally supports running **hundreds of lightweight agents simultaneously**, each occupying a distinct golden-ratio axis in the shared hypervector space. This is a future extension, not required for the current submission.

### The Core Idea

Each agent `i` is a `SpiralDSVLanguageModel` whose codebook is a golden-ratio rotation of the shared base codebook:

```python
agent_codebook[i] = partner_hv(base_codebook, k=i)   # axis k=i rotation
```

All agents are trained on the same 500M tokens via `EigenTrainer.build_bilateral_from_tokens()` but with their own rotated codebook. Because the rotation is metric-preserving (`cosine(partner_hv(A,k), partner_hv(B,k)) = cosine(A,B)`), each agent encodes the same semantic relationships — but from a different geometric frame of reference. Their XOR-bundle confidence errors are **maximally decorrelated** by the golden-ratio equidistribution property.

### Memory Footprint per Agent

| n_words | n_bits | sem_fwd + sem_bwd + codebook | Agents in 16 MB |
|---|---|---|---|
| 4 | 256 | 96 KB | ~170 agents |
| 8 | 512 | 192 KB | ~85 agents |
| 16 | 1,024 | 384 KB | ~42 agents |

### Ensemble Prediction

```python
ensemble_scores = sum(agent.vote_scores_all_vocab(prev_tokens) for agent in agents) / len(agents)
p_correct = float(ensemble_scores[0, target_token])
```

Each agent's errors are independent (different axis rotations → different XOR-bundle noise patterns). The ensemble average reduces variance as `O(1/sqrt(N_agents))`.

### Why This Is Deferred

The current 16 MB budget is better spent on a single high-dimensional agent (`n_words=1024`, 65,536 bits) than on many low-dimensional agents (`n_words=8`, 512 bits). The single-agent SNR gain from 64× more bits outweighs the ensemble variance reduction from ~85 agents at 512 bits each. This trade-off should be revisited if the budget increases or if the single-agent approach plateaus.

### Relationship to SpiralPointerMemory

The [`SpiralPointerMemory`](_spiral_dsv_lm.py:165) composite address `(k1, k2, ..., kL)` is the hierarchical generalisation of this idea — each level adds another golden-ratio rotation, giving D^L addressable agent slots. For language modeling, the 1D sequential structure means L=1 (one axis per lag) is sufficient. The multi-agent ensemble above uses L=0 (agents differ only in their base axis, not in lag structure).
