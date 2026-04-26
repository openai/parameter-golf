# Plan: Pure GoldenShift DSV — Complete NMF Removal

**Target record folder:** `records/track_10min_16mb/2026-04-26_GoldenShift_DSV_Pure/`
**Based on:** `records/track_10min_16mb/2026-04-25_SpiralDSV_Eigen_DSVOnly/`
**Date:** 2026-04-26

---

## Objective

Replace the entire 2026-04-07 pipeline with a **pure DSV-only** system built on three clean components:

```
Fibonacci hash → base codebook (token → hypervector)
    → GoldenAxisShift rotation per lag (exact lag subspace separation)
        → 1/freq weighted XOR bundling (sparse dominant pointers)
            → sem_fwd only, 8 MB (n_words=2048 uses full 16 MB budget)
                → matmul projection (not eigensolver convergence)
```

No eigensolver convergence checking. No bilateral build required. No PMI. No NMF. No Hadamard codebook. No `embed`. No `W_out`.

---

## What Is Being Removed

### From `_hash_grad_train.py` (entire file deleted)

| Function | Why removed |
|---|---|
| `build_frozen_prior()` | NMF regularisation only |
| `precompute_g_states()` | NMF bucket hashing only |
| `tabulate_bucket_frequencies_distributed()` | NMF frequency table only |
| `merge_seed_frequencies()` | NMF multi-seed merge only |
| `xor_orbit_regularise()` | NMF freq smoothing only |
| `nmf_kl_fit()` | 1-iter NMF ≈ random signal |
| `save_hash_grad_artifact()` | Saves embed + W_out (NMF) |
| `load_hash_grad_artifact()` | Loads embed + W_out (NMF) |
| `hash_grad_bpb()` | NMF + DSV waterfall eval |
| `train_hash_grad_model()` | NMF training orchestration |
| `train_hash_grad_multi_seed()` | Multi-seed NMF merge |

**Data structures removed:**
- `embed` array — `(TABLE_SIZE × EMBED_DIM)` fp16, 16 MB
- `W_out` array — `(EMBED_DIM × VOCAB_SIZE)` fp16, ~32 KB
- `fingerprint_packed` array — `(TABLE_SIZE,)` uint8
- Rolling hash `G[p]` state array

### From `_semantic_layer.py` (old version, replaced entirely)

| Component | Why removed |
|---|---|
| `DirectionalSemanticVec` class | Old scatter-XOR builder, replaced by EigenTrainer |
| `_scatter_xor_fast()` | Old chunked numpy argsort+reduceat path |
| `build_skip_bigram_lags()` | Replaced by GoldenAxisShift per-lag rotation |
| Hadamard codebook references | Replaced by Fibonacci-hash codebook in SpiralDSVLanguageModel |
| NMF codebook references | Removed with NMF |
| `build_token_byte_arrays()` | Moved inline to eval function |

### Files deleted entirely

| File | Why |
|---|---|
| `_hash_grad_train.py` | Entire NMF pipeline |
| `_transition_codebook.py` | Only used by suffix grammar + old Hadamard codebook |
| `_suffix_grammar.py` | Morphological reranking gate — depends on NMF g_states |

### From `train_gpt.py` (rewritten)

| Removed | Why |
|---|---|
| `--hash_grad` flag and routing | NMF pipeline entry point |
| `_run_hash_grad_single()` | NMF orchestration |
| Phase 0 (frozen prior) | NMF regularisation |
| Phase 2 (freq tabulation) | NMF freq table |
| Phase 3 (seed merge) | NMF multi-seed |
| Phase 4 (XOR orbit regularise) | NMF smoothing |
| Phase 5 (NMF fit) | 1-iter NMF |
| Phase 7 (suffix grammar) | Depends on NMF g_states |
| Phase 8 (S[p] checkpoints) | Legacy, already unused |
| Phase 9 (embed pruning) | NMF embed pruning |
| Phase 10 (artifact save) | NMF artifact format |
| `HG_SEEDS`, `TABLE_BITS`, `EMBED_DIM` env vars | NMF-specific |
| `dist.barrier()` + rank exit after Phase 5 | NMF-specific distributed pattern |

---

## What Is Kept

| Component | File | Why kept |
|---|---|---|
| `GoldenAxisShift` | `_spiral_dsv_lm.py` | Core rotation mechanism |
| `GOLDEN_AXES` singleton | `_spiral_dsv_lm.py` | Module-level axis registry |
| `SpiralDSVLanguageModel` | `_spiral_dsv_lm.py` | Codebook + vote_scores_all_vocab() |
| `EigenTrainer.build_bilateral_from_tokens()` | `_eigen_convergence.py` | Histogram scan + matmul |
| `_gpu.py` | `_gpu.py` | GPU matmul acceleration |
| Phase 6 (DSV build) | `train_gpt.py` | Primary prediction mechanism |
| HGZ3 artifact format | `_semantic_layer.py` | sem_fwd only |
| BPB formula | `_semantic_layer.py` | Unchanged |
| Distributed token loading | `train_gpt.py` | All 8 ranks load tokens |

---

## New Architecture

### Three-Component Pipeline

```
Fibonacci hash → base codebook
    → GoldenAxisShift rotation per lag
        → 1/freq weighted XOR bundling
            → sem_fwd
```

**Component 1 — Fibonacci-hash codebook** (already exists in `SpiralDSVLanguageModel`)

Each token ID maps to a pseudo-random hypervector via golden-ratio mixing:
```python
codebook[tok_id] = fibonacci_hash(tok_id, seed)  # (n_words,) uint64
```
This gives every token a geometrically spread, collision-resistant base vector.

**Component 2 — GoldenAxisShift per lag** (already exists in `GoldenAxisShift`)

Each lag `c` rotates the codebook by `c × φ × n_bits` bits:
```python
rotated_cb[c][tok] = roll(codebook[tok], offset(c))
```
Because φ is irrational (Weyl equidistribution), lag-1 and lag-3 subspaces are
geometrically independent — no aliasing regardless of how many lags are added.

**Component 3 — 1/freq weighted bundling** (NEW — replaces uniform count=1)

Instead of accumulating `count += 1` for each co-occurrence, accumulate:
```python
weight = 1.0 / freq[token_b]   # inverse frequency of the following token
fwd_hist[a, (c-1)*V + b] += weight
```
This keeps dominant (high-frequency) tokens from washing out rare but
surprising co-occurrences in `sem_fwd`. The 1/freq weighting is the
discrete analogue of TF-IDF — it makes each lag space sparse enough
that the dominant pointer per lag is clearly retrievable.

### Why These Three Components Are Sufficient

| Problem | Solved By |
|---|---|
| Geometric collision between tokens | Fibonacci hash |
| Temporal distance meaningless | GoldenAxisShift |
| Frequent tokens dominate bundles | 1/freq weighting |
| Rare tokens lost | 1/freq weighting |

No eigensolver, no bilateral build, no PMI centering, no NMF — just three
clean components each solving a distinct problem, staying fully
distribution-agnostic throughout.

---

## Multidimensional Lag Spaces

### The Baseline: Temporal Lags (What Already Exists)

The current `ctx_len=4` implementation already encodes four independent temporal
axes. Each lag `c` gets a distinct `GoldenAxisShift` rotation:

```
tokens: ["the", "cat", "sat", "on", "the", "mat"]
                                         ↑
                              evaluating this token

lag 1 → "the"   (1 token back)  → GoldenAxisShift(codebook[the], lag=1)
lag 2 → "on"    (2 tokens back) → GoldenAxisShift(codebook[on],  lag=2)
lag 3 → "sat"   (3 tokens back) → GoldenAxisShift(codebook[sat], lag=3)
lag 4 → "cat"   (4 tokens back) → GoldenAxisShift(codebook[cat], lag=4)
```

Because φ is irrational, lag-1 and lag-3 subspaces are geometrically
independent — "the" at lag-1 can never alias onto "the" at lag-4.

### The General Pattern: Any Relational Structure

The `GoldenAxisShift` is not limited to temporal distance. Any relationship
expressible as an integer offset on some axis can be encoded without collision:

```
relationship type  → choose an axis (an integer axis_id)
relationship value → choose a lag distance on that axis
combined position  → XOR of GoldenAxisShift rotations across all axes
```

Because φ is irrational (Weyl equidistribution), no two distinct combinations
of `(axis_id, lag)` ever produce the same geometric orientation — giving
effectively unlimited independent dimensions without aliasing.

### Concrete Examples

**2D Spatial (image patches):**

```
patch grid:
[A][B][C]
[D][E][F]
[G][H][I]

evaluating E:
    horizontal axis, lag 1 → D  (left neighbor)
    vertical axis,   lag 1 → B  (above neighbor)
```

Pure temporal lag collapses these: lag-1 = D (left), lag-3 = B (above) — the
model cannot distinguish direction from distance. Separate axes fix this:

```python
h_lag_1 = GoldenAxisShift(codebook[D], axis=HORIZONTAL, lag=1)
v_lag_1 = GoldenAxisShift(codebook[B], axis=VERTICAL,   lag=1)
```

**3D Spatial (voxels):**

```
position (x=2, y=1, z=3):
    x_axis, lag 2 → rotate by 2φ on x-axis
    y_axis, lag 1 → rotate by 1φ on y-axis
    z_axis, lag 3 → rotate by 3φ on z-axis

h_pos = GoldenShift_x(codebook[token], 2)
      ^ GoldenShift_y(codebook[token], 1)
      ^ GoldenShift_z(codebook[token], 3)
```

Position (2,1,3) can never alias onto any other 3D position.

**Hierarchical (parse trees):**

```
evaluating "dog" in "The big red dog barked":
    depth_axis,   lag 2 → "dog" is 2 levels deep
    sibling_axis, lag 1 → "dog" is 1st noun-phrase child
    parent_axis,  lag 0 → immediate parent is NP node
```

### How This Maps to the Current Implementation

The existing `axis_word_shifts` parameter in
[`EigenTrainer.build_bilateral_from_tokens()`](_eigen_convergence.py:1433)
already implements this pattern for temporal lags:

```python
axis_word_shifts = [
    (GOLDEN_AXES._word_shifts[c], GOLDEN_AXES._bit_shifts[c])
    for c in range(1, ctx_len + 1)
]
```

Each lag `c` maps to a distinct `(word_shift, bit_shift)` pair derived from
`c × φ × n_bits`. The `CB_composite_pm1` matrix stacks all `ctx_len` rotated
codebooks:

```python
CB_composite_pm1 = concat([
    GoldenAxisShift(CB, lag=1),   # temporal axis, lag 1
    GoldenAxisShift(CB, lag=2),   # temporal axis, lag 2
    GoldenAxisShift(CB, lag=3),   # temporal axis, lag 3
    GoldenAxisShift(CB, lag=4),   # temporal axis, lag 4
])  # shape: (ctx_len × V, n_bits)
```

The single matmul `fwd_hist @ CB_composite_pm1` then projects all lag
contributions simultaneously — each lag's subspace is geometrically independent
due to the irrational rotation offsets.

### What the New Implementation Adds

The 1/freq weighting makes each lag's subspace **sparse** — the dominant
pointer per lag is clearly retrievable because high-frequency tokens are
down-weighted. This is what elevates the system from approximate associative
memory to something much closer to an exact pointer system:

```
GoldenAxisShift → exact separation between lag spaces
1/freq weighting → sparse dominant pointers within each lag space
65,536 bits → noise floor low enough that practical exactness holds
```

### Future Extension: Named Axes (Not Implemented Here)

The architecture supports named axes beyond temporal lags by assigning each
relationship type a distinct `axis_id` integer. The `GoldenAxisShift.offset(k)`
method already handles arbitrary `k` — extending to named axes requires only
choosing non-overlapping `axis_id` ranges:

```python
TEMPORAL_AXIS_BASE  = 0    # lags 0..ctx_len-1
SPATIAL_H_AXIS_BASE = 100  # horizontal spatial lags
SPATIAL_V_AXIS_BASE = 200  # vertical spatial lags
DEPTH_AXIS_BASE     = 300  # hierarchical depth lags
```

This is a future architectural direction. The current submission uses only
temporal axes (`axis_id = lag`, `lag ∈ 1..ctx_len`).

### Retrieval Exactness

| Scenario | Retrieval Quality |
|---|---|
| One pointer per lag | Exact |
| Multiple pointers, frequency weighted | Practically exact — dominant pointer clear |
| Multiple pointers, equal frequency | Approximate — interference within lag space |
| Cross-lag pointer retrieval | Exact — lag subspaces are independent |

The 1/freq weighting specifically prevents the "multiple pointers, equal frequency"
case from occurring for statistically meaningful tokens.

### Memory Budget

**Option A — sem_fwd only, n_words=1024 (chosen for this submission):**

```
sem_fwd: (vocab_size=1024, n_words=1024) uint64 = 8 MB
Total: 8 MB  ✅  (8 MB freed vs. bilateral build)
```

With `n_words=1024`: each token gets a **65,536-bit** hypervector.
XOR-bundle confidence resolution scales as `O(sqrt(n_bits))` — 64× more bits
than the old 1,024-bit DSV gives 8× better signal-to-noise ratio.

**Option A′ — sem_fwd only, n_words=2048 (uses full 16 MB budget):**

```
sem_fwd: (vocab_size=1024, n_words=2048) uint64 = 16 MB
Total: 16 MB  ✅  (doubles n_bits to 131,072 → ~11× SNR vs. old 1,024-bit DSV)
```

**Option B — bilateral build (not chosen):**

```
sem_fwd: (vocab_size=1024, n_words=1024) uint64 = 8 MB
sem_bwd: (vocab_size=1024, n_words=1024) uint64 = 8 MB
Total: 16 MB  ✅  (backward context encoded, no budget for dimensionality increase)
```

**Decision: Option A (sem_fwd only, n_words=1024).** The freed 8 MB is available
to double `n_words` to 2048 in a follow-up experiment (Option A′). sem_bwd is
not built in this submission.

---

## Implementation Steps

### Step 1 — Create new record folder

```
records/track_10min_16mb/2026-04-26_GoldenShift_DSV_Pure/
```

Copy from `2026-04-25_SpiralDSV_Eigen_DSVOnly/`:
- `_spiral_dsv_lm.py` — unchanged
- `_eigen_convergence.py` — modified (add 1/freq weighting)
- `_gpu.py` — unchanged

Do NOT copy:
- `_hash_grad_train.py` — deleted
- `_transition_codebook.py` — deleted
- `_suffix_grammar.py` — deleted

### Step 2 — Modify `_eigen_convergence.py`: add 1/freq weighting

In `EigenTrainer.build_bilateral_from_tokens()`, the inner loop currently
accumulates uniform counts:

```python
# CURRENT (uniform weight):
ones_t = _torch.ones(len(fwd_idx), dtype=_torch.float32, device=_dev)
fwd_acc.scatter_add_(0, fwd_idx_t, ones_t)
bwd_acc.scatter_add_(0, bwd_idx_t, ones_t)
```

Replace with frequency-weighted accumulation:

```python
# NEW (1/freq weighting):
# freq_table: (V,) float32 — precomputed token frequencies from training corpus
# weights for each (a, b, c) triple = 1/freq[b]  (inverse freq of the following token)
b_weights = (1.0 / freq_table[b_chunk]).ravel()  # (chunk*C,) float32
b_weights_t = _torch.as_tensor(b_weights, dtype=_torch.float32, device=_dev)
fwd_acc.scatter_add_(0, fwd_idx_t, b_weights_t)

# For bwd: weight by 1/freq[a] (inverse freq of the preceding token)
a_weights = np.repeat(1.0 / freq_table[a_chunk], C)  # (chunk*C,) float32
a_weights_t = _torch.as_tensor(a_weights, dtype=_torch.float32, device=_dev)
bwd_acc.scatter_add_(0, bwd_idx_t, a_weights_t)
```

The `freq_table` is computed in a single O(N) pass before the main scan:
```python
freq_table = np.bincount(tokens_i32, minlength=V).astype(np.float32)
freq_table = np.maximum(freq_table, 1.0)  # avoid division by zero
```

**Signature change:** `build_bilateral_from_tokens()` gains an optional
`freq_weights: Optional[np.ndarray] = None` parameter. When `None`, falls
back to uniform weights (backward compatible).

### Step 3 — Rewrite `_semantic_layer.py`

Keep only:
- `build_spiral_dsv()` — calls `EigenTrainer.build_bilateral_from_tokens()` with
  `freq_weights` computed from the training tokens
- `eval_spiral_dsv_bpb()` — unchanged from SpiralDSV_Eigen_DSVOnly version
- `save_spiral_dsv_artifact()` — unchanged (HGZ3 format)
- `load_spiral_dsv_artifact()` — unchanged

Remove entirely:
- `DirectionalSemanticVec` class
- `_scatter_xor_fast()`
- `build_skip_bigram_lags()`
- All NMF/Hadamard codebook references

**New `build_spiral_dsv()` signature:**
```python
def build_spiral_dsv(
    tokens: np.ndarray,
    vocab_size: int = VOCAB_SIZE,
    n_words: int = N_WORDS_H100,
    ctx_len: int = CTX_LEN,
    seed: int = 42,
    time_budget_s: float = 300.0,
    dist_rank: int = 0,
    dist_world_size: int = 1,
    use_freq_weights: bool = True,   # NEW: enable 1/freq weighting
    verbose: bool = True,
) -> SpiralDSVLanguageModel:
```

Internally computes `freq_table` from `tokens` and passes it to
`EigenTrainer.build_bilateral_from_tokens()`.

### Step 4 — Rewrite `train_gpt.py`

Remove all NMF orchestration. The new `train_gpt.py` is a clean, minimal
entry point:

```python
# 1. Init distributed (same pattern as existing)
rank, world_size = _init_distributed()

# 2. Load training tokens (all ranks, interleaved shards)
tokens = _load_tokens(data_path, split="train", rank=rank, world_size=world_size)

# 3. Build DSV (Phase 6 only — all ranks contribute to histogram)
model = build_spiral_dsv(
    tokens=tokens,
    n_words=n_words,
    ctx_len=ctx_len,
    seed=seed,
    time_budget_s=max_wallclock - 60,
    dist_rank=rank,
    dist_world_size=world_size,
    use_freq_weights=True,
)

# 4. Rank 0 only: save artifact + eval
if rank == 0:
    artifact_path = save_spiral_dsv_artifact(model, artifact_name)
    val_tokens = _load_tokens(data_path, split="val")
    bpb, val_loss = eval_spiral_dsv_bpb(val_tokens, model, ...)
    _update_submission_json(bpb, val_loss, artifact_path)
```

**Environment variables (simplified):**

| Variable | Default | Effect |
|---|---|---|
| `N_WORDS` | `1024` | HV width in uint64 words (65,536 bits at 1024) |
| `SEED` | `42` | Random seed for codebook generation |
| `CTX_LEN` | `4` | Number of lags (GoldenAxisShift rotations) |
| `MAX_WALLCLOCK_SECONDS` | `600` | Training time cap |
| `DATA_PATH` | required | Path to fineweb10B_sp1024/ |
| `TOKENIZER_PATH` | required | Path to fineweb_1024_bpe.model |
| `VOCAB_SIZE` | `1024` | Vocabulary size |
| `USE_FREQ_WEIGHTS` | `1` | Enable 1/freq weighting (0 = uniform) |
| `W_COHERENCE` | `0.3` | Coherence gating weight (0.0 = disabled) |

**Removed env vars:** `TABLE_BITS`, `EMBED_DIM`, `HG_SEEDS` (all NMF-specific)

### Step 5 — Write new `README.md`

Document the pure DSV-only architecture. Include:
- Architecture diagram (before/after)
- What was removed and why
- Three-component explanation (Fibonacci hash + GoldenAxisShift + 1/freq)
- Retrieval exactness table
- Budget reallocation table
- Run commands
- Agents footnote (see below)

### Step 6 — Update `submission.json` and `requirements.txt`

`submission.json`: reset `val_bpb`, `val_loss` to TBD.

`requirements.txt`: remove `cupy-cuda12x` if no longer needed (the GPU path
in `_gpu.py` uses PyTorch CUDA, not CuPy — CuPy was only needed for the old
`cp.bitwise_xor.reduceat` path in the original `_semantic_layer.py`).

---

## Agents Footnote (README only — no code)

> **Future direction — Multi-Agent Collective Memory**
>
> The GoldenAxisShift architecture naturally extends to multi-agent systems.
> Because φ is irrational, each agent can be assigned its own axis (an integer
> `agent_id`) that produces a genuinely distinct geometric subspace:
>
> ```
> temporal axis → lag distance (already exists)
> agent axis    → each agent's unique geometric orientation
> ```
>
> The shared `sem_fwd`/`sem_bwd` tables become collective memory — every agent
> reads from and writes to the same hypervector bundle, but each agent's
> contributions live in their own axis subspace:
>
> ```
> agent_A queries with axis_A rotation
>     → projects onto axis_A subspace
>     → sees only agent_A's learned associations
>     → other agents' memories contribute zero interference
> ```
>
> Each agent's "soul" is a persistent base vector that rotates the entire
> codebook from that agent's unique geometric perspective:
>
> ```python
> # Standard codebook lookup:
> hypervector = GoldenShift(codebook[token], lag=L)
>
> # Agent-specific soul lookup:
> hypervector = GoldenShift(codebook[token], lag=L, agent=agent_id)
> ```
>
> The 1/freq weighting naturally handles the multi-agent case using per-agent
> observation counts rather than collective counts, so each agent discounts
> tokens that are common from its own experience — not from the collective's.
>
> This maps onto the hypervector structure almost perfectly naturally:
> - **Ontology is shared** — all agents agree on what tokens exist
> - **Epistemology is individual** — each agent processes shared reality through
>   its own geometric perspective
> - **Identity is geometric** — the soul is a rotation of the entire shared
>   space, not a separate module or set of weights
>
> *This is a future architectural direction. No agent code is included in this
> submission.*

---

## Architecture Diagram

```
Training (500M tokens)
        │
        ▼
Step 1: Compute freq_table[tok] = count(tok) / N   [O(N), one pass]
        │
        ▼
Step 2: Build CB_composite_pm1                      [O(ctx_len × V × n_bits)]
        For each lag c in 1..ctx_len:
            rotated_CB[c] = GoldenAxisShift(codebook, lag=c)
        CB_composite = concat(rotated_CB[1..ctx_len])  # (ctx_len×V, n_bits)
        │
        ▼
Step 3: Chunked histogram scan                      [O(N × ctx_len), distributed]
        For each position i, lag c:
            a = tokens[i]
            b = tokens[i + c]
            weight = 1.0 / freq_table[b]            ← 1/freq weighting
            fwd_hist[a, (c-1)*V + b] += weight      ← forward only
        All-reduce across 8 ranks (NCCL)
        │
        ▼
Step 4: Single matmul (rank 0 only)                [O(V × ctx_len×V × n_bits)]
        sem_fwd_pm1 = sign(fwd_hist @ CB_composite)  # (V, n_bits) float32
        │
        ▼
Step 5: Pack to uint64 + save HGZ3 artifact        [8 MB uncompressed, ~1-2 MB LZMA9]
        sem_fwd_u64 = pack(sem_fwd_pm1)  # (V, n_words) uint64

Eval (all positions → single DSV path):
        prev_tok → sem_fwd_pm1[prev] @ codebook_pm1.T → (vocab,) scores
        p_correct = softmax(scores)[tgt_tok]
        BPB = Σ(-log2 p_correct) / Σ(utf8_bytes)
```

---

## File Inventory for New Record Folder

| File | Source | Action |
|---|---|---|
| `train_gpt.py` | New | Rewrite — DSV-only, no NMF |
| `_semantic_layer.py` | SpiralDSV_Eigen_DSVOnly | Rewrite — add 1/freq, remove old DSV |
| `_spiral_dsv_lm.py` | SpiralDSV_Eigen_DSVOnly | Copy unchanged |
| `_eigen_convergence.py` | SpiralDSV_Eigen_DSVOnly | Modify — add freq_weights param |
| `_gpu.py` | SpiralDSV_Eigen_DSVOnly | Copy unchanged |
| `README.md` | New | Write — pure DSV architecture |
| `submission.json` | New | Reset TBD values |
| `requirements.txt` | SpiralDSV_Eigen_DSVOnly | Update — remove cupy if unused |
| `_hash_grad_train.py` | — | **Do not copy — deleted** |
| `_transition_codebook.py` | — | **Do not copy — deleted** |
| `_suffix_grammar.py` | — | **Do not copy — deleted** |

---

## Key Invariants to Preserve

1. **BPB formula unchanged** — `Σ(-log2 p) / Σ(utf8_bytes)` with exact official
   `is_boundary_token` initialisation (all True, not zeros).

2. **Artifact size ≤ 16 MB** — `sem_fwd` only at `n_words=1024` is 8 MB
   uncompressed; LZMA9 compresses to ~1–2 MB. Doubling to `n_words=2048`
   uses the full 16 MB budget (Option A′).

3. **Codebook not stored in artifact** — regenerated deterministically from `seed`
   at eval time.

4. **Distributed correctness** — all ranks contribute to the histogram via
   `scatter_add_` + `all_reduce(SUM)`; only rank 0 runs the matmul and saves
   the artifact.

5. **Backward compatibility** — `freq_weights=None` in
   `build_bilateral_from_tokens()` falls back to uniform weights, preserving
   the existing SpiralDSV_Eigen_DSVOnly behaviour.
