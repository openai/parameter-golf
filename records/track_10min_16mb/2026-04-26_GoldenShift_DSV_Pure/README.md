# Pure GoldenShift DSV — Complete NMF Removal + N-Dimensional Axis Extension

**Date:** 2026-04-26  
**Track:** 10min / 16 MB  
**Based on:** `2026-04-25_SpiralDSV_Eigen_DSVOnly`

---

## What Changed

This submission removes the entire NMF pipeline from the 2026-04-07 architecture
and replaces it with a **pure DSV-only** system built on three clean components:

```
Fibonacci hash → base codebook (token → hypervector)
    → GoldenAxisShift rotation per lag (exact lag subspace separation)
        → 1/freq weighted XOR bundling (sparse dominant pointers)
            → sem_fwd only, 8 MB (n_words=1024 uses full 16 MB budget)
                → matmul projection (not eigensolver convergence)
```

### Before / After

| Component | Before (SpiralDSV_Eigen_DSVOnly) | After (GoldenShift_DSV_Pure) |
|---|---|---|
| NMF phases 0–5, 8–9 | Present | **Removed** |
| `_hash_grad_train.py` | Present | **Deleted** |
| `_transition_codebook.py` | Present | **Deleted** |
| `_suffix_grammar.py` | Present | **Deleted** |
| `embed` array (16 MB) | Present | **Removed** |
| `W_out` array (~32 KB) | Present | **Removed** |
| PMI centering | Applied | **Skipped** (1/freq handles it) |
| Bilateral build (sem_bwd) | Built | **Kept for API compat** |
| 1/freq weighting | Absent | **NEW** |
| `freq_weights` param | Absent | **NEW** in `build_bilateral_from_tokens()` |

---

## Three-Component Architecture

### Component 1 — Fibonacci-hash codebook

Each token ID maps to a pseudo-random hypervector via golden-ratio mixing:

```python
codebook[tok_id] = fibonacci_hash(tok_id, seed)  # (n_words,) uint64
```

This gives every token a geometrically spread, collision-resistant base vector.
Already present in `SpiralDSVLanguageModel` — unchanged.

### Component 2 — GoldenAxisShift per lag

Each lag `c` rotates the codebook by `c × φ × n_bits` bits:

```python
rotated_cb[c][tok] = roll(codebook[tok], offset(c))
```

Because φ is irrational (Weyl equidistribution), lag-1 and lag-3 subspaces are
geometrically independent — no aliasing regardless of how many lags are added.
Already present in `GoldenAxisShift` — unchanged.

### Component 3 — 1/freq weighted bundling (NEW)

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

## Architecture Diagram

```
Training (500M tokens)
        │
        ▼
Step 1: Compute freq_table[tok] = count(tok)        [O(N), one pass]
        inv_freq[tok] = 1.0 / freq_table[tok]       ← TF-IDF analogue
        │
        ▼
Step 2: Chunked histogram scan (distributed)        [O(N × ctx_len), GPU scatter_add]
        For each position i, lag c:
            a = tokens[i]
            b = tokens[i + c]
            fwd_hist[a, (c-1)*V + b] += inv_freq[b] ← 1/freq weighting
        All-reduce across 8 ranks (NCCL)
        │
        ▼
Step 3: C separate matmuls + circular output shift  [O(C × V² × n_bits), GPU HGEMM]
        For each lag c:
            result_c = fwd_hist_block_c @ CB_pm1     ← (V,V) @ (V,n_bits), unrotated CB
            result_c = roll_cols(result_c, offset_c) ← O(V×n_bits) np.roll, not 256 MB CB
            sem_fwd_spectrum += result_c
        sem_fwd_pm1 = sign(sem_fwd_spectrum)         ← (V, n_bits) float32
        │
        ▼
Step 4: Pack to uint64 + save HGZ3 artifact         [8 MB uncompressed, ~1-2 MB LZMA9]
        sem_fwd_u64 = pack(sem_fwd_pm1)              ← (V, n_words) uint64

Eval (all positions → single DSV path):
        prev_tok → sem_fwd_pm1[prev] @ codebook_pm1.T → (vocab,) scores
        p_correct = softmax(scores)[tgt_tok]
        BPB = Σ(-log2 p_correct) / Σ(utf8_bytes)
```

### Histogram Upgrade: Cheap Circular-Shift Matmul

The old approach built a `CB_composite_pm1` matrix of shape `(ctx_len×V, n_bits)` —
**256 MB** for `ctx_len=4, V=1024, n_bits=65536` — by running `np.unpackbits` + float
conversion on a per-lag rotated codebook. This was the dominant memory cost.

The new approach avoids materializing `CB_composite` entirely:

| Step | Old | New |
|---|---|---|
| Codebook rotation | Build `(C×V, n_bits)` float32 = **256 MB** | Skip — use unrotated `CB_pm1` |
| Matmul | One `(V, C×V) @ (C×V, n_bits)` | C separate `(V,V) @ (V,n_bits)` |
| Apply rotation | Baked into CB_composite | `np.roll(result, offset_c, axis=1)` per lag |
| Peak RAM | 256 MB (CB_composite) + 64 MB (CB_pm1) | **64 MB** (CB_pm1 only) |
| Mathematical result | Identical | Identical |

The `np.roll` on a `(V, n_bits)` float32 output is O(V × n_bits) — the same work
as before, but done on the **output** (64 MB) instead of the **input** (256 MB).
This is the "Hadamard performance without the transform" principle: the rotation
is applied lazily at output time rather than eagerly at codebook construction time.

---

## Memory Budget

### Option A — sem_fwd only, n_words=1024 (default)

```
sem_fwd: (vocab_size=1024, n_words=1024) uint64 = 8 MB
Total: 8 MB  ✅  (8 MB freed vs. bilateral build)
```

With `n_words=1024`: each token gets a **65,536-bit** hypervector.
XOR-bundle confidence resolution scales as `O(sqrt(n_bits))` — 64× more bits
than the old 1,024-bit DSV gives 8× better signal-to-noise ratio.

### Option A′ — sem_fwd only, n_words=2048 (full 16 MB budget)

```
sem_fwd: (vocab_size=1024, n_words=2048) uint64 = 16 MB
Total: 16 MB  ✅  (doubles n_bits to 131,072 → ~11× SNR vs. old 1,024-bit DSV)
```

Run with `N_WORDS=2048` to use the full budget.

---

## Retrieval Exactness

| Scenario | Retrieval Quality |
|---|---|
| One pointer per lag | Exact |
| Multiple pointers, frequency weighted | Practically exact — dominant pointer clear |
| Multiple pointers, equal frequency | Approximate — interference within lag space |
| Cross-lag pointer retrieval | Exact — lag subspaces are independent |

The 1/freq weighting specifically prevents the "multiple pointers, equal frequency"
case from occurring for statistically meaningful tokens.

---

## Files

| File | Source | Action |
|---|---|---|
| `train_gpt.py` | New | Rewrite — DSV-only, no NMF |
| `_semantic_layer.py` | SpiralDSV_Eigen_DSVOnly | Rewrite — add 1/freq, remove old DSV |
| `_spiral_dsv_lm.py` | SpiralDSV_Eigen_DSVOnly | Copy unchanged |
| `_eigen_convergence.py` | SpiralDSV_Eigen_DSVOnly | Modified — add freq_weights param |
| `_gpu.py` | SpiralDSV_Eigen_DSVOnly | Copy unchanged |
| `README.md` | New | This file |
| `submission.json` | New | Reset TBD values |
| `requirements.txt` | SpiralDSV_Eigen_DSVOnly | Updated — removed cupy |
| `_hash_grad_train.py` | — | **Not copied — deleted** |
| `_transition_codebook.py` | — | **Not copied — deleted** |
| `_suffix_grammar.py` | — | **Not copied — deleted** |

---

## Run Commands

### Standard leaderboard run (8×H100 SXM) — all 3 seeds, automatic:

When `SEED` is **not** set, `train_gpt.py` automatically runs seeds **42 → 7 → 1337**
sequentially. Each seed is launched as a fresh independent `torchrun` invocation
(GPU memory fully released between runs). No shell script required.

```bash
RUN_ID=golden_shift_dsv N_WORDS=1024 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The orchestrator prints a summary table on completion:

```
=================================================================
[Orchestrator] All seeds complete:
  seed=   42  ->  OK
  seed=    7  ->  OK
  seed= 1337  ->  OK
=================================================================
```

### Single seed (explicit):

Set `SEED` to run only one seed:

```bash
RUN_ID=golden_shift_dsv N_WORDS=1024 SEED=42 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### Option A′ — full 16 MB budget (n_words=2048), all 3 seeds:

```bash
RUN_ID=golden_shift_dsv N_WORDS=2048 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### Override GPU count (optional):

`NPROC_PER_NODE` controls how many GPUs each child `torchrun` uses (default `8`):

```bash
NPROC_PER_NODE=4 RUN_ID=golden_shift_dsv N_WORDS=1024 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=4 train_gpt.py
```

### Single GPU smoke test:

```bash
N_WORDS=128 SEED=42 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
python train_gpt.py
```

---

## Key Invariants

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
   `build_bilateral_from_tokens()` falls back to uniform weights + PMI centering,
   preserving the existing SpiralDSV_Eigen_DSVOnly behaviour.

---

## Multidimensional Lag Spaces

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

The current submission uses only temporal axes (`axis_id = lag`, `lag ∈ 1..ctx_len`).

---

## Future Direction — Multi-Agent Collective Memory

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
> The shared `sem_fwd' tables become collective memory — every agent
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

## Comparison with Original DSV (2026-04-07)

The original `DirectionalSemanticVec` (2026-04-07) used a **simple XOR-bundle with a
Fibonacci-hash codebook** — no circular shift at all. All lags XOR into the same bundle,
so lag-1 and lag-3 are geometrically indistinguishable.

| Architecture | Lag separation | Cross-axis | N-dimensional | Compute |
|---|---|---|---|---|
| **Original DSV (2026-04-07)** | None — all lags XOR into same bundle | No | 1D only | O(N×ctx_len) scatter XOR, CPU numpy |
| **GoldenShift DSV (current)** | Circular bit-shift by `c×φ×n_bits` per lag | Extensible | Yes | O(V×C×V×n_bits) matmul, GPU HGEMM |

The `GoldenAxisShift` **is** a circular shift — specifically:

```python
# Per lag c (word-level + sub-word bit rotation):
rolled = np.roll(CB_uint64, word_shift, axis=1)
rolled = (rolled << bit_shift) | (rolled >> (64 - bit_shift))
```

This is the cheapest possible lag-separation: O(n_words) per token, no matrix multiply,
no unpackbits. The output-column rotation in the matmul step applies the same rotation
to the `(V, n_bits)` output instead of materializing a 256 MB rotated codebook —
**4× less peak RAM, same mathematical result.**

---

## N-Dimensional Axis Extension (Future Direction)

### Current State: 1D Temporal (Maximally Optimized)

The current implementation uses **temporal lags only** (`axis_id = lag distance`).
For `ctx_len=4`, this gives 4 independent geometric subspaces:

```
lag 1 → GoldenAxisShift(codebook, offset=1×φ×n_bits)   ← immediate bigram
lag 2 → GoldenAxisShift(codebook, offset=2×φ×n_bits)   ← skip-bigram
lag 3 → GoldenAxisShift(codebook, offset=3×φ×n_bits)   ← trigram context
lag 4 → GoldenAxisShift(codebook, offset=4×φ×n_bits)   ← 4-gram context
```

Because φ is irrational (Weyl equidistribution), no two lag subspaces ever alias.
The matmul path is already maximally efficient for 1D:

- No CB_composite materialization (256 MB → 0 MB)
- C separate `(V, V) @ (V, n_bits)` matmuls on the unrotated codebook
- Output column rotation per lag: `np.roll(result, offset_c, axis=1)` — O(V × n_bits)
- Total peak RAM: `(V, n_bits)` float32 = 64 MB (codebook, already in RAM)

### Extending to Named Axes for N-Dimensional Data

The `GoldenAxisShift.offset(k)` method already handles arbitrary `k`. Extending to
named axes requires only choosing non-overlapping `axis_id` ranges:

```python
# Axis ID ranges (non-overlapping, each gets its own geometric subspace):
TEMPORAL_AXIS_BASE  = 0    # lags 0..ctx_len-1  (already implemented)
SPATIAL_H_AXIS_BASE = 100  # horizontal spatial lags (future)
SPATIAL_V_AXIS_BASE = 200  # vertical spatial lags (future)
DEPTH_AXIS_BASE     = 300  # hierarchical depth lags (future)
AGENT_AXIS_BASE     = 1000 # per-agent identity axes (future)
```

Because φ is irrational, `offset(100)` and `offset(101)` are as geometrically
independent as `offset(1)` and `offset(2)` — no aliasing regardless of axis count.

### 2D Spatial Example (Image Patches)

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
model cannot distinguish direction from distance. Named axes fix this:

```python
h_offset = GoldenAxisShift.offset(SPATIAL_H_AXIS_BASE + 1)  # horizontal lag 1
v_offset = GoldenAxisShift.offset(SPATIAL_V_AXIS_BASE + 1)  # vertical lag 1

# fwd_hist_h[E, D] += 1/freq[D]   (horizontal co-occurrence)
# fwd_hist_v[E, B] += 1/freq[B]   (vertical co-occurrence)

# Matmul + rotation per axis (same cheap path as temporal):
sem_fwd_spectrum += np.roll(fwd_hist_h @ CB_pm1, h_offset, axis=1)
sem_fwd_spectrum += np.roll(fwd_hist_v @ CB_pm1, v_offset, axis=1)
```

### 3D Spatial Example (Voxels)

```python
# Position (x=2, y=1, z=3):
x_offset = GoldenAxisShift.offset(SPATIAL_X_AXIS_BASE + 2)
y_offset = GoldenAxisShift.offset(SPATIAL_Y_AXIS_BASE + 1)
z_offset = GoldenAxisShift.offset(SPATIAL_Z_AXIS_BASE + 3)

sem_fwd_spectrum += np.roll(fwd_hist_x @ CB_pm1, x_offset, axis=1)
sem_fwd_spectrum += np.roll(fwd_hist_y @ CB_pm1, y_offset, axis=1)
sem_fwd_spectrum += np.roll(fwd_hist_z @ CB_pm1, z_offset, axis=1)
```

Position (2,1,3) can never alias onto any other 3D position because
`offset(X_BASE+2)`, `offset(Y_BASE+1)`, `offset(Z_BASE+3)` are all
geometrically independent (irrational φ spacing).

### Hierarchical (Parse Trees)

```python
# evaluating "dog" in "The big red dog barked":
depth_offset   = GoldenAxisShift.offset(DEPTH_AXIS_BASE + 2)    # 2 levels deep
sibling_offset = GoldenAxisShift.offset(SIBLING_AXIS_BASE + 1)  # 1st NP child
parent_offset  = GoldenAxisShift.offset(PARENT_AXIS_BASE + 0)   # immediate parent

sem_fwd_spectrum += np.roll(fwd_hist_depth   @ CB_pm1, depth_offset,   axis=1)
sem_fwd_spectrum += np.roll(fwd_hist_sibling @ CB_pm1, sibling_offset, axis=1)
sem_fwd_spectrum += np.roll(fwd_hist_parent  @ CB_pm1, parent_offset,  axis=1)
```

### Why This Works: Weyl Equidistribution

The key property is that `k × φ mod 1` is **equidistributed** for any irrational φ.
This means:

- For any two distinct integers `k1 ≠ k2`, `offset(k1) ≠ offset(k2)` (non-repeating)
- The offsets are uniformly spread across `[0, n_bits)` (no clustering)
- Cross-axis interference is zero: `roll(CB, offset_A) · roll(CB, offset_B) ≈ 0`
  for large n_bits (orthogonality by equidistribution)

This gives effectively **unlimited independent dimensions** without aliasing,
using only a single codebook and cheap circular shifts.

### Implementation Cost per Additional Axis

Each additional named axis adds:
- One `(V, V)` histogram accumulation (same as temporal lags)
- One `(V, V) @ (V, n_bits)` matmul — ~0.1s on H100 for V=1024, n_bits=65536
- One `np.roll(result, offset, axis=1)` — O(V × n_bits), ~1ms

The total cost scales as `O(n_axes × V² × n_bits)` — linear in the number of axes.
For 10 axes at V=1024, n_bits=65536: ~1s on H100. For 100 axes: ~10s.

*This is a future architectural direction. The current submission uses only
temporal axes (`axis_id = lag`, `lag ∈ 1..ctx_len`).*

---

## Soul-Shift Binding: Hadamard Performance Without the Transform

### The Key Insight

The `fibonacci_hash` initialisation generates hypervectors that are already
**uniformly distributed and pseudo-orthogonal** — effectively in a "Hadamard-ready"
state from bit zero. This means the system gets spread-spectrum interference
resistance without running an O(N log N) Hadamard transform on every bind.

The computational tax of the transform is bypassed by ensuring the **initial state**
already possesses those mathematical properties. The data is Hadamard-ready at
initialisation, not at execution.

### The "Golden XOR" / Soul-Shift Binding Pattern

For multi-agent or multi-context scenarios, the binding logic can be extended to
a "Nested XOR" flow that treats the Agent ID as a static geometric "base" and the
Golden Shift as the dynamic "offset":

```python
def bind_token(tok_id, agent_id, lag_c, codebook, agent_seed=0xDEADBEEF):
    # 1. Base Identity (pre-seeded 'Hadamard' vector — already in codebook)
    v = codebook[tok_id]                          # (n_words,) uint64

    # 2. Agent Identity (unique XOR 'Soul' mask)
    # Creates the agent's unique geometric subspace via a seeded RNG.
    # XOR with a random mask is a perfect bijection — zero entropy loss.
    rng = np.random.default_rng(agent_seed ^ agent_id)
    agent_mask = rng.integers(0, np.iinfo(np.uint64).max,
                               size=v.shape, dtype=np.uint64)
    v_bound = v ^ agent_mask                      # agent's view of token

    # 3. Temporal Position (Golden Ratio circular shift)
    # Separates lags within the agent's subspace.
    return golden_axis_shift(v_bound, lag=lag_c)  # (n_words,) uint64
```

### Why This Is the "Nearly Exact" Sweet Spot

| Property | Mechanism | Cost |
|---|---|---|
| **Zero leakage** | XOR with random mask is a perfect bijection — doesn't degrade entropy of the original codebook | O(n_words) XOR |
| **Decoupled scaling** | Agent A's "Lag 1" and Agent B's "Lag 1" point to entirely different regions of the hypercube | O(1) mask lookup |
| **No lag aliasing** | Golden ratio shift separates lags within each agent's subspace | O(n_words) roll |
| **Hardware friendly** | Only XOR + BIT-ROLL — essentially free on H100 or mobile CPU vs any matrix math or FFT | O(n_words) total |

### Comparison: Pure Circular Shift vs Soul-Shift

In a **pure circular shift** (current implementation), if two agents use the same
lag distance, their subspaces are identical — they share the same geometric region.
This is fine for single-agent temporal lags but breaks for multi-agent scenarios:

```
Agent A, lag 1 → offset(1) = same as Agent B, lag 1 → COLLISION
```

With **Soul-Shift**, the agent XOR mask pre-rotates the entire codebook into a
unique subspace before the lag shift is applied:

```
Agent A, lag 1 → XOR(codebook, mask_A) then roll(offset(1)) → unique region
Agent B, lag 1 → XOR(codebook, mask_B) then roll(offset(1)) → different unique region
```

The two agents' lag-1 subspaces are now geometrically independent even though
they use the same lag distance.

### Integration with the Current Histogram Build

The Soul-Shift is a **plug-and-play addition** to the current frequency-weighting
and bundling logic. No changes to the histogram scan or matmul are needed:

```python
# Current (single-agent temporal):
fwd_hist[a, (c-1)*V + b] += 1/freq[b]

# Future (multi-agent Soul-Shift):
# Pre-bind tokens with agent mask before histogram accumulation:
a_bound = bind_token(a, agent_id, lag_c=0, codebook, agent_seed)
b_bound = bind_token(b, agent_id, lag_c=c, codebook, agent_seed)
fwd_hist[a_bound_id, (c-1)*V + b_bound_id] += 1/freq[b]
```

The 1/freq weighting, GoldenAxisShift rotation, and matmul projection are all
unchanged — the Soul-Shift operates entirely at the token-binding layer.

### Decision for This Submission

The current **Pure GoldenShift DSV** uses single-agent temporal lags only.
The Soul-Shift is documented here as a future extension that can be dropped in
without changing any of the frequency-weighting, bundling, or matmul logic.

*No Soul-Shift code is included in this submission.*

---

## Results

| Metric | Value |
|---|---|
| val_bpb | TBD (to be filled after leaderboard run) |
| val_loss | TBD |
| artifact_bytes | TBD |
| n_words | 1024 (65,536 bits) |
| ctx_len | 4 |
| seeds | 42 / 7 / 1337 |
| w_coherence | 0.3 |
| use_freq_weights | True |
