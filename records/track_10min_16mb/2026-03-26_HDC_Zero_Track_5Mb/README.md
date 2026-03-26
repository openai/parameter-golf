# HDC/VSA Zero-Weight Language Model

**val_bpb: TBD** (Pure Hyperdimensional Computing - no learned weights)

## Run Command

```bash
# Install dependencies
pip install -r requirements.txt

# Setup (once)
python data/cached_challenge_fineweb.py

# Multi-seed training (recommended for statistical significance)
python train_gpt.py --multi_seed --seeds 42 7 1337

# Single run (development/testing)
python train_gpt.py --seed 42
```

## Key Techniques

### Sparse Projection Encoding

The central architectural idea: the full 2²⁰-dimensional hypervector space is always
addressable, but each sequence position only **reads and writes a small window of W=64
uint64 blocks** (4096 bits). This is called sparse projection.

Each position `p` has a fixed **circular_shift address** = `p % uint64_count`. Its
window covers blocks `[shift, shift+1, ..., shift+W-1] mod uint64_count`. Because
every position uses a different starting address, the XOR bundling across an entire
sequence assembles a full 2²⁰-dim vector without any single step touching the whole
thing.

```
pos 0 → shift 0  → writes blocks [0 .. W]
pos 1 → shift 1  → writes blocks [1 .. W+1]
pos 2 → shift 2  → writes blocks [2 .. W+2]
...
bundled output = all positions co-occupy the full vector, one window each
```

**Why this matters:**

| Metric | Dense (original) | Sparse (W=64) |
|--------|-----------------|---------------|
| Intermediate tensor (batch=64, seq=512) | ~4.3 GB | ~17 MB |
| CUDA block size needed | 16,384 — **illegal** | 64 — valid |
| Metacognitive correction cost | O(dim) = O(16,384 uint64s) | O(W) = O(64 uint64s) |

The `SPARSE_WINDOW_SIZE = 64` constant controls W and can be tuned in `HDCConfig`
via `sparse_window_size`.

### Three-Path Encoding Pipeline

All three encoding paths use the same sparse `win_idx` array, pre-computed once
per call as `(shifts[:, None] + offsets[None, :]) % uint64_count`:

| Path | Condition | How sparse is applied |
|------|-----------|----------------------|
| **PATH 1** `sparse_encode` kernel | GPU + kernel compiled | `atomicXor` writes W blocks per position; block=(W,) |
| **PATH 2** `sparse_encode` kernel (fallback) | GPU, PATH 1 failed | Same kernel, same W blocks |
| **PATH 3** Vectorised NumPy/CuPy | CPU or GPU kernel failure | Gather `(C, seq, W)` then scatter-XOR into output |

All three paths produce identical output. The dense `(C, seq, uint64_count)` gather
that appeared in the original fallback is fully removed.

### H100 Tensor Core Acceleration

Custom CUDA kernels compiled via CuPy `RawKernel`:

| Kernel | Purpose | Block size |
|--------|---------|-----------|
| `sparse_encode` | Main sparse encoding — W blocks per position via `atomicXor` | `(W,)` = 64 |
| `sparse_meta_correct` | O(W) in-place residual correction at `circular_shift` | `(W,)` = 64 |
| `tensor_core_xor_similarity` | Batch Hamming similarity via XOR+popcount | 256 |
| `tensor_core_fp16_similarity` | FP16 dot-product similarity | per vocab |
| `tensor_core_fused_xor_popcount` | Fused XOR + popcount with warp reduction | 256 |
| `tensor_core_full_encode` | Dense encode (compiled, available for extension) | `min(1024, uint64_count)` |

**Block size fix:** The original `tensor_core_full_encode` launch used
`block = (uint64_count,) = (16384,)` for 2²⁰-dim, which exceeds CUDA's hard limit of
1024 threads per block and caused every kernel launch to fail silently. The fixed
version shards `uint64_count` across `blockIdx.y` so each block uses at most
`MAX_CUDA_THREADS = 1024` threads. In practice `sparse_encode` is preferred and
`tensor_core_full_encode` serves as a dense fallback.

### Multi-GPU Distributed Training

**Execution Model:**
- **Multi-seed runs are SEQUENTIAL**: Each seed trains one after another (not parallel)
- **Within each run, all GPUs work together**: 8 GPUs share the training workload

```bash
# 8 GPUs train seed 42 together (up to 10 min)
# Then 8 GPUs train seed 7 together (up to 10 min)
# Then 8 GPUs train seed 1337 together (up to 10 min)
# Total: ~30 minutes for 3 seeds
torchrun --nproc_per_node=8 train_gpt.py --multi_seed --seeds 42 7 1337
```

**How 8 GPUs divide work:**

| Feature | Description |
|---------|-------------|
| Data sharding | Each GPU processes different portions of training data |
| Recipe sync | All-gather every 100 iterations to share learned patterns |
| N-gram sync | Distributed aggregation of n-gram statistics |
| No gradient sync | HDC is weight-free, no gradients to synchronize |

**Estimated training time on 8x H100s:**
- Single seed: ~2–3 minutes (without time limit), or completes all 20K iterations within 10-minute limit
- Multi-seed (3 seeds): ~6–9 minutes total (sequential execution)

### Training Approach: Time-Bounded, Not Epoch-Based

**Important:** This model does NOT iterate through the full training dataset. Instead:

| Aspect | Behavior |
|--------|----------|
| **Stopping condition** | Time limit (10 min) OR max iterations (20K), whichever comes first |
| **Data processing** | Processes as many batches as possible within time limit |
| **No epochs** | Training stops when time runs out, not after full dataset pass |
| **HDC advantage** | Each pattern learned is ~50 bytes, so 20K iterations ≈ ~1MB of recipes |

```python
while iteration < config.iterations:  # Max 20,000 iterations
    if elapsed >= config.max_wallclock_seconds:  # 10-minute limit
        break
    # Process batch, learn patterns...
```

### Zero-Weight Architecture
- No learned weight matrices — all vectors procedurally generated
- Knowledge stored as "recipes" (~50 bytes each) describing XOR operations
- Instant Hadamard Projection for mathematically guaranteed orthogonality

### Instant Hadamard Projection
- Token vectors: `hash(token_id) mod dim → Hadamard row`
- Position vectors: `position mod dim → Hadamard row` (direct indexing)
- 100% orthogonal (vs ~50% statistical for pseudo-random)
- BLAKE3 hashing for fast, deterministic token-to-row mapping

### XOR-Based Algebra
- Binding: `a ⊕ b` (XOR for reversible composition)
- Unbinding: `bound ⊕ key` (self-inverse property)
- Bundling: Element-wise sum with sign thresholding
- Context encoding: XOR chain of token⊕position pairs, written into sparse windows

### Difficulty-Aware Learning
- Adaptive time budgeting based on pattern complexity
- Difficulty memory tracks solve times and success rates
- Hierarchical search: XOR peeling → resonator → n-gram fallback

## Accuracy Engine Methods

### Resonator Network
A resonator network is an iterative factorization algorithm that "resonates" when it finds the correct decomposition of a bound hypervector. Given a composite vector `C = A ⊕ B`, it finds the original components:

```
Input: bound_vector (composite), codebook_A, codebook_B
Output: best_A, best_B such that A ⊕ B ≈ bound_vector

Algorithm:
1. Initialize estimates: â = random, b̂ = random
2. Iterate until convergence:
   - Update â = argmax_{a∈codebook_A} similarity(a, bound_vector ⊕ b̂)
   - Update b̂ = argmax_{b∈codebook_B} similarity(b, bound_vector ⊕ â)
3. Return best estimates when similarity exceeds threshold
```

### XOR Peeling Search
Factorizes bound vectors by iteratively "peeling" the most likely component:

```
1. Compute similarity of bound_vector to all codebook entries
2. Select best_match with highest similarity
3. Unbind: remaining = bound_vector ⊕ best_match
4. Repeat until remaining is near-zero or max iterations
5. Return list of peeled factors
```

### Hierarchical Search
Multi-level search strategy for complex factorizations:

1. **Level 1 — Recipe Recall**: Direct lookup of previously learned patterns
2. **Level 2 — XOR Peeling**: Fast factorization for simple bindings
3. **Level 3 — Resonator**: Iterative refinement for complex bindings
4. **Level 4 — N-gram Fallback**: Statistical prediction from frequency counts
5. **Level 5 — Similarity**: Nearest-neighbour in hypervector space

### Semantic Codebook
Maintains clusters of semantically similar vectors for generalization:
- Groups tokens with similar hypervector representations
- Enables prediction of unseen tokens via similarity to known patterns
- Expands automatically during training as new patterns are learned

### Iterative Refinement Engine
```
1. Run XOR Peeling and Resonator in parallel
2. Combine estimates using weighted confidence
3. Reconstruct: predicted = â ⊕ b̂
4. Compare to original bound_vector
5. If error > threshold, perturb estimates and retry
```

## Architecture

```
HDCLanguageModel (hdc_dim=2^20, vocab_size=1024, sparse_window=64)
├── WalshHadamardBasis
│   ├── Token vectors: hash(token) → Hadamard row (packed uint64)
│   └── Position vectors: pos → Hadamard row + circular shift
├── TensorCoreBatchOperations
│   ├── sparse_encode kernel  (PATH 1/2)  block=(W=64,), O(seq*W) memory
│   ├── apply_sparse_update               O(W) metacognitive correction
│   └── Vectorised NumPy/CuPy fallback    (PATH 3) O(seq*W) memory
├── RecipeStorage
│   └── Recipes: seed_sequence + XOR ops → predicted token (~50 bytes each)
├── RecipeDeduplicator
│   ├── Three-tier deduplication (signature → hash → similarity)
│   ├── Pattern clustering with centroids
│   └── LRU cache for vector computations
├── XORRelationshipGraph
│   ├── Relationship edges between patterns
│   ├── Trajectory tracking (temporal sequences)
│   └── Cluster membership management
├── MetaResidualRecipeStorage
│   ├── O(1) lookup by state hash
│   ├── O(W) correction via apply_sparse_update
│   └── optimal_shift = last_pos % uint64_count
├── LearnablePositionEncoder
│   ├── PositionRecipe: context_fingerprint → (hadamard_index, circular_shift)
│   └── Hadamard search on prediction failure
├── SelfObservation (metacognition)
│   ├── Convergence signal detection
│   └── Trajectory action selection
├── AccuracyEngine
│   ├── HierarchicalSearch
│   ├── ResonatorNetwork
│   ├── XORPeelingSearch
│   └── IterativeRefinement
└── DifficultyMemory (adaptive time budgeting)
```

## Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| HDC dimension | 2²⁰ = 1,048,576 |
| Sparse window size (W) | 64 uint64 blocks = 4096 bits |
| Vocabulary size | 1,024 tokens |
| Max sequence length | 512 tokens |
| Batch tokens | 524,288 |
| Training time limit | 10 minutes |
| Max iterations | 20,000 |
| Seeds for multi-run | 42, 7, 1337 |
| GPU batch size | 1,024 |

## Output Files

- `submission.json` — Competition submission with val_bpb
- `train_seed{N}.log` — Training logs per seed
- `recipes.json` — Learned pattern recipes (optional)

## Dependencies

```bash
pip install numpy sentencepiece blake3
# Optional (GPU acceleration):
pip install cupy-cuda12x
```

## How It Works

1. **Encoding**: Each token is mapped to a hypervector via Instant Hadamard Projection
2. **Sparse context**: For each position `p`, only `W=64` blocks at address `p % uint64_count` are written — never the full vector
3. **Learning**: Patterns stored as recipes (seed sequences + XOR operations)
4. **Prediction**: Multi-path search (recipe recall → resonator → n-gram → similarity fallback)
5. **Metacognition**: STUCK states trigger an O(W) sparse correction jump to the right window
6. **Evaluation**: Bits Per Byte (BPB) computed on FineWeb validation set

## Bipolar Ternary Representation

The model uses a **two-bit ternary encoding** `{-1, 0, +1}` for hypervector operations:

| State | Bit Pattern | Meaning |
|-------|-------------|---------|
| +1 (Excited) | `pos_vec=1, neg_vec=0` | Positive correlation |
| -1 (Inhibited) | `pos_vec=0, neg_vec=1` | Negative correlation |
| 0 (Neutral) | `pos_vec=0, neg_vec=0` | Unknown/masked |

**Key Functions:**
- `seed_to_ternary_hypervector()` — Generates orthogonal (pos, neg) vector pairs
- `ternary_xor()` — XOR binding for ternary vectors
- `ternary_similarity()` — Similarity computation with mismatch detection

## Advanced Deduplication System

The `RecipeDeduplicator` implements a **three-tier deduplication pipeline**:

### Tier 1: O(1) Signature Lookup
```
seed_sequence → BLAKE3 hash → exact match check
```

### Tier 2: O(1) Content Hash
```
seed_string → content_hash → near-duplicate detection
```

### Tier 3: O(n) Hadamard Similarity
```
vector × vector → cosine similarity via Walsh-Hadamard transform
```

### Configuration
```python
DeduplicationConfig(
    similarity_threshold=0.95,
    enable_clustering=True,
    cluster_threshold=0.85,
    max_cluster_size=1000
)
```

## XOR Relationship Graph

The `XORRelationshipGraph` tracks semantic relationships between patterns:

| Type | Description |
|------|-------------|
| `SIMILAR` | High Hadamard similarity (>0.85) |
| `OPPOSITE` | Negative correlation (ternary -1) |
| `COMPOSED_FROM` | Pattern A is XOR combination of others |
| `PREDICTS` | Pattern A predicts pattern B |
| `TEMPORAL_PREDECESSOR` | Sequential pattern relationship |
| `SEMANTIC_CLUSTER` | Same semantic cluster |

## Self-Observation / Metacognition

The `SelfObservation` class enables the model to "see" its own encoded state:

### Convergence Signals
| Signal | Detection Criteria | Action |
|--------|-------------------|--------|
| `CONVERGING` | Steady improvement in similarity | Continue current path |
| `BREAKTHROUGH` | Rapid similarity gain (>0.1) | Early exit with confidence |
| `STUCK` | Variance < 0.02 over 5 iterations | Apply sparse residual jump |
| `OSCILLATING` | ≥2 sign changes in similarity | Explore new directions |
| `DIVERGING` | Similarity dropping > 0.1 | Random restart |

```python
observer = SelfObservation(dim=DEFAULT_HDC_DIM, known_patterns=patterns)
state = observer.observe(current_vector, iteration=10)

if state.trajectory_action == TrajectoryAction.STUCK:
    # Sparse O(W) correction — not a full-vector XOR
    current_guess = model.apply_residual_to_vec(current_guess, recipe)
elif state.trajectory_action == TrajectoryAction.RECALL:
    recipe = recall_recipe(state.detected_patterns[0])
elif state.trajectory_action == TrajectoryAction.RESTART:
    current_estimate = initial_estimate.copy()
```

## Metacognitive Residual Learning

When STUCK is detected, the model learns a **MetaResidualRecipe** — a shortcut that
jumps directly to the correct answer in future predictions.

### Key Components

| Component | Purpose |
|-----------|---------|
| `CognitiveBudget` | Dynamic compute allocation based on difficulty |
| `MetaResidualRecipe` | Sparse residual shortcut with O(1) state-hash lookup |
| `MetaResidualRecipeStorage` | Multi-index storage: by state hash, context sig, shift |
| `predict_with_metacognitive_gating()` | Main prediction pipeline |

### MetaResidualRecipe

```python
MetaResidualRecipe(
    observed_state_hash=12345,              # O(1) lookup key
    optimal_shift=last_pos % uint64_count,  # sparse window address
    residual_seeds=["residual_42_shift7"],  # XOR correction seeds
    target_token=42,
    replaces_iterations=50                  # compute savings
)
```

`optimal_shift` is derived from `last_pos % uint64_count` — the same circular address
that `batch_encode_context` used when encoding that position. This guarantees that
the correction lands in exactly the right window.

### Sparse Residual Jump (O(W) not O(dim))

When STUCK is detected, the correction is applied via `apply_residual_to_vec`, which
calls `apply_sparse_update`:

```python
# O(W) — only touches W blocks at recipe.optimal_shift
win_idx = (arange(W) + shift) % uint64_count
vec[win_idx] ^= correction[win_idx]
```

This replaces the original O(dim) full-vector XOR:
```python
# OLD — O(16,384 uint64s) per correction
current_guess = np.bitwise_xor(current_guess, correction)  # removed
```

The GPU path uses the `sparse_meta_correct` CUDA kernel with `block=(W,)` threads,
touching only those W locations in-place.

### Prediction Pipeline
1. **Phase 1**: Check DifficultyMemory for time budget
2. **Phase 2**: Fast path — existing residual → instant return O(1)
3. **Phase 3**: Initialize SelfObservation for monitoring
4. **Phase 4**: Metacognitive search loop with trajectory actions:
   - `RECALL` → Pattern recognised, instant return
   - `BREAKTHROUGH` → Early exit with confidence
   - `STUCK` → `apply_residual_to_vec` (O(W) sparse jump)
   - `DIVERGING` → Random restart
5. **Phase 5**: Fallback to standard prediction + learn new residual

## Learnable Position Encoding

### Key Components

| Component | Purpose |
|-----------|---------|
| `PositionRecipe` | `context_fingerprint → (hadamard_index, circular_shift)` |
| `PositionSearchConfig` | Search depth, confidence threshold, context window |
| `LearnablePositionEncoder` | Recipe storage and position vector generation |
| `PositionLearningIntegrator` | Bridges HDC model with position learning |

### How It Works

1. **Context Fingerprinting**: BLAKE3 hash of surrounding tokens identifies unique contexts
2. **Hadamard Search**: When prediction fails, search Hadamard rows for better position vectors
3. **Recipe Storage**: Store successful `(context → hadamard_index + circular_shift)` for O(1) lookup
4. **Sparse Consistency**: `circular_shift` stored in a `PositionRecipe` is the same address used by `batch_encode_context`, so learned positions and sparse encoding are always aligned

### Configuration

```python
PositionSearchConfig(
    search_depth=100,           # Hadamard rows to search
    min_confidence=0.7,         # Threshold to store recipe
    context_window=3,           # Tokens before/after for fingerprint
    enable_roles=True,          # Temporal/spatial/semantic roles
    learning_rate=0.1,
    improvement_threshold=0.1,
    max_shifts=16
)
```

## GPU Parallelism & Multi-GPU Support

### Sparse GPU Batch Processing

All GPU operations operate on `(batch, seq, W)` tensors — never `(batch, seq, uint64_count)`:

- **Sparse Context Encoding**: `sparse_encode` kernel with `block=(W=64,)` per batch item
- **Sparse Metacognitive Correction**: `sparse_meta_correct` kernel, O(W) in-place
- **Batch Token Matrices**: Pre-built token/position matrices enable O(1) vector lookup by index
- **Chunked Processing**: Memory-efficient chunks (batch_chunk_size=64 default)

### Multi-GPU Distributed Training

```bash
torchrun --nproc_per_node=8 train_gpt.py --multi_seed --seeds 42 7 1337
```

| Feature | Description |
|---------|-------------|
| `DistributedTokenLoader` | Each GPU processes disjoint data shards |
| `AsyncTokenLoader` | Prefetch pipeline for GPU runs (prefetch_batches=2) |
| Recipe sync | `all_gather_recipes` every `sync_recipes_every` iterations |
| N-gram sync | `all_gather_ngrams` distributed aggregation |
| Gradient-free | No gradient synchronisation needed |

**Configuration:**
```python
HDCConfig(
    world_size=8,
    rank=0,
    distributed_backend="nccl",
    sync_recipes_every=100,
    gpu_batch_size=1024,
    sparse_window_size=64       # W — controls all three encoding paths
)
```

### Performance Characteristics

| Operation | CPU | Single H100 | Multi-GPU (8x) |
|-----------|-----|-------------|----------------|
| Context encoding (sparse) | O(seq·W) | O(seq·W) kernel | O(seq·W) + sync |
| Metacognitive correction | O(W) | O(W) kernel | O(W) per GPU |
| Pattern learning | O(1) per pattern | O(1) batch | O(1) + all-gather |
| Intermediate memory (batch=64, seq=512) | ~17 MB | ~17 MB | ~17 MB per GPU |

## Bidirectional Memory Traversal

The `BidirectionalMemory` class enables O(1) access to any position in the encoded sequence:

```python
memory = BidirectionalMemory(dim=DEFAULT_HDC_DIM)
event_id = memory.add_event("token_42", vector)

future = memory.traverse_forward(from_timestamp=5, n_events=10)
past   = memory.traverse_backward(from_timestamp=15, n_events=5)
state  = memory.reconstruct_up_to_timestamp(10)
```

Circular encoding keeps memory footprint bounded regardless of sequence length:
```python
shift = timestamp % uint64_count   # always bounded
shifted_vector = np.roll(vector, shift)
```

## Recipe Reconstruction from Seeds

The `RecipeReconstructor` implements zero-weight procedural generation:

| Seed Format | Example | Reconstruction |
|-------------|---------|----------------|
| Token | `"token_42"` | Hadamard row from BLAKE3 hash |
| Position | `"pos_5"` | Hadamard + circular shift |
| Hadamard | `"hadamard_123"` | Direct Hadamard row |
| Residual | `"residual_42_shift7"` | BLAKE3 → hypervector (sparse correction) |
| Custom | `"my_pattern"` | BLAKE3 hash → hypervector |

```python
reconstructor = RecipeReconstructor(dim=DEFAULT_HDC_DIM)
vector     = reconstructor.reconstruct_from_recipe(recipe)
vectors    = reconstructor.reconstruct_batch(recipes)
similarity = reconstructor.verify_reconstruction(recipe, original_vector)
```

## Ablation Summary

| Component | Purpose |
|-----------|---------|
| Instant Hadamard | Guaranteed orthogonal vectors |
| **Sparse Projection (W=64)** | 250× smaller intermediates; valid CUDA block sizes |
| **sparse_encode kernel** | O(seq·W) GPU encoding — main fast path |
| **apply_sparse_update** | O(W) metacognitive correction — not O(dim) |
| **sparse_meta_correct kernel** | In-place GPU residual jump |
| XOR Peeling Search | Factorize bound vectors |
| Resonator Network | Iterative factorization refinement |
| Difficulty Memory | Adaptive time allocation |
| Semantic Codebook | Pattern generalization |
| Bipolar Ternary | Negative correlation support |
| RecipeDeduplicator | Three-tier pattern deduplication |
| XORRelationshipGraph | Semantic relationship tracking |
| TensorCoreBatchOperations | Sparse parallel context encoding |
| DistributedContext | Multi-GPU recipe/n-gram synchronisation |
| CognitiveBudget | Dynamic compute allocation |
| MetaResidualRecipe | Sparse shift-invariant shortcut learning |
| MetaResidualRecipeStorage | O(1) residual lookup by state hash |
| SelfObservation | Metacognitive convergence detection |
| LearnablePositionEncoder | Adaptive position encodings |
| PositionLearningIntegrator | Position learning feedback loop |
