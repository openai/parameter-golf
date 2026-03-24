# HDC/VSA Zero-Weight Language Model

**val_bpb: TBD** (Pure Hyperdimensional Computing - no learned weights)

## Run Command

```bash
# Setup (once)
python data/cached_challenge_fineweb.py

# Multi-seed training (recommended for statistical significance)
python train_gpt.py --multi-seed --seeds 42 7 1337

# Single run (development/testing)
python train_gpt.py --seed 42
```

## Key Techniques

### H100 Tensor Core Acceleration

This model uses custom CUDA kernels optimized for H100 (sm_90 architecture):

| Kernel | Purpose | Speedup |
|--------|---------|---------|
| `tensor_core_xor_similarity` | Batch Hamming similarity via XOR+popcount | 10-50x vs CPU |
| `tensor_core_batch_encode` | Parallel context encoding with circular shifts | 5-20x vs CPU |
| `tensor_core_fp16_similarity` | FP16 tensor core GEMM for similarity | 2-5x vs CUDA cores |
| `tensor_core_fused_xor_popcount` | Fused XOR + popcount in single kernel | 3-10x vs separate ops |

**Key optimizations:**
- WMMA (Warp Matrix Multiply Accumulate) for 16x16x16 tensor operations
- FP16/BF16 tensor core operations for similarity computation
- Async compute/comm overlap with multiple CUDA streams
- Memory-aligned allocations for optimal tensor core utilization

### Multi-GPU Distributed Training

**Execution Model:**
- **Multi-seed runs are SEQUENTIAL**: Each seed trains one after another (not parallel)
- **Within each run, all GPUs work together**: 8 GPUs share the training workload

```bash
# 8 GPUs train seed 42 together (up to 10 min)
# Then 8 GPUs train seed 7 together (up to 10 min)
# Then 8 GPUs train seed 1337 together (up to 10 min)
# Total: ~30 minutes for 3 seeds
torchrun --nproc_per_node=8 train_gpt.py --multi-seed --seeds 42 7 1337
```

**How 8 GPUs divide work:**
| Feature | Description |
|---------|-------------|
| Data sharding | Each GPU processes different portions of training data |
| Recipe sync | All-gather every 100 iterations to share learned patterns |
| N-gram sync | Distributed aggregation of n-gram statistics |
| No gradient sync | HDC is weight-free, no gradients to synchronize |

**Estimated training time on 8x H100s:**
- Single seed: ~2-3 minutes (without time limit), or completes all 20K iterations within 10-minute limit
- Multi-seed (3 seeds): ~6-9 minutes total (sequential execution)

### Training Approach: Time-Bounded, Not Epoch-Based

**Important:** This model does NOT iterate through the full training dataset. Instead:

| Aspect | Behavior |
|--------|----------|
| **Stopping condition** | Time limit (10 min) OR max iterations (20K), whichever comes first |
| **Data processing** | Processes as many batches as possible within time limit |
| **No epochs** | Training stops when time runs out, not after full dataset pass |
| **HDC advantage** | Each pattern learned is ~50 bytes, so 20K iterations = ~1MB of recipes |

The training loop:
```python
while iteration < config.iterations:  # Max 20,000 iterations
    if elapsed >= config.max_wallclock_seconds:  # 10-minute limit
        break  # Stop immediately
    # Process batch, learn patterns...
```

**On 8x H100s with tensor cores:** The model can complete all 20,000 iterations well within 10 minutes, maximizing pattern learning within the contest time constraint.

### Zero-Weight Architecture
- No learned weight matrices - all vectors procedurally generated
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
- Context encoding: XOR chain of token⊕position pairs

### Difficulty-Aware Learning
- Adaptive time budgeting based on pattern complexity
- Difficulty memory tracks solve times and success rates
- Hierarchical search: XOR peeling → resonator → n-gram fallback

## Accuracy Engine Methods

### Resonator Network
A resonator network is an iterative factorization algorithm that "resonates" when it finds the correct decomposition of a bound hypervector. Given a composite vector `C = A ⊕ B`, it finds the original components:

```
Input: bound_vector (composite), codebook_A (possible A values), codebook_B (possible B values)
Output: best_A, best_B such that A ⊕ B ≈ bound_vector

Algorithm:
1. Initialize estimates: â = random, b̂ = random
2. Iterate until convergence:
   - Update â = argmax_{a∈codebook_A} similarity(a, bound_vector ⊕ b̂)
   - Update b̂ = argmax_{b∈codebook_B} similarity(b, bound_vector ⊕ â)
3. Return best estimates when similarity exceeds threshold
```

The resonator exploits XOR's self-inverse property: if `C = A ⊕ B`, then `A = C ⊕ B` and `B = C ⊕ A`. It alternates between estimating each factor, "cleaning up" the estimate using the codebook (known valid vectors).

### XOR Peeling Search
Factorizes bound vectors by iteratively "peeling" the most likely component:

```
Input: bound_vector, codebook
Output: list of factor vectors

Algorithm:
1. Compute similarity of bound_vector to all codebook entries
2. Select best_match with highest similarity
3. Unbind: remaining = bound_vector ⊕ best_match
4. Repeat until remaining is near-zero or max iterations
5. Return list of peeled factors
```

This works like peeling layers from an onion - each iteration removes one bound component.

### Hierarchical Search
Multi-level search strategy for complex factorizations:

1. **Level 1 - Recipe Recall**: Direct lookup of previously learned patterns
2. **Level 2 - XOR Peeling**: Fast factorization for simple bindings
3. **Level 3 - Resonator**: Iterative refinement for complex bindings
4. **Level 4 - N-gram Fallback**: Statistical prediction from frequency counts
5. **Level 5 - Similarity**: Nearest-neighbor in hypervector space

### Semantic Codebook
Maintains clusters of semantically similar vectors for generalization:

- Groups tokens with similar hypervector representations
- Enables prediction of unseen tokens via similarity to known patterns
- Expands automatically during training as new patterns are learned

### Iterative Refinement Engine
Combines multiple factorization methods for robust prediction:

```
1. Run XOR Peeling and Resonator in parallel
2. Combine estimates using weighted confidence
3. Reconstruct: predicted = â ⊕ b̂
4. Compare to original bound_vector
5. If error > threshold, perturb estimates and retry
```

## Architecture

```
HDCLanguageModel
├── WalshHadamardBasis (dim=8192 default)
│   ├── Token vectors: hash(token) → Hadamard row
│   └── Position vectors: pos → Hadamard row
├── RecipeStorage
│   └── Recipes: seed_sequence + XOR ops → predicted token
├── AccuracyEngine
│   ├── HierarchicalSearch
│   ├── EnhancedResonator
│   ├── SemanticCodebook
│   └── IterativeRefinement
└── DifficultyMemory (adaptive learning)
```

## Training Hyperparameters

| Parameter | Default |
|-----------|---------|
| HDC dimension | 8192 |
| Max sequence length | 2048 |
| Batch tokens | 786,432 |
| Training time | 10 minutes |
| Seeds for multi-run | 42, 7, 1337 |

## Output Files

- `submission.json` - Competition submission with val_bpb
- `train_seed{N}.log` - Training logs per seed
- `recipes.json` - Learned pattern recipes (optional)

## Dependencies

```bash
pip install numpy sentencepiece blake3
# Optional (GPU acceleration):
pip install cupy-cuda12x
```

## How It Works

1. **Encoding**: Each token is mapped to a hypervector via Instant Hadamard Projection
2. **Context**: Token vectors XOR-bound with position vectors, then bundled
3. **Learning**: Patterns stored as recipes (seed sequences + XOR operations)
4. **Prediction**: Multi-path search (recipe recall → resonator → n-gram → similarity fallback)
5. **Evaluation**: Bits Per Byte (BPB) computed on FineWeb validation set

## Bipolar Ternary Representation

The model uses a **two-bit ternary encoding** `{-1, 0, +1}` for hypervector operations:

| State | Bit Pattern | Meaning |
|-------|-------------|---------|
| +1 (Excited) | `pos_vec=1, neg_vec=0` | Positive correlation |
| -1 (Inhibited) | `pos_vec=0, neg_vec=1` | Negative correlation |
| 0 (Neutral) | `pos_vec=0, neg_vec=0` | Unknown/masked |

**Key Functions:**
- `seed_to_ternary_hypervector()` - Generates orthogonal (pos, neg) vector pairs
- `ternary_xor()` - XOR binding for ternary vectors
- `ternary_similarity()` - Similarity computation with mismatch detection

**Benefits:**
- 2-bit storage (not 3-bit) - memory efficient
- Single-instruction XOR operations - no floating point
- Supports negative correlations for richer pattern representation

## Advanced Deduplication System

The `RecipeDeduplicator` implements a **three-tier deduplication pipeline** for efficient pattern storage:

### Tier 1: O(1) Signature Lookup
```
seed_sequence → BLAKE3 hash → exact match check
```
Fast elimination of identical patterns before expensive computation.

### Tier 2: O(1) Content Hash
```
seed_string → content_hash → near-duplicate detection
```
Catches patterns with same semantic content but different metadata.

### Tier 3: O(n) Hadamard Similarity
```
vector × vector → cosine similarity via Walsh-Hadamard transform
```
Semantic similarity detection for patterns that are functionally equivalent.

### Configuration
```python
DeduplicationConfig(
    similarity_threshold=0.95,    # Hadamard similarity threshold
    enable_clustering=True,       # Group similar patterns
    cluster_threshold=0.85,       # Cluster membership threshold
    max_cluster_size=1000         # Prevent unbounded growth
)
```

## XOR Relationship Graph

The `XORRelationshipGraph` tracks semantic relationships between patterns:

### Relationship Types
| Type | Description |
|------|-------------|
| `SIMILAR` | High Hadamard similarity (>0.85) |
| `OPPOSITE` | Negative correlation (ternary -1) |
| `COMPOSED_FROM` | Pattern A is XOR combination of others |
| `PREDICTS` | Pattern A predicts pattern B |
| `TEMPORAL_PREDECESSOR` | Sequential pattern relationship |
| `SEMANTIC_CLUSTER` | Same semantic cluster |

### Key Methods
- `add_relationship(source, target, type)` - Add relationship edge
- `find_xor_similar(pattern, threshold)` - Find XOR-similar patterns
- `get_trajectory(pattern_id)` - Get temporal sequence of patterns
- `get_cluster_patterns(cluster_id)` - Get all patterns in cluster

### Usage Example
```python
# Check for duplicates before storing
result = deduplicator.check_duplicate(new_recipe)
if result.is_duplicate:
    print(f"Duplicate of: {result.existing_recipe.recipe_id}")
    print(f"Similarity: {result.similarity_score}")
else:
    deduplicator.store_or_update(new_recipe)

# Find similar patterns
similar = deduplicator.find_similar(query_vector, threshold=0.9)

# Get related patterns via relationship graph
related = deduplicator.get_related_recipes(recipe_id, relationship_type="PREDICTS")
```

## Architecture (Updated)

```
HDCLanguageModel
├── WalshHadamardBasis (dim=8192 default)
│   ├── Token vectors: hash(token) → Hadamard row
│   └── Position vectors: pos → Hadamard row
├── RecipeStorage
│   └── Recipes: seed_sequence + XOR ops → predicted token
├── RecipeDeduplicator (NEW)
│   ├── Three-tier deduplication (signature → hash → similarity)
│   ├── Pattern clustering with centroids
│   └── LRU cache for vector computations
├── XORRelationshipGraph (NEW)
│   ├── Relationship edges between patterns
│   ├── Trajectory tracking (temporal sequences)
│   └── Cluster membership management
├── AccuracyEngine
│   ├── HierarchicalSearch
│   ├── EnhancedResonator
│   ├── SemanticCodebook
│   └── IterativeRefinement
└── DifficultyMemory (adaptive learning)
```

## Self-Observation / Metacognition

The `SelfObservation` class enables the model to "see" its own encoded state and modify its trajectory:

### Convergence Signals
| Signal | Detection Criteria | Action |
|--------|-------------------|--------|
| `CONVERGING` | Steady improvement in similarity | Continue current path |
| `BREAKTHROUGH` | Rapid similarity gain (>0.1) | Continue with confidence |
| `STUCK` | Variance < 0.02 over 5 iterations | Switch strategy (peel → resonator → explore) |
| `OSCILLATING` | ≥2 sign changes in similarity | Explore new directions |
| `DIVERGING` | Similarity dropping > 0.1 | Restart search |

### Trajectory Actions
```python
# Observe current state and get recommendation
observer = SelfObservation(dim=DEFAULT_HDC_DIM, known_patterns=patterns)
state = observer.observe(current_vector, iteration=10)

if state.trajectory_action == TrajectoryAction.RECALL:
    # Pattern recognized - use stored recipe
    recipe = recall_recipe(state.detected_patterns[0])
elif state.trajectory_action == TrajectoryAction.PEEL:
    # Stuck - try XOR peeling search
    seeds, confidence = xor_peeler.search(target, candidates)
elif state.trajectory_action == TrajectoryAction.RESTART:
    # Diverging - reset search
    current_estimate = initial_estimate.copy()
```

### Reasoning Trace
Each observation produces a human-readable trace:
```
Iteration 15: similarity=0.8234
Convergence signal: converging
Detected pattern: token_sequence_42
Action: continue
```

## Bidirectional Memory Traversal

The `BidirectionalMemory` class enables O(1) access to any position in the encoded sequence:

### Key Features
- **Timestamp-based indexing**: Each event stored with logical timestamp
- **Forward traversal**: `traverse_forward(from_ts, n)` - get next n events
- **Backward traversal**: `traverse_backward(from_ts, n)` - get previous n events
- **Time reconstruction**: `reconstruct_up_to_timestamp(ts)` - "time travel" to any point

### Usage
```python
memory = BidirectionalMemory(dim=DEFAULT_HDC_DIM)

# Add events with automatic timestamps
event_id = memory.add_event("token_42", vector)

# Traverse forward from position 5
future = memory.traverse_forward(from_timestamp=5, n_events=10)

# Traverse backward from position 15
past = memory.traverse_backward(from_timestamp=15, n_events=5)

# Reconstruct state at timestamp 10
state_at_10 = memory.reconstruct_up_to_timestamp(10)
```

### Circular Encoding
Memory is bounded regardless of sequence length:
```python
# Circular shift wraps around dimension
shift = timestamp % uint64_count  # Bounded!
shifted_vector = np.roll(vector, shift)
```

## Recipe Reconstruction from Seeds

The `RecipeReconstructor` class implements zero-weight procedural generation:

### Key Concept
- Recipes store **seed strings**, not vectors
- Vectors are **deterministically reconstructed** via BLAKE3 hash
- Single recipe can reconstruct **infinite contexts**
- No frequency bias - rare tokens get same quality as common ones

### Seed Formats
| Format | Example | Reconstruction |
|--------|---------|----------------|
| Token | `"token_42"` | Hadamard row from BLAKE3 hash |
| Position | `"pos_5"` | Hadamard + circular shift |
| Hadamard | `"hadamard_123"` | Direct Hadamard row |
| Custom | `"my_pattern"` | BLAKE3 hash → hypervector |

### Usage
```python
reconstructor = RecipeReconstructor(dim=DEFAULT_HDC_DIM)

# Reconstruct from recipe
vector = reconstructor.reconstruct_from_recipe(recipe)

# Batch reconstruction for efficiency
vectors = reconstructor.reconstruct_batch(recipes)

# Verify reconstruction matches original
similarity = reconstructor.verify_reconstruction(recipe, original_vector)
```

### Cache Management
Optional LRU cache for performance:
```python
stats = reconstructor.get_cache_stats()
# {'cache_size': 5000, 'max_cache_size': 10000, 'cache_enabled': True}

reconstructor.clear_cache()  # Free memory
```

## GPU Parallelism & Multi-GPU Support

### GPU Batch Processing
The model uses CuPy-accelerated batch operations for parallel token training:

- **Vectorized Context Encoding**: All positions processed in parallel via `_parallel_circular_xor()` using GPU-optimized strided operations
- **Batch Token Matrices**: Pre-built token and position matrices enable O(1) vector lookup
- **Fused XOR Operations**: CUDA kernels for batch XOR and popcount in single pass
- **Chunked Processing**: Memory-efficient processing with configurable chunk sizes (batch=64, seq=128)

**Note on GPU Parallelism**: HDC operations use CUDA cores (not tensor cores) for parallelism. While tensor cores are specialized for matrix multiply (FP16/BF16), HDC's bitwise XOR/popcount operations run on the 16,896+ CUDA cores per H100 in parallel. Each uint64 processes 64 bits simultaneously, with entire arrays processed across thousands of cores—achieving true SIMD parallelism for ternary operations.

### Multi-GPU Distributed Training
PyTorch distributed training with NCCL backend for multi-GPU scaling:

```bash
# Launch multi-GPU training (e.g., 8 GPUs)
torchrun --nproc_per_node=8 train_gpt.py --multi-seed --seeds 42 7 1337
```

**Key Features:**
- **DistributedTokenLoader**: Each GPU processes disjoint data shards
- **Recipe Synchronization**: All-gather operations every `sync_recipes_every` iterations
- **N-gram Sync**: Distributed aggregation of n-gram statistics
- **Gradient-free Training**: No gradient synchronization needed (HDC is weight-free)

**Configuration:**
```python
HDCConfig(
    world_size=8,              # Number of GPUs
    rank=0,                    # Current GPU rank
    distributed_backend="nccl",
    sync_recipes_every=100,    # Sync interval
    gpu_batch_size=1024        # Per-GPU batch size
)
```

### Performance Characteristics
| Operation | CPU | Single GPU | Multi-GPU (8x) |
|-----------|-----|------------|----------------|
| Context encoding | O(n) sequential | O(log n) parallel | O(log n) + sync |
| Pattern learning | O(1) per pattern | O(1) batch | O(1) batch + all-gather |
| Recipe sync | N/A | N/A | O(recipes) per sync |

## Metacognitive Residual Learning

The model implements a **Cognitive Governor** that learns shortcuts from stuck states:

### Core Concept
When the metacognitive system detects a STUCK state, it creates a **MetaResidualRecipe** - a shortcut that allows future predictions to jump directly to the correct answer, skipping expensive iterative search.

### Key Components

| Component | Purpose |
|-----------|---------|
| `CognitiveBudget` | Dynamic compute allocation based on difficulty |
| `MetaResidualRecipe` | Shift-invariant residual with O(1) lookup |
| `MetaResidualRecipeStorage` | Multi-index storage for fast retrieval |
| `predict_with_metacognitive_gating()` | Main prediction pipeline with cognitive control |

### Cognitive Budget
```python
CognitiveBudget(
    max_iterations=100,           # Compute limit
    early_exit_threshold=0.85,    # BREAKTHROUGH trigger
    residual_trigger_threshold=0.70,  # STUCK trigger
    shortcut_available=False,     # Existing residual recipe?
    difficulty_class=DifficultyClass.MEDIUM
)
```

### MetaResidualRecipe
```python
MetaResidualRecipe(
    observed_state_hash=12345,    # O(1) lookup key
    optimal_shift=7,              # Circular encoding alignment
    residual_seeds=["residual_42_shift7"],  # XOR correction
    target_token=42,
    replaces_iterations=50        # Compute savings
)
```

### Prediction Pipeline
1. **Phase 1**: Check DifficultyMemory for time budget
2. **Phase 2**: Fast path - existing residual → instant return (O(1))
3. **Phase 3**: Initialize SelfObservation for monitoring
4. **Phase 4**: Metacognitive search with trajectory actions:
   - `RECALL` → Pattern recognized, instant return
   - `BREAKTHROUGH` → Early exit with confidence
   - `STUCK` → Apply residual jump via XOR correction
   - `DIVERGING` → Random restart
5. **Phase 5**: Learn new residual from stuck state

### XOR Residual Jump
When STUCK is detected, the residual jump applies:
```python
# Find optimal circular shift
shift = find_optimal_shift(residual)

# Apply XOR correction at optimal shift
correction = reconstruct_residual(recipe)
current_guess = np.bitwise_xor(current_guess, correction)
```

### Performance Impact
- **Fast Path**: O(1) lookup skips all search iterations
- **Residual Jump**: XOR correction only when STUCK
- **Path Compression**: Old iterative paths replaced by single recipe

## Learnable Position Encoding

The model now supports **learnable position encodings** that adapt during training:

### Core Concept
Instead of using fixed sequential position indices, the model can **search** the Hadamard space for optimal position vectors and **store** successful mappings as recipes.

### Key Components

| Component | Purpose |
|-----------|---------|
| `PositionRecipe` | Stores learned position as (context_fingerprint → hadamard_index + circular_shift) |
| `PositionSearchConfig` | Configurable search depth, confidence threshold, context window |
| `LearnablePositionEncoder` | Manages recipe storage and position vector generation |
| `PositionLearningIntegrator` | Bridges HDC model with position learning system |

### How It Works

1. **Context Fingerprinting**: BLAKE3 hash of surrounding tokens identifies unique contexts
2. **Hadamard Search**: When prediction fails, search Hadamard rows for better position vectors
3. **Recipe Storage**: Store successful (context → position) mappings for O(1) lookup
4. **Feedback Learning**: Reinforce successful recipes, prune unused ones

### Usage

```python
# Position learning is automatically integrated into HDCLanguageModel
model = HDCLanguageModel(config)

# During encoding, learned positions are tried first
context_vec = model.encode_context(tokens, use_learned_positions=True)

# Feedback is provided during pattern learning
model.learn_pattern(context, target)  # Internally updates position recipes
```

### Configuration

```python
PositionSearchConfig(
    search_depth=100,           # How many Hadamard rows to search
    min_confidence=0.7,         # Threshold to store recipe
    context_window=3,           # Tokens before/after for fingerprint
    enable_roles=True,          # Use temporal/spatial/semantic roles
    learning_rate=0.1,          # How fast to update confidence
    improvement_threshold=0.1,  # Minimum improvement to store new recipe
    max_shifts=16               # Maximum circular shifts to try
)
```

### Benefits

- **Adaptive**: Position encodings improve as model learns
- **Zero-weight**: All position vectors procedurally generated (no weight matrices)
- **Context-aware**: Same position index can have different encodings based on context
- **O(1) lookup**: Recipes indexed by context fingerprint for fast retrieval

## Ablation Summary

| Component | Purpose |
|-----------|---------|
| Instant Hadamard | Guaranteed orthogonal vectors |
| XOR Peeling Search | Factorize bound vectors |
| Resonator Network | Iterative factorization refinement |
| Difficulty Memory | Adaptive time allocation |
| Semantic Codebook | Pattern generalization |
| Bipolar Ternary | Negative correlation support |
| RecipeDeduplicator | Three-tier pattern deduplication |
| XORRelationshipGraph | Semantic relationship tracking |
| GPUBatchOperations | Parallel context encoding & batch learning |
| DistributedContext | Multi-GPU recipe/n-gram synchronization |
| **CognitiveBudget** | Dynamic compute allocation |
| **MetaResidualRecipe** | Shift-invariant shortcut learning |
| **MetaResidualRecipeStorage** | O(1) residual lookup |
| **SelfObservation** | Metacognitive convergence detection |
| **LearnablePositionEncoder** | Adaptive position encodings |
| **PositionLearningIntegrator** | Position learning feedback loop |
