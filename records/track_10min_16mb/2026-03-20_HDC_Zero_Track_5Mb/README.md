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

## Ablation Summary

| Component | Purpose |
|-----------|---------|
| Instant Hadamard | Guaranteed orthogonal vectors |
| XOR Peeling Search | Factorize bound vectors |
| Resonator Network | Iterative factorization refinement |
| Difficulty Memory | Adaptive time allocation |
| Semantic Codebook | Pattern generalization |
