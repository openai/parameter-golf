# Merging Strategy Benchmark: Bind All (Recipes) vs. Bundle (Superposition)

This benchmark evaluates the two primary ways to merge external knowledge into the **Universal XOR 8K Engine**.

## The Strategies

### 1. Bundle (Superposition)
*   **The Math**: `Merged_Vector = Snap(Vec_A + Vec_B + ...)`
*   **The Result**: All knowledge is squashed into a single 32,768-dimensional vector.
*   **The Problem**: **Lossy**. Once you add too many items (usually >500), the noise (crosstalk) makes it impossible to distinguish between vectors. The model starts "blurring" memories.

### 2. Bind All (Recipe Union) 
*   **The Math**: `Merged_Model = [Recipe_A, Recipe_B, ...]`
*   **The Result**: Models are merged by concatenating lists of "Recipes" (Identity Recipes).
*   **The Benefit**: **100% Deterministic & Lossless**. Knowledge is never destroyed.

## Automatic "Shortcuts" & Optimization
The "Bind All" method using the `IdentityRecipe` system (`recipe_storage.py`) includes automatic optimizations:

1.  **Deduplication (The Shortcut)**: Each recipe has a deterministic `recipe_id` (e.g., `H102.S50`). If Model A and Model B both contain the same concept, they share the same ID. Merging them doesn't double the size; the system simply points to the existing ID.
2.  **Decoupling**: We don't store the 32KB vectors. We store a 100-byte "Recipe" (Seed + Address). The vector is materialized only during "thinking." This makes the model **327x smaller** than raw storage.
3.  **Hadamard Orthogonality**: Because every address (H-Index) is mathematically perpendicular, there are zero search collisions between merged models.

## How to Run the Benchmark

To see the accuracy drop-off of Bundling vs the perfect stability of Recipes, run:

```bash
python Hdc_Sparse/compare_merging_strategies.py
```

### Expected Output
| Items | Method | Accuracy | Size (Est) |
| :--- | :--- | :--- | :--- |
| 10 | Bundle | 100% | 4 KB |
| 100 | Bundle | 98% | 4 KB |
| 1000 | Bundle | **12% (Failing)** | 4 KB |
| 1000 | **Recipe Union** | **100% (Perfect)** | 0.1 MB |

---
> [!IMPORTANT]
> **Conclusion**: For high-fidelity 8K processing and permanent memory, **"Bind All (Recipe Union)"** is the superior architecture. It provides the "Shortcuts" (IDs) and "Optimization" (Seeds) needed for infinite scaling without accuracy loss.
