# Path A: Token-Normalized PPM Mixture — Archived (Computationally Intractable)

## Status: ❌ Computationally Intractable

Path A requires computing PPM-D byte-string scores for **every token in the vocabulary** (V=8192) at each scoring position, then normalizing over the full vocabulary before mixing with the neural softmax.

### Why it's intractable

- **O(V) PPM-D evaluations per token position** where V = 8,192
- Full validation has 40,540,160 scored token positions
- Total PPM-D evaluations: ~332 billion byte-level scores
- CPU-only full eval projected at **~38 days** on 32 cores
- Even with the C++ pybind11 backend (17-50× speedup), this exceeds practical budgets
- A CUDA backend was planned but not implemented

### What was built

- ✅ Python reference evaluator: `eval_path_a_ppmd.py` (10 tests passing)
- ✅ C++ pybind11 backend: repo-root `scripts/ppmd_cpp/` (18 tests, bit-exact BPB equivalence)
- ✅ SLURM CPU benchmark: 2.28M probes/s on 32 cpu_short threads
- 🟡 CUDA backend plan exists but was not implemented

### Resolution

Path B (byte-level trie marginalization) was pursued instead, achieving a full-validation audited result. Path B avoids the O(V) bottleneck by operating at the byte level (O(256) per position) rather than the token level.

### Archived materials

This directory contains the Path A evaluator script and all associated planning documents, preserved for reference.
