# MUD + Int5 MLP + BigramHash(10240) + SWA

## Key Innovation: MUD Optimizer

Replaces Muon's 5-step Newton-Schulz iteration with MUD's triangular Gram preconditioning.

**Paper:** [Beyond Muon: MUD (MomentUm Decorrelation) for Faster Transformer Training](https://arxiv.org/abs/2603.17970) (Southworth & Thomas, Mar 2026)

### Why MUD?

| Method | FLOPs per step | Relative cost |
|--------|---------------|---------------|
| Muon5 (quintic NS) | ~30k²d | 1x (baseline) |
| MUD1 (1 pass) | ~2.5k²d | **12x cheaper** |
| MUD2 (2 passes) | ~5k²d | **6x cheaper** |

MUD replaces the expensive `X @ X.T` Gram formation + polynomial iteration loop with:
1. Row-normalize the momentum matrix
2. Form the k×k Gram matrix `Q @ Q.T`
3. Extract lower-triangular part `T = tril(G)`
4. Forward triangular solve `Q = T^{-1} Q` (TRSM)
5. Row-normalize again

The paper reports **1.3-2.6x throughput improvement** over Muon and **10-50% wall-clock improvements** across GPT-2 small/medium/large on A100, MI250, and GH200.

### Algorithm (MUD1)

```
def mud_whiten(G, passes=1, eps=1e-8):
    n, m = G.shape
    k = min(n, m)
    if n > m:
        M = G.T
    Q = M.bfloat16()
    for _ in range(passes):
        r = Q.norm(dim=1)
        Q = Q / (r[:, None] + eps)        # row-normalize
        Gk = Q @ Q.T                       # Gram matrix k×k
        T = torch.tril(Gk)                 # lower-triangular
        Q = torch.linalg.solve_triangular(T, Q, upper=False)  # TRSM
        r = Q.norm(dim=1)
        Q = Q / (r[:, None] + eps)        # row-normalize again
    if n > m:
        Q = Q.T
    return Q
```

## Architecture

Based on the SOTA submission (thwu1, 1.1428 BPB) with MUD replacing Muon:

- 10 layers, 512 dim, 8 heads, 4 KV heads (GQA)
- MLP 3x expansion (hidden=1536), relu² activation
- Int5 MLP weights / Int6 attention weights (mixed quantization)
- SmearGate + BigramHash(10240, dim=128)
- U-Net skip connections, tied embeddings
- SWA with start_frac=0.4
- Sliding window eval stride=64

## Expected Improvement

The SOTA (1.1428 BPB) is limited by training steps within the 10-minute window. MUD's ~12x cheaper optimizer should allow **significantly more gradient updates** in the same wall-clock time, potentially:

- ~15-25% more training steps (accounting for TRSM being slower than GEMM in practice)
- Better convergence per wall-clock second

## Hyperparameters

Same as SOTA with `MUON_BACKEND_STEPS=1` (MUD1 passes).

## Run Command

```bash
SEED=42 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## References

- Southworth & Thomas. "Beyond Muon: MUD (MomentUm Decorrelation) for Faster Transformer Training." arXiv:2603.17970, Mar 2026.
- Jordan. "Muon: An optimizer for hidden layers in neural networks." Dec 2024.
- Liu et al. "Muon is scalable for LLM training." arXiv:2502.16982, 2025.
