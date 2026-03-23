# MUD + Int5 MLP + BigramHash(10240) + SWA (Non-Record)

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

```python
def mud_whiten(G, passes=1, eps=1e-7):
    n, m = G.shape
    k = min(n, m)
    if n > m:
        M = G.T.contiguous()
    else:
        M = G.contiguous()
    Q = M.float()
    for _ in range(passes):
        r = Q.norm(dim=1)
        Q = Q / (r[:, None] + eps)        # row-normalize
        Gk = Q @ Q.T                       # Gram matrix k×k
        T = torch.tril(Gk)                 # lower-triangular
        T.diagonal().add_(eps)             # numerical stability
        Q = torch.linalg.solve_triangular(T, Q, upper=False)  # TRSM
        r = Q.norm(dim=1)
        Q = Q / (r[:, None] + eps)        # row-normalize again
    Q = Q.bfloat16()
    if n > m:
        Q = Q.T.contiguous()
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

## Results (Single Seed, 8xH100 SXM)

| Metric | Value |
|--------|-------|
| Final val_bpb | **1.1989** |
| Final val_loss | 2.0243 |
| Steps in 10 min | 5,087 |
| step_avg | 118ms |
| Peak memory | 18,866 MiB |
| Artifact size | 15.9 MB |

### Convergence Curve

| Step | val_bpb |
|------|---------|
| 500 | 1.4604 |
| 1000 | 1.3649 |
| 2000 | 1.3191 |
| 3000 | 1.2647 |
| 4000 | 1.2291 |
| 5000 | 1.1945 |
| Final (post-quant) | **1.1989** |

### vs. Muon SOTA

| Metric | Muon (SOTA) | MUD (this) |
|--------|-------------|------------|
| step_avg | ~26ms | 118ms |
| Steps in 10 min | ~20,000 | 5,087 |
| Final val_bpb | 1.1428 | 1.1989 |
| Convergence quality | — | Strong (4x fewer steps, within 0.056 BPB) |

### Key Finding

MUD achieves **strong convergence** (1.1989 BPB in only 5,087 steps) but is **4.5x slower per step** than Muon on H100s. The paper's throughput claims (A100/MI250/GH200) do not transfer to H100s due to `torch.linalg.solve_triangular` CUDA overhead — TRSM is not as well-optimized as GEMM on Hopper architecture.

If MUD could match Muon's step time (~26ms), extrapolating the convergence curve suggests it could reach ~1.10 BPB in 20,000 steps.

## Run Command

```bash
SEED=42 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## References

- Southworth & Thomas. "Beyond Muon: MUD (MomentUm Decorrelation) for Faster Transformer Training." arXiv:2603.17970, Mar 2026.
- Jordan. "Muon: An optimizer for hidden layers in neural networks." Dec 2024.
- Liu et al. "Muon is scalable for LLM training." arXiv:2502.16982, 2025.
