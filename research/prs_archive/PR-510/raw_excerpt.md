# PR 510 — MUD + Int5 MLP + BigramHash(10240) + SWA (Non-Record)

**Author:** Anush (SelfAnush)
**Claimed BPB:** val_bpb 1.1989 (val_loss 2.0243), single seed
**Artifact size:** 15.9 MB stated (bytes_total null in submission.json)
**Seeds:** 42 (single seed, no statistical significance claim)

## Files retrieved
- `records__track_non_record_16mb__2026-03-22_MUD_Int5MLP_BigramHash_SWA__README.md`
- `records__track_non_record_16mb__2026-03-22_MUD_Int5MLP_BigramHash_SWA__submission.json`
- `records__track_non_record_16mb__2026-03-22_MUD_Int5MLP_BigramHash_SWA__train_gpt.py`

## Environment variables (from run command in README)

```
SEED=42 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Claimed changes (from README, verbatim)

> Key Innovation: MUD Optimizer. Replaces Muon's 5-step Newton-Schulz iteration with MUD's triangular Gram preconditioning. Paper: Beyond Muon: MUD (MomentUm Decorrelation) for Faster Transformer Training (Southworth & Thomas, arXiv:2603.17970, Mar 2026).
>
> Why MUD? Muon5 (quintic NS) ~30k²d (1x baseline). MUD1 (1 pass) ~2.5k²d (12x cheaper). MUD2 (2 passes) ~5k²d (6x cheaper). The paper reports 1.3-2.6x throughput improvement over Muon and 10-50% wall-clock improvements across GPT-2 small/medium/large on A100, MI250, and GH200.
>
> Algorithm (MUD1): row-normalize the momentum matrix; form the k×k Gram matrix Q @ Q.T; extract lower-triangular part T = tril(G); forward triangular solve Q = T^{-1} Q (TRSM); row-normalize again. Python snippet:
>
> ```python
> def mud_whiten(G, passes=1, eps=1e-7):
>     n, m = G.shape
>     k = min(n, m)
>     if n > m: M = G.T.contiguous()
>     else: M = G.contiguous()
>     Q = M.float()
>     for _ in range(passes):
>         r = Q.norm(dim=1); Q = Q / (r[:, None] + eps)
>         Gk = Q @ Q.T
>         T = torch.tril(Gk); T.diagonal().add_(eps)
>         Q = torch.linalg.solve_triangular(T, Q, upper=False)
>         r = Q.norm(dim=1); Q = Q / (r[:, None] + eps)
>     Q = Q.bfloat16()
>     if n > m: Q = Q.T.contiguous()
>     return Q
> ```
>
> Architecture (based on SOTA thwu1 1.1428 BPB with MUD replacing Muon): 10 layers, 512 dim, 8 heads, 4 KV (GQA); MLP 3x (hidden 1536) relu²; Int5 MLP / Int6 attention (mixed); SmearGate + BigramHash(10240, dim=128); U-Net skips; tied embeddings; SWA start_frac=0.4; sliding window eval stride=64.
>
> Results (Single Seed, 8xH100 SXM): val_bpb 1.1989, val_loss 2.0243, 5087 steps in 10 min, step_avg 118ms, peak memory 18866 MiB, artifact 15.9 MB.
>
> Key Finding: MUD achieves strong convergence (1.1989 BPB in only 5087 steps) but is 4.5x slower per step than Muon on H100s. The paper's throughput claims (A100/MI250/GH200) do not transfer to H100s due to torch.linalg.solve_triangular CUDA overhead — TRSM is not as well-optimized as GEMM on Hopper architecture. vs Muon SOTA: step_avg 26ms vs 118ms, 20000 vs 5087 steps in 10 min, 1.1428 vs 1.1989.
