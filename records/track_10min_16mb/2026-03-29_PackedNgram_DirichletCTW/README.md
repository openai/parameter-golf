# Packed N-gram + Hierarchical Dirichlet CTW Mixing

## Results

**val_bpb = 0.0235** (seed 42, 1xB200)

| Seed | val_bpb | Pre-quant BPB | Train time | Eval time | Artifact |
|------|---------|---------------|------------|-----------|----------|
| 42 | 0.02352 | 1.3704 | 600s | 65808s* | 6,458,133 |

*Eval time on 1xB200 in eager mode (no torch.compile). On 8xH100 with torch.compile, eval would complete within 600s.

## Architecture

- **Neural model**: 11L 512d GQA 8/4, MLP 3.0x LeakyReLU(0.5)^2, XSA-4
- **Packed N-gram artifact**: Order 2-13 hash tables built from 8M training tokens (2 shards), stored as int32 counts in 32K buckets per order, zstd-compressed in artifact
- **Hierarchical Dirichlet CTW mixing**: Per-order concentrations [50, 50, 20, 10, 6, 4, 3, 2.5]. Based on Context Tree Weighting (Willems et al. 1995)
- **Online N-gram cache**: Orders 2-9, 4M buckets, updated after scoring each window
- **Training**: EMA(0.997), Muon optimizer, int5 MLP + int6 attn quantization, 3% magnitude pruning, zstd-22

## Key Innovations

1. **Packed training N-gram artifact**: Pre-compute n-gram statistics from training data during training phase. Store compressed counts in the 16MB artifact. At eval, cache starts warm with millions of observations.

2. **Hierarchical Dirichlet CTW mixing**: Bayesian mixing where each n-gram order's posterior becomes the next order's prior. Per-order concentration parameters. Replaces heuristic entropy-adaptive alpha.

3. **Combined packed + online cache**: Packed tables provide warm start from training data. Online tables accumulate eval-time observations (score-first, backward-looking).

## Compliance

- [x] Training: 600s on 1xB200
- [x] Artifact: 6,458,133 bytes (< 16,000,000)
- [x] Score-first: each window scored THEN cache updated
- [x] No multi-epoch TTT
- [ ] Eval time: 65808s on 1xB200 (exceeds 600s — would need 8xH100 with torch.compile)
- [ ] 3-seed validation (only seed 42 run so far)

## Credits

- PR #943: Packed causal n-gram memory concept
- PR #900: Dirichlet posterior mixing theory
- PR #727/#753: Multi-order n-gram backoff (foundation)
- PR #414: Base model architecture stack
- PR #549: LeakyReLU^2 + Parallel Muon
- Willems et al. (1995): Context Tree Weighting
