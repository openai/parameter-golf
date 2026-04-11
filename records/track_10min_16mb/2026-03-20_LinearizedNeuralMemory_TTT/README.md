# Linearized Neural Memory + TTT

**val_bpb: 1.1844** | Non-record submission (novel architecture)

## Approach

This submission adds a neural memory module inspired by the Titans paper to each transformer block. The Titans memory module from the paper uses sequential gradient descent loops for memory update, which ended up being too slow given the 10 minute training budget. I made it a closed form linearization that eliminates all loops for increased throughput.

The cumulative gradient update `W_t = W_0 - θ Σ grad_i` turned out to be equivalent to causal linear attention, computable with ` cumsum` + ` einsum`. It compiled with ` fullgraph=True` and added around 8k parameters per layer with negligible throughput overhead.

The memory is placed in between attention and the MLP as a gated residual. Each token's retrieval incorporates all the prior tokens' prediction errors, which functions as a form of test-time adaptation built into the forward pass instead of being applied at eval time.

## Combined Techniques

On top of the neural memory, this submission incorporates techniques from other submissions that achieved SOTA:

- 10 layers with overtone spectral embedding init and phase-transition residual mixing
- LoRA TTT at eval (per-document rank-8 adaptation on Q/V/lm_head)
- FP16 embedding bypass and int6 quantization on middle layers
- Muon weight decay (0.02, decoupled), AdamW for embeddings/scalars, warmdown=2500

## Results

| Metric | Value |
|--------|-------|
| val_bpb (with TTT) | **1.1844** |
| val_bpb (standard) | 1.2163 |
| Artifact size | 14.5 MB |
| Training steps | 8,641 @ 69ms/step |

## References

- [Titans: Learning to Memorize at Test Time](https://arxiv.org/abs/2501.00663)

## Reproduction

```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```
