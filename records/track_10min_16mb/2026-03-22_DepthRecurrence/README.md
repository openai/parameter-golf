# Depth Recurrence (3+3 unique x 2 loops)

## Summary

Replace 10 unique transformer layers with 6 unique layers (3 encoder + 3 decoder) looped 2 times each, yielding 12 effective layers from half the unique parameters. The freed byte budget (~7.5MB) can be reinvested into wider MLPs or more BigramHash buckets.

Built on top of the current #1 submission (10L Int5-MLP, 1.1428 bpb).

## Architecture Changes

- **Depth recurrence**: `encoder_blocks` and `decoder_blocks` ModuleLists, each looped `num_loops` times
- **Per-loop conditioning**: Learned scale + bias per (loop, block) so layers differentiate passes
- **U-Net preserved**: Skip connections work across effective layers (LIFO), looping within each half separately
- **LR scaling**: Matrix and scalar LRs scaled by `1/sqrt(num_loops)` to compensate for gradient accumulation through tied weights

## Training HW Optimizations

- **Async data prefetch**: Background thread + separate CUDA stream loads next batch during forward+backward
- **Pinned memory**: CPU tensors use `pin_memory()` for fast async H2D transfers
- **NCCL tuning**: `NCCL_NVLS_ENABLE=1`, `NCCL_NET_GDR_LEVEL=5` for H100 NVLink topology
- **GPU-resident SWA**: Accumulate SWA state on GPU in float32, avoiding D2H sync per checkpoint
- **Cache cleanup**: `torch.cuda.empty_cache()` after warmup to reduce memory fragmentation

## Configurations

| Config | Unique layers | Effective depth | MLP mult | Expected artifact |
|--------|:---:|:---:|:---:|:---:|
| 3+3 x 2 loops (default) | 6 | 12 | 3x | ~10MB |
| 3+3 x 2 loops, wider | 6 | 12 | 5x | ~14MB |
| 2+2 x 3 loops | 4 | 12 | 3x | ~7MB |

## Environment Variables

```bash
NUM_UNIQUE_ENCODER=3    # unique encoder blocks
NUM_UNIQUE_DECODER=3    # unique decoder blocks
NUM_LOOPS=2             # times to loop each half
RECURRENCE_LR_SCALE=0   # 0 = auto (1/sqrt(num_loops))
NUM_LAYERS=0            # >0 disables recurrence (backward compat)
```

## Run

```bash
torchrun --nproc_per_node=8 train_gpt.py
```

## References

- ALBERT (Lan et al., 2019): Cross-layer parameter sharing
- Universal Transformers (Dehghani et al., 2018): Adaptive depth via recurrence
