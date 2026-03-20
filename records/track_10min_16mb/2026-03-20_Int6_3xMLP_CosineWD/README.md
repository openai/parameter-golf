# Int6 3xMLP + Cosine Warmdown + OrthoInit

**val_bpb: 1.1704** (single seed, 8xH100 SXM)

## Key Techniques

1. **Int6 STE quantization**: Quantize weights to 6-bit range ([-31,31]) during both training (straight-through estimator) and export. Saves ~3MB vs int8, enabling a wider model within the 16MB budget.

2. **3x MLP width** (hidden=1536): With int6 freeing parameter budget, MLP hidden dimension expanded from 2x (1024) to 3x (1536), adding ~4.8M parameters for better factual knowledge storage.

3. **Cosine warmdown schedule** (novel): Replaces linear warmdown with `0.5 * (1 + cos(pi * progress))`. Keeps LR higher for longer during early warmdown, then decays faster near the end. More effective use of the warmdown phase.

4. **Orthogonal initialization**: All non-zero-init linear layers initialized with orthogonal matrices (gain=1.0) for better gradient flow.

5. **Zstd level 22 compression**: Better compression ratio than zlib, contributing to the headroom that enables the wider model.

6. **RoPE base 50000**: Increased from 10000 for better position allocation.

Built on the merged SOTA (SlidingWindow_FP16Emb_10L_MuonWD_OvertoneInit, 1.1748 bpb) which provides:
- 10 transformer layers
- Sliding window eval (stride=64)
- FP16 tied embedding passthrough
- Decoupled Muon weight decay (0.02)
- Overtone spectral embedding init
- Phase-transition residual mixing

## Results

| Seed | val_loss | val_bpb | Steps | ms/step |
|------|----------|---------|-------|---------|
| 1337 | 1.9762 | **1.1704** | 7630 | 79.04 |

- Artifact: 13,520,480 bytes (13.5MB, well under 16MB)
- Model int8+zlib: 13,463,803 bytes
- Code size: 56,677 bytes
- Training time: 603,111ms (10 min wallclock cap)
- Peak memory: 15,732 MiB
- Eval time: 61,387ms (sliding window, stride=64)

## Run Command

```bash
EVAL_STRIDE=64 ROPE_BASE=50000 RUN_ID=final_v2 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

All other hyperparameters use script defaults. No additional env overrides needed.

## Ablation Notes

- Int6 + 3x MLP is the single biggest contributor (~-0.022 bpb per PR #212 ablations)
- Cosine warmdown provides smoother convergence vs linear warmdown
- OrthoInit improves gradient flow at initialization
- Sliding window eval adds ~0.03 bpb improvement at eval time
