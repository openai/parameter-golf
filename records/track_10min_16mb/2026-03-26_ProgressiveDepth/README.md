## Progressive Depth Training via Shared-Weight Recurrence

val_bpb = **1.1980** (sliding window, stride=256, int8+zstd22 roundtrip)
val_bpb = 1.2315 (standard int8+zstd22 roundtrip)

Progressive Depth is a training-time advantage unique to shared-weight recurrence — flat architectures cannot dynamically adjust their depth during training.

Because the same 3 blocks are reused at every depth, we can start training with 2 repeats (fast, cheap steps), then progressively increase to 3 and 4 repeats as training progresses. The model learns coarse representations quickly at shallow depth, then refines them at full depth. This is structurally impossible with flat architectures where each layer has unique parameters — you cannot add or remove layers mid-training without changing the parameter space.

### Progressive Depth Schedule

| Phase | Time | Repeats | Eff. depth | ms/step | Steps | val_bpb at end |
|-------|------|---------|------------|---------|-------|----------------|
| 1 | 0–40% | 2 | 6 | ~75 | ~3200 | 1.319 |
| 2 | 40–65% | 3 | 9 | ~86 | ~1200 | 1.298 |
| 3 | 65–100% | 4 | 12 | ~96 | ~1800 | 1.229 |

**Total: 5861 steps** in 600s vs ~4300 steps at constant depth 4 (+36% more gradient updates).

SWA (Stochastic Weight Averaging) collects checkpoints only during Phase 3 at full depth to avoid mixing representations from different depths. 18 checkpoints averaged.

### Ablation Trajectory

Each change isolated and measured on 8xH100 (sliding window eval):

| Change | val_bpb | Delta |
|--------|---------|-------|
| OpenAI Naive Baseline (9×512, unique layers) | 1.2244 | — |
| Depth Recurrence 3×4 + Cross-Repeat Skip (PR [#148](https://github.com/openai/parameter-golf/pull/148)) | 1.2213 | -0.003 |
| + XSA (Exclusive Self-Attention, last 4 layers) | 1.2110 | -0.010 |
| + LeakyReLU(0.5)² MLP | 1.2069 | -0.004 |
| + Progressive Depth (2→3→4 schedule) | 1.1980 | -0.009 |
| **Total** | **1.1980** | **-0.026** |

### Cross-Repeat Skip (Novel, PR [#148](https://github.com/openai/parameter-golf/pull/148))

Standard depth recurrence is stateless — each repeat starts fresh with no memory of previous passes. Cross-Repeat Skip turns this into stateful recurrence: each block receives a weighted residual of its own output from the previous repeat. Per-block, per-repeat learned scales (~7.5K params). This gives the model a direct gradient path across repeats without the overhead of unique parameters.

### Architecture

- 3 shared blocks × 4 repeats = 12 effective layers
- dim=832, 8 heads, 4 KV heads (GQA), MLP 2×, tied embeddings
- **XSA**: Subtracts self-value projection from attention output on last 4 effective layers (reduces attention collapse in deep recurrence)
- **LeakyReLU(0.5)²**: Replaces ReLU² — preserves gradient flow on negative activations through 4 recurrence passes
- 2 Value Embedding tables with per-layer learned scales
- Loop Embedding (depth-wise positional encoding)
- Logit softcap=30, RoPE, RMSNorm
- GPTQ-lite int8 quantization (per-row clip percentile search) + zstd-22 compression
- 17.14M params, 15.88MB artifact

### Training

Muon optimizer (momentum=0.95, 5 Newton-Schulz steps, WD=0.04) for matrix params, Adam for scalars/embeddings.

MATRIX_LR=0.012, SCALAR_LR=0.012, TIED_EMBED_LR=0.015, GRAD_CLIP_NORM=0.3, WARMDOWN_ITERS=3000.

Phase switching synchronized across DDP ranks via `all_reduce` (max elapsed time) to prevent race conditions during `torch.compile` recompilation.

### Command

```
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### Results

5861 steps, 600s on 8xH100. Roundtrip val_bpb 1.2315, sliding window 1.1980. Peak memory 25.5 GB/GPU.
