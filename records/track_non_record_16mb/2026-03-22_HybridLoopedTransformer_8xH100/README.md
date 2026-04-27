# Hybrid Depth-Recurrent Transformer (Non-Record)

**val_bpb: 1.2334** (sliding window stride=64, 8xH100 SXM, 600s)

## Architecture: Depth Recurrence with Solved Quantization Compounding

The challenge description calls out depth recurrence as a promising direction. The core idea: run N physical layers K times for N*K effective depth at N layers of parameter cost. This lets you trade the abundant 10-minute compute budget for the scarce 16MB parameter budget.

**The catch:** prior attempts (PR #341) found that quantization errors compound across loop iterations — a 0.40 BPB gap that made the approach unviable.

**Our solution: Hybrid entry/exit architecture + int8 shared layers.**

```
[UNIQUE Entry Layer]  ← int6 quantization, sees raw embeddings
[SHARED Layer 1]  ←┐
[SHARED Layer 2]   │  int8 quantization (higher precision)
[SHARED Layer 3]   │  Run 2x (looped)
[SHARED Layer 4]   │  Per-loop learnable scale parameters
[SHARED Layer 5]   │  differentiate each iteration
[SHARED Layer 6]  ←┘
[UNIQUE Exit Layer]   ← int6 quantization, produces output
```

**Result: quantization gap reduced from 0.40 BPB to 0.007 BPB** — matching non-looped SOTA.

## Novel Input Features (Zero Parameter Cost)

Three precomputed features injected at the embedding layer:

1. **Word-position index** (32-dim one-hot): distance from last word boundary. Eliminates the need for the model to discover word-internal position patterns.

2. **Copy flags** (5-dim binary): whether this token appeared 1/2/4/8/16 positions ago. Copy mechanisms are hard for small transformers but trivial to precompute.

3. **Unigram log-frequency** (1-dim scalar): base rate of each token. Model only needs to learn contextual deltas.

Total: 38 features → projected to model_dim via zero-initialized linear. ~19K additional parameters.

## Results (8xH100 SXM, seed 42)

| Metric | Value |
|--------|-------|
| Physical layers | 8 |
| Loop count | 2 |
| **Effective depth** | **16** |
| Stored params | 20,041,794 |
| Steps (10 min) | 2,169 |
| Step time | 277ms |
| Pre-quant val_bpb | 1.2499 |
| Post-quant standard | 1.2571 |
| **Post-quant sliding s64** | **1.2334** |
| Quant degradation | 0.007 BPB |
| Artifact size | 14,156,560 bytes |

## Why It Doesn't Beat SOTA (Yet)

The 277ms/step (vs SOTA's 85ms) means only 2169 steps instead of ~7000. Block-level torch.compile can't fuse as aggressively as full-model compile due to the looping control flow. The parameter savings (20M vs 27M) don't compensate for 3x fewer training steps within the 10-minute window.

**The approach would dominate in an unlimited-compute setting** where the 16MB constraint matters but training time doesn't.

## Key Engineering Contributions

1. **Solved DDP + shared parameters**: `find_unused_parameters=True` handles the per-loop scale params that don't all participate in every forward pass.

2. **Solved quantization compounding**: Hybrid unique/shared architecture with int8 for looped layers eliminates the 0.40 BPB gap reported by PR #341.

3. **Block-level torch.compile**: Individual block compilation avoids triton register OOM while still accelerating attention/MLP kernels.

4. **Portable state dict**: Strip `_orig_mod.` prefix from compiled module keys for clean save/load roundtrip.

## Run Command

```bash
RUN_ID=looped SEED=42 \
NUM_LAYERS=8 LOOP_COUNT=2 USE_INPUT_FEATURES=1 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Built on PR #315 by @alertcat (XSA, EMA, Partial RoPE, LN Scale).
