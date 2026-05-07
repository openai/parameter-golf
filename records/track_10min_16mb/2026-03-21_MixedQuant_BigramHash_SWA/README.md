# Optimized Baseline with Mixed Quantization and BigramHash

Tried a bunch of things on top of the baseline script. Some worked, some didn't. Here's what stuck.

## What I changed

**Architecture:**
- Bumped to 10 layers (from 9) with U-Net skip connections
- ReLU² MLP with 3x expansion (hidden=1536) — same param count per layer as SwiGLU but faster since it's only 2 matmuls
- Added BigramHash embeddings — hash table of 10240 bigram pairs mapped to 128-dim vectors, projected to model dim. Gives the model cheap access to "what was the previous token" without burning attention compute on it
- Orthogonal init for all weight matrices, SVD-based init for embeddings

**Quantization:**
- Mixed precision: INT6 for all weight matrices, INT8 for embeddings
- Straight-Through Estimator during training so the model learns to deal with quantization from the start
- zstd level 22 compression instead of zlib — squeezes out a few more percent

**Training:**
- Muon optimizer with weight decay 0.04 and gradient clipping at 0.3
- Stochastic Weight Averaging over the last 50% of training (every 50 steps) — this smooths out the weight distribution which helps quantization a lot
- Momentum warmup from 0.85 to 0.99 over 1500 steps

## Result

- **val_bpb: 1.2421** (post-roundtrip)
- Pre-roundtrip was 1.1924, so quantization costs about 0.05 bpb
- 11,070 steps in 600 seconds on 8xH100 SXM (~54 ms/step)
- Artifact: 13.28 MB (well under 16 MB limit)

## What I'd do differently

The main bottleneck is quantization degradation. Pre-roundtrip score is 1.19 which would be competitive, but INT6 quantization adds ~0.05 bpb. The top submissions get this down to 0.01-0.02. I think better STE scheduling or per-channel quantization could help here.

Also didn't get to try longer training with 80 shards — only used 10 due to disk constraints on the cloud setup. More data would probably help.
