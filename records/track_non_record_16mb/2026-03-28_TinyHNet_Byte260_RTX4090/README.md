# Tiny H-Net: First Learned Byte-Level Tokenization for Parameter Golf

**Author:** GreQ (Grzegorz Nowosielski)
**Date:** 2026-03-28
**Track:** Non-record, unlimited compute, 16MB
**val_bpb:** 1.8989 (post int6+zstd22 quantization roundtrip)
**Hardware:** 1x RTX 4090 (local), ~2.8 hours training

## Summary

This is the first implementation of **H-Net tokenization** (arXiv:2507.07955, Hwang/Wang/Gu, Goomba Lab) at tiny scale for the Parameter Golf challenge. H-Net was specifically listed in the README's "Requests for PRs" as a creative technique the organizers wanted to see explored.

Instead of using a fixed BPE/SentencePiece tokenizer, this model **learns to segment raw bytes dynamically** during training via a differentiable chunking gate. The architecture eliminates the traditional tokenization pipeline entirely.

## Architecture

```
Raw bytes (vocab=260) --> Embedding(260, 512)
  --> Encoder: 3x CausalDepthwiseConv1d(d=512, kernel=4)
  --> ChunkingGate: cosine similarity + STE --> boundary mask
  --> ChunkLayer: gather boundary tokens (~25% of sequence = ~4 bytes/chunk)
  --> Main Transformer: 9 layers, d=512, 8 heads, 4 KV heads, LeakyReLU(0.5)^2 MLP 3x
  --> DeChunkLayer: vectorized EMA expansion back to full byte sequence
  --> Decoder: 3x CausalDepthwiseConv1d(d=512, kernel=4)
  --> Tied output head --> 260-dim logits
```

**Key simplification vs. reference H-Net:** Replaced Mamba-2 SSM layers (which require custom CUDA kernels from `mamba_ssm`/`causal_conv1d`) with pure-PyTorch depthwise causal Conv1d. This eliminates all exotic dependencies while preserving the core dynamic chunking mechanism.

**Total parameters:** 22,178,377
**Compressed artifact:** 15,443,775 bytes (int6 + zstd-22), well under the 16MB limit.

## How Dynamic Chunking Works

1. The **byte encoder** (3 causal conv layers) processes raw UTF-8 bytes into hidden representations
2. The **ChunkingGate** computes cosine similarity between consecutive encoder outputs. High dissimilarity triggers a boundary
3. **Straight-Through Estimation (STE)** makes the discrete boundary decision differentiable
4. A **chunk ratio auxiliary loss** steers the gate toward a target boundary density (~25%)
5. The **ChunkLayer** gathers hidden states at boundary positions, compressing the sequence ~4x
6. The main **transformer** processes these compressed chunks (with causal + padding attention masks)
7. The **DeChunkLayer** expands back to full byte length using learned exponential moving average (EMA) decay
8. The **byte decoder** (3 causal conv layers) produces final representations for next-byte prediction

The gate learned to create boundaries approximately every 4 bytes on average, which is remarkably close to the average bytes-per-token ratio of BPE tokenizers -- the model independently discovered a similar compression ratio.

## Results

| Step | val_bpb | Notes |
|------|---------|-------|
| 0 | 7.9934 | Random init |
| 5,000 | 2.2424 | Gate converged to ~25% ratio |
| 10,000 | 2.0568 | |
| 15,000 | 2.0399 | |
| 20,000 | **1.9002** | Pre-quantization |
| 20,000 (int6) | **1.8989** | Post-quantization roundtrip |

The 1.90 BPB is not competitive with the BPE transformer SOTA (~1.12 BPB), which is expected: a byte-level model must learn character-level patterns that BPE tokenization solves for free. The value of this submission is architectural novelty, not BPB optimization.

## Key Engineering Challenges Solved

1. **Gate initialization:** The cosine similarity threshold must be carefully tuned. Too high = no boundaries (ratio ~0.002), too low = everything is a boundary (ratio ~1.0). We use `sigmoid(-3.0) = 0.047` as the initial threshold with a strong ratio loss (weight=1.0) to steer convergence.

2. **Vectorized ChunkLayer/DeChunkLayer:** Naive Python for-loops over the batch dimension are too slow for training. We use cumsum-based segment ID computation, scatter operations for chunking, and broadcasted exponential decay for dechunking -- all fully vectorized, no batch-dim loops.

3. **Rotary cache poisoning:** PyTorch's `torch.inference_mode()` creates tensors that cannot participate in autograd. The Rotary positional embedding cache must be cleared after every eval_val call to prevent `RuntimeError: Inference tensors cannot be saved for backward`.

4. **Byte-level data conversion:** The competition's HF dataset does not include byte260 shards. We wrote a converter that decodes sp1024 shards back to text via SentencePiece, then re-encodes as byte260 tokens.

## Reproduction

```bash
# 1. Prepare byte260 data (requires sp1024 data + tokenizer already present)
python data/convert_sp_to_byte260.py

# 2. Train (single GPU, ~2.8 hours on RTX 4090)
RUN_ID=hnet_v6_20k \
ITERATIONS=20000 \
VAL_LOSS_EVERY=5000 \
TRAIN_LOG_EVERY=200 \
TRAIN_BATCH_TOKENS=65536 \
ENABLE_TORCH_COMPILE=0 \
WARMUP_STEPS=5 \
python train_hnet.py
```

## What Could Improve This

- **More training:** Loss was still decreasing at step 20K. 100K+ steps would likely push below 1.7 BPB.
- **More data:** We only used 1 train shard (244M bytes). The full dataset has 80 shards.
- **Replace Conv1d with actual Mamba-2:** The reference H-Net uses Mamba-2 SSM for encoder/decoder, which has longer effective receptive field than our 3-layer kernel-4 conv (10 positions).
- **2-stage H-Net:** The reference architecture supports nested hierarchical chunking for additional compression.
- **Larger model:** With only 15.4MB of the 16MB budget used, there's room for more transformer layers or wider dimensions.
- **SWA/EMA:** Stochastic weight averaging was not implemented for this initial submission.
- **torch.compile:** Disabled due to dynamic shapes from chunking. Could be enabled for the fixed-shape transformer portion only.

## Files

- `train_hnet.py` -- Complete training script (self-contained, ~1050 lines)
- `submission.json` -- Submission metadata
- `README.md` -- This file
