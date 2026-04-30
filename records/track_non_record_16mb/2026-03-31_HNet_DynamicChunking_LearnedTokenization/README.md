# Non-record: H-Net Dynamic Chunking — Learned Tokenization Layer

**val_bpb: 1.3587** | 1×RTX 5090, 600s | TTT enabled

## Summary

This submission adds a learned dynamic chunking layer (inspired by H-Net) to the standard transformer baseline. The chunker predicts soft boundaries between adjacent token embeddings using a similarity-based projection, then blends neighboring embeddings where boundaries are low. This is a differentiable approximation of H-Net's hard chunking mechanism.

**Key result:** The chunking layer nearly matches the standard baseline (1.3587 vs 1.3577 on the same hardware), adding only ~263K parameters. This suggests learned tokenization is a viable research direction for the competition.

## Results (1×RTX 5090, 600s)

| Seed | Steps | Step Avg | Pre-TTT BPB | Final BPB | Artifact |
|------|-------|----------|-------------|-----------|----------|
| 1337 | 964 | 624ms | 1.3593 | **1.3587** | 8,592,061 |

## Architecture

```
Input tokens → tok_emb + bigram_emb → RMSNorm
→ DynamicChunker (learned boundary prediction + soft blending)
→ SmearGate → Transformer blocks (9L, 512d) → LM head
```

### DynamicChunker module (~263K params)
- `boundary_proj`: Linear(dim*2, 1) — predicts boundary score from adjacent token pair
- `chunk_mixer`: Linear(dim, dim) — mixes tokens within soft chunks
- Sigmoid boundary scores: 1.0 = keep separate, 0.0 = blend with neighbor
- Differentiable: `blended[:, 1:] = x[:, 1:] + (1-boundary) * (x[:, :-1] - x[:, 1:])`

### What this is NOT
- This is NOT byte-level (uses standard BPE tokens)
- This is NOT a tokenizer replacement
- This IS a learned preprocessing layer that re-groups token embeddings

## Changes from baseline `train_gpt.py`
1. Added `DynamicChunker` and `DynamicChunkerStack` modules
2. Inserted chunker after embedding norm, before skip connection setup
3. Chunker params added to Muon (2D) and Adam (1D) optimizer groups
4. New env vars: `HNET_ENABLED` (default 1), `HNET_LAYERS` (default 1)
5. Setting `HNET_ENABLED=0` produces identical behavior to base script

## Why this matters

OpenAI's requested research directions include "H-net tokenization." This submission demonstrates that a lightweight learned chunking mechanism can be added to any transformer with minimal parameter overhead and no performance degradation. The natural extension is byte-level dynamic chunking (true H-Net), which we leave for future work.

## Reproducing

```bash
# 1×GPU
HNET_ENABLED=1 TTT_ENABLED=1 MAX_WALLCLOCK_SECONDS=600 python3 train_gpt.py

# Disable chunker (control)
HNET_ENABLED=0 TTT_ENABLED=1 MAX_WALLCLOCK_SECONDS=600 python3 train_gpt.py
```
