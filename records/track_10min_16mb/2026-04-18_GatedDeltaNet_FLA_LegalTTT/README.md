# GatedDeltaNet (FLA) + Legal Score-First TTT

**val_bpb: 1.00995** (3-seed mean, std 0.0012) | **~15.8 MB** | 8xH100 SXM

## Results (8xH100 80GB SXM, PyTorch 2.9.1+cu128)

| Seed | Steps | EMA BPB | Pre-TTT BPB | **Post-TTT BPB** | TTT Gain | Artifact |
|------|-------|---------|-------------|-----------------|----------|----------|
| 42 | 2,364 | 1.001693 | 1.021422 | **1.011302** | -0.010120 | 16,600,916 |
| 314 | 2,398 | 0.999552 | 1.018725 | **1.008960** | -0.009764 | 16,548,775 |
| 999 | 2,370 | 1.000492 | 1.019672 | **1.009589** | -0.010083 | 16,474,250 |
| **Mean** | **2,377** | **1.000579** | **1.019940** | **1.009950 (std 0.0012)** | **-0.009989** | |

## Key Innovation: GatedDeltaNet Linear Attention

This submission replaces softmax attention with **GatedDeltaNet** from the [Flash Linear Attention](https://github.com/sustcsonglin/flash-linear-attention) library (`fla-core==0.4.2`). GDN provides O(n) sequence complexity through a gated delta rule recurrence, enabling:

- **More parameters per FLOP**: No quadratic attention cost means more budget for model width/depth
- **Implicit state compression**: Recurrent state captures long-range dependencies without explicit KV cache
- **TTT-friendly architecture**: All parameters participate meaningfully in adaptation (no frozen attention matrices)

Architecture: `K_KVShare_Wider` config from PR #1687 — 10 GDN layers, 544d, 8 heads, KV sharing stride=2.

## Legal TTT Protocol

Score-first TTT following the framework from PR #461, adapted for GDN:

1. Val tokens split into non-overlapping 32K-token chunks
2. **For each chunk**:
   - **SCORE**: Sliding window eval under `torch.no_grad()` — no gradients, no weight mutation
   - **TRAIN**: SGD(lr=0.005, momentum=0.9) on the already-scored chunk. 3 epochs, freeze first 2 blocks, cosine LR decay, grad clip 1.0
3. Last chunk scored but never trained on
4. Chunk N scored by model adapted only on chunks 0..N-1

### GDN-Specific Adaptations

- No `torch.compile` on backward pass (Triton kernel compatibility with FLA)
- Uses `model(x, y)` for training (returns loss directly) and `model.forward_logits(x)` for scoring
- All recurrent and MLP parameters adapt (recurrent state is implicit in weight matrices)

### TTT Hyperparameters

| Parameter | Value |
|-----------|-------|
| Chunk size | 32,768 tokens |
| Optimizer | SGD + momentum(0.9) |
| Learning rate | 0.005 (cosine decay across chunks) |
| Epochs per chunk | 3 |
| Frozen blocks | 2 (first 2 blocks frozen) |
| Gradient clip | 1.0 |

### Timing Budget

| Phase | Time |
|-------|------|
| Training (7000 max steps, 600s wallclock) | 600s |
| Standard eval (int6 roundtrip + sliding window) | ~120s |
| Legal TTT (score-first sliding + adaptation) | ~200s |
| **Total eval** | **~320s (< 10 min)** |

## Training Architecture

PR #1687 `K_KVShare_Wider` with full production recipe:

| Component | Setting |
|-----------|---------|
| Layers | 10 GDN (544d, 8H) |
| KV Sharing | Stride 2 |
| MLP | 3x width |
| BigramHash | 5120 |
| SmearGate | Enabled |
| Weight avg | EMA(0.997) + SWA(every 50) |
| Late QAT | Threshold 0.15 |
| Quantization | Int6 matrices + Int8 embeddings + zstd-22 |
| Optimizer | Muon (matrices) + Adam (scalars/embeds) |
| Attention | GatedDeltaNet (FLA) — O(n) linear |

## Run Command

```bash
ARCH_MODE=K TTT_ENABLED=1 TTT_LR=0.005 TTT_EPOCHS=3 \
TTT_CHUNK_TOKENS=32768 TTT_FREEZE_BLOCKS=2 TTT_MOMENTUM=0.9 \
TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
SEED=42 MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Comparison with Prior Work

| Submission | BPB | Delta vs Ours |
|-----------|-----|---------------|
| **This (GDN + TTT)** | **1.00995** | — |
| PR #1687 (GDN, no TTT) | 1.04090 | +0.031 |
| #1 3-Layer Recurrence + TTT | 1.08100 | +0.071 |
| #2 Parallel Residuals + TTT | 1.08220 | +0.073 |

## Dependencies

- `flash-linear-attention==0.4.2`
- `fla-core==0.4.2`
- PyTorch >= 2.6.0
- `triton`, `einops`, `zstandard`, `sentencepiece`

## Credits

- **GatedDeltaNet architecture**: [PR #1687](https://github.com/openai/parameter-golf/pull/1687) by @resouer — K_KVShare_Wider config, FLA integration, full training recipe
- **TTT recipe**: [PR #461](https://github.com/openai/parameter-golf/pull/461) by @Christopher-Lee-McClendon — score-first legal TTT framework (adapted for GDN)
- **Flash Linear Attention**: [FLA](https://github.com/sustcsonglin/flash-linear-attention) by @sustcsonglin — GatedDeltaNet implementation
