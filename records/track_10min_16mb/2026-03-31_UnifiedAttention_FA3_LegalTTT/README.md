# Unified Attention + FA3 Head-Dim Padding + Legal Score-First TTT

**val_bpb: 1.1412** (3-seed mean, std 0.0008) | **~15.97 MB** | 8×H100 SXM

## Results (8×H100 80GB SXM, PyTorch 2.8.0+cu128)

| Seed | step_avg | steps | Pre-TTT bpb | **Post-TTT bpb** | TTT gain | TTT time | Artifact |
|------|----------|-------|-------------|-----------------|----------|----------|----------|
| 1337 | 49.6ms | 12,088 | 1.1647 | **1.1416** | -0.0231 | 408s | 15,991,687 |
| 42 | 49.6ms | 12,109 | 1.1647 | **1.1416** | -0.0231 | 407s | 15,988,916 |
| 2025 | 49.6ms | 12,103 | 1.1635 | **1.1403** | -0.0232 | 408s | 15,962,515 |
| **Mean** | **49.6ms** | **12,100** | **1.1643** | **1.1412 (std 0.0008)** | **-0.0231** | **~408s** | |

## Key Innovation: Unified Attention

Unified Attention ([Deshwal, 2026](https://github.com/ReinforceAI/yocto)) replaces the three separate Q, K, V projections in standard self-attention with a single W_unified projection. The output splits into three functional bands after the matmul. In our research, we found that one matrix carries enough geometric structure for all three roles. The amplitude and phase rotation across output dimensions create functional differentiation naturally, and three bands form on their own during training.

```python
# Standard: 3 separate projections for routing
q = W_q @ x;  k = W_k @ x;  v = W_v @ x

# Unified Attention: 1 projection, bands form naturally
unified = W_unified @ x
seeking, offering, content = unified.split(d//3, dim=-1)
```

The core insight is that attention is a routing mechanism, which decides which tokens talk to each other. The FFN does the heavy lifting and actually transforms information at each position. Unified attention cuts the routing budget and gives those bytes to the FFN.

This matters most in parameter-constrained settings. In the 16 MB challenge, the bytes we save on attention go straight to the MLP:

| | Standard (SOTA) | Unified (Ours) |
|---|---|---|
| Attention (compressed) | 5.10 MB | **2.82 MB** |
| MLP (compressed) | 10.21 MB | **12.70 MB** |

We trade 2.28 MB of routing for 2.49 MB of computation. The MLP gets a bigger budget, and it shows.

## Key Innovation: FA3 Head-Dim Padding

Flash Attention 3 (Hopper) requires head_dim to be a multiple of 8. Our architecture uses head_dim=44 (from d=528, 4 heads). Rather than constraining the architecture, we **zero-pad to 48 dims before FA3 and slice back after**:

```python
pad_n = (8 - head_dim % 8) % 8  # 4 for head_dim=44
if pad_n > 0:
    q = F.pad(q, (0, pad_n))  # [B,T,H,44] → [B,T,H,48]
    k = F.pad(k, (0, pad_n))
    v = F.pad(v, (0, pad_n))
out = flash_attn_func(q, k, v, causal=True)
y = out[..., :head_dim]       # [B,T,H,48] → [B,T,H,44]
```

**Mathematically lossless**. Padded zeros contribute nothing to dot products or weighted sums. The 9% compute overhead from 44→48 dims is overwhelmed by FA3's 1.57× speedup over FA2/SDPA, giving a net **51ms/step** (vs 67ms SDPA, 65ms FA2). This unlocks **11,714 training steps** in 10 minutes, 40% more than FA2.

## Legal TTT Protocol

Backward-looking, score-first TTT following the PR #461 / PR #549 framework:

1. Val tokens split into 1,893 non-overlapping 32K-token chunks
2. **For each chunk**:
   - **SCORE**: Sliding window eval under `torch.no_grad()`. No weight mutation
   - **TRAIN**: SGD(lr=0.002, momentum=0.9) on the already-scored chunk. 3 epochs, all blocks unfrozen, cosine LR decay, grad clip 1.0
3. Last chunk scored but never trained on

### TTT Hyperparameters

| Parameter | Value |
|-----------|-------|
| Chunk size | 32,768 tokens |
| Optimizer | SGD + momentum(0.9) |
| Learning rate | 0.002 (cosine decay across chunks) |
| Epochs per chunk | 3 |
| Frozen blocks | None (all blocks adapt) |
| Gradient clip | 1.0 |
| Eval stride | 64 |

### Timing Budget

| Phase | Time |
|-------|------|
| Training | 600s (10 min) |
| Quantization + roundtrip validation | ~70s |
| Legal TTT (score-first + adaptation) | ~490s |
| **Total eval** | **~560s (< 10 min)** |

## Training Architecture

| Component | Setting |
|-----------|---------|
| **Attention** | **Unified Attention (single W_unified, 67% fewer attn params)** |
| Layers | 11 unique (K=11, R=1) |
| Dimension | 528 (component_dim=176, head_dim=44) |
| Heads | 4 |
| MLP | 3× (1584) with LeakyReLU(0.5)² |
| SmearGate | Position-mixing gate (zero-init sigmoid) |
| LN Scale | 1/√(layer+1) on norm outputs |
| VE128 | Value embedding on layers 9-10 |
| U-Net skips | Encoder-decoder connections |
| Weight avg | EMA(0.997) + Tight SWA(every 50) |
| Quantization | GPTQ-lite int6 + int8 embeddings + LZMA-6 |
| QAT | STE int6 at fraction ≥ 0.15 |
| Optimizer | Parallel Muon (batched NS5, async reduce-scatter) |
| **Flash Attention** | **FA3 (Hopper) with head_dim zero-padding 44→48** |
| Total params | 23,209,295 |

### Parallel Muon with Parameter Banking

4 contiguous 3D `nn.Parameter` banks replace 44 separate weight matrices:
- `unified_bank[K, d, d]`: unified attention projections
- `output_bank[K, d, comp]`: attention output projections
- `fc_bank[K, 3d, d]`: MLP up-projections
- `proj_bank[K, d, 3d]`: MLP down-projections

Batched Newton-Schulz orthogonalization via 3D tensor operations. DDP replaced with async reduce-scatter → local NS5 → async all-gather.

## Ablation

Incremental contribution of each technique:

| Change | BPB | Delta |
|--------|-----|-------|
| Baseline (9L, 512d, relu²) | 1.2244 | |
| + 10L, 3× MLP, seq 2048 | ~1.17 | ~-0.05 |
| + SmearGate | ~1.167 | -0.003 |
| + LN Scale | ~1.165 | -0.002 |
| + VE128 (layers 8-9) | ~1.161 | -0.004 |
| + Unified Attention (replaces Q/K/V) | ~1.161 | ±0.000 (same BPB, 67% fewer attn params) |
| + K=11 (depth > width) | ~1.155 | -0.006 |
| + FA3 with head-dim padding (51ms/step) | 1.1649 | -0.010 (more steps) |
| + Legal Score-First TTT | **1.1414** | -0.024 |

## Negative Results

| Technique | Impact | Root Cause |
|-----------|--------|------------|
| XSA (last 4 layers) | +0.0015 worse | Content band coupled to seeking/offering in shared projection |
| BigramHash at input | +0.009 worse | Single matrix can't route bigram across 3 functional bands |
| EMA 0.999 | +0.007 worse | Over-smooths weight distribution |
| Soft-sigmoid QAT | Training stall steps 2000-6000 | Ramping alpha creates unstable gradients; simple STE works |
| K=11 d=552 (mixed int5/int6) | Over budget (16.9 MB) | Unified attention weights have higher entropy than Q/K/V |

## Requirements

Flash Attention 3 (Hopper) is required:

```bash
pip install flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch280
pip install sentencepiece zstandard
```

## Run Command

```bash
NUM_UNIQUE_LAYERS=11 MODEL_DIM=528 NUM_HEADS=4 \
VE_LAYERS=9,10 \
EMA_DECAY=0.997 QAT_START_FRACTION=0.15 \
TRAIN_BATCH_TOKENS=524288 \
SLIDING_WINDOW_EVAL=0 \
VAL_LOSS_EVERY=3000 TRAIN_LOG_EVERY=500 \
LEGAL_TTT_EPOCHS=3 \
TTT_LORA_ATTN=0 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Credits

- **Unified Attention architecture**: This work (Viraj Deshwal, Reinforce AI)
- **FA3 head-dim padding**: This work
- **LeakyReLU² activation**: [PR #493](https://github.com/openai/parameter-golf/pull/493) by @parinzee
- **SmearGate**: [PR #65](https://github.com/openai/parameter-golf/pull/65) by @aquariouseworkman
- **LN Scale**: [PR #315](https://github.com/openai/parameter-golf/pull/315) by @jfprincz
- **VE128**: [PR #374](https://github.com/openai/parameter-golf/pull/374) by @unnir
- **Parameter Banking + Parallel Muon**: [PR #399](https://github.com/openai/parameter-golf/pull/399) by @abaybektursun
- **Legal TTT recipe**: [PR #461](https://github.com/openai/parameter-golf/pull/461) by @Christopher-Lee-McClendon
- **FA3 prebuilt wheels**: [windreamer/flash-attention3-wheels](https://github.com/windreamer/flash-attention3-wheels)