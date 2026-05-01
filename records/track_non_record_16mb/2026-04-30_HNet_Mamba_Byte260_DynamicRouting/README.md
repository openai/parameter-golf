# HNet Mamba Byte-Level LM with Dynamic Routing

**Track:** Non-Record (Research Contribution)  
**Author:** firfre  
**Date:** 2026-04-30  
**val_bpb:** 3.4922 | **val_loss:** 2.4206 | **Compressed size:** 13.25 MB

---

## Overview

A hierarchical byte-level language model based on the H-net architecture, using Mamba2 (SSM) for byte-level encoding/decoding and Transformer attention for the word-level chunk processor. Dynamic boundary routing learns to segment the byte stream into variable-length chunks.

## Architecture

```
byte input (vocab=260)
→ Embedding (dim=512)
→ Encoder: 2× Mamba2 layers (byte-level)
→ RoutingModule: cosine similarity boundary detection
→ ChunkLayer: dynamic downsampling (~1/8 rate target)
→ Main: 4× Attention layers (word-level, dim=576)
→ DeChunkLayer: EMA upsample back to byte-level
→ Residual connection
→ Decoder: 2× Mamba2 layers (byte-level)
→ LM Head
```

**Arch layout:** `m2-T4-m2`  
**Byte dim:** 512 | **Chunk dim:** 576 | **Chunk FFN:** 2304  
**Mamba2:** state=32, expand=2, conv=4  
**Routing downsample target:** 8× (1 boundary per 8 bytes)

## Key Design Choices

- **byte260 tokenizer**: Pure byte-level, vocab=260, no BPE needed
- **Dynamic routing**: RoutingModule uses cosine similarity between adjacent hidden states to detect natural boundaries (word/phrase boundaries)
- **STE gradient**: Straight-Through Estimator allows gradients to flow through hard boundary selection
- **EMA upsample**: DeChunkLayer uses Mamba2's `mamba_chunk_scan_combined` kernel as an EMA to spread word-level context back to byte positions
- **int6 + Brotli-11**: Large transformer matrices quantized to int6, Mamba SSM dynamics (A_log, dt_proj, conv1d, D) kept at int8, compressed with Brotli quality=11

## Observations

- Dynamic routing convergence is challenging with byte-level input: `boundary_mean` stabilized at ~0.31 instead of the target 0.125 (1/8). The encoder (2 Mamba layers) may be too shallow to build semantic representations sufficient for reliable boundary detection before routing.
- Increasing `ROUTING_LOSS_WEIGHT` from 1.0 to 3.0 made training worse, suggesting the load-balancing loss fights the LM objective at high weights.
- A fixed-stride variant (selecting every 8th byte deterministically) achieved 1.9733 bpb with the same architecture, suggesting the dynamic routing is the bottleneck rather than the hierarchical structure itself.

## Training

```bash
VOCAB_SIZE=260 \
TOKENIZER_PATH=./data/tokenizers/fineweb_pure_byte_260.json \
DATA_PATH=./data/datasets/fineweb_byte260 \
MAMBA_STATE=32 CHUNK_MODEL_DIM=576 CHUNK_ROTARY_DIM=48 \
NUM_LAYERS=8 ENCODER_LAYERS=2 CHUNK_LAYERS=4 DECODER_LAYERS=2 \
ROUTING_LOSS_WEIGHT=1.0 ROUTING_DOWNSAMPLE=8.0 \
python train_hnet_repo_pg.py
```

**Hardware:** 1× H100 80GB  
**Steps:** 925 / 20000 (600s wall-clock cap)  
**Step time:** ~649 ms/step  
**Peak memory:** 10,377 MiB

## Compression

| Format | Size |
|--------|------|
| Raw torch (float16) | 51.4 MB |
| int6 + Brotli-11 | **13.25 MB** |
| Compression ratio | ~3.9× |

Mamba SSM dynamics kept at int8 (sensitive to int6 quantization): `A_log`, `dt_proj`, `conv1d.weight`, `.D`
