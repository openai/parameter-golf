# Non-Record: U-Net Transformer + Int8 QAT + LeakyReLU² + Muon

**U-Net Skip Connections + Muon Optimizer + Int8 QAT + Leaky ReLU Squared + EMA + zlib**

**val_bpb: 1.6632** (best seed) | **artifact: 8.77 MB** | DGX Spark (1x GB10 Blackwell, 128GB unified)

> **Non-record submission** — trained on NVIDIA DGX Spark (single GB10 GPU) rather than 8xH100. Submitted to demonstrate Blackwell consumer-class GPU training viability and compression techniques on unified memory architecture. Entire project developed with AI-assisted coding (Claude, GPT, Gemini).

## Results

| Metric | Seed 42 | Seed 314 | Seed 999 |
|--------|---------|----------|----------|
| val_bpb (int8 roundtrip) | 1.6642 | 1.6694 | 1.6632 |
| Steps | 319 | 321 | 319 |
| Artifact size | 8,772,205 B | 8,785,874 B | 8,777,249 B |
| Training time | ~80 min | ~80 min | ~80 min |

## Architecture

- **9 transformer layers**, dim=512, 8 heads, 4 KV heads (GQA), head_dim=64
- **U-Net skip connections**: encoder/decoder halves with learned skip weights and per-block residual mixing from input embedding
- **Leaky ReLU squared** MLP activation (negative_slope=0.5, then squared)
- **Tied embeddings** with separate tied_embed_lr
- **Logit softcap** (tanh-based, cap=30)
- **RoPE** positional encoding (base=10000)
- Vocab size 1024 (SentencePiece BPE)
- Sequence length 1024
- 17,059,912 parameters

## Training

- **Muon optimizer** for matrix parameters with Newton-Schulz orthogonalization
- **Adam** for embeddings and scalar/vector parameters
- **Quantization-aware training (QAT)**: int8 straight-through estimator during forward pass
- **Late QAT**: optional delayed activation based on LR warmdown progress
- **EMA** (exponential moving average) of model weights
- **Warmdown** LR schedule based on wallclock time remaining
- **Muon momentum warmup** from 0.85 to 0.95 over first 500 steps
- 524,288 tokens per step, 8 gradient accumulation steps

## Quantization & Serialization

- **Int8 per-row quantization** for all weight matrices (clip percentile 99.99984%)
- **Float16 passthrough** for small tensors (<65536 elements) and control parameters
- **zlib compression** (level 9) on serialized checkpoint
- Roundtrip validation: decompress, dequantize, and re-evaluate to verify BPB integrity
- Quantization degradation: ~0.04 BPB (pre-quant ~1.625 vs post-quant ~1.664)

## Hardware

- **NVIDIA DGX Spark**: ARM64 (aarch64), GB10 GPU (Blackwell sm_121)
- 128 GB unified CPU+GPU memory (no discrete VRAM)
- CUDA 13.0, PyTorch 2.11+cu128
- Single GPU training via torchrun
- ~15s/step, ~319 steps in 80 min wallclock

## Development

This submission was developed entirely with AI-assisted development using Claude, GPT, and Gemini for architecture exploration, hyperparameter tuning, and code implementation. The project demonstrates the viability of using consumer-class Blackwell hardware (DGX Spark) combined with AI coding assistants for competitive language model compression research. All training and evaluation ran on DGX Spark hardware.
