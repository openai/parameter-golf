# SwiGLU + VE128 + U-Net Skip Gates (No TTT)

**val_bpb (3-seed mean): 1.11807945** (std: 0.000836, best: 1.11691774)

| Seed | val_bpb |
|------|---------|
| 42   | 1.11885136 |
| 123  | 1.11691774 |
| 7    | 1.11846924 |

## Architecture

11-layer transformer with the following components:

- **SwiGLU FFN** with Star-ReLU activation (hidden=1792). Multiple idea banks listed SwiGLU as a dead end for non-TTT; this result demonstrates otherwise when paired with the right training configuration.
- **U-Net Skip Gates**: 5 encoder layers, 6 decoder layers with learned gating
- **XSA4**: Extended Self-Attention in last 4 layers
- **Value Embeddings (VE128)**: 128-dim shared embedding table, per-layer scales on layers 9-10
- **BigramHash**: 8192 buckets, 128-dim embeddings
- **EMA** (decay=0.997)
- **Partial RoPE** (16 dims)
- **LN Scale**: Layer-dependent normalization scaling
- **Late QAT@0.15**: Quantization-aware training enabled when LR scale < 0.15
- **Int6 + GPTQ-lite + zstd-22** compression

## Training Configuration

- **Sequence length: 2048** (key finding: +0.008 bpb over seq_len=1024)
- **Batch tokens: 786,432**
- **Warmdown: 3,500 steps**
- **8xH100 SXM** (Modal)

## Key Ablation: Sequence Length

| Config | val_bpb | Notes |
|--------|---------|-------|
| Full arch, seq_len=1024 | 1.12670 | Wave 7 baseline |
| Full arch, seq_len=2048 | **1.11808** | This submission |
| Improvement | **-0.00862** | Longer context alone |

## Provenance

All architectural components (SwiGLU, U-Net skip gates, Star-ReLU, XSA4, BigramHash) were discovered through systematic ablation search (GEPA) and Codex-guided exploration across Waves 1-7. Value Embeddings adapted from the competition community. No test-time training.

## Compute

Approximately 18 minutes per seed on 8xH100 SXM (Modal). Three seeds for verification.
