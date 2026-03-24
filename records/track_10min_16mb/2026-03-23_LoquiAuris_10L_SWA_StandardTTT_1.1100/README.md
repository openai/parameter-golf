# Loqui Auris — 10L + SWA + Standard TTT

**Built by Loqui Auris**

## Result

**val_bpb: 1.1100** (seed 1337)

## Architecture

- 10 transformer layers, dim=512, 8 heads / 4 KV heads (GQA)
- MLP 3x expansion (1536 hidden), ReLU² activation
- SmearGate + BigramHash(4096, dim=128) + OrthoInit
- U-Net skip connections, tied embeddings
- Logit softcap: 30.0
- ~24.7M parameters

## Training

- Muon optimizer (matrix_lr=0.02, WD=0.04, momentum=0.99)
- AdamW for embeddings/scalars
- Batch size: 786,432 tokens, seq_len=2048
- SWA: 29 checkpoints averaged
- ~5992 steps in 600s on 8xH100

## Quantization

- INT6 per-row for weight matrices
- FP16 for tied embeddings
- zstd level 22 compression
- Artifact: 15.69 MB (250KB headroom)

## Test-Time Training

Standard AdamW TTT on validation data (10 epochs, lr=0.0005).

## Acknowledgments

- Training stack: PR #162 (raahilshah), PR #180 (thwu1)
- TTT approach: PR #77 (samacqua), PR #442 (sjp611)

## Platform

RunPod 8xH100 SXM 80GB.
