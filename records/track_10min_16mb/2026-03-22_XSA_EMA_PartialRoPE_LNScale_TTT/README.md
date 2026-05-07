# 10L XSA + EMA + Partial RoPE + LN Scale

**val_bpb: 1.1365** (mean of seeds 42, 1337)

## Results
| Seed | val_bpb | Artifact | Steps |
|------|---------|----------|-------|
| 42 | 1.1365 | 15,759,319 | 6491 |
| 1337 | 1.1366 | 15,820,386 | 6520 |

## Architecture
- 10 layers, 512d, 8/4 GQA, 3x MLP (1536), ReLU^2
- XSA last 4 layers, EMA 0.997, Partial RoPE 16/64, LN Scale
- SmearGate + BigramHash(10240, 128), U-Net skips, softcap=30

## Training
- BF16, Muon (WD=0.04, momentum=0.99) + AdamW
- 786K batch tokens, seq 2048, warmdown 3000

## Quantization
- Int5 MLP / Int6 attention, FP16 embeds, 3.2% pruning, zstd-22

## Eval
- Sliding window stride=64

## Environment
- PyTorch 2.7.0+cu128, FlashAttention 2.8.3, 8xH100 SXM
