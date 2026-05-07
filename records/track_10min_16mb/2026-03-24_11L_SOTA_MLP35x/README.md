# Record: 11L MLP3.5x LeakyReLU(0.5)^2 + Full SOTA Stack (mean val_bpb=1.1330)

**3-seed mean val_bpb: 1.1330** (std=0.0007)

| Seed | val_bpb | val_loss | Steps |
|------|---------|----------|-------|
| 1337 | 1.1334 | 1.9136 | 3842 |
| 42 | 1.1322 | 1.9116 | 3885 |
| 2024 | 1.1334 | 1.9136 | 3857 |

## Architecture (31.4M parameters)
- 11 transformer layers, dim=512, 8 heads / 4 KV heads (GQA)
- MLP 3.5x expansion (hidden=1792) with **LeakyReLU(0.5)^2** activation
- **SmearGate** + **BigramHash(10240, dim=128)** + **TrigramHash(4096, dim=128)**
- **Value Residual (ResFormer)** — cache V from layer 0, blend via learned lambda
- **Gated Attention** — per-head sigmoid gate (nn.Linear, bias init 4.0)
- **XSA on all 11 layers** — exclusive self-attention
- **Partial RoPE** — 16/64 head dimensions
- Tied FP16 embeddings, U-Net skip connections, orthogonal initialization

## Training
- Muon optimizer: lr=0.03, momentum 0.92→0.99/1500 steps, WD=0.04
- Adam for embeddings (lr=0.035) and scalars (lr=0.03)
- Batch 786,432 tokens, seq_len 2048
- EMA (decay=0.997), warmdown 3500 iterations
- Late QAT via STE (final 15% of wallclock)
- Gradient clipping 0.3

## Quantization
- Int6 uniform per-row with GPTQ-lite (5-percentile clip search per row)
- FP16 passthrough for tied embeddings
- zstd-22 compression

## Evaluation
- Sliding window eval, stride=64

## Development Process
30-experiment autoresearch loop on 1xH100 (~8 hours), then validated on 8xH100 SXM.

### Feature ablation (measured on 1xH100):

| Feature | BPB Impact |
|---------|-----------|
| Value Residual | -0.017 |
| SmearGate | -0.010 |
| XSA all 11 layers | -0.005 |
| Gated Attention | -0.004 |
| Partial RoPE (16/64) | -0.004 |
| TrigramHash | -0.002 |
| Late QAT | -0.002 |
