## Midnight 12L

Midnight 12L is a 12-layer Rascal II submission that uses mixed-int quantization plus Brotli
packing to add one extra transformer layer while staying under the 16,000,000-byte artifact cap.

## Architecture summary

- Backbone: 12-layer Rascal II decoder
- Attention: GQA (`num_heads=8`, `num_kv_heads=4`)
- Context features: Bigram hash 2048, RoPE dims 16, XSA on last 11 layers
- Quantization: `attn=int5`, `mlp=int6`, `aux=int6`, `embed=int8`, `other=int8`
- Compression: mixed-int checkpoint + Brotli
- Hardware: 8xH100 SXM
- Train wallclock: 600s
- `bytes_code`: 124,698

## 3-seed results

| Seed | val_bpb_exact (sliding window) | Steps | Train time (s) | bytes_total |
|------|--------------------------------|------:|---------------:|------------:|
| 444  | 1.10567949                     | 6160  | 600            | 15631603    |
| 300  | 1.10582448                     | 6154  | 600            | 15624171    |
| 42   | 1.10641160                     | 6153  | 600            | 15619003    |
| **mean** | **1.10597186**             |       |                |             |
| **std (population)** | **0.00031653** |       |                |             |
| **max bytes_total** |                |       |                | **15631603** |

## Technique description

Compared to the prior 11-layer stack, this run spends compression headroom on depth:
the model is extended to 12 layers while preserving submission legality through mixed-int
quantization and Brotli artifact compression. Training and scoring remain standard score-first
evaluation, with no validation-set leakage.

## Reproduce

```bash
SKIP_GPTQ=1 SEED=444 torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-04-07_Midnight_12L_8xH100/train_gpt.py
```
