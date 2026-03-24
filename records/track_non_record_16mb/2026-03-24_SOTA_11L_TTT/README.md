# Non-record: 11L Full SOTA Stack + Score-First TTT (1xH100)

**val_bpb: 1.1383** (1xH100, 985 steps + 3-epoch TTT)
**Artifact: 8.5 MB** (well under 16 MB budget)

## Method

### Architecture (28.7M parameters)
- 11 transformer layers, dim=512, 8 heads / 4 KV heads (GQA)
- MLP 3x expansion (hidden=1536) with **LeakyReLU(0.5)^2** activation
- **SmearGate** — learned per-dim gate blending each token with previous
- **BigramHash(10240, dim=128)** — XOR hash-based bigram embeddings
- **TrigramHash(4096, dim=128)** — XOR hash-based trigram embeddings
- **Value Residual (ResFormer)** — cache V from layer 0, blend into all layers via learned lambda
- **Gated Attention** — per-head sigmoid gate (nn.Linear, bias init 4.0)
- **XSA on all 11 layers** — exclusive self-attention subtracting self-value projection
- **Partial RoPE** — rotary embeddings on 16/64 head dimensions only
- Tied FP16 embeddings, U-Net skip connections, orthogonal initialization
- Logit softcap 30.0, QK gain init 1.5

### Training
- Muon optimizer: lr=0.03, momentum 0.92→0.99 over 1500 steps, WD=0.04
- Adam for embeddings (lr=0.035) and scalars (lr=0.03)
- Batch 524,288 tokens, seq_len 2048
- Warmdown 300 iterations (wallclock-based)
- Late QAT via STE (final 15% of wallclock time)
- Gradient clipping 0.3

### Quantization
- Int6 uniform per-row quantization with GPTQ-lite (5-percentile clip search)
- FP16 passthrough for tied embeddings (most quant-sensitive tensor)
- zstd-22 compression

### Evaluation
- Sliding window eval, stride=256 (on 1xH100 for speed; stride=64 for competition)
- **Score-first TTT (3 epochs)**: frozen embeddings, only block params updated, per-layer LR groups (3x for mlp.proj, 0.5x for mlp.fc), cosine LR decay

## Development Process

Used Karpathy-style **autoresearch** methodology: autonomous experiment loop with 30 experiments over ~8 hours on 1xH100. Each experiment modifies one variable, runs training, and keeps/discards based on val_bpb improvement.

### Key findings from autoresearch (30 experiments):

| Experiment | BPB | Delta | Status |
|---|---|---|---|
| Initial baseline | 1.4527 | — | start |
| Disable EMA+SWA (raw weights better) | 1.4027 | -0.050 | keep |
| Batch 786K→524K (more steps) | 1.3432 | -0.060 | keep |
| XSA all 11 layers | 1.3379 | -0.005 | keep |
| LR 0.025→0.03 | 1.3271 | -0.011 | keep |
| Warmdown 3500→300 | 1.2567 | -0.070 | keep (series) |
| Remove VE128 (not helping) | 1.2567 | 0.000 | keep (simpler) |
| Disable LN Scale (neutral) | 1.2563 | -0.000 | keep (simpler) |
| + 3-epoch score-first TTT | **1.1383** | -0.116 | **final** |

### Feature ablation (impact on our stack):

| Feature | BPB Impact |
|---|---|
| Value Residual | -0.017 |
| SmearGate | -0.010 |
| XSA all 11 layers | -0.005 |
| Gated Attention | -0.004 |
| Partial RoPE (16/64) | -0.004 |
| TrigramHash | -0.002 |
| Late QAT | -0.002 |

### Confirmed anti-patterns (tested and rejected):
- Warmdown 4500: too much decay for limited steps
- Batch 262K: per-step quality drops
- LR 0.04: too high, diverges
- seq_len=1024: worse despite more steps
- Fast momentum warmup: unstable
- No grad clipping: diverges
- EMA on 1xH100: too slow for ~960 steps

## Scaling to 8xH100

This submission was developed on 1xH100 (~960 steps). On 8xH100 (~7000 steps), recommended changes:
- Re-enable EMA (decay=0.997)
- Warmdown 3500, batch 786K
- MLP 3.5x (hidden=1792)
- EVAL_STRIDE=64

Expected 8xH100 BPB: ~1.05-1.10 with TTT.
