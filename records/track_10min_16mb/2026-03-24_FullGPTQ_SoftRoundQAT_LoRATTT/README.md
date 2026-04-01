# Full GPTQ + Soft-Round QAT + LoRA TTT + Entropy-Coded Compression

**val_bpb: TBD** (pending 8xH100 evaluation)

## Overview

This submission builds on the PR #549 LeakyReLU + Legal TTT + Parallel Muon base
(1.1194 BPB) and introduces four novel improvements:

1. **Full GPTQ Quantization** — Hessian-guided column-by-column error compensation
   replaces the GPTQ-lite clip search, reducing quantization error especially for
   attention Q/K/V and MLP up projections.

2. **Soft-Round QAT** — Differentiable soft-rounding in `CastedLinear` during the
   warmdown QAT phase using `tanh`-based smooth approximation with temperature
   annealing (1.0 → 0.05), allowing gradients to flow through the quantization
   operation more effectively than the hard-round STE baseline.

3. **LoRA-Based Test-Time Training** — Low-rank adapters (rank-8, α=16) on Q/K/V/O
   projections, trained with AdamW instead of full-parameter SGD. This prevents
   catastrophic forgetting of base model knowledge while allowing aggressive per-chunk
   adaptation with cosine-annealed learning rates.

4. **Entropy-Coded Compression** — Huffman coding pre-pass before LZMA to exploit
   non-uniform distribution of quantized int6 weight values. Automatically falls back
   to plain LZMA if Huffman doesn't help. Saved bytes can be reinvested in more
   parameters.

## Architecture

Same as PR #549 base:
- 11 layers, dim=512, 8 heads, 2 KV heads, 3x MLP multiplier
- LeakyReLU(0.5)² activation
- U-Net skip connections (5 encoder + 6 decoder layers)
- Grouped Query Attention with partial RoPE (16/64 dims)
- Cross-Self Attention (XSA) on last 4 layers
- BigramHash embedding (vocab=3072, dim=128)
- Value Embedding (dim=128, layers 9-10)
- RMSNorm with learnable scale
- Parameter banking (all 2D weights in contiguous 3D banks)

## Training

- Muon optimizer for 2D weights + AdamW for embeddings/scalars
- EMA (decay=0.95, start=step 800) + SWA
- Late QAT with soft-rounding (activated at warmdown scale < 0.18)
- 4600 iterations, seq_len=1024, batch_size=64
- Learning rate: 0.0036 (Adam), 0.0126 (Muon)
- Warmup: 200 steps, Warmdown: 26% of total steps

## Quantization

- Full GPTQ int6 for attention and MLP weights (Hessian-guided)
- GPTQ-lite int6 fallback for output projections and MLP down-projections
- Int8 for embeddings and small tensors
- Entropy-coded (Huffman+LZMA or plain LZMA, whichever is smaller)

## Evaluation

- Sliding window with legal score-first TTT protocol
- LoRA adapters (rank=8) on Q/K/V/O trained per-chunk with AdamW
- 3 TTT epochs per chunk, chunk size 32K tokens

## Expected Ablation Table

| Configuration | val_bpb | Δ from baseline |
|---|---|---|
| Baseline (PR #549 GPTQ-lite, no TTT) | ~1.1234 | — |
| + Full GPTQ | ~1.1210 | −0.0024 |
| + Soft-Round QAT | ~1.1195 | −0.0015 |
| + Standard TTT (SGD) | ~1.1194 | −0.0001 |
| + LoRA TTT (instead of SGD) | ~1.1160 | −0.0034 |
| **All combined** | **~1.10xx** | **−0.02xx** |

*Note: Expected values based on ablation studies from related PRs (#593, #606, #620).
Actual values will be populated after 8xH100 runs with 3 seeds.*

## Reproduction

```bash
# On 8xH100 SXM with the parameter-golf environment:
cd records/track_10min_16mb/2026-03-24_FullGPTQ_SoftRoundQAT_LoRATTT/
bash run_8xh100.sh 42   # seed 42
bash run_8xh100.sh 43   # seed 43
bash run_8xh100.sh 44   # seed 44
```

## Key Files

- `train_gpt.py` — Self-contained training + evaluation script (2418 lines)
- `run_8xh100.sh` — Full 8xH100 launch script with all hyperparameters
- `run_ablation.sh` — Per-improvement ablation testing (1xH100)
