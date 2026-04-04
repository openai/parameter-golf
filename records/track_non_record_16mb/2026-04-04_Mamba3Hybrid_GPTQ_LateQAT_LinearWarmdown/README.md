# Mamba-3 Hybrid + Full Hessian GPTQ + Late QAT + Linear Warmdown

## Summary

Best SSM submission: **1.1526 bpb** post-quantization (seed 1337, 8×H100). Closes the 174 mBPB quantization gap from our previous GPTQ run to effectively zero through late QAT with linear warmdown scheduling.

- **Training bpb:** 1.1546 (BF16)
- **Post-quant bpb:** 1.1526 (INT6 + GPTQ + Late QAT)
- **Artifact size:** 15.78MB (within 16MB limit)
- **Architecture:** 7× Mamba-3 SISO + 1× attention hybrid, dim=512, d_state=64, seq_len=4096

---

## Key Improvements Over Previous Submission (PR #1107, 1.5633 bpb)

| Change | Impact |
|--------|--------|
| **Full Hessian GPTQ** (replaces QAT-only) | Better weight quantization — no training regression from fake quant |
| **Late QAT** (`LATE_QAT_THRESHOLD=0.15`) | Fake quantization only in final ~500 steps when LR is low |
| **Linear warmdown** (`WARMDOWN_SHAPE=linear, WARMDOWN_ITERS=3500`) | Fixes cosine warmdown bug that prevented late QAT from triggering |
| **LZMA compression** | Better compression than zlib-9 |
| Combined | **408 mBPB improvement** (1.5633 → 1.1526) |

### Quantization Gap: Root Cause and Fix

Our previous run used cosine warmdown with `WARMDOWN_ITERS=22000`, which meant `lr_mul ≈ 0.13` from step 1. Late QAT (which triggers when `lr_mul < threshold`) could never activate because LR was already too low to allow meaningful fine-tuning with fake quantization noise.

**Fix:** `WARMDOWN_SHAPE=linear` + `WARMDOWN_ITERS=3500` keeps LR at 1.0 for the first ~33% of training, then linearly decays to 0. Late QAT triggers at `lr_mul < 0.15` (~step 4686), giving ~500 steps of QAT at low LR. Result: quantization gap goes from +174 mBPB to **-2 mBPB** (post-quant is slightly *better* than training bpb due to sliding window eval).

---

## Architecture

8-layer U-net hybrid: 7× Mamba-3 SISO blocks + 1 attention layer at position 4.

```
[Mamba-3 SSD] + [MLP]   ×7 layers
[Attention]   + [MLP]   ×1 layer (at layer 4)
dim=512, d_state=64, mlp_mult=3, seq_len=4096
```

- **Mamba-3 SISO:** pure Triton kernels, chunked SSD, expand=2, headdim=64
- **Attention:** causal GQA (8/4 heads), RoPE, q_gain, GLU values
- **MLP:** LeakyReLU², mlp_mult=3
- **Other:** U-net skip connections, SmearGate, BigramHash, tied embeddings, Muon optimizer

---

## Quantization Pipeline

1. **Train** 5179 steps in BF16 with linear warmdown + late QAT (last ~500 steps)
2. **GPTQ** with AR self-generated calibration data (32 seqs × 4096 tokens)
3. **INT6** per-row quantization with full Hessian error minimization
4. **LZMA** compression of quantized state dict
5. **Selective ±1 pruning** (safety net, not activated — model already fits)

---

## Results

| Metric | Value |
|--------|-------|
| Training steps | 5,179 |
| Step time | 115.86ms |
| Training val_bpb | 1.1546 |
| Post-quant val_bpb | **1.1526** |
| Quantization gap | **-2 mBPB** |
| Artifact size | 15,781,278 bytes |
| Hardware | 8×H100 80GB SXM |

### Comparison

| Submission | bpb | Type |
|-----------|-----|------|
| SOTA (PR #1019) | 1.1147 | Transformer |
| Our best transformer (PR #768) | 1.1201 | Transformer |
| **This submission** | **1.1526** | **SSM hybrid** |
| Previous SSM (PR #1107) | 1.5633 | SSM hybrid |

---

## Run Command

```bash
WARMDOWN_ITERS=3500 WARMDOWN_SHAPE=linear LATE_QAT_THRESHOLD=0.15 \
FP16_INPROJ_ROWS=0 TARGET_MB=15.9 \
QUANT_BITS=6 QAT_START_FRAC=0.0 USE_GPTQ=1 \
TTT_ENABLED=0 EVAL_STRIDE=32 USE_LZMA=1 EVAL_TEMP=0.9 \
WEIGHT_DECAY=0.04 MUON_MOMENTUM=0.99 MATRIX_LR=0.025 \
torchrun --nproc_per_node=8 train_mamba3_hybrid.py
```

## Setup

See `requirements.txt`. The `mamba-ssm` wheel requires manual Mamba-3 source file installation — see [mamba repo](https://github.com/state-spaces/mamba) v2.3.1.
