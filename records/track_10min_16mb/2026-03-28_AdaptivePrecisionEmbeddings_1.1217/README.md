# Frequency-Weighted GPTQ Calibration + Adaptive Precision Embedding Quantization

**val_bpb: 1.0980 (3-seed mean) | 14.46 MB | 8×H100 SXM**

## Checklist
- [x] Artifact < 16,000,000 bytes (all 3 seeds)
- [x] Training < 600s, eval < 600s
- [x] Causal sliding-window evaluation (stride=64)

## Results

| Seed | val_bpb | Size |
|------|---------|------|
| 1337 | 1.09820924 | < 14.5 MB |
| 42   | 1.09775873 | < 14.5 MB |
| 2024 | 1.09798646 | < 14.5 MB |
| **Mean** | **1.09798481** | **< 14.5 MB** |

## Files
- `trainFreqGPTQ_gpt.py` - Training script with Frequency-Weighted GPTQ Calibration
- `submission.json` - Submission metadata
- `freqgptq_seed_1337.log` - Training log seed 1337
- `freqgptq_seed_42.log` - Training log seed 42
- `freqgptq_seed_2024.log` - Training log seed 2024

## Core Innovations

### 1. Frequency-Weighted GPTQ Calibration (New)

Natural language follows Zipf's law: the top 100 tokens cover ~53% of all text.
Standard GPTQ treats all tokens equally during Hessian collection — but
quantization errors on frequent tokens propagate far more into the final BPB.

**Implementation:** Activations from top-100 most frequent tokens receive 2×
weight in Hessian accumulation during GPTQ calibration:

```python
is_top = torch.isin(token_ids, top_ids_tensor)
weights = (1.0 + is_top.float()).unsqueeze(1)
x_weighted = x * weights.sqrt()  # sqrt because H = X^T X
hessians[name].addmm_(x_weighted.T, x_weighted)
```

Zero artifact size cost. Log confirmation:
```
[FreqGPTQ] Frequency-weighted Hessians collected: 66 layers, top-token boost=2.0x
```

### 2. Adaptive Precision Embedding Quantization (from PR #1042)

Top-100 frequent tokens → **int8** (higher precision)
Remaining 924 tokens → **int6** (standard compression)

Log confirmation:
```
[FreqQuant] Embedding: 100 top tokens -> int8, 924 rare tokens -> int6
```

## Architecture Base

Built on **PR #1435** (AbhayAnandUCSD). Full credit for base architecture.

Key components:
- 11 physical layers, 512d, 8 heads, 4 KV heads (GQA)
- Depth recurrence: layers 4,5 repeat (13 virtual layers), activates at step 3000
- Skip gates on U-Net skip connections
- Parallel residuals from layer 7 (attention + MLP run simultaneously)
- EMA decay = 0.9965
- Full GPTQ (64 calibration batches, 10s reserved)
- Selective ±1 pruning
- Brotli + byte shuffle compression
- BigramHash (1536 buckets, dim 112)
- Value Embedding (dim 128, layers 9,10)
- QK-Gain init = 5.0, Weight decay = 0.09

## Training Command

```bash
RUN_ID=freqgptq_s1337 \
SEED=1337 \
MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=8 trainFreqGPTQ_gpt.py
```

## Key Findings

- **Recurrence start step is robust:** Values from 2000-4000 produce identical BPB
- **TTT hurts GPTQ models:** SGD TTT increased BPB by +0.09 (1.098→1.19)
- **Loop 3-5 vs 4-5:** No measurable improvement due to fewer warmdown steps
- **FreqGPTQ consistently beats standard GPTQ** by ~0.001 BPB across all seeds

## Hardware

8× NVIDIA H100 80GB SXM | Training: ~590s | Eval: ~120s

## Credits
Base architecture: PR #1435 by AbhayAnandUCSD
Frequency-Weighted Embedding Quantization: PR #1042 (my PR NothingLiVa)
Frequency-Weighted GPTQ Calibration: new contribution (this PR)

- Base architecture: PR #1435 by AbhayAnandUCSD
- Frequency-Weighted Embedding Quantization: PR #1042 (NothingLiVa)
- Frequency-Weighted GPTQ Calibration: new contribution (this PR)
