# Warmdown-Tuned Training (1xRTX 5090)

## Summary

Systematic experiment-driven optimization of the baseline 9-layer, 512-dim GPT. 9+ experiments tested batch size, sequence length, architecture changes, training duration, warmdown tuning, register tokens, and layer looping. Key discovery: the LR warmdown phase produces disproportionate BPB improvement (2.6x per-step efficiency vs normal training), with warmdown_iters=3000 being optimal.

**Result: val_bpb 1.2987** on 1xRTX 5090 (1200s wallclock, 11,603 steps).

Next steps: layer looping + wider model to eliminate dead middle layers and increase attention capacity.

## Configuration (Best Result — Exp 8a)

```
VOCAB_SIZE=1024 NUM_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4
MLP_MULT=2 TIE_EMBEDDINGS=1
MATRIX_LR=0.04 SCALAR_LR=0.04 TIED_EMBED_LR=0.05
MUON_MOMENTUM=0.95 MUON_MOMENTUM_WARMUP_STEPS=50
WARMDOWN_ITERS=3000 TRAIN_BATCH_TOKENS=65536 TRAIN_SEQ_LEN=1024
MAX_WALLCLOCK_SECONDS=1200
```

## Command

```bash
python train_gpt.py  # 1xRTX 5090
```

## Key Metrics

- Training stopped at step **11,603/20,000** due to wallclock cap (1200s, ~103.4ms/step)
- Model params: **17,059,912**
- Pre-quant: `val_loss:2.1858 val_bpb:1.2945`
- Int8+zlib roundtrip: `val_loss:2.1928 val_bpb:1.2987`
- Compressed model (int8+zlib): **15,777,669 bytes**
- Code: **50,529 bytes**
- **Total: 15,828,198 bytes** (under 16,000,000 limit by 171,802 bytes)
- Peak memory: 1,552 MiB

## Experiment History

| Exp | Change | Steps | val_bpb | Finding |
|-----|--------|-------|---------|---------|
| 1 | Baseline (batch=524K) | 94 | 3.10 | Batch too large for 1xRTX 5090 |
| 2 | batch=65K, 180s | 1,785 | 1.46 | 12x speedup unlocked training |
| 3 | seq_len=512 | 1,925 | 1.49 | Shorter sequences WORSE |
| 4 | 600s training | 5,964 | 1.35 | More training helps, warmdown effective |
| 5a | 6L/624d architecture | 4,500 | 3.57 | FAILED — LR doesn't transfer |
| 5 | batch=32K | 8,533 | 1.38 | Smaller batch WORSE |
| 6 | 1200s training | 11,909 | 1.312 | Warmdown 2.6x per-step efficiency |
| 7 | warmdown=2000 | 11,597 | 1.306 | Layer 3 activated, position degradation worsened |
| 7+ | Register token A/B/C | ~300 | — | Register token doesn't help overall BPB |
| **8a** | **warmdown=3000** | **11,603** | **1.299** | **Best result. Warmdown=3000 optimal.** |
| 8b | warmdown=3000 + register | 11,200 | 1.301 | Register token conclusively dead |
| 9 | Layer looping + wider (in progress) | — | — | 6 unique layers / 608 dim / looped to 9 |

## Key Discoveries

### 1. Warmdown is the highest-leverage parameter
warmdown_iters=3000 > 2000 > 600. The cosine LR decay phase produces 2.6x the per-step BPB improvement. Extending warmdown from 600 to 3000 steps improved BPB from 1.312 to 1.299.

### 2. Layer activation via warmdown
Longer warmdown revived dead layer 3 (impact 0.1→0.58). However, warmdown=3000 did NOT activate layers 4-6, suggesting diminishing returns for layer activation.

### 3. Register token is ineffective at this scale
Tested 3 approaches (causal position 0, append + per-layer gate, append + final gate) plus context injection. None improved overall BPB — marginal juncture benefit eaten by step overhead.

### 4. Hard tokens are word-initial at syntactic boundaries
~65% of very-hard tokens follow `,` `.` `the` `and` etc. This is invariant across all training approaches tested.

### 5. Position degradation is architectural
context_benefit stable at ~-1.20 across all experiments. The model degrades sharply past position 256.

### 6. Architecture changes need LR sweeps
Exp 5a failed without one. Exp 9 (layer looping) includes a proper LR sweep.

## Next Steps

1. **Restore batch=524288** on multi-GPU — 8x more tokens per step, ~49ms/step, ~12K steps in 600s
2. **Apply warmdown_iters=3000** — validated as optimal
3. **Layer looping + wider model** — if Exp 9 confirms benefit
4. **Implement int6 quantization + zstd** to fit MLP 3x (~21M params in 16MB)
5. **Add sliding window evaluation** (stride=64) for ~0.03 BPB improvement

Expected result: **~1.18-1.22 BPB** based on competition baselines + my findings.

## Analysis & Evaluation

Detailed analysis documents drove every experiment decision:

- **[Experiment 6 Deep Analysis](https://github.com/swapp1990/parameter-golf/blob/main/dashboard/experiment6_analysis.md)** — Three-phase learning curve, warmdown efficiency, compression trajectory
- **[Experiment 7 Deep Analysis](https://github.com/swapp1990/parameter-golf/blob/main/dashboard/experiment7_analysis.md)** — Layer 3 activation, position degradation, entropy analysis
- **[Hard Token Analysis](https://github.com/swapp1990/parameter-golf/blob/main/dashboard/hard_token_analysis.md)** — Word-initial dominance, entropy quadrants, bigram transitions
- **[Structural Evaluation](https://github.com/swapp1990/parameter-golf/blob/main/dashboard/evaluation_analysis.md)** — Position degradation, layer utilization, loss distribution
- **[Experiment Plan](https://github.com/swapp1990/parameter-golf/blob/main/dashboard/experiments_plan.md)** — Full experiment framework with lessons learned and Exp 9 plan

Tools:
- **[deep_eval.py](https://github.com/swapp1990/parameter-golf/blob/main/dashboard/backend/deep_eval.py)**: Loss distribution, position-by-position loss, layer ablation, bits budget
- **[hard_token_analyzer.py](https://github.com/swapp1990/parameter-golf/blob/main/dashboard/backend/hard_token_analyzer.py)**: Token-level analysis — word position, frequency, entropy, bigram transitions
