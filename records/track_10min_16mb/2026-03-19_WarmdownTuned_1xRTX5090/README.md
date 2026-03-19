# Warmdown-Tuned Training (1xRTX 5090)

## Summary

Systematic experiment-driven optimization of the baseline 9-layer, 512-dim GPT. Eight experiments tested batch size, sequence length, architecture changes, and training duration. Key discovery: the LR warmdown phase produces disproportionate BPB improvement (2.6x per-step efficiency vs normal training).

**Result: val_bpb 1.312** on 1xRTX 5090 (1200s wallclock, 11,909 steps).

This is a compute grant application — we expect ~1.22 BPB with tuned warmdown + full batch on 8xH100.

## Configuration

```
VOCAB_SIZE=1024 NUM_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4
MLP_MULT=2 TIE_EMBEDDINGS=1
MATRIX_LR=0.04 SCALAR_LR=0.04 TIED_EMBED_LR=0.05
MUON_MOMENTUM=0.95 MUON_MOMENTUM_WARMUP_STEPS=50
WARMDOWN_ITERS=600 TRAIN_BATCH_TOKENS=65536 TRAIN_SEQ_LEN=1024
MAX_WALLCLOCK_SECONDS=1200
```

## Command

```bash
python train_gpt.py  # 1xRTX 5090
```

## Key Metrics

- Training stopped at step **11,909/20,000** due to wallclock cap (1200s, ~100.8ms/step)
- Model params: **17,059,912**
- Pre-quant: `val_loss:2.2090 val_bpb:1.3083`
- Int8+zlib roundtrip: `val_loss:2.2159 val_bpb:1.3124`
- Compressed model (int8+zlib): **15,747,327 bytes**
- Code: **48,344 bytes**
- **Total: 15,795,671 bytes** (under 16,000,000 limit by 204,329 bytes)
- Peak memory: 1,552 MiB

## Experiment History

| Exp | Change | Steps | val_bpb | Finding |
|-----|--------|-------|---------|---------|
| 1 | Baseline (batch=524K) | 94 | 3.10 | Batch too large for 1xRTX 5090 |
| 2 | batch=65K, 180s | 1,785 | 1.46 | 12x speedup unlocked training |
| 3 | seq_len=512 | 1,925 | 1.49 | Shorter sequences WORSE - model uses long context |
| 4 | 600s training | 5,964 | 1.35 | More training helps, warmdown phase effective |
| 5a | 6L/624d architecture | 4,500 | 3.57 | FAILED - optimizer LR doesn't transfer across architectures |
| 5 | batch=32K | 8,533 | 1.38 | Smaller batch WORSE despite more steps |
| 6 | **1200s training** | **11,909** | **1.31** | **Best result. Warmdown disproportionately effective.** |
| 7 | warmdown=2000, 1200s | - | - | Interrupted (RunPod outage) |

## Key Discoveries

### 1. Warmdown is the highest-leverage parameter
The LR warmdown phase (cosine decay to zero) produces 2.6x the per-step BPB improvement compared to normal training. In Exp 6, the last 900 steps (warmdown) dropped BPB by 0.036 - more efficient than 3000 steps of normal training. This suggests longer warmdown periods (e.g., 2000-3000 iters) could significantly improve results.

### 2. Hard tokens are word-initial predictions at syntactic boundaries
Custom hard-token analysis revealed that ~65% of "very hard" tokens are word-initial (e.g., predicting which word follows "the" or ","). This is a context utilization problem, not a vocabulary or architecture issue.

### 3. Position degradation is architectural, not training-dependent
The model consistently gets WORSE at later sequence positions (context_benefit = -1.1). This persisted unchanged across 3x more training. The attention mechanism favors short-range patterns at this scale.

### 4. Architecture changes require optimizer re-tuning
Attempting a shallower/wider model (6L/624d) with the same Muon optimizer settings completely failed. The optimizer hyperparameters are architecture-specific.

## Plan with H100 Compute

With 8xH100, we would:
1. **Restore batch=524288** - 8x more tokens per step, ~49ms/step, ~12K steps in 600s
2. **Apply tuned warmdown** (warmdown_iters=3000) based on our discovery
3. **Tune matrix_lr** (try 0.02-0.06 range based on competition findings)
4. **Implement int6 quantization + zstd** to fit MLP 3x (~21M params in 16MB)
5. **Add sliding window evaluation** (stride=64) for ~0.03 BPB improvement
6. **Test SmearGate** - simple bigram blending for ~0.01 BPB

Expected result: **~1.18-1.22 BPB** based on competition baselines + our warmdown insights.

## Custom Evaluation Tools

We built two evaluation tools during this project:
- **deep_eval.py**: Structural analysis - loss distribution, position-by-position loss, layer ablation, bits budget
- **hard_token_analyzer.py**: Token-level analysis - what makes tokens hard (word position, frequency, entropy, bigram transitions)

These tools drove every experiment decision and are available in our [dashboard repo](https://github.com/swapp1990/parameter-golf/tree/main/dashboard/backend).
