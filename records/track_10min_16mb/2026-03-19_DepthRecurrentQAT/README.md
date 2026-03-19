# Depth-Recurrent QAT Transformer

**Author:** hacksurvivor
**Status:** WIP — awaiting 8xH100 validation runs

## Approach

Depth-recurrent transformer with quantization-aware training (QAT).

### Key optimizations

- **Depth recurrence:** 3 shared transformer blocks looped 4x = 12 effective layers (vs baseline's 9 unique blocks). Frees ~66% of stored parameter budget while increasing effective depth by 33%.
- **Wider model:** dim 512 → 768 using freed parameter budget. 12 attention heads, 6 KV heads (GQA).
- **Per-loop LoRA adapters:** Rank-4 LoRA deltas on Q and V projections for loop iterations 1-3. Allows each "virtual layer" to specialize without storing full separate blocks. LoRA params trained with Adam (not Muon).
- **QAT:** STE fake-quantize in CastedLinear forward pass during training. Model learns to be robust to int8 quantization noise, reducing post-quantization BPB degradation.
- **fp16 tied embedding:** Embedding kept in fp16 during int8 export for better output head quality (~500KB cost).
- **LAWA:** Linearly averaged weights during warmdown phase for free quality boost.
- **Hyperparameter tuning:** MATRIX_LR 0.04 → 0.06, WARMDOWN_ITERS proportional to reduced step count (~400).

### Architecture

| Component | Baseline | Ours |
|-----------|----------|------|
| Stored blocks | 9 | 3 |
| Effective layers | 9 | 12 |
| Model dim | 512 | 768 |
| Attention heads | 8 | 12 |
| KV heads | 4 | 6 |
| Total stored params | ~17M | ~13.3M |

### A10G smoke test results (200 iterations, 1 shard)

| Metric | Value |
|--------|-------|
| val_bpb (200 steps) | 2.0799 |
| Post-quant val_bpb | 2.0485 |
| Artifact size (int8+zlib) | 15.6 MB |
| GPU memory | 3.6 GB |

Training loss converges (6.96 → 3.54 over 200 steps). Full 8xH100 multi-seed results pending.

## Config

```
NUM_LAYERS=3 NUM_LOOPS=4 MODEL_DIM=768 NUM_HEADS=12 NUM_KV_HEADS=6 MLP_MULT=2 LORA_RANK=4 VOCAB_SIZE=1024
```
