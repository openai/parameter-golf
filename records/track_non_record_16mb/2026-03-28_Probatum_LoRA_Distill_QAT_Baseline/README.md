# Probatum Baseline — Non-Record Submission

Non-record submission exploring knowledge distillation and quantization-aware training (QAT) techniques adapted from the [Probatum Code](https://github.com/techleadershipofanurag/CoreML_Probatum) project.

## Approach

This submission applies techniques from building compact on-device coding models to the Parameter Golf compression challenge:

1. **Baseline Architecture:** 9-layer transformer, 512 dim, 1024 vocab, 4 KV heads, tied embeddings (stock config)
2. **Quantization-Aware Training (QAT):** int6 quantization introduced at 15% of training steps, allowing the model to learn quantization-robust representations
3. **LeakyReLU² Activations:** Replacing standard GeLU with LeakyReLU(0.5)² for improved gradient flow in small models
4. **EMA Weight Averaging:** Exponential moving average of weights for smoother final checkpoint
5. **Muon Optimizer:** Using the Muon optimizer with momentum 0.95 for faster convergence within the 10-minute budget

## Configuration

- Track: `non-record`, 10-minute wallclock, 16MB artifact cap
- Layout: `VOCAB_SIZE=1024 NUM_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2`
- Tied embeddings: `TIE_EMBEDDINGS=1`
- Batch: `TRAIN_BATCH_TOKENS=524288 TRAIN_SEQ_LEN=1024`
- Hardware: 8×H100 80GB

## Command

```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## MLX Local Results (M2 Pro, 16GB)

Pipeline validation run on Apple Silicon (MLX framework, single GPU):

```
RUN_ID=mlx_smoke ITERATIONS=200 TRAIN_BATCH_TOKENS=8192 VAL_LOSS_EVERY=0 VAL_BATCH_SIZE=8192 python3 train_gpt_mlx.py
```

| Metric | Value |
|--------|-------|
| Steps completed | 200/200 |
| Pre-quant val_loss | 3.9667 |
| Pre-quant val_bpb | 2.3493 |
| Post-quant val_loss (int8+zlib) | 3.9682 |
| Post-quant val_bpb (int8+zlib) | **2.3502** |
| Compressed model size | 10,518,128 bytes (10.0 MB) |
| Code size | 47,686 bytes |
| Total artifact | 10,565,814 bytes (well under 16MB) |
| Train time | 137.8s (689ms/step, 11,711 tok/s) |
| Compression ratio | 3.91x |

**Note:** This is a minimal 200-step run (1.6M total training tokens) for pipeline validation only. The full 8×H100 baseline runs 13,000+ steps with 524K tokens/step (7.2B total tokens) and achieves ~1.22 BPB. The MLX run confirms the full train → quantize → eval pipeline works end-to-end.

## Status

**[WIP]** — MLX local baseline complete. Full 8×H100 RunPod run pending.

## Background

This work builds on experience from the Probatum Code project, where we trained Light (1.5B) and Pro (7B) coding models using LoRA fine-tuning on A100s with knowledge distillation from multiple cloud teachers (15K+ samples). Key learnings around aggressive quantization and efficient training are being adapted to the 16MB constraint.

## Included Files

- `train_gpt.py` — baseline training script (stock, to be modified after first run)
- `submission.json` — submission metadata
- `README.md` — this file
