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

## Status

**[WIP]** — Training run pending. Results and train.log will be added after RunPod execution.

## Background

This work builds on experience from the Probatum Code project, where we trained Light (1.5B) and Pro (7B) coding models using LoRA fine-tuning on A100s with knowledge distillation from multiple cloud teachers (15K+ samples). Key learnings around aggressive quantization and efficient training are being adapted to the 16MB constraint.

## Included Files

- `train_gpt.py` — baseline training script (stock, to be modified after first run)
- `submission.json` — submission metadata
- `README.md` — this file
