# Combined SOTA: INT6 + SmearGate + BigramHash + OrthoInit + SWA + Sliding Window

## Summary

This submission combines six orthogonal improvements over the naive baseline, achieving **val_bpb = 1.1754** with sliding window evaluation (stride=64, seq=1024). The compressed artifact is **13.98 MB** (well under the 16 MB limit), leaving room for future model scaling.

## Techniques

1. **Per-row INT6 Quantization + zstd-22**: 6-bit per-row quantization with zstandard compression (level 22) instead of INT8+zlib. Saves ~25% model bytes, enabling a wider MLP.

2. **FP16 Tied Embedding Export**: Tied embeddings kept in FP16 instead of INT8, reducing quantization gap from ~0.007 to ~0.001 BPB since errors compound in both input and output paths.

3. **MLP 2.5x Expansion**: Hidden dimension increased from 1024 (2x) to 1280 (2.5x), enabled by INT6 byte savings. Provides more model capacity.

4. **SmearGate**: Lightweight bigram-aware gating module (~512 params) that blends each token's embedding with the previous token via a learned sigmoid gate.

5. **BigramHash Embedding**: 4096-bucket hash table (dim=128, projected to 512) encoding consecutive token pairs. Adds explicit bigram context at minimal cost (~524K params).

6. **Orthogonal Initialization + muP Scaling**: All large weight matrices initialized with orthogonal init. Output projections scaled by 1/sqrt(2*num_layers) for stable training at depth.

7. **Phase-Transition Residual Mixing**: Sigmoid-scheduled residual mix initialization — early layers favor the original embedding, later layers favor the residual stream.

8. **Muon Weight Decay (0.02)**: Decoupled weight decay added to the Muon optimizer, improving both generalization and post-quantization robustness.

9. **Stochastic Weight Averaging (SWA)**: Checkpoint averaging during the warmdown phase (every 150 steps when LR scale < 0.5). 4 checkpoints averaged in the final model.

10. **Sliding Window Evaluation** (stride=64): Each token evaluated with near-full context, improving BPB measurement accuracy by ~0.03 BPB.

## Configuration

```
VOCAB_SIZE=1024
NUM_LAYERS=9
MODEL_DIM=512
NUM_HEADS=8
NUM_KV_HEADS=4
MLP_MULT=2.5
TIE_EMBEDDINGS=1
TRAIN_SEQ_LEN=1024
TRAIN_BATCH_TOKENS=524288
MATRIX_LR=0.02
SCALAR_LR=0.02
TIED_EMBED_LR=0.03
MUON_MOMENTUM=0.99
MUON_MOMENTUM_WARMUP_START=0.92
MUON_MOMENTUM_WARMUP_STEPS=1500
MUON_WEIGHT_DECAY=0.02
GRAD_CLIP_NORM=0.3
WARMDOWN_ITERS=3000
SWA_ENABLED=1
SWA_EVERY=150
BIGRAM_VOCAB_SIZE=4096
BIGRAM_DIM=128
USE_SMEAR_GATE=1
USE_INT6=1
EVAL_STRIDE=64
```

## Training Command

```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Results

| Metric | Value |
|--------|-------|
| Pre-quant val_bpb | 1.2038 |
| Post-quant val_bpb (INT6+zstd) | 1.2097 |
| **Sliding window val_bpb** | **1.1754** |
| Quantization gap | 0.0059 |
| Compressed artifact size | 13.98 MB |
| Code size | 50,629 bytes |
| Total submission | 13,979,624 bytes |
| Training steps | 9,919 |
| Training time | 609 seconds |
| Step avg | ~55 ms |
| Peak memory | 11,055 MiB per GPU |
| SWA checkpoints | 4 |

## Training Dynamics

- Training stopped at step 9,919 (hit 10-minute wallclock cap)
- SWA started at step 9,200 (LR warmdown phase)
- Validation BPB progression: 1.2770 (4K) -> 1.2544 (8K) -> 1.2038 (final)
- Loss was still decreasing when wallclock cap was reached

## Ablation Notes

We also ran a variant with MLP=3.0x and seq_len=2048, which achieved 1.1790 BPB sliding window but only completed 4,324 steps due to 3.2x slower step time. The MLP=2.5x + seq=1024 variant converges better within the 10-minute budget by achieving 2.3x more training steps.

## Dependencies

- PyTorch 2.9.1+ (CUDA 12.8)
- zstandard (for zstd-22 compression)
- sentencepiece
- numpy
