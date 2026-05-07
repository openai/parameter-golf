# Submission: Int6 MLP3x + QAT + SlidingWindow (val_bpb: 1.1702)

## Summary
Stacked int6 per-row quantization + zstd22 compression, 3x MLP expansion, QAT weight-snapping at 70% training, tuned Muon optimizer (momentum=0.99 with warmup), extended warmdown (3000 iters), and stride-64 sliding window evaluation.

## Configuration
MLP_MULT=3 VAL_LOSS_EVERY=0 MATRIX_LR=0.02 SCALAR_LR=0.02 TIED_EMBED_LR=0.03 MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_STEPS=1500 MUON_MOMENTUM_WARMUP_START=0.92 WARMDOWN_ITERS=3000 ENABLE_QAT=1 QAT_START_FRAC=0.7 USE_SLIDING_EVAL=1 EVAL_STRIDE=64

## Key Metrics
- Training steps: 9,540/20,000 (600s wallclock cap)
- Average step time: 62.86ms
- Model params: 21,778,504
- Standard val_bpb: 1.2040
- Sliding val_bpb: 1.1702
- Total artifact: 15,306,777 bytes (int6+zstd22)
- Under 16MB: YES (693,223 bytes headroom)

## Command
torchrun --standalone --nproc_per_node=8 train_gpt.py

## Seed
SEED=1337
