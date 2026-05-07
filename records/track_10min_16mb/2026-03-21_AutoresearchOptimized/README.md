# Autoresearch-Optimized Submission

**Score: 1.2459 val_bpb**
**Size: 15,900,785 bytes (int8+zlib)**
**Steps: 16,562 in 600s on 8xH100 SXM**

## Approach

Used Karpathy's autoresearch pattern — an autonomous AI coding agent iteratively modified hyperparameters on a single RTX 4080 (97 experiments over ~12 hours), then validated on 8xH100 SXM.

## Key changes from baseline
- Tied embeddings (reduced model size from 16.2MB to 15.9MB)
- Optimizer tuning (muon momentum, warmup schedule, gradient clipping)
- Attention configuration (4 heads, 2 KV heads via GQA)
- Learning rate adjustments across all parameter groups

## Hardware
- Development: RTX 4080 16GB (97 experiments, ~12 hours)
- Submission: 8x H100 SXM (RunPod, single 10-minute run)
