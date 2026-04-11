# 10-Layer 4xMLP Baseline

**Author:** [@hmhm0](https://github.com/hmhm0)

### Key Idea
This submission maximizes the 16MB constraint by expanding the standard 9-layer architecture to 10 layers and increasing the MLP multiplier from 2x to 4x. Utilizing standard INT8 per-row post-training quantization and zlib compression, this architecture utilizes 91.7% of the strict 16MB file size limit (14.68 MB).

Evaluated using an overlapping sliding window and batched LoRA test-time training. 

### Results
| Metric | This Submission |
|--------|-----------------|
| Post-quant `val_bpb` | **1.4444** |
| Training steps | 851 |
| Artifact size | 14,682,364 bytes |
| Hardware | 1xH100 PCIe (Scaled Sprint) |
