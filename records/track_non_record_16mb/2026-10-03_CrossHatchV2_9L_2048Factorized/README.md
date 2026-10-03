# Cross-Hatch V2 — Factorized 2048D Language Model

## Status
**NON-RECORD SUBMISSION**

This submission documents the existing Cross-Hatch V2 architecture. It is NOT claiming a new SOTA. It serves to preserve a successful architectural experiment within the Parameter Golf 16MB track constraints.

The 2048-dimensional representation is factorized as:
`128 channels × 16 lanes`.

## Architecture

| Parameter                  | Value |
|----------------------------|-------|
| Vocabulary                 | 1024 |
| Layers                     | 9 |
| Channel dimension          | 128 |
| Lanes                      | 16 |
| Working dimension          | 2048 |
| FFN hidden                 | 1408 |
| Sequence length            | 1024 |
| Attention                  | shared channel projections |
| Lane mixer                 | 16x16 |
| FFN                        | SwiGLU |
| RoPE                       | yes |
| RMSNorm                    | yes |
| Tied embedding             | yes |

## Training Configuration

| Parameter                  | Value |
|----------------------------|-------|
| Optimizer                  | Muon (matrices) + Adam (scalars/embeddings) |
| Muon LR                    | 0.025 |
| Embedding LR               | 0.05 |
| Scalar LR                  | 0.04 |
| Momentum                   | 0.99 (starts at 0.92) |
| Warmup                     | 20 steps (momentum warmup 1500 steps) |
| EMA Decay                  | 0.9965 |
| Wallclock                  | 600 seconds |
| Target Global Batch        | 524,288 tokens |

### Notes on Hardware and Execution
The recorded values below reflect the *original development run* on a single NVIDIA T4 GPU. 
The accompanying `train_gpt.py` script has been updated to support `torchrun` DistributedDataParallel (DDP) across 8×H100 GPUs with a global batch of 524,288 tokens, adhering strictly to current official Parameter Golf conventions.

**Reproduction Command for Target 8xH100 Hardware:**
```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Results (Original T4 Run)
*Note: These are exact measured numbers from the T4 environment. H100 benchmarks must be run on target hardware.*

* **Hardware:** 1x NVIDIA T4
* **Training time:** 600.48 s
* **Optimizer steps:** 72
* **Tokens seen:** 4,763,648
* **Training throughput:** ~7,933 tok/s
* **Raw validation loss:** 4.43828
* **EMA validation loss:** 4.63267

*(Official BPB has not been measured on the official evaluator yet; approximate BPB was 2.64657)*

## Artifact
Because the raw FP16 payload comfortably fits within the 16,000,000-byte cap, this submission utilizes a **pure FP16 serialization format** without INT8 quantization or zlib compression, fulfilling the principle of minimal complexity.

* **Raw FP16 parameters:** 15,202,404 bytes
* **Serialized FP16 model:** ~15,202,404 bytes
* **Code bytes:** 26,135 bytes
* **Total submission bytes:** 15,228,539 bytes
* **Headroom:** 771,461 bytes

The `train_gpt.py` script performs a full round-trip verification: it saves the FP16 model, loads the bytes back from disk, maps them into the architecture, and re-evaluates validation loss to confirm exact reconstruction.
