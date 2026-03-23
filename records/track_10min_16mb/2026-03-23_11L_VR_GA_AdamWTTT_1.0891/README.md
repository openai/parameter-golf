# Record: 11L EMA + Value Residual + Gated Attention + AdamW TTT (val_bpb=1.0891)

**val_bpb = 1.0891** (sliding window stride=64, seed 1337) | **14.2 MB** artifact | 8xH100 SXM, 600s

## Approach

Two architecture changes on top of the PR #442 recipe (11L EMA + AdamW TTT):

**Value Residual** (ResFormer, arXiv:2410.17897): Each attention block receives the raw V from the first block. A learned 2-element lambda blends first-block V with current V before attention. Block 0 passes V through unchanged (no lambda parameter). Adds 2 params per layer (layers 1-10 only).

**Gated Attention** (arXiv:2505.06708): Per-head sigmoid gate on attention output. Learned weight matrix (dim x num_heads) + bias initialized to 4.0 (near-open gate at init). Adds 4104 params per layer.

Both techniques were ablated individually in PR #413 (-0.015 and -0.003 bpb respectively, -0.017 combined). This is the first validation on the full competitive stack with AdamW TTT.

## Results (seed 1337, 8xH100 SXM)

| Metric | Value |
|--------|-------|
| Training steps | 6,021 (wallclock capped at 600s) |
| Step time | 99.66 ms/step |
| Pre-quant val_bpb | 1.1545 |
| Post-quant roundtrip val_bpb | 1.0964 |
| **Sliding window val_bpb (s=64)** | **1.0891** |
| Artifact size | 14,195,825 bytes |
| Peak GPU memory | 21,374 MiB |
| TTT time | 171.8s |

## Comparison to prior SOTA

| Submission | Best BPB | Steps | Step time |
|-----------|----------|-------|-----------|
| **Ours** | **1.0891** | 6,021 | 99.7 ms |
| PR #442 (sjp611) | 1.0992 | 4,612 | ~137 ms |
| PR #481 (mrdavtan) | 1.0959 | 7,101 | ~84 ms |

## Key findings

1. VR+GA adds ~300K params (27.1M vs 26.8M) with negligible throughput cost
2. Faster step time (99.7ms vs PR #442's 137ms) yields 38% more training steps
3. AdamW TTT recovers 0.065 bpb from quantized model (1.1545 -> 1.0891 with sliding window)

## Config

All hyperparameters set as defaults in train_gpt.py. Key settings:

```
NUM_LAYERS=11  MODEL_DIM=512  NUM_HEADS=8  NUM_KV_HEADS=4
MATRIX_LR=0.025  SCALAR_LR=0.025  TIED_EMBED_LR=0.035
ITERATIONS=9000  WARMDOWN_ITERS=1200
EMA_ENABLED=1  EMA_DECAY=0.997
VALUE_RESIDUAL=1  GATED_ATTENTION=1
TTT_ENABLED=1  TTT_LR=0.0005  TTT_EPOCHS=10
EVAL_STRIDE=64
```

## Run command

```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Credits

- **PR #442** (sjp611): AdamW TTT, base recipe
- **PR #398** (felipe-parodi): EMA, aggressive TTT findings
- **PR #413**: Value Residual + Gated Attention ablation
- **PR #315** (jfprincz): Foundation architecture (U-Net skips, SmearGate, orthogonal init)
