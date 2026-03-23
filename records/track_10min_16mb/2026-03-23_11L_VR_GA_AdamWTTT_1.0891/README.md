# Record: 11L VR + GA + LeakyReLU² + Legal Score-First TTT (val_bpb=pending)

**val_bpb = pending rerun** | 8xH100 SXM, 600s training + legal TTT eval

## Approach

Architecture improvements on the standard 11L competitive stack:

**Value Residual** (ResFormer, arXiv:2410.17897): Each attention block receives the raw V from the first block. A learned 2-element lambda blends first-block V with current V before attention. Block 0 passes V through unchanged (no lambda parameter). Adds 2 params per layer (layers 1-10 only).

**Gated Attention** (arXiv:2505.06708): Per-head sigmoid gate on attention output. Learned weight matrix (dim x num_heads) + bias initialized to 4.0 (near-open gate at init). Adds 4104 params per layer.

**LeakyReLU(0.5)²**: Replaces relu² in MLP. Preserves negative gradient flow. Proven by PR #569 and #535.

**Legal score-first TTT**: Score each validation chunk before training on it. Every token evaluated BEFORE the model has seen it. AdamW optimizer, cosine LR across chunks, last 2 blocks + norms unfrozen.

Both VR and GA were ablated individually in PR #413 (-0.015 and -0.003 bpb respectively, -0.017 combined). This is the first validation with legal TTT + LeakyReLU².

## Previous result (pre-eval TTT, non-compliant)

The initial submission used pre-eval TTT (training on all val data before scoring), which is not competition-legal per issue #402. That result (1.0891) is invalid. This update switches to legal score-first TTT. Score pending rerun.

## Config

All hyperparameters set as defaults in train_gpt.py. Key settings:

```
NUM_LAYERS=11  MODEL_DIM=512  NUM_HEADS=8  NUM_KV_HEADS=4
MATRIX_LR=0.025  SCALAR_LR=0.025  TIED_EMBED_LR=0.035
ITERATIONS=9000  WARMDOWN_ITERS=1200
EMA_ENABLED=1  EMA_DECAY=0.997
VALUE_RESIDUAL=1  GATED_ATTENTION=1
TTT_ENABLED=1  TTT_LR=0.0001  TTT_EPOCHS=3  TTT_UNFREEZE_BLOCKS=2
EVAL_STRIDE=64
```

## Run command

```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Credits

- **PR #576** (cmcdnd): Legal score-first TTT implementation, temperature calibration
- **PR #569** (gowtham0992): VRL + LeakyReLU² + Full GPTQ (best non-TTT)
- **PR #413**: Value Residual + Gated Attention ablation
- **PR #315** (jfprincz): Foundation architecture (U-Net skips, SmearGate, orthogonal init)
