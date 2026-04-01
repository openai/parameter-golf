# NorMuon + Int6 STE + SWA + Sliding Window

**val_bpb: 1.1624** | 11917 steps in 600s on 8xH100 | artifact: 15.5MB

stacks five orthogonal improvements over the naive baseline (1.2244):

## techniques

**int6 fake quantization w/ STE** - CastedLinear applies per-row int6 fake quantization during the forward pass with straight-through estimator. the model learns to be robust to the post-training quantization that brings us under 16MB. block weights get int6 ([-31, 31] stored in int8, compresses well with zstd-22). quant penalty is only +0.002 bpb.

**fp16 embedding passthrough** - the tied embedding/logit head is the most quantization-sensitive tensor and has no STE protection during training (it's an nn.Embedding, not a CastedLinear). keeping it in fp16 instead of int6/int8 eliminates its quant penalty entirely at ~0.5MB cost.

**3x wider MLP** - hidden dim 1536 (up from 1024). the int6 compression savings free up enough bytes in the artifact budget to fit the wider model. relu^2 activation. 21.8M total params.

**NorMuon optimizer** - replaces vanilla Muon with NorMuon (from modded-nanogpt). adds per-row second-moment normalization on top of the Newton-Schulz orthogonalization, similar to how Adam normalizes by second moment. slightly better optimization quality per step.

**stochastic weight averaging** - collects model checkpoints every 200 steps during the final warmdown phase and averages them. the averaged weights generalize slightly better than the final point estimate. 7 checkpoints averaged in the submitted run. zero artifact cost.

**sliding window eval (stride=64)** - instead of non-overlapping chunks, slides a 1024-token window by 64 tokens and only scores the last 64 tokens per window. every scored token gets 960 tokens of preceding context. ~0.033 bpb improvement over standard eval, zero artifact cost.

## optimizer tuning vs baseline

- muon momentum: 0.99 (vs 0.95)
- matrix_lr / scalar_lr: 0.020 (vs 0.04)
- tied_embed_lr: 0.030 (vs 0.05)
- warmdown_iters: 3000 (vs 1200)
- muon_momentum_warmup_steps: 1500 (vs 500)
- muon_momentum_warmup_start: 0.92 (vs 0.85)

## results

| run | seed | steps | post-quant bpb | sliding window bpb | artifact bytes |
|-----|------|-------|----------------|-------------------|----------------|
| 1 | 1337 | 11917 | 1.1956 | **1.1624** | 15,518,709 |
| 2 | 42 | 11925 | 1.1955 | **1.1623** | 15,118,439 |
| 3 | 2025 | 11917 | 1.1951 | **1.1618** | - |
| **mean** | | | 1.1954 | **1.1622** | |

quant gap across runs: +0.002 bpb (consistent, STE doing its job)
all runs on 8xH100, ~50ms/step, 600s wall clock

## run command

```bash
RUN_ID=submission \
TRAIN_LOG_EVERY=200 \
VAL_LOSS_EVERY=0 \
EVAL_STRIDE=64 \
EVAL_SEQ_LEN=1024 \
SWA_ENABLED=1 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```
