# Seq2048 — Longer Training Context

**val_bpb: 1.2101** (post-quant int8+zlib roundtrip)

Baseline: 1.2244. Improvement: **-0.0143 BPB / -0.025 nats.**

15.87MB artifact. 600s wallclock. 11,417 steps at 52.56ms/step on 8xH100 SXM.

## Approach

One change: train and evaluate at sequence length 2048 instead of 1024. No architecture modifications, no optimizer tuning — the standard baseline model with longer context.

The model learns real long-range dependencies during training rather than relying on RoPE position extrapolation at eval time. Each training step processes the same number of tokens (524K) but in 256 sequences of 2048 instead of 512 sequences of 1024.

## Configuration

```bash
NCCL_IB_DISABLE=1 \
RUN_ID=sub_8x_relu2_seq2048 \
TRAIN_SEQ_LEN=2048 \
EVAL_SEQ_LEN=2048 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

All other hyperparameters are defaults:
- Layout: 9×512, 8 heads, 4 KV heads, MLP_MULT=2 (relu²)
- Tied embeddings, TRAIN_BATCH_TOKENS=524288
- Muon momentum=0.95, MATRIX_LR=0.04, WARMDOWN_ITERS=1200

## Results

| Metric | Value |
|--------|-------|
| Pre-quant val_bpb | 1.2033 |
| **Post-quant val_bpb** | **1.2101** |
| Post-quant val_loss | 2.0431 |
| Quant gap | 0.0068 |
| Training steps | 11,417 (wallclock capped) |
| Step avg | 52.56ms |
| Train time | 600.035s |
| Peak memory | 10,301 MiB per GPU |
| Model int8+zlib | 15,810,588 bytes |
| Code size | 58,477 bytes |
| **Total artifact** | **15,869,065 bytes** |

## Hardware

8×H100 80GB HBM3 (RunPod Secure Cloud), PyTorch 2.9.1+cu128.

## Included files

- `train_gpt.py` — code snapshot from the run
- `train.log` — full training log with periodic validation
- `submission.json` — metadata
