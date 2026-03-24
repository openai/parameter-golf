This submission captures a compact 11-layer GPT trained on 8x H100 with plain DDP, EMA final weights, and an int6+zstd compressed artifact. The main practical finding was that on the available H100 infrastructure, smaller global batch size worked better under the fixed 600s budget.

## Method

Training uses `train_merged_gpt_gold6.py`.

Model:
- 11 transformer layers
- model dim 512
- 8 attention heads / 4 KV heads
- tied embeddings
- RoPE with `rope_dims=16`
- XSA enabled on the last 4 layers
- bigram hash embedding
- value embeddings enabled
- layerwise residual mixing
- LN scaling enabled
- EMA applied at the end of training

Optimization:
- 8x H100 with PyTorch DDP
- one process per GPU via `torchrun --standalone --nproc_per_node=8`
- Muon for matrix parameters
- Adam/AdamW for embeddings / scalar params / head params
- warmup for 20 steps
- wallclock-capped training at 600s

Compression:
- export BF16 checkpoint
- quantize submission weights with mixed low-bit export
- submit `final_model_gold6.int6.ptz`
- compression backend: zstd

## Practical Notes

This same code path previously achieved a much stronger result on a good 8xA100 node using `TRAIN_BATCH_TOKENS=524288`, with post-EMA validation around `1.1764` bpb. On the H100 side, however, the available rented pods showed substantially worse per-step latency despite being rule-compliant H100 hardware.

In practice, the H100 nodes available for this run were inconsistent and often slower step-for-step than the strong A100 baseline node. Because of that, batch-size tuning under the 600s wallclock limit mattered more than raw tokens/step, and smaller batch size was consistently better on the H100 infrastructure we could access.

The main empirical result from this H100 run family was:

| TRAIN_BATCH_TOKENS | post-EMA val_bpb |
|---|---:|
| 1048576 | 1.3510 |
| 524288 | 1.2362 |
| 262144 | 1.2073 |

This suggests that on degraded multi-GPU H100 infrastructure, more optimizer steps within the time budget beat higher token throughput.

## Results

Best confirmed H100 run on this setup:
- `TRAIN_BATCH_TOKENS=262144`
- `DIAGNOSTIC post_ema val_bpb: 1.2073`

Timed eval proof on the saved artifact:
- BF16 eval:
  - `val_bpb: 1.20740615`
  - `eval_time: 63907ms`
- INT6 artifact eval:
  - `val_bpb: 1.21083708`
  - `eval_time: 63493ms`

Artifact size:
- `final_model_gold6.int6.ptz`: about 15 MB
- total submission size with code: under 16 MB

## Command

```bash
torchrun --standalone --nproc_per_node=8 train_merged_gpt_gold6.py
Key environment for the best H100 run:

SEED=42
TRAIN_BATCH_TOKENS=262144
TRAIN_SEQ_LEN=2048
MAX_WALLCLOCK_SECONDS=600
VAL_LOSS_EVERY=0
TRAIN_LOG_EVERY=200
TORCH_COMPILE=0