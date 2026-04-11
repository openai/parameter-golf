# EMA+SWA Tight Averaging with Fused TTT LoRA + Sliding Window

**val_bpb = 1.1371** (at 600s eval mark) | Artifact: 15.88 MB

Non-record submission. Training fits 10-minute budget (600s on 8xH100). Eval exceeded budget by ~50s due to an adaptive-skip timing bug (now fixed in the included script).

## Method

Four main additions over baseline:

1. **EMA + SWA tight averaging** (decay=0.997, no qgrid). SWA collects from EMA state, not raw weights. Disabling `qgrid_lambda` was critical — grid nudging corrupts EMA by snapping weights to quantization grid during training.

2. **Fused TTT LoRA + Sliding Window**. Rank-8 LoRA test-time training fused with sliding window eval (stride=256) in a single pass. Score each chunk, then train LoRA on it.

3. **Muon 0.99 momentum** with warmup from 0.92 over 1500 steps, warmdown 3500 iters.

4. **Full GPTQ Hessians** for all tensors including attention projection and MLP down-projection.

## Configuration

11L D512, MLP 3x, 8/4 heads, BigramHash 4096, VE 128, XSA last 4, Partial RoPE 25%.
QAT int5 MLP + int6 attention. EMA 0.997, SWA enabled, qgrid disabled. Seed 1337.

```bash
torchrun --nproc_per_node=8 --master_port=29500 train_gpt.py
```

All hyperparameters baked into defaults. No environment variables needed.

## Results

| Metric | Value |
|--------|-------|
| Training steps | 5922 (600s on 8xH100, 101 ms/step) |
| Raw val_bpb | 1.3548 |
| Roundtrip val_bpb | 1.1967 |
| SWA count | 15 |
| Fused TTT+sliding at 600s eval | **1.1371** |
| Fused TTT+sliding full (651s) | 1.1301 |
| Artifact | 15,876,912 bytes (5 int5 fallback layers) |
| Model params | 27,333,833 |

## Eval Time

Full fused eval completed in 651s, exceeding the 600s eval budget by ~50s. At the 600s mark the score was 1.1371 (84% chunks evaluated). The included script contains a hard-timeout fix that guarantees eval termination within budget.

## Included Files

- `train_gpt.py` — training + eval script (114 KB)
- `train.log` — full 8xH100 log
- `submission.json` — metadata
- `README.md` — this file
