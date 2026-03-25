# LongContext 4096 + Int6 QAT + Full SOTA Stack (Safe)

**val_bpb: TBD** (3-seed mean, post int6+lzma, sliding window stride=64 + TTT)

## Summary

This is the **safe baseline** for the int4 bank-QAT risky experiment. Same as `LongContext4096_FullSOTA`
but with QAT enabled from step 1 instead of only at the tail of warmdown:

- `QAT_ENABLED=1` by default (was 0)
- `late_qat_threshold=0.0` (disabled — QAT is always on, no late trigger needed)
- CastedLinear applies 6-bit STE fake-quant (`/31.0`, clamp `[-32, 31]`) throughout training
- Export: GPTQ-lite int6, lzma — QAT and export are consistent

**Hypothesis**: Training with int6 noise from step 1 (vs only the final ~15% of steps) gives the
model more time to adapt, reducing the post-quantization degradation.

## Run Command

```bash
SEED=1337 TTT_ENABLED=1 torchrun --standalone --nproc_per_node=8 train_gpt.py
SEED=42   TTT_ENABLED=1 torchrun --standalone --nproc_per_node=8 train_gpt.py
SEED=2025 TTT_ENABLED=1 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Results

*(Pending H100 runs)*

| Seed | Steps | Pre-TTT bpb | Post-TTT bpb | Artifact (bytes) |
|------|-------|-------------|--------------|-----------------|
| 1337 | — | — | — | — |
| 42   | — | — | — | — |
| 2025 | — | — | — | — |
| **Mean** | | | | |
