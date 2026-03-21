# MLA + SmearGate + BigramHash + SWA

## Summary

Non-record submission demonstrating a stacked architecture combining:
- Multi-Head Latent Attention (MLA) with kv_rank=128
- SmearGate MLP (relu^2 gated, mlp_mult=3)
- BigramHash embeddings (10240 buckets, dim=128)
- Stochastic Weight Averaging (start_frac=0.4, every=50 steps)
- Muon optimizer (momentum=0.99, WD=0.04)
- Mixed int5/int6 quantization + zstd-22
- Sliding-window evaluation (stride=64)

## Results

| Metric | Value |
|---|---|
| Pre-quantization val_bpb | 1.2838 |
| Roundtrip val_bpb | 1.3559 |
| Model size | 14.449MB |
| Training steps | 7001 / 20000 |
| Tokens seen | ~3.7B |
| Step time | ~83ms |

## Key Finding

MLA attention, while parameter-efficient, adds significant compute overhead
per step (~83ms vs ~43ms for the baseline). In a fixed 10-minute window on
8xH100s this reduces token throughput from ~7.2B (baseline) to ~3.7B —
roughly half the training data. The pre-quantization bpb of 1.2838 suggests
the architecture itself is competitive; the bottleneck is throughput, not
capacity.

Replacing MLA with standard GQA would recover the full step budget (~11,500
steps at ~52ms/step) and likely push final bpb below 1.15.

## Architecture

- vocab_size=1024, num_layers=13, model_dim=512
- num_heads=8, num_kv_heads=4, kv_rank=128
- mlp_mult=3, bigram_buckets=10240, bigram_dim=128
- SWA: start_frac=0.4, every=50
- Quantization: int5 MLP, int6 attention, fp16 embeddings
- Compression: zstd-22

## Run Command
```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Files

- `train_gpt.py` — training script
- `*.log` — full training log for this run (UUID-named in this directory)
- `submission.json` — metadata
