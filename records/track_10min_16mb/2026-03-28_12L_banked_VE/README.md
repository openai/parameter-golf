# 12L Banked + Parallel Muon + Value Embeddings

**val_bpb: 1.1571** (new best) | **16.47 MB** | 8×H100 SXM

## Results

| Seed | step_avg | steps | Post-quant bpb | Post-TTT bpb | Artifact |
|------|----------|-------|----------------|--------------|----------|
| 1    | 138ms    | 4358  | 1.1668         | **1.1571**   | 16,465,445 |

## Architecture

| Component | Setting |
|-----------|---------|
| Layers | 12 (512d, 8H, 4KV) |
| MLP | 3× with LeakyReLU(0.5)² |
| BigramHash | 10240 buckets, INT4 bQAT |
| XSA | Last 4 layers |
| RoPE | Partial (16/64 dims) |
| LN Scale | 1/√(layer+1) |
| Weight avg | EMA(0.997) with QAT-activation reset |
| Quantization | INT4 MLP + INT4 bigram + INT6 attn + zstd |
| QAT trigger | Wallclock fraction (65% of budget) |
| Value Embeddings | ve_dim=128, layers 10-11 |
| TTT | Legal score-first, lr=0.002, 3 epochs |
| **Model Banking** | 4 banked 3D params (qo/kv/mlp_up/mlp_down) |
| **Parallel Muon** | Async reduce-scatter on banked grads, no DDP |

## Key Change: Model Banking

Previous approach used per-layer `nn.Linear` modules. Each layer's grad was a separate 2D tensor, so Parallel Muon required copying grads into a stacked buffer — overhead that negated NCCL savings.

Model banking stores weights as 3D tensors `[num_layers, M, K]`. Grad accumulates directly in banked shape — reduce-scatter operates on it with zero copy.

**Result:** 138ms/step vs 148ms for unbanked VE — 10ms improvement, 300 extra steps in 10 min, new best.

## Comparison

| Run | ms/step | Steps | ttt_bpb |
|-----|---------|-------|---------|
| v7_ve seed 2 | 148ms | 4058 | 1.15738 |
| v7_ve seed 3 | 149ms | 4034 | 1.15796 |
| **v10_banked seed 1** | **138ms** | **4358** | **1.15711** |

## Training File

`train_gpt_v3.py` — full rewrite with banked weight storage and parallel Muon.

## Size Budget

| Component | Bytes |
|-----------|-------|
| Total artifact | 16,465,445 |
| Budget | 16,777,216 |
| Margin | 311,771 (304KB) |
