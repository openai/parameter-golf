# AutoResearch: LeakyReLU² + MLP3x + Batch Optimization

Non-record submission (1×RTX 4090, 5-minute training budget).

## Score

**val_bpb: 1.1801** on FineWeb validation set.

## Method

Built on the [autoresearch](https://github.com/openai/autoresearch) framework, iteratively optimizing architecture and hyperparameters through automated ablation sweeps on a single RTX 4090 (24GB).

### Architecture
- **8 layers, 1024d, 8 heads** (MHA, no GQA)
- **MLP 3x** hidden dim (3072) — saves parameters vs 4x, allowing more training steps
- **LeakyReLU(0.75)²** activation (inspired by PR #1185 SOTA)
- **Value Embeddings** (alternating layers, gated per-head)
- **RoPE** positional encoding
- **RMSNorm** pre-norm with residual lambdas + x0 skip connections
- **Window pattern: L** (all long-range attention via Flash Attention 3)
- **Sequence length: 4096** at train and eval

### Optimization
- **Muon optimizer** for matrix params (LR=0.10, WD=0.2)
- **Adam** for embeddings (LR=0.6) and scalars (LR=0.5)
- **Total batch size: 64K tokens** (device_batch=8, grad_accum=2)
- **No warmup**, 50% cosine warmdown
- **467 steps** in 300s training budget

### Key Findings (from 50+ ablation runs)
1. **MLP 3x > 4x**: Fewer params per layer → more steps in budget → better final BPB
2. **LeakyReLU(0.75)²** beats ReLU² by ~0.002 bpb
3. **Window pattern L** (all long-range) slightly beats SSSL
4. **Batch size 64K** sweet spot: enough steps (467) with good gradient estimates
5. **device_batch=8** with more grad accumulation beats device_batch=16
6. **seq_len=4096** marginal improvement over 2048

### Hardware
- 1× NVIDIA RTX 4090 (24GB GDDR6X)
- Peak VRAM: 8.8 GB
- MFU: 4.47%
- Training: 300.6s, Eval: 93.3s

## Progression

| Round | Best BPB | Key Change |
|-------|----------|------------|
| R1 | 1.4789 | Baseline + batch/LR tuning |
| R2 | 1.2056 | Total batch 2^17 |
| R3 | 1.1934 | LeakyReLU(0.75)², window=L, batch 2^16 |
| R4 | **1.1801** | MLP 3x, device_batch=8, seq_len=4096 |

## Results

Validated on the full FineWeb validation set. Final score: **val_bpb = 1.1801**.

## Notes

This is a single-GPU (4090) submission — no H100 cluster. The autoresearch framework's automated sweep approach finds good architectures efficiently even on consumer hardware.
