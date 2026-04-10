# SubSixteen v2: Int6 QAT + MLP 3x + SWA + Sliding Window

**val_bpb: 1.1708** | Artifact: 14,603,588 bytes (under 16MB)

## Architecture

- 9-layer GPT with U-Net skip connections, GQA (8 heads, 4 KV heads)
- MLP 3x expansion (hidden=1536), relu² activation
- 512-dim, 1024 vocab (SentencePiece BPE)
- Tied embeddings with Overtone SVD init
- Phase-transition resid_mix, logit soft-capping (tanh, cap=30)
- NTK-aware RoPE

## Training

- 8xH100 SXM, 600s wallclock cap → 9,722 steps at 61.72ms/step
- Muon optimizer (momentum=0.99, lr=0.02) for matrix params
- AdamW for embeddings (lr=0.03) and scalars (lr=0.02)
- Decoupled weight decay on Muon params
- seq_len=4096, batch=393,216 tokens/step
- Warmdown: 3000 iters, wallclock-aware schedule
- Momentum warmup: 0.92 → 0.99 over 1500 steps

## Key Techniques (v2 additions)

1. **STE fake-int6 QAT**: CastedLinear weights fake-quantized to [-31,31] during forward via Straight-Through Estimator. Model learns distributions that survive 6-bit post-training quantization. Quant penalty: ~0.002 BPB.
2. **MLP 3x expansion**: Hidden dim 1536 (up from 1024). Enabled by int6 saving ~4MB artifact space. 9 layers (down from 10) to fit under 16MB.
3. **SWA (Stochastic Weight Averaging)**: 16 checkpoints collected every 200 steps during warmdown, averaged for export. Smoother generalization.
4. **zstd-22 compression**: Better compression ratio than zlib-9 for the quantized artifact.
5. **Tuned hyperparameters**: Muon momentum 0.99, matrix_lr 0.02, warmdown 3000, seq_len 4096 — from top competition submissions.

## Techniques (carried from v1)

- Sliding window eval (stride=64, seq_len=4096)
- FP16 tied embedding passthrough (no embedding quantization)
- Int6 per-row quantization on block weights, fp32 control tensors
- Overtone spectral init (SVD power-law shaping)
- Phase-transition resid_mix initialization

## Metrics

| Metric | Value |
|---|---|
| Post-quant val_bpb (sliding window) | 1.1708 |
| Pre-quant val_bpb (step 9722) | 1.1732 |
| Quantization penalty | +0.002 BPB |
| Artifact size | 14,603,588 bytes |
| Training steps | 9,722 (wallclock-limited) |
| Step avg | 61.72ms |
| Eval time (sliding window) | 226s |
| Peak memory | 10,138 MiB |
| SWA checkpoints | 16 |

## Run command

```bash
pip install zstandard
RUN_ID=subsixteen_v2 MAX_WALLCLOCK_SECONDS=600 VAL_LOSS_EVERY=2000 TRAIN_LOG_EVERY=200 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Trained and evaluated on 8xH100 SXM (Modal).
