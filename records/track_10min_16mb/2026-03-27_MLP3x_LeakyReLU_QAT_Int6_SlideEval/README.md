## Record: 9L MLP3x LeakyReLU(0.5)^2 + QAT Int6 + zstd (val_bpb: 1.1653)

**val_bpb = 1.1653** (sliding window stride=64, post int6+zstd roundtrip) | **15.03 MB** artifact | 8xH100 SXM, 600s

### Key Changes from Baseline

1. **3x MLP expansion** (hidden=1536 vs baseline 1024): Larger MLP capacity, enabled by int6 compression savings.
2. **LeakyReLU(0.5)^2 activation**: Preserves negative gradient flow through the MLP while maintaining the relu^2 inductive bias. Eliminates dead neurons. One-line change: `F.leaky_relu(x, negative_slope=0.5).square()`.
3. **Int6 quantization** ([-32, 31] per-row): Narrower quantization range stored as int8, compresses much better with zstd. Enables fitting 21.8M params (vs 17M at int8) under 16MB.
4. **QAT (Quantization-Aware Training)**: STE fake-quantize applied to all CastedLinear weight matrices during training. The model learns to be robust to int6 rounding noise, reducing post-quant degradation.
5. **zstd-22 compression**: Replaces zlib-9. ~30% better compression ratio on int6 data.
6. **FP16 embedding passthrough**: Tied embedding kept in fp16 during quantization (not int6). Nearly eliminates quant error for this sensitive tensor.
7. **Sliding window evaluation** (stride=64): Each token scored with 1984+ tokens of context instead of avg ~1024.
8. **Seq2048 training**: Longer context during training improves model quality.
9. **Optimizer tuning**: warmdown=10000, muon_backend_steps=10, grad_clip=1.0, beta2=0.99, scalar_lr=0.02.

### Architecture

- 9 transformer layers, 512-dim, 8 heads, 4 KV heads (GQA)
- **3x MLP** expansion (hidden=1536), **LeakyReLU(0.5)^2** activation
- U-Net skip connections (4 encoder + 5 decoder)
- Tied embeddings, logit softcap=30.0, RoPE base=10000
- 21.8M parameters

### Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| train_seq_len | 2048 |
| train_batch_tokens | 524,288 |
| warmdown_iters | 10,000 |
| matrix_lr | 0.04 |
| scalar_lr | 0.02 |
| tied_embed_lr | 0.05 |
| muon_momentum | 0.95 (warmup 0.85→0.95 over 500 steps) |
| muon_backend_steps | 10 |
| grad_clip_norm | 1.0 |
| beta2 | 0.99 |

### Quantization

- Int6 per-row [-32, 31] for all block weight matrices (stored as int8, compresses better)
- FP16 passthrough for tied embedding
- FP32/FP16 passthrough for small control tensors
- zstd level 22 compression
- QAT: STE fake-quantize during training simulates int6 noise

### Command

```bash
RUN_ID=submission \
MAX_WALLCLOCK_SECONDS=600 \
TRAIN_LOG_EVERY=200 \
VAL_LOSS_EVERY=2000 \
EVAL_BATCH_SEQS=1024 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Requires `zstandard` package (`pip install zstandard`).

### Key Metrics (from `train.log`)

- Timed training stopped at `10503/20000` steps due to the wallclock cap.
- Pre-quant eval at stop: `val_loss:2.0041`, `val_bpb:1.1869`
- Post-quant sliding window eval: `val_loss:1.9676`, `val_bpb:1.1653`
- Exact printed metric: `final_int8_zlib_roundtrip_exact val_bpb:1.16531391`
- Train time: `600038ms` (`step_avg:57.13ms`)
- Peak memory: `11250 MiB allocated`, `11398 MiB reserved`
- Eval time: `147279ms` (sliding window, stride=64, batch_seqs=1024)
- Serialized model int6+zstd: `14980068 bytes`
- Code size: `53965 bytes`
- Total submission size: `15034033 bytes`

### Training Volume

- Global batch: `524288` tokens/step
- Total train tokens seen: `5,511,536,640`

### Methodology

Developed through 54+ iterations of automated experimentation (autoresearch) on an RTX 3090 proxy, validated on 8xH100. Key learnings:
- Int6+zstd compression frees ~5MB of artifact budget vs int8+zlib, enabling 3x MLP
- QAT is essential — without it, int6 quantization degrades BPB significantly
- LeakyReLU(0.5)^2 provides consistent improvement by eliminating dead neurons
- 11 layers MLP3x achieves better BPB (1.1505) but blows the 16MB budget at convergence
- The 16MB constraint at full H100 convergence is ~0.69 bytes/param (much tighter than short proxy runs suggest)

### Included Files

- `train_gpt.py` (code snapshot used for the run)
- `train.log` (training log, SEED=1337)
- `submission.json` (leaderboard metadata)
