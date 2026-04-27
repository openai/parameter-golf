## Record: 10L LeakyReLU(0.5)² + EMA + GPTQ-lite (target val_bpb: ~1.138)

**Architecture**: 10 layers, 512-dim, 8 heads (4 KV heads, GQA) | **Expected artifact**: ~15 MB | 8×H100 SXM, 600s

### Key Innovations Over Previous MixedQAT Submission (1.1478 bpb)

| Change | Previous | This | Expected Impact |
|--------|----------|------|-----------------|
| **LeakyReLU(0.5)²** | relu² | leaky_relu(0.5)² | -0.003 BPB |
| **EMA** (decay=0.997) | None | EMA shadow every step, blended 50/50 with SWA | -0.0006 BPB |
| **GPTQ-lite** | Fixed row-max clip | 5 clip percentiles, min-MSE per row | -0.0006 BPB |
| **No QAT** (pure PTQ) | Full QAT (int5+int6 STE) | PTQ only → ~7500 steps vs 6137 | -0.005 BPB |
| **Warmdown 3500** | 3000 | 3500 | -0.0002 BPB |
| **Native GQA** | repeat_interleave workaround | enable_gqa=True (PyTorch 2.4+) | faster SDPA |
| **Total** | 1.1478 | **~1.137–1.140** | **~-0.009 BPB** |

### LeakyReLU(0.5)²

```python
# Before (relu²)
x = torch.relu(self.fc(x))
return self.proj(x.square())

# After (leaky relu²)
x = F.leaky_relu(self.fc(x), negative_slope=0.5)
return self.proj(x.square())
```

Preserves negative gradient flow through the MLP (eliminates dead neurons). Squaring still produces non-negative outputs. Ablated at **-0.003 BPB** in PR #493 and PR #518.

### EMA (Exponential Moving Average)

Shadow copy of model parameters updated every training step with `decay=0.997`. At the end of training, EMA is blended 50/50 with SWA-averaged weights. Overhead: ~50µs/step on H100 (negligible).

### GPTQ-lite: Per-Row Optimal Clip Search

Instead of using `row_max` as quantization scale, 5 clip percentiles [0.999, 0.9995, 0.9999, 0.99999, 1.0] are tried per row and the one minimizing reconstruction MSE is selected. Applied at quantization time — **zero training cost**.

### Architecture

- 10 transformer layers, 512-dim, 8 heads (4 KV heads, GQA)
- 3× MLP expansion (1536 hidden), **LeakyReLU(0.5)²** activation
- U-Net skip connections
- SmearGate + BigramHash (10240 buckets, dim=128)
- Tied embeddings, logit softcap=30.0

### Training

- Muon optimizer (matrices): lr=0.02, momentum=0.99 (warmup 0.92→0.99 over 1500 steps), WD=0.04
- AdamW (embeddings/scalars): lr=0.03/0.02
- Batch: 786,432 tokens/step, seq_len=2048
- Warmdown: **3500 iterations** (wallclock-based)
- **EMA**: decay=0.997, every step
- **SWA**: every 50 steps when scale<0.4
- **No QAT** — pure PTQ (int5 MLP, int6 attention, int8 embeddings) applied only at export
- GPTQ-lite clip search at export time

### Quantization

- Int5 per-row for MLP weights (GPTQ-lite clip search)
- Int6 per-row for attention + bigram projection weights (GPTQ-lite clip search)
- Int8 per-row for embeddings
- Control tensors in fp32
- zstd level 22 compression

### Run Command

**Important**: Use a RunPod image with PyTorch ≥2.4 for native GQA support (e.g. `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`).

```bash
RUN_ID=leakyrelu2_ema_gptqlite SEED=1337 \
torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-03-23_10L_LeakyReLU2_EMA_GPTQlite/train_gpt.py
```

For PyTorch <2.4 (fallback to repeat_interleave GQA — slightly slower):
```bash
RUN_ID=leakyrelu2_ema_gptqlite SEED=1337 \
torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-03-23_10L_LeakyReLU2_EMA_GPTQlite/train_gpt.py
```
(Same command — code auto-detects `enable_gqa` availability.)
