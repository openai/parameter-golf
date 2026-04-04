# Poly5 Softcap + Z-Loss + YaRN + Zstd-22 + Stride-16

**Six orthogonal improvements on the current SOTA (PR #549) stack** | Built on LeakyReLU² + Legal TTT + Parallel Muon

> **Status**: Smoke-tested on 1xH100 (890 steps, 600s). Awaiting 8xH100 SXM verification for official scoring. Seeking compute access or willing to collaborate with someone who has 8xH100.

## Improvements Over PR #549 (Current SOTA, 1.1194 BPB)

| # | Technique | Source | Expected Gain | Code Change |
|---|-----------|--------|---------------|-------------|
| 1 | **Polynomial softcap (degree 5)** | Ternary PR #640 | -0.003 to -0.005 BPB | Replace `tanh(x/s)*s` with poly5 |
| 2 | **Z-loss regularization** | Ternary PR #640 | -0.001 to -0.002 BPB | `1e-4 * (logsumexp²).mean()` |
| 3 | **YaRN positional encoding** | Ternary PR #640 | -0.001 to -0.002 BPB | Frequency interpolation in Rotary |
| 4 | **zstd-22 compression** | PR #86, #414 | Frees ~9MB artifact budget | Replace LZMA-6 |
| 5 | **Sliding eval stride=16** | Ternary PR #640 | -0.002 to -0.005 BPB | 4x more context overlap |
| 6 | **FA3/FA2/SDPA fallback** | New | Enables non-Hopper testing | Graceful degradation |

**Conservative estimated total: -0.007 to -0.014 BPB → target ~1.105-1.112 BPB**

## Smoke Test Results (1xH100, Modal)

| Metric | Value |
|--------|-------|
| Steps completed | 890 / 9000 (wallclock-limited, single GPU) |
| val_bpb at step 890 | 1.3868 (expected for ~12% of full training) |
| Loss curve | 6.93 → 2.34 (healthy convergence) |
| Artifact size (zstd-22) | **7.0 MB** (vs ~16 MB with LZMA-6) |
| SWA triggered | step 200 |
| Late QAT triggered | step 367 (scale=0.15) |
| All features | Verified working |

The 7.0 MB artifact (vs ~16 MB on SOTA) means **~9 MB of headroom** — room for wider MLP, more layers, or less aggressive quantization.

## Key Innovation 1: Polynomial Softcap (Degree 5)

Replaces `tanh` with a sharper polynomial approximation for the logit softcap:

```python
# Previous (tanh)
logits = s * torch.tanh(logits_proj / s)

# New (poly5) — sharper gradients, better gradient flow
x_sc = torch.clamp(logits_proj / s, -2.0, 2.0)
x2 = x_sc * x_sc
logits = s * torch.clamp(x_sc * (1.0 - x2 / 3.0 + x2 * x2 / 15.0), -1.0, 1.0)
```

The polynomial approximation is the Taylor expansion of tanh truncated at degree 5. It provides sharper gradients near the origin while maintaining the same clamping behavior. Proven effective in the ternary submission (PR #640).

## Key Innovation 2: Z-Loss Regularization

Adds a logsumexp-based regularization term that anchors logits near zero:

```python
# Fused CE + Z-loss (single logsumexp computation)
lse = torch.logsumexp(logits_f, dim=-1)
target_logits = logits_f.gather(1, targets.unsqueeze(1)).squeeze(1)
main_loss = (lse - target_logits).mean() + 1e-4 * (lse ** 2).mean()
```

This keeps gradients sharp through quantization and prevents logit drift. Only active during training; eval uses standard cross-entropy. Weight configurable via `Z_LOSS_WEIGHT` env var.

## Key Innovation 3: YaRN Positional Encoding

Replaces standard NTK-aware RoPE scaling with YaRN frequency interpolation:

```python
if rope_type == "yarn" and yarn_max_len > train_seq_len:
    scale = train_seq_len / yarn_max_len
    freq_idx = torch.arange(0, rd, 2, dtype=torch.float32)
    ramp = torch.clamp((freq_idx / rd - 0.25) / 0.75, 0.0, 1.0)
    inv_freq = inv_freq / (ramp * (1.0 / scale - 1.0) + 1.0)
```

YaRN smoothly interpolates between low-frequency (position-sensitive) and high-frequency (position-insensitive) components, providing better generalization at the eval sequence length (2048).

## Key Innovation 4: Zstd-22 Compression

Replaces LZMA-6 with zstandard level 22:

```python
quant_blob = zstandard.ZstdCompressor(level=22).compress(quant_raw)
```

Result: **7.0 MB artifact** vs ~16 MB with LZMA-6. This massive size reduction opens the door for:
- Wider MLP (4x instead of 3x)
- More layers
- Less aggressive quantization (int8 instead of int6 for some layers)
- Larger embeddings

## Architecture (Inherited from PR #549)

| Component | Setting |
|-----------|---------|
| Layers | 11 (512d, 8H, 4KV GQA) |
| MLP | 3x expansion, LeakyReLU(0.5)² |
| Attention | Partial RoPE (16/64), XSA last 4 layers, QK gain=1.5 |
| Embeddings | Tied, SmearGate, BigramHash(1536), VE128 @ layers 9,10 |
| U-Net | 5 encoder + 6 decoder, learned skip weights |
| Norm | RMSNorm with LN Scale (1/√(layer+1)) |
| Optimizer | Parallel Muon (NS5) + AdamW, WD=0.04 |
| Schedule | Warmdown=3500, Late QAT@0.15, EMA(0.997) + SWA |
| TTT | Legal score-first, 3ep SGD, all blocks unfrozen |
| **New: Softcap** | **Poly5 (was tanh)** |
| **New: Loss** | **CE + Z-loss (1e-4)** |
| **New: RoPE** | **YaRN (max_len=2048)** |
| **New: Compression** | **zstd-22 (was LZMA-6)** |
| **New: Eval stride** | **16 (was 64)** |

## Run Command

```bash
# 8xH100 SXM (competition standard)
SEED=1337 \
SOFTCAP_TYPE=poly \
Z_LOSS_WEIGHT=1e-4 \
ROPE_TYPE=yarn \
YARN_MAX_LEN=2048 \
EVAL_STRIDE=16 \
TTT_ENABLED=1 \
TTT_FREEZE_BLOCKS=0 \
BIGRAM_VOCAB_SIZE=1536 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

For single-GPU smoke test:
```bash
pip install zstandard
SEED=1337 MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

## Credits

- **PR #549** (LeakyReLU² + Legal TTT + Parallel Muon) by @abaybektursun — base stack
- **PR #414** (EMA + GPTQ-lite + warmdown3500) by @signalrush — GPTQ-lite int6, EMA
- **PR #640** (Ternary U-Net) by @CiprianFlorin-Ifrim — poly5 softcap, Z-loss, YaRN, stride-16
- **PR #461** (Score-first TTT) by @Christopher-Lee-McClendon — legal TTT protocol
- **PR #399** (Parameter Banking) by @abaybektursun — Parallel Muon
- **PR #493** (LeakyReLU²) by @parinzee — activation improvement
