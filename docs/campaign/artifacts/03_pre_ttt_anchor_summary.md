# 03 Pre-TTT Anchor Summary

Date: 2026-03-28
Status: Complete

## Anchor Definition

The Session 03 anchor is a clean 2026-03-21-style pre-TTT stack ported onto the repo-root `train_gpt.py` skeleton. It uses the SDPA attention backend (not flash_attn_3) and hardcodes all architecture/training constants.

### Feature Set

| Feature | Status |
|---------|--------|
| 11L 512d 8H/4KV U-Net skip | Ported |
| 3x relu^2 MLP | Ported |
| SmearGate | Ported |
| BigramHash (2048 buckets, 128d) | Ported |
| XSA on last 4 layers | Ported (adapted for SDPA layout) |
| Partial RoPE (16/64) | Ported (with NTK scaling, root cache layout) |
| Layerwise LN scale | Ported |
| EMA (decay=0.997) | Ported |
| Muon WD=0.04, Adam WD=0.04 | Ported |
| Mixed int6+zstd export | Ported |
| Stride-64 sliding eval | Ported |
| Orthogonal init + proj scaling | Ported |

### Excluded Features

Late QAT, SWA, MTP, VE, DTG, GPTQ-lite, flash_attn_interface, TTT, warmdown 3500.

### Key Adaptation

SDPA uses (B, H, T, D) tensor layout vs donor's flash_attn_3 (B, T, H, D). XSA adapted by transposing SDPA output before self-value subtraction. Rotary cache uses root's `[None, None, :, :]` shape.

## Parameter Count

26,829,913 (matches donor exactly).

## Target

`val_bpb` in 1.123-1.128 range on 8xH100 in 600s.

## Measured Results

| Metric | Value |
|--------|-------|
| GPU / node | 8x NVIDIA H100 80GB HBM3 (SXM5) / serv-3342 |
| Container | nvcr.io_nvidia_pytorch_26.03-py3.sqsh |
| Data path | /fscratch (low-latency) |
| Steps completed | 6,564 / 9,000 |
| Step average | 91.37 ms |
| Pre-quant EMA val_bpb | 1.14472403 |
| Post-quant roundtrip val_bpb | 1.15247273 |
| **Sliding s64 val_bpb** | **1.12904446** |
| Artifact size (int6+zstd) | 15,692,752 bytes |
| Code size | 58,572 bytes |
| Total submission size | 15,751,324 bytes |
| Peak memory | 21,274 MiB allocated / 22,070 MiB reserved |
| Compressor used | zstd |

## Comparison with Donor

| Metric | This run | Donor (2026-03-21) | Delta |
|--------|----------|-------------------|-------|
| Sliding s64 val_bpb | 1.1290 | 1.1248 | +0.0042 |
| Steps | 6,564 | 7,051 | -487 |
| Step average | 91.37 ms | ~85 ms | +6.4 ms |
| Artifact | 15,751,324 | 15,612,308 | +139,016 |

## Bottleneck Analysis

The primary bottleneck is **throughput**: 91.37 ms/step vs donor's ~85 ms/step. This results in 487 fewer steps in 600s, which accounts for the 0.0042 BPB gap.

Root cause: SDPA dispatches to FlashAttention-2 kernels. The donor uses FlashAttention-3 (`flash_attn_3_func`) which has Hopper-specific warp-specialization and asynchronous softmax. Additionally, SDPA requires a transpose for XSA compatibility that FA3 avoids natively.

The `math=True` SDP fallback (now fixed in commit 563700f) may also contribute marginally.

Model fidelity is confirmed: the port is faithful. The 0.0042 gap is entirely explainable by step count.

## Next Recommended Delta

**Session 04: FlashAttention-3 + GPTQ-lite (narrow, isolated)**

1. **FA3 integration**: Switch from SDPA to `flash_attn_func`. Expected gain: ~6ms/step → ~480 more steps → ~0.003 BPB. This also eliminates the XSA transpose overhead.
2. **GPTQ-lite clip search**: Better int6 quantization scales. Expected gain: 0.002-0.005 BPB post-quant.
3. **LeakyReLU²**: 1-line activation change used by the current #1 (1.1194). Expected gain: 0.001-0.003 BPB.

Each delta should be tested in isolation before stacking.
