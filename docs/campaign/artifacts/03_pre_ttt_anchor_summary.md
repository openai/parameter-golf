# 03 Pre-TTT Anchor Summary

Date: 2026-03-28
Status: Script complete, pre-run

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

## Expected Parameter Count

~26.8M parameters (donor: 26,829,913).

## Target

`val_bpb` in 1.123-1.128 range on 8xH100 in 600s.

## Measured Results

(To be filled after first Pegasus run)

| Metric | Value |
|--------|-------|
| GPU / node | |
| Steps completed | |
| Step average | |
| Pre-quant EMA val_bpb | |
| Post-quant roundtrip val_bpb | |
| Sliding s64 val_bpb | |
| Artifact size (int6+zstd) | |
| Code size | |
| Total submission size | |
| Peak memory | |
| Compressor used | |

## Bottleneck Analysis

(To be filled after run)

## Next Recommended Delta

(To be determined by bottleneck readout)

- If step_avg is materially slower than expected: backend/kernel parity (flash_attn_3)
- If step_avg is acceptable but BPB is far from target: port-fidelity gap investigation
- If results are in-band: GPTQ-lite as first post-anchor export refinement
