# PROTEUS v4 — Parameter Golf Submission

**Built with [PROTEUS](https://lightspeedup.com) by LightSpeedUp**

Non-record submission. See [PR #95](https://github.com/openai/parameter-golf/pull/95) for our earlier submission and documented negative results on INT4, depth recurrence, and EMA.

## Approach

10-layer transformer with mixed INT5/INT6 quantization, consensus optimizer settings, and eval-time sliding window.

### Techniques

- 10 layers, dim=512, MLP 3x (1536), 24.5M params
- Mixed quantization: INT5 for MLP weights, INT6 for attention weights
- SmearGate + BigramHash(2048) + OrthoInit
- FP16 tied embedding + Late-K passthrough (last 2 layers c_k)
- Muon optimizer with decoupled weight decay (0.04 on all params)
- AdamW (WD=0.04) for embeddings and scalars
- EMA weight averaging (decay=0.999, every 10 steps)
- 3% magnitude pruning before export
- zstd-22 compression
- Sliding window eval (stride=64, seq=2048)
- RoPE base 50K
- Grad clip 0.3, warmdown 3000

## Key Metrics

- Training stopped at `10493/20000` steps due to wallclock cap
- Pre-quant eval: `val_loss:1.9992`, `val_bpb:1.1840`
- Post-quant sliding window eval: `val_loss:2.0324`, `val_bpb:1.2037`
- Exact: `final_int8_zlib_roundtrip_exact val_bpb:1.20368943`
- Train time: `600049ms` (`step_avg:57.19ms`)
- Eval time: `82174ms`
- Artifact: `12,499,612 bytes` (78.1% of 16MB cap)
- Pruned: 726,371 weights (3.0%)

## Platform

Run on Modal 8×H100.

## Credits

Techniques drawn from published submissions by @unnir (SmearGate, PR #162), @notapplica (Muon WD, PR #60), @mattqlf (sliding window, PR #50), @nanlliu (mixed quantization, PR #39).

## Included Files

- `train_gpt.py` — training script
- `train.log` — full training log
- `submission.json` — leaderboard metadata
- `README.md` — this file
