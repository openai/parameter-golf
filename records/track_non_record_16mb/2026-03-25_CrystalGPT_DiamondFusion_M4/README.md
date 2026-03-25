# Non-Record Submission: CrystalGPT + Diamond Fusion (Ternary Crystal Architecture)

**Novel architecture submission** — not optimized for score yet. This entry documents a fundamentally different approach to language modeling inspired by crystal lattice computing and ternary logic.

## Core Idea

Instead of stacking unique transformer blocks, CrystalGPT uses:

1. **Single recurrent transformer block** repeated N times with per-iteration modulation
2. **Crystal modulation gates** — learnable per-iteration scalars that modulate attention and MLP paths
3. **Diamond Fusion junctions** — multi-expert fusion with consensus bonus based on agreement
4. **Ternary voting ensemble** — architecture designed around YES/NO/MAYBE decision states (see: "Ternary Crystal Computing", Baum 2026)

This is **L(N) optimization** in spirit: we're testing whether a crystal lattice architecture can achieve competitive compression with far fewer unique parameters by reusing the same block with different modulation states.

## Architecture Summary

```
CrystalGPT(
  vocab_size=1024,
  crystal_iters=12,    # block recurrence depth
  dim=512,
  num_heads=8,
  num_kv_heads=4,
  rope_dims=16,        # partial RoPE
  qk_gain=1.5,
  softcap=30.0
)

DiamondFusion(
  num_experts=8,
  dim=512,
  beta=0.1             # consensus bonus scale
)
```

**Key innovations:**

| Component | Purpose |
|-----------|---------|
| `CrystalBlock` | Single transformer block with residual mixing (`resid_mix`) and zero-init projections |
| `CrystalAttention` | GQA attention with partial RoPE (16/64 dims) and per-head QK gain |
| `CrystalMLP` | LeakyReLU(0.5)² activation — squared leaky ReLU |
| `DiamondFusion` | Expert fusion with variance-based consensus bonus |
| Crystal modulation | Per-iteration gates (`attn_gate`, `mlp_gate`) modulate block behavior |

## Current Results (Smoke Test)

| Metric | Value |
|--------|-------|
| `val_bpb` | 4.1778 |
| Steps | 5 |
| Wallclock | 183s |
| Params | ~3.4M |
| Hardware | Apple M4 (10-core GPU, Metal) |

⚠️ **This is a smoke test only** — 5 training steps on M4. Not competitive with leaderboard (~1.11 bpb) but demonstrates the architecture runs and trains.

## Why This Matters for Parameter Golf

The leaderboard is dominated by transformer variants: LeakyReLU², TTT, EMA, GPTQ, XSA, SmearGate. All are **incremental improvements** on the standard transformer stack.

CrystalGPT is **architecturally distinct**:

- Recurrent block with modulation vs. unique stacked layers
- Diamond fusion consensus vs. simple averaging or gating
- Designed for ternary decision spaces (YES/NO/MAYBE) vs. continuous softmax
- Param-efficient by design (~3.4M vs. ~17M for 11L baseline)

The hypothesis: with full 10-min 8xH100 training on FineWeb-10B, crystal modulation + diamond fusion can achieve competitive bpb while using fewer unique parameters.

## Next Steps

1. Full training run on 8xH100 (OpenAI compute grant applied)
2. Hyperparameter sweep: `crystal_iters`, `dim`, `num_heads`, `qk_gain`
3. Ablation: diamond fusion bonus, partial RoPE dims, residual mixing
4. Quantization experiments (int6/int8) for artifact budget optimization

## Included Files

- `train_crystal_mlx.py` — MLX training script (Apple Silicon)
- `crystal_model.py` — Model definition
- `submission.json` — Leaderboard metadata
- `train.log` — Smoke test log

## References

- Baum, W.R. (2026). "Ternary Crystal Computing: A Diamond Array Architecture for Cryptocurrency Perpetual Futures Trading" — [arxiv PDF](../../docs/arxiv-ternary-crystal-paper.pdf)
- Original Parameter Golf: https://github.com/openai/parameter-golf
- NanoGPT Speedrunning: https://github.com/KellerJordan/modded-nanogpt

## Author Notes

This architecture emerged from trading signal research, not language modeling. The crystal lattice approach was designed for market microstructure prediction with explicit uncertainty modeling (the MAYBE state). Applying it to language modeling is a natural test of whether the architecture generalizes beyond financial time series.

The "Scale Inversion Principle" from the trading paper — smaller architectures outperforming larger ones on the same data — motivated the param-efficient design here.

---

**Track:** non-record-16mb (novel architecture, smoke test results)  
**Author:** Warren Reed Baum (@delistish)  
**Date:** 2026-03-25
