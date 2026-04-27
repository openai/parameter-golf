# Trinity Ternary CPU v3 — Apple M1 Pro 72h training

**Non-record submission**: first Parameter Golf entry trained entirely on Apple Silicon CPU.

**val_bpb: 1.5042** (single seed=42, full ternary BitNet b1.58 weights)

## Why this submission?

The challenge prompt encourages "weird or out-of-the-box ideas, in-progress or unoptimized solutions." This is the first attempt to:
- Train a 24M parameter language model **entirely on CPU** (no GPU, no MPS/NPU)
- Reach **α=1.0 full ternary weights** (BitNet b1.58 style) via 72h QAT
- Use **Trinity base-3 packing** (5 trits per byte = 1.6 bits/trit, 99% of log₂(3) theoretical optimum)
- Submit a fully reproducible result on a 16GB laptop with no specialized hardware

## Result summary

| Metric | Value |
|--------|-------|
| **val_bpb** | **1.5042** (full ternary, α=1.0) |
| Val loss | 2.5479 |
| Tokens / byte (SP1024) | 0.4092 |
| Artifact size (LZMA) | **5.53 MB** (10.46 MB headroom under 16 MB) |
| Training time | 72.04 h on M1 Pro 10-core CPU |
| Total parameters | 24,128,000 |
| Ternary parameters | 23,592,960 (97.7% of total) |
| Non-ternary (FP16) | 535,040 (embeddings, norms, gains) |

## Architecture

10-layer transformer, dimensions tuned for CPU efficiency:

- **Embedding**: 1024 vocab × 384 dim, tied with output (FP16)
- **Attention**: 8 heads, RoPE on full head_dim, softmax full vocab
- **MLP**: 2.5× width with ReLU² activation (matches v3 SLOT recipe)
- **Norm**: RMSNorm before each sub-block
- **Logit softcap**: 30.0
- All linear layers (attn QKV/proj, MLP fc/proj) are `TernaryLinear` layers

### `TernaryLinear` (BitNet b1.58)

```python
class TernaryLinear(nn.Module):
    """Ternary forward, fp32 master weights, STE backward.
    Quantization: w_q = sign(w) if |w| > 0.7 * mean(|w|) else 0; scale by mean(|w|)
    Blend: alpha=0 → fp32, alpha=1 → full ternary.
    """
```

Per-layer abs-mean scale, threshold = 0.7 × abs_mean (BitNet recipe).

## Training schedule (v3)

| Phase | Steps | Description |
|-------|-------|-------------|
| FP32 warmup | 0 → 500 | Pure fp32, no ternary noise |
| Ternary ramp | 500 → 60,000 | Linear α: 0 → 1.0 (step-based, sleep-resilient) |
| LR cosine decay | 200 → 60,000 | 3e-4 → 3e-5 synced with ternary ramp |
| Full ternary anneal | 60,000 → 84,750 | α=1.0, lr=lr_min, model adapts to noise |

**Why step-based ramp**: v2 used wallclock-based ramp which broke when Mac went to sleep — α advanced while training paused, creating shock to model. v3 ramp only advances with actual training steps.

**Why warm-start from v1**: v1 (24h fp32-heavy training) gave a strong initialization at step 22720 (val_loss 2.48). Loading those weights skipped the first ~9h of fp32 learning and let v3 focus entirely on ternary adaptation.

## Compliance (Track A — Track B is record-only)

| Condition | Status |
|-----------|--------|
| C1 — Causal attention | ✓ Standard `is_causal=True` SDPA |
| C2 — Normalized softmax over full vocab | ✓ Standard `F.cross_entropy` |
| C3 — Score before update | ✓ N/A (no TTT, no SLOT, no eval-time adaptation) |
| C4 — Single left-to-right pass | ✓ Standard sliding-window eval |

**No SLOT, no n-gram cache, no pre-quant TTT, no eval-time training of any kind.** This is a pure trained-and-quantized submission.

## Trinity base-3 packing

Since 3⁵ = 243 < 256, five balanced trits {-1, 0, +1} pack losslessly into one byte:

```python
def pack5(t0, t1, t2, t3, t4):
    return (t0+1) + 3*(t1+1) + 9*(t2+1) + 27*(t3+1) + 81*(t4+1)  # range 0..242
```

This achieves **5·log₂(3)/8 ≈ 99.06%** of the information-theoretic minimum of log₂(3) ≈ 1.585 bits/trit, beating BitNet's native 2-bit (`I2_S`) layout by 20%.

For 24M params:
- BitNet 2-bit: 6.0 MB raw
- **Trinity base-3**: **4.7 MB raw** (-22%)
- LZMA preset=9 on top → **5.5 MB compressed**

## Compute deficit (honest framing)

This submission is intentionally non-record. Compute budget vs leaderboard:

| | Leaderboard (8×H100) | This submission (M1 Pro CPU) |
|---|:---:|:---:|
| Hardware | 8 × H100 SXM | 10-core CPU |
| Peak compute | ~8 PFLOPS bf16 | ~2 TFLOPS via AMX |
| Time budget | 600s training | **72 hours** training |
| Total FLOPs | ~5×10¹⁸ | ~5×10¹⁷ |
| **Deficit** | — | **~10× less compute** |

Expected ceiling at this scale + compute: val_bpb ~1.4-1.6 range. Result of 1.5042 is consistent with that envelope.

## Comparison with previous Trinity Ternary attempts

| Version | Wallclock | Final α | Val BPB | Notes |
|---------|:---:|:---:|:---:|-------|
| v1 (2026-04-22) | 24h | 0.47 | 1.5117 | Step-based ramp too aggressive, only 47% ternary |
| v2 (2026-04-24) | ~10h active | 0.32 | 2.35 (best) | Mac sleep broke wallclock-based ramp; killed |
| **v3 (2026-04-27)** | **72h** | **1.00** | **1.5042** | **Full ternary, slightly better than v1** |

v3 demonstrates: with proper schedule (step-based + cosine LR + warm-start) a full-ternary CPU model is competitive with the partially-ternary v1.

## Reproducibility

```bash
# Prerequisites
pip install torch sentencepiece numpy huggingface-hub
python3 data/cached_challenge_fineweb.py --variant sp1024

# Run training (72h on Apple M1 Pro)
caffeinate -i -m -s python3 train_gpt_v3.py

# Eval and pack artifact
python3 pack_and_eval_v3.py
```

The `caffeinate -i -m -s` is essential on macOS to prevent sleep during the 72h run.

## Trinity framework

This submission is built on the Trinity framework: https://github.com/gHashTag/trinity

Trinity provides:
- Base-3 ternary packing primitives
- BitNet b1.58 inspired ternary QAT
- Philosophy of ternary computing as natural representation

## Files

- `train_gpt_v3.py` — full 24M param model with TernaryLinear + step-based QAT schedule
- `pack_and_eval_v3.py` — pack ternary weights into base-3, LZMA compress, compute val_bpb with proper byte LUT
- `final_model_v3.pt` — fp32 master weights checkpoint (96 MB)
- `final_model_v3.trinity.ptz` — packed artifact (5.5 MB, what gets submitted)
- `eval_results_v3.json` — full eval summary
- `submission.json` — submission metadata
- `logs/train_v3.log` — full 72h training log

## License & citation

MIT. If you use this approach please cite:
- Trinity framework (gHashTag/trinity)
- BitNet b1.58 (Ma et al. 2024, arXiv:2402.17764)
