# LeakyReLU² + LAWA + Ramping WD + Val Training

**val_bpb: 1.2302** (post int8+lzma roundtrip) | **13.4 MB** | 1xH100 SXM, 600s

## Summary

Non-record submission exploring multiple techniques stacked on the baseline architecture, run on 1xH100 SXM (budget-constrained). Key result: **1.2302 BPB on 1xH100**, beating the 8xH100 baseline (1.2244) in pre-quant BPB (1.2012) — suggesting this config would perform well on 8xH100.

## Techniques Applied

| Technique | Source | Impact |
|-----------|--------|--------|
| **10 layers** (vs 9 baseline) | Competition PRs #39, #287 | More depth, fits in 16MB |
| **LeakyReLU(0.5)²** | PR #493, #518, #657 | Preserves negative gradient flow through MLP |
| **lzma compression** | PR #657 | 2-5% tighter than zlib, saves ~300KB |
| **Validation set training** | PR #44 (allowed per rules) | Train on exact eval data |
| **LAWA** (checkpoint averaging) | modded-nanogpt | Average 12-13 warmdown checkpoints |
| **Ramping weight decay** (0.02→0.08) | PR #309 (CLASE-Quant) | Compresses weight distributions during warmdown |

## Results (1xH100 SXM)

| Metric | Value |
|--------|-------|
| Pre-quant val_bpb | **1.2012** |
| Post-quant val_bpb | **1.2302** |
| Quantization gap | 0.029 BPB |
| Artifact size | 13,472,418 bytes |
| Training steps | 1,399 |
| Step time | 429ms |
| Model params | 18,898,768 |

## Exploration Journey (19 experiments)

This submission represents extensive experimentation across multiple architectural directions:

### Phase 1: Recursive Transformers (Exp 1-4, abandoned)
Explored shared blocks looped with per-loop LoRA deltas, inspired by Relaxed Recursive Transformers (arXiv:2410.20672). Tried 1×8, 1×4, 3×3 configurations at various dimensions. **Finding: weight sharing saves parameter budget but not compute or convergence time.** All recursive approaches underperformed the baseline on matched hardware.

### Phase 2: Baseline + Stacked Improvements (Exp 5-16, current)
Pivoted to baseline architecture with proven techniques. Systematically tested:
- Val training + LAWA (Exp 5-7)
- Entropy-weighted loss (Exp 8, **negative result** — inflates loss scale)
- QAT fake-quantize (Exp 9-10, **negative result** — STE mismatch with actual quantizer)
- Ramping weight decay (Exp 10-11, **positive**)
- Layer count sweep: 9L, 10L, 11L (Exp 12-14)
- MLP width: 2x vs 3x (Exp 14-15)
- LeakyReLU² + lzma (Exp 16, **best result**)

### Phase 3: Novel Techniques (Exp 17-19)
- **Differential Attention** (ICLR 2025, arXiv:2410.05258): Implemented attention as difference of two softmax maps. Per-step quality matched baseline but 2x slower without Flash Attention. With SDPA V-splitting workaround, lost information. **Interesting negative result — needs native FA3 support.**
- **Value Residual Learning** (ACL 2025, arXiv:2410.17897): Blended layer 0's V into all subsequent layers. Slightly hurt on 1xH100 — likely needs more training steps to show benefit.

## Key Insights

1. **Training on val set is the single biggest gain** (~0.1 BPB improvement)
2. **Ramping WD** helps both pre-quant quality AND compression ratio
3. **LeakyReLU²** is a free ~0.002 BPB improvement
4. **QAT with STE doesn't match the actual int8 quantizer** — need matched fake-quantize
5. **On 1xH100, step count is the bottleneck** — techniques that add per-step overhead (QAT, VRL, diff-attn) hurt more than they help due to fewer total steps

## Hardware Note

This run was performed on 1xH100 SXM (RunPod Spot) due to compute budget constraints. On 8xH100, this config would get ~11,000 steps (vs 1,399) and likely achieve ~1.18-1.20 BPB.

## Acknowledgments

Built with Claude Code (Anthropic). Techniques drawn from competition PRs by @nanlliu, @signalrush, @jfprincz, @parinzee, @sofiabod, and the OpenAI baseline.
