# SP8192 + 3-Layer Recurrence + Parallel Residuals + QK-Gain 5.25 + Pre-Quant TTT + Scored-Position SLOT

**val_bpb: 0.94569** (3-seed mean: seeds 1337, 42, 2025)

| Seed | val_bpb |
|------|---------|
| 1337 | 0.95036466 |
| 42   | 0.96014543 |
| 2025 | 0.92656976 |

## What this submission does

This builds on the kilojoules/alertcat PR #1738 stack and adds **Scored-Position SLOT** at evaluation time:

- Per-sample delta `[bsz, 1, d_model]` and logit_bias `[bsz, 1, vocab]` in fp32
- AdamW optimizer, 24 steps, cosine LR 0.008 → 0.0008
- Optimization target: scored positions only (past tokens, no look-ahead)
- **Cross-batch EMA warmup** (decay=0.5): converged delta/logit_bias means are carried forward as initialization for the next batch, giving each batch a head start on convergence at zero extra parameter cost
- SLOT runs only during `eval_val_sliding` on the quantized model; training is unmodified

Training wall-clock: ~588s (within 600s budget).
Eval wall-clock: ~1405s (SLOT optimization over full val set).
Artifact size: ~15.87MB (within 16MB budget).

## Base stack

- SP8192 + CaseOps tokenizer (@romeerp PR #1729)
- GPTQ SDClip INT6 + Brotli compression (@clarkkev PR #1394)
- 3-Layer Depth Recurrence L3-5 (@dexhunter PR #1331)
- Parallel Residuals L7+ (@Robby955 PR #1412)
- QK-Gain 5.25 (@stukenov PR #1364)
- EMA 0.9965 + Muon with Huber-WD (@bigbag PR #1493)
- Legal score-first TTT framework (@AjAnubolu PR #1735)
- 3-epoch parallel pre-quant AdamW TTT + budget guard (@alertcat PR #1738, @kilojoules)

## SLOT lineage

Scored-Position SLOT is inspired by @resouer PR #1229. Key differences: per-sample (not shared) delta, cross-batch EMA prior warmup, and restriction to scored positions only via a boolean mask.
