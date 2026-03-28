# QAT Dead-Code Bug + 7 Untried Techniques: What I Found

**Non-Record Submission (Research Contribution)**
**Author:** [@wfproc](https://github.com/wfproc)
**Base:** PR #549 by @abaybektursun (1.1194 bpb, current SOTA)
**Hardware:** 1xH100 80GB SXM, 600s wallclock (~940 steps per run)
**Best pre-quant:** 1.3646 val_bpb | **Best post-quant (int6 sliding):** 2.3411 val_bpb

Not competitive with the 8xH100 leaderboard (7x fewer steps), but the findings transfer to any hardware.

---

## The Short Version

I spent two days running 20+ experiments on the SOTA #549 stack, testing techniques nobody else had submitted results for. The main finding: Late QAT is dead code in #315-derived submissions due to a torch.compile bug. Fixing it actually made the int6 gap *worse*, not better. I also tested 7 novel techniques from recent papers (Muon-VS, Deep Delta Learning, stable depth recurrence, anti-layer removal, NS step reduction, spectral SVD compression, wider models). All negative. The throughput tax at this scale kills everything that adds overhead.

---

## The QAT Dead-Code Bug

PR #315's README flagged this already, but I wanted to confirm it and try to fix it.

The issue: `CastedLinear._qat_enabled` is a class attribute set to `False`. When `torch.compile(fullgraph=True)` traces the model, it sees `_qat_enabled=False` and eliminates the STE branch from the compiled graph. Setting it to `True` later (when Late QAT is supposed to activate) does nothing. The compiled code is frozen.

This is present in the SOTA (#549). I verified by checking step times: with the original code, step time stays flat at ~625ms throughout training. If QAT was actually running, you'd see a jump when it activates (STE adds overhead).

I tried three ways to fix it:

| Fix | What happened |
|---|---|
| Mutable list `[False]` | `FailOnRecompileLimitHit` error, `fullgraph=True` won't recompile |
| Instance attribute per module | Same recompilation error |
| Tensor multiplier (0.0 or 1.0) | Works. Graph shape stays the same, no recompile needed |

The tensor approach: `w = w + qat_scale * (w_q - w).detach()`. When scale is 0.0, it's a no-op. When 1.0, full STE. The compiled graph handles both without recompilation.

With the fix, step time jumps from ~625ms to ~670ms when QAT activates. So QAT is actually running now. But the results got worse:

| Metric | Dead QAT (original) | Working QAT (tensor fix) |
|---|---|---|
| val_bpb pre-quant | 1.3631 | 1.3646 |
| val_bpb int6 sliding | 2.3107 | 2.3411 |

Why? A few theories:

1. At only ~940 steps (1xH100), the model hasn't converged enough for STE to help. The weights are still moving too fast.
2. WD=0.04 + EMA 0.997 already push weights toward int6-friendly distributions without needing explicit fake quantization.
3. The tensor multiply `qat_scale * (w_q - w).detach()` always executes (even at scale=0.0), adding a tiny overhead that costs a few training steps.

I can't rule out that QAT would help at 7000 steps (8xH100). But it's also possible that the dead code is genuinely the right call for this stack.

---

## Novel Technique Sweep

All tested on the SOTA #549 stack on 1xH100. Baseline: 1.3631 val_bpb (pre-quant, slope=0.75, 947 steps).

| # | Technique | Paper | val_bpb | Delta | Why it failed |
|---|---|---|---|---|---|
| 1 | Muon-VS (variance-adaptive) | 2601.14603 | 1.3884 | +0.019 | Variance buffer needs 100s of steps to warm up, +2.2% overhead |
| 2 | Deep Delta Learning | 2601.00417 | 1.3720 | +0.003 | Beta params stay near zero at init, within noise |
| 3 | Thinking Deeper recurrence | 2603.21676 | 1.4163 | +0.053 | 21% step overhead, 163 fewer steps |
| 4 | Anti-layer removal | 2603.19348 | n/a | n/a | No anti-layers found in 11L stack |
| 5 | Newton-Schulz steps=3 | - | 1.3798 | +0.017 | Worse orthogonalization quality, +9 steps doesn't compensate |
| 6 | Spectral SVD (640d 12L) | Novel | 1.4646 | +0.102 | 46% slower, only 649 steps, SVD can't separate signal from noise |
| 7 | Wider model (576d 11L) | - | 1.4504 | +0.087 | 33% slower, only 711 steps |

Also confirmed LeakyReLU(0.75)^2 > LeakyReLU(0.5)^2 by 0.006 bpb, matching PR #977.

### The throughput tax

This keeps coming up: at ~83ms/step on H100, each millisecond of added overhead costs roughly 0.007 bpb. Techniques 1, 3, 6, and 7 all added overhead. None of them delivered enough quality improvement per step to compensate. The only technique that worked (LeakyReLU slope change) adds zero overhead.

### Anti-layer results (full table)

Trained 943 steps, then zeroed each layer's attn_scale + mlp_scale and re-evaluated:

| Layer | Delta bpb when removed | Role |
|---|---|---|
| 0 | +1.127 | Critical (embedding interaction) |
| 1 | +0.346 | Critical |
| 2 | +0.237 | Important |
| 3 | +0.087 | Moderate |
| 4 | +0.081 | Moderate |
| 5-6 | +0.036 | Low |
| 7-8 | +0.031-0.034 | Low (layer 8 least important) |
| 9-10 | +0.040-0.044 | Low |

No layer hurts performance when present. The LN Scale (1/sqrt(L+1)) in the stack may prevent anti-layer formation by dampening deep layers.

---

## Other Findings

**SWA during QAT:** PR #989 found SWA sabotages QAT. I added a one-line fix that stops SWA accumulation when QAT activates. Worth adopting if you use both.

**Prune-then-quantize:** Implemented but only partially tested. Prunes smallest-magnitude weights before int6 quantization (arXiv:2603.18426). One partial run showed neutral results. The pruned zeros don't compress better than quantized values under lzma, so the artifact size didn't shrink. Needs full evaluation.

---

## Reproducing These Results

From the repo root:
```bash
# Download data (if not already present)
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10

# Main run (slope=0.75, includes QAT tensor-scale fix)
RUN_ID=main \
LEAKY_SLOPE=0.75 \
torchrun --standalone --nproc_per_node=1 \
records/track_non_record_16mb/2026-03-28_QAT_DeadCode_Analysis_NovelTechniques_1xH100/train_gpt.py

# With prune-then-quantize (5%)
RUN_ID=prune5 \
LEAKY_SLOPE=0.75 \
PRUNE_FRACTION=0.05 \
torchrun --standalone --nproc_per_node=1 \
records/track_non_record_16mb/2026-03-28_QAT_DeadCode_Analysis_NovelTechniques_1xH100/train_gpt.py

# With anti-layer diagnostic (runs layer ablation after training)
RUN_ID=antilayer \
LEAKY_SLOPE=0.75 \
ANTILAYER_DIAGNOSTIC=1 \
torchrun --standalone --nproc_per_node=1 \
records/track_non_record_16mb/2026-03-28_QAT_DeadCode_Analysis_NovelTechniques_1xH100/train_gpt.py
```

Other toggles: `MUON_VS=1`, `DEEP_DELTA=1`, `RECURRENCE_LAYERS=2 RECURRENCE_STEPS=2`, `SVD_RANK=128 MODEL_DIM=640 NUM_LAYERS=12`.

---

## What I'd try next with more compute

1. QAT fix at 7000 steps (8xH100) to see if the STE actually helps when the model is better converged
2. Spectral SVD compression at full scale (the concept needs 7000+ steps for weight matrices to develop spectral structure)
3. Prune-then-quantize sweep (5%, 10%, 15%)
4. LeakyReLU slope sweep at 0.80, 0.85, 0.90

---

## Included Files

- `train_gpt.py` - modified SOTA #549 script with all techniques as env var toggles
- `submission.json` - metadata
- `train.log` - QAT fix run output (slope=0.75 with tensor-scale QAT, 1xH100, 943 steps)

## Acknowledgments

Built on PR #549 (@abaybektursun), PR #315 (@jfprincz, first to flag the QAT torch.compile issue), PR #977 (@michaelwinczuk, LeakyReLU 0.75), PR #989 (@alexanderaperry-arch, SWA-QAT finding). Issue #140 commentary by @notapplica was really helpful for prioritizing what to try.
