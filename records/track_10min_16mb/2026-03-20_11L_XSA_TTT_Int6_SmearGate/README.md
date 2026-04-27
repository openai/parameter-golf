# WIP: Depth Recurrence + SwiGLU + XSA-all + Parallel Residuals + AR GPTQ + Legal TTT

## Status: Verified runs coming mid-April

Building toward a sub-1.08 submission. Script in active development, incorporating the latest proven techniques.

## Planned Architecture

Combining the strongest signals from the current frontier:

| Component | Source | Impact |
|-----------|--------|--------|
| **3-layer depth recurrence** (layers 3,4,5) | PR #1331, #1445 | 14 virtual layers from 11 physical |
| **SwiGLU FFN** | PR #462 | Smoother loss landscape for TTT |
| **XSA on all layers** | PR #1019, #478 | Better than XSA-4 |
| **Parallel residuals** (layers 7+) | PR #1412 | Improved gradient flow |
| **EMA** (decay ~0.9965) | PR #1421 | Cleaner quantization |
| **AR self-generated GPTQ** | PR #1019 | Better calibration than STE QAT |
| **Legal score-first TTT** | PR #461 | Causal-legal eval-time adaptation |
| **SP8192 tokenizer** | PR #1394 | Larger vocab helps |
| **Partial RoPE** (16 dims) | PR #398 | Proven marginal gain |
| **LN Scale** | PR #398 | Layer-wise normalization scaling |
| **N-gram tilt** (causal, token-only) | PR #1420 | Eval-time boost |

## Prior Results (unoptimized, March 20)

Ran an earlier version of the script (pre-depth-recurrence, pre-GPTQ) on 8xH100:
- **val_bpb: 1.1429** (sliding window, stride=64) — tied verified #1 at the time
- Artifact was 16.1MB (over limit due to WD=0.04, now fixed)

## Credits

Built on shoulders of: @abaybektursun (PR #549, #1019), @JoeProAI (PR #462), @dexhunter (PR #1437), @X-Abhishek-X (PR #1445), @felipe-parodi (PR #398), @sjp611 (PR #442)

## Checklist
- [x] Submission folder
- [x] README.md
- [x] submission.json
- [x] train_gpt.py (base script, updating)
- [ ] Training log (mid-April)
- [ ] Verified BPB score (mid-April)
