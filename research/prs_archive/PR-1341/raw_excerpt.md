# PR 1341 — TTT and GPTQ Are Fundamentally Incompatible

**Author:** himanshudongre
**Claimed BPB:** not stated (research/analysis submission); LoRA TTT experiment -0.0013 BPB
**Artifact size:** not stated
**Seeds:** not stated
**Track:** non_record_16mb
**Hardware:** 1xH100 80GB (RunPod)
**Compute cost:** $5

## Files retrieved
- `records__track_non_record_16mb__2026-04-04_TTT_GPTQ_Incompatibility__README.md`
- `records__track_non_record_16mb__2026-04-04_TTT_GPTQ_Incompatibility__submission.json`
- `records__track_non_record_16mb__2026-04-04_TTT_GPTQ_Incompatibility__clark_ttt_eval.py`
- `records__track_non_record_16mb__2026-04-04_TTT_GPTQ_Incompatibility__sgd_ttt_eval.py`

## Claimed changes (from README, verbatim)
"Test-time training (TTT) provides substantial BPB improvement on simple quantization but is fundamentally ineffective on GPTQ-quantized models. Evidence table: PR #461 (SGD, simple int6) -0.0165; PR #601 (SGD, GPTQ int5) +0.030 WORSE; This work (LoRA rank-8 Q,V, GPTQ int6) -0.0013; PR #1326 (Score-first SGD, GPTQ int6) -0.0001. Root Cause: GPTQ's Compensatory Weight Structure — each quantized weight compensates for errors in previously quantized weights; SGD updates individual weights based on local gradients, ignoring the compensatory structure, destroying error cancellation. Proposed Fix Directions: (1) Quantization-aware TTT with full-precision master weights + re-quantization; (2) Structured TTT respecting GPTQ block boundaries; (3) Higher-rank LoRA (32, 64); (4) Simple int6 + larger model. SGD TTT implementation: momentum=0.9, lr=0.002, cosine decay, 32K-token chunks, 3 epochs/chunk, freeze first 2 blocks, grad clip 1.0."
