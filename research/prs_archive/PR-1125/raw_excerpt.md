# PR 1125 — Non-record: XSA-All + QK Gain 4.0 + LN Scale on 1×RTX 5090

**Author:** Pranjal Jain (@jainpranjal97)
**Branch date:** 2026-03-30
**Claimed BPB:** 1.1946 (60-min, 1×RTX 5090, int8+zlib)
**Artifact size:** 18.1 MB (exceeds 16MB cap — non-record)
**Seeds:** not stated (single-seed experiments)
**Hardware:** 1×RTX 5090 (32GB, Blackwell), vast.ai

## Files retrieved
- `README.md`
- `records__track_non_record_16mb__2026-03-30_XSA-All_QKGain4_LNScale_1x5090__README.md`
- `records__track_non_record_16mb__2026-03-30_XSA-All_QKGain4_LNScale_1x5090__train_gpt.py`

## Claimed changes (from README, verbatim)
"I ran 45 systematic experiments on a single RTX 5090 to find the best configuration for this architecture. Three findings stand out as potentially novel or underexplored:

1. XSA on ALL layers beats XSA on last 4 (-0.0018 BPB). Every top entry uses XSA on the deepest 3-4 layers. I found that applying XSA to every layer helps, even the shallowest ones.
2. qk_gain_init = 4.0 (-0.0039 BPB cumulative from default 1.5). Sharper initial attention patterns significantly help small models. I swept 1.5 → 2.0 → 3.0 → 4.0 with consistent gains.
3. Warmdown calibration for wallclock-capped training is critical. The default warmdown_iters=1200 with a 10-min cap means the LR never reaches full strength. Reducing to 200 gave -0.0078 BPB.

I also tested four novel architectural ideas (Progressive Layer Growing, Depth Recurrence + LoRA, Cosine Warmdown, XSA Gating) — all failed, with documented reasons why."
