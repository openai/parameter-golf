---
name: Competition state apr6
description: Parameter Golf SOTA is 1.1147 BPB (PR #1019), our baseline train_gpt.py has most Tier 1-2 techniques
type: project
---

As of 2026-04-06:
- **SOTA**: 1.1147 BPB by abaybektursun (PR #1019) — AR Self-Gen GPTQ + XSA-all + BigramHash 3072×112
- **Our train_gpt.py baseline** already includes: 11L, 512d, MLP 3x LeakyReLU², int6 quant, XSA last 4, BigramHash 2048×128, Muon optimizer, EMA, late QAT, sliding window eval, zstd-22, SmearGate, partial RoPE 16/64, LN scale
- **Key gaps vs SOTA**: Full Hessian GPTQ (we have naive int6), XSA-all (we do last 4), BigramHash 3072×112, LZMA compression, selective pruning, VE128, U-Net skips not leveraged, Parameter Banking
- Previous experiments (on different hardware): baseline ~1.3463, int6+mlp3x ~1.3526, sliding window gave free -0.034 BPB

**Why:** Tracking what the frontier looks like so we can prioritize experiments effectively.
**How to apply:** When suggesting next experiments, focus on gaps between our baseline and SOTA.
