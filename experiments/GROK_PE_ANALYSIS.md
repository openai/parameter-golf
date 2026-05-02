# Grok's Positional Encoding Recommendations — Assessment

## Context
Our ROPE_BASE=200K test (exp066) showed ROPE_BASE change is NEUTRAL for BPB (1.1469 vs 1.1474).
This suggests standard RoPE at base=10K is already fine, and the issue isn't RoPE frequency.

## Grok's Recommendations Assessed

### 1. ALiBi — MODERATE RISK, worth testing
- Grok claims "+0.02-0.05 BPB" and "zero risk"
- BUT: ALiBi doesn't work with FlashAttention's efficient GQA path (needs attention bias support)
- F.scaled_dot_product_attention doesn't support attention bias with enable_gqa=True
- Would need custom attention kernel or flash_attn with bias — adds complexity
- Also: no one in the competition uses ALiBi. All top PRs use RoPE.
- **VERDICT: HIGH RISK despite Grok's confidence. Skip for now.**

### 2. DroPE (drop RoPE late in training) — INTERESTING but untested
- Drop RoPE after 90% of training + 200 step recalibration
- Theory: model learns to be position-agnostic → better sliding window extrapolation
- Risk: could destabilize in our tiny 7300 step budget
- 200 recalibration steps = ~16 seconds, fits in budget
- **VERDICT: MEDIUM priority. Test after we solve artifact size.**

### 3. Partial RoPE (50% of head dim) — LOW RISK, easy test
- Only apply rotary to first 50% of head dimensions
- Saves compute, keeps most positional signal
- Already proven in modded-nanogpt speedruns
- **VERDICT: WORTH TESTING. Easy change, low risk.**

### 4. QK-Norm — ALREADY HAVE IT
- We already have q_gain in attention. PR135 has it via qk_gain_init=1.5.
- Not much more to do here.
- **VERDICT: Already implemented.**

## Priority for future experiments
1. Partial RoPE (50% head dim) — easy, low risk
2. DroPE two-phase — interesting but risky with few steps
3. ALiBi — skip (incompatible with GQA flash attention)

## Key insight from our ROPE_BASE=200K test
RoPE base doesn't matter much for sliding window eval. The bottleneck isn't positional encoding —
it's model capacity and training steps. Focus on compression and more params, not PE tricks.
