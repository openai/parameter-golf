# Parameter Golf Council Brief — March 25, 2026 (End of Day)

## Where We Stand

**PR #175**: val_bpb = 1.1229 (3-seed mean, std 0.0005). March 19 timestamp. Clean, valid, pending review.

**Budget**: ~$30 remaining (~2 full runs).

**Competition frontier**: 0.9625 (n-gram cache), 1.0222 (Hedge Mixer + our VRL). Our 1.1229 is pure neural, no eval-time tricks.

## What We Tried Today (All Failed or Net Zero)

| Experiment | Result | Verdict |
|---|---|---|
| Gated Attention | 1.1239 vs 1.1229 base (+4ms overhead) | Net zero. Strip. |
| BigramHash 3072 | Artifact 16.12MB (over limit) | Doesn't fit. Keep 2048. |
| CROWN-Q quant penalty | Bundled with GA, no isolated gain | Net zero. Strip. |
| AdamW TTT (PR #688 recipe) | running_bpb never dropped below pre-TTT over 700 chunks | Dead. VRL stack rejects all weight-modifying TTT. |
| N-gram cache (our impl) | 1.167 bpb vs 1.124 base — WORSE | Alpha=0.3 too aggressive. Needs entropy-adaptive mixing. |

## The Key Insight: Hedge Mixer Bypasses Our TTT Problem

All our TTT failures share one root cause: modifying model weights mid-eval destabilizes VRL gates and SmearGate's compiled state.

Hedge Mixer (PR #688) doesn't modify model weights. It only updates scalar mixing weights via multiplicative updates. The transformer stays frozen. This bypasses every failure mode we've hit:
- No VRL gate desync (weights unchanged)
- No compiled graph invalidation (no weight mutations)
- No optimizer state issues (Hedge uses loss-based updates, not gradients)

PR #745 (1.0222 bpb) uses Hedge Mixer ON TOP of our VRL and gets massive gains. Their pre-TTT is 1.1348, ours is 1.1229 — our base is stronger.

## What We Need the Council to Research

### 1. PR #727's Exact N-gram Mixing Formula
Our n-gram cache implementation uses fixed alpha=0.3 which makes things worse. PR #727 gets 0.9674 with "entropy-adaptive alpha." We need:
- The exact formula for computing alpha per token
- How they handle the cold-start problem (few tokens scored = weak n-gram stats)
- Whether they use backoff (try 7-gram, fall back to 6, 5, ... unigram) or blend all orders simultaneously
- The exact mixing: linear interpolation `(1-a)*p_neural + a*p_ngram` or log-domain `logsumexp`?

### 2. Hedge Algorithm Implementation Details
From PR #688's 5-expert Hedge Mixer:
- How are expert weights initialized? (neural bias=2.0, others=0?)
- What learning rate (eta) for the multiplicative update?
- Is the update `log_w -= eta * loss` or `w *= exp(-eta * loss)`?
- How does the entropy expert work? It's not a proper probability model.
- Do they normalize expert weights after each update?

### 3. Legal Compliance: Causal vs Precomputed N-gram Tables
The council flagged this: building n-gram tables from already-scored eval tokens is clearly legal. But what about:
- Can we hash bigrams/trigrams into a fixed-size table or does it need to be exact counts?
- Is there a minimum count threshold before we trust an n-gram (e.g., count >= 2)?
- Do the top PRs (#727, #753) use smoothing (add-alpha) or raw counts?

### 4. Liger-Kernel Compatibility
The council recommended `pip install liger-kernel` for 20-43% throughput. But:
- Does it work with our custom CastedLinear (fp32 weights, bf16 forward)?
- Does it conflict with torch.compile?
- Does it work on the RunPod parameter-golf template (PyTorch 2.9.1)?
- Which specific ops should we fuse? (RMSNorm, linear+CE, residual+norm)

### 5. Can We Beat PR #745 With Just Hedge + Our Better Base?
PR #745's stack:
- Pre-TTT: 1.1348 (their neural base)
- Post-TTT with Hedge: 1.0222

Our pre-TTT: 1.1229 (0.012 better neural base)

If Hedge gives the same absolute delta, we'd hit ~1.0100. But there might be diminishing returns — a better base means less room for n-gram improvement. What should we realistically expect?

## Proposed Plan for Tomorrow

1. **Implement Hedge Mixer** (~170 lines, offline, $0)
2. **Add Liger-Kernel** to setup ($0)
3. **One test run** on fast pod: train + Hedge eval ($15)
4. **If it works**: 3-seed run, update PR #175 ($15)

Total budget: $30. Exactly what we have.

## Strategic Context

- Competition runs until April 30 (5 weeks left)
- N-gram techniques dominating the frontier (0.96-1.03)
- Our VRL contribution is being adopted by others
- PR #175 has the earliest timestamp of any competitive PR (March 19)
- If Hedge works: we could jump from 1.1229 to ~1.03-1.05 in one run
- If it doesn't: we still have 1.1229 as a valid non-record submission
