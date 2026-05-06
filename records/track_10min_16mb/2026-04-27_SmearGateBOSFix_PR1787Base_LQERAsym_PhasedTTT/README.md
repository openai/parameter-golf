# Record: SmearGate BOS Fix + PR #1787 Base + Smear Gate + LQER Asymmetric + Phased TTT

**val_bpb = 1.06128** | **~15.95 MB** | 8xH100 SXM

## Result

| Seed | Pre-TTT BPB | Post-TTT BPB | Artifact (bytes) |
|------|-------------|--------------|------------------|
| 42   | 1.07406     | **1.06128**  | 15,952,086       |

Merged SOTA (PR #1493): **1.0810 BPP**. Delta: **-0.0197 BPP**. Clears the 0.005-nat threshold.

## Key Change: SmearGate BOS Document Boundary Fix

Builds on PR #1797 stack (PR #1787 base + SmearGate + LQER Asymmetric) but fixes the **SmearGate cross-document leakage bug** identified by @cocohearts in PR #1797 audit.

The bug: SmearGate 1-token causal lookback does not mask BOS positions, so the final token of document N smears into BOS of document N+1.

The fix (applied in both forward_logits and forward_ttt):

    bos_mask = (input_ids[:, 1:] == 1).unsqueeze(-1)
    g = g.masked_fill(bos_mask, 0.0)

## Technique Stack

| Component | Origin |
|-----------|--------|
| CaseOps bijective case transform | PR #1729 / PR #1736 |
| SparseAttnGate | PR #1787 (nprime06) |
| SmearGate + BOS fix | PR #1797 + this submission |
| LQER asymmetric rank-4 | PR #1797 |
| Phased TTT (score-first, 3 phases) | PR #1394 / PR #1736 |
| PolarNS + MIN_LR=0.1 + FusedCE | PR #1787 |
| Full Hessian GPTQ + Brotli | PR #1019 / PR #1530 |

## Architecture

11L x 512d x 8H/4KV, MLP 4x, LeakyReLU(0.5)^2, Partial RoPE (16/64 dims), layerwise LN scale, tied embeddings, logit softcap=30.0. Depth recurrence: layers 3-5 looped x2 (activated at frac=0.35). Parallel residuals from layer 8. XSA on all 11 layers. SmearGate window=12.

## Compliance

- Artifact <= 16,000,000 bytes: 15,952,086 bytes
- train_time <= 600s: 599.6s
- eval_time <= 600s: 519.5s
- Issue #1017 Conditions 1-4: All satisfied. SmearGate BOS mask ensures no cross-document leakage.

## Credits

- @nprime06 -- PR #1787 base stack
- @romeerp -- CaseOps transform (PR #1729)
- @dexhunter -- SmearGate + LQER (PR #1797)
- @cocohearts -- Identifying SmearGate BOS bug
- @abaybektursun -- Score-first TTT (PR #549)
- @clarkkev -- GPTQ SDClip + SP8192 (PR #1394)
