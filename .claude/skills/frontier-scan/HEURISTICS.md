# Frontier-scan heuristics

Used by the frontier-scan skill to classify each PR. Update this file when new
rulings land or new patterns emerge.

## Our baseline

**PR #1736** (dexhunter) — val_bpb **1.06549** (3-seed mean).
Stack: SP8192 + CaseOps tokenizer + SmearGate + AttnOutGate + QuantGate +
Loop45 depth-recurrence + phased TTT (multi-phase global SGD + doc-indep LoRA).

## Legitimacy categories

### `clean`
No open disputes. Training and eval procedures follow competition rules.

### `tokenizer-disputed`
Uses a non-standard tokenizer. Apply Issue #1604 lossless-roundtrip test:
- **Likely legal:** lossless CaseOps — `decode(encode(s)) == s` holds; byte
  sidecar recovers original bytes; exact byte count recoverable from token IDs.
- **Likely illegal:** lossy casefold (`.lower()`) — destroys case information;
  byte count cannot be reproduced from token IDs alone.

### `prequant-ttt-disputed`
Issue #1017 Condition 3: "you may not test-time train on validation data before
the artifact is frozen." A pre-quant TTT pass runs AdamW/SGD updates on val
tokens, then freezes weights — this is the disallowed pattern.
- Physics ceiling check: bigbag's empirical corpus-level TTT ceiling is
  **~0.0003 bpb**. Any submission claiming >0.005 bpb from TTT-related levers
  deserves scrutiny. A claim of ~0.038 bpb (as in #1735) is ~100× over the
  ceiling — assign 85-90% probability of illegal ruling.

### `byte-bug-suspect`
GatedDeltaNet / FLA cluster (Issue #1719): `build_sentencepiece_luts` double-
counts the byte denominator by ~17.46%. Corrected bpb is ~18% higher than
claimed. PRs: #1698, #1711, #1712, #1734 (three self-closed). Flag any PR
using flash-linear-attention (`from fla.` import) for this check.

### `broken`
Artifact oversized, training crash, or claim definitively refuted.

### `other`
Non-record track, WIP, negative-result, or reproduction only. Skip unless it
contains an extractable novel lever.

## Absorption rule

Training-time levers (architecture changes, optimizer changes) are at risk of
being absorbed by #1736's phased TTT — TTT adapts the model at eval time and
can compensate for a missing training-time improvement. Quant-side levers have
been empirically shown absorbed (specs 009/010/010b). Eval-time levers
(Tap-In, score-first TTT, hash embeddings trained only via TTT) are downstream
of TTT and **cannot be absorbed**.

## Already-in-#1736 (exclude from "novel lever" classification)

SP8192 tokenizer, CaseOps, SmearGate, AttnOutGate, QuantGate, Loop45, phased
TTT (multi-phase SGD + LoRA), VarLen attention + fused MLP (TMA), Muon
matrix_lr=0.026, per-layer adaptive GPTQ clip (MLP/ATTN/MATRIX sigmas), int8
embeddings, logit_softcap, XSA (`F.normalize(v, dim=-1)`), RMSNorm Q/K,
`leaky_relu(x,0.5).square()` MLP activation, GPTQ mixed int6/int8.

## Already-specced (don't re-flag as new)

| Spec | Lever |
|------|-------|
| 011  | Tapered weight decay |
| 012  | GradPower Muon p=0.9 |
| 013  | xIELU + per-layer QK gain |
| 014  | Tap-In min_match=1 |

## Banned mechanisms (always illegal)

- Lossy casefold (Issue #1604)
- Pre-quant TTT (Issue #1017 Condition 3)
- N-gram eval cache
- Trinity/SLOT
- GatedDeltaNet / FLA byte-bug cluster — treat any `from fla.` import as a
  red flag requiring byte-accounting audit before accepting claimed bpb.

## Actionable-delta threshold

Flag a PR as **actionable** if its clean claimed bpb is below our baseline
**1.06549**. Flag as **watch** if between 1.065 and 1.075. Above 1.075 is
background noise unless it contains an isolated novel lever.
