# Idea — BPB-weighted training loss (port from #1519)

**Status:** 📝 CANDIDATE, queued behind specs 011 + 012. Recommendation: **test it**, but after 011 lands.
**Source:** PR #1519 (elliottdehn, open, non-record, created 2026-04-10).

## Core

Weight each token's cross-entropy loss by the number of UTF-8 bytes it encodes. Aligns training objective with the eval metric (bits-per-byte), which currently weights all tokens equally under standard CE.

Three lines:

```python
if self.training and hasattr(self, '_byte_weights'):
    per_token_loss = F.cross_entropy(logits.float(), targets, reduction="none")
    w = self._byte_weights[targets]
    return (per_token_loss * w).sum() / w.sum()
return F.cross_entropy(logits.float(), targets, reduction="mean")
```

`_byte_weights` is a buffer: `base_bytes_lut.float().clamp(min=1.0)`, token_id → byte-count mapping.

## Why it might help on #1736

Under uniform CE, a token "the " (4 bytes) contributes the same gradient as a single-byte punctuation token, but at eval it contributes 4× to BPB. The model's gradient signal is therefore misaligned with what we actually care about. BPB-weighting pushes the gradient toward multi-byte tokens that carry the majority of eval-metric weight.

Class T (training-time loss). Upstream of TTT. Should survive TTT absorption.

## Why it might not (or why it's risky)

1. **Vocab-size transfer risk.** Author explicitly verifies this works on SP1024 (gentle byte-length variance, 1-8×) and **fails on GPT-2-50K** (extreme variance destabilizes training). #1736 uses SP8192 — between the two extremes. Plausibly safe, not guaranteed.
2. **CaseOps breaks the naive byte-mapping.** #1736's CaseOps tokenizer emits case-flag sidecar tokens (e.g., one token for the chunk + a case token that together produce some number of source bytes). A naive `_byte_weights[token_id]` LUT will misattribute bytes between the chunk token and its sidecar. The weighting would be biased in a way that may or may not matter.
3. **Claimed Δ is vs ancient baseline.** Author's 2×RTX 5090 data shows −0.019 bpb vs a pre-#1493 baseline. On top of #1736's 1.0655, expect a much smaller delta — maybe −0.002 to −0.005 if it transfers, null if vocab/CaseOps break the assumption, positive if it destabilizes.

## Legality

Community-reviewed clean (MatoTeziTanka: "LOOKS CLEAN — pure-neural submission, no TTT/SLOT/n-gram-cache"). Training-time CE reweight doesn't touch Issue #1017's eval-time conditions (causality, full distribution, score-before-update). Not on any banned list.

## Implementation notes (for when this becomes a spec)

**~30-60 min of careful work, not 3 lines as in the source PR:**

1. **Byte-weight LUT construction.** #1736's tokenizer is CaseOps. Need to compute `bytes(token_id)` correctly — specifically, for case-flag sidecar tokens, decide whether the bytes "belong" to the main chunk, the flag, or are split. Simplest approach: for every `token_id`, decode the token to its surface string via the tokenizer, compute `len(surface.encode('utf-8'))`, and store. For case-flag tokens that emit no visible bytes alone, assign 0 (or 1 with `clamp_min(1)` per author).

2. **Where to register the buffer.** After model creation, before training starts. Must be on `device` and float.

3. **Env-gated** (`BPB_WEIGHTED_LOSS=0/1`, default 0) so the code is byte-identical to baseline when disabled.

4. **Verify `base_bytes_lut` already exists in #1736's code path.** If it's already being computed for eval, we can reuse it. If not, we need to build it in the training path — still trivial.

5. **Smoke test is critical.** Instability on SP8192 is the headline risk. 2×H100 short run to confirm train_loss curve is well-behaved.

## Recommendation

**Test it, but queue behind specs 011 and 012.**

Order of operations:
1. Spec 011 (WD taper + GradPower) → lands or doesn't
2. Spec 012 (softer QK_GAIN) → lands or doesn't
3. **Spec 013 (BPB-weighted CE)** → run on whichever 011/012 winning config looks best, or on bare #1736 if neither lands

Reasoning:
- This is arguably the most *principled* change in the whole candidate pool (objective-alignment > optimizer tweak on expected Δ). Worth pursuing.
- But it has real implementation work (byte LUT for CaseOps) and real regression risk (SP8192 destabilization). We want a known-good stack to add it to.
- Running it in parallel with spec 011 confounds attribution. Sequential is cheaper in interpretation cost even at same dollar cost.

Rough cost: ~$1 smoke + ~$20 full = **~$21**. Comparable to spec 011.

## Cross-references

- Source: PR #1519 (elliottdehn), community review in comments.
- Queued behind: `research/specs/011-training-bundle.md`, `research/ideas/per-layer-qk-gain.md` (→ spec 012).
- Related training-time levers: `research/ideas/gradpower-muon.md`.
