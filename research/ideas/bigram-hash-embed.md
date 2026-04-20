# Idea — BigramHash auxiliary embedding (port from #1716)

**Status:** 📝 CANDIDATE, not yet specced. Recommendation: **test after 011**, likely bundled with BPB-weighted loss.
**Source:** PR #1716 (himanshudongre, val_bpb 1.07882, Z=−2.998 p=0.00136 vs #1493).

## Core

Add a small auxiliary embedding table (16384 × 32) keyed by a hash of the current + previous token pair. Project to model dim, add to the token embedding before block 0.

```python
hash_ids = (36313 * t[..., 1:] ^ 27191 * t[..., :-1]) % 16383
h = bigram_embed(hash_ids)      # (B, T-1, 32)
h = proj(h)                      # (B, T-1, 512)
x = tok_emb(ids) + bigram(ids)   # additive, first block input
```

Zero-init the projection so the model starts byte-identical to no-bigram baseline; gradient pressure discovers which bigram buckets matter.

## Why it might help on #1736

First-order token co-occurrence is the simplest useful signal any language model exploits. Standard transformers *must* learn this implicitly in attention + token embeddings. A 540K-param explicit bigram table offloads that work to dedicated capacity, freeing attention/MLP to learn higher-order structure. #1736 does **not** have one — verified by grep on the current code.

Prior art: bigram hash embeddings appear in many competitive submissions' lineages (#1716 itself, #1571, #1669, #1675, variants elsewhere). The technique is proven repeatedly; #1736's absence is genuinely unusual among the strong submissions.

## Why it might not

1. **Artifact budget.** 540K params × ~0.75 bytes/param (int6-ish) = ~400 KB. #1736 is already fitting ≤16 MB. Need a size dry-run before running — if we're already at ~15.9 MB, adding 400 KB puts us over.
2. **CaseOps collision profile.** #1716's numbers are on plain SP8192. #1736 uses CaseOps, which changes token IDs. The hash `(36313*t[...,1:] ^ 27191*t[...,:-1]) % 16383` will collide differently on CaseOps's vocab. Collision rate is probably fine (16384 buckets for 8192² pairs), but worth spot-checking.
3. **Absorbed by existing capacity?** #1736 has SmearGate + AttnOutGate + parallel residuals + depth recurrence. Some of that capacity may already be implicitly learning bigrams; adding an explicit table could be redundant.
4. **Training-time discovery vs init.** Zero-init means gradient pressure has to discover useful buckets in 4500 steps. On a 10-min run, that may or may not be enough to exploit the table.

## Legality

Clean. No TTT/SLOT/n-gram-cache-at-eval. The bigram embedding is a *training-time* structural addition — the learned weights ship as part of the model artifact, so eval is just normal forward-pass with an extra embedding. No byte-bug. Not on banned list.

## Implementation notes

**~30 LOC port:**

1. **New module** `BigramHashEmbedding(vocab_size=8192, buckets=16384, dim=32, proj_dim=512)`. Two parameters: `embed (16384,32)` + `proj (32,512)`. Zero-init `proj`.
2. **Hash computation** in `forward`: the `ids[..., :-1]` causal dependence is fine for Condition 1 (uses only past tokens). Edge case: first token has no previous — pad with a sentinel token or use `ids[..., 0]` twice. Check #1716's handling.
3. **Wire into GPT.forward** before block 0: `x = tok_emb(ids) + bigram_embed(ids)`.
4. **Quantization plumbing.** The 16384×32 table should quant as int6 like matrices; the 32×512 proj as a standard CastedLinear. Needs addition to the GPTQ Hessian collection path.
5. **Env-gated** (`BIGRAM_HASH_ENABLED=0/1`, default 0) so baseline is byte-identical when off.

## Cost

~$1 smoke + ~$20 full run = **~$21**. Comparable to spec 011.

## Recommendation on testing

**Yes, test it.** But sequencing matters:

1. Spec 011 (WD + GradPower) — currently queued
2. Spec 012 (softer QK_GAIN) — trivial, queued
3. **Spec 013 candidate: bundle BigramHash + BPB-weighted CE.** Both are orthogonal training-time levers, both add minor artifact budget (BigramHash: +400 KB; BPB-weighted: 0 KB). If artifact fits, bundle is cheaper than running sequentially.
4. If 013 regresses, run each alone to attribute.

Alternative: treat each as its own spec. BigramHash is lower implementation-risk (no CaseOps byte accounting headache); BPB-weighted has the SP8192 destabilization question. Running them separately gives cleaner attribution at +$20 cost.

**My recommendation:** sequential, not bundled. BigramHash first (spec 013), BPB-weighted second (spec 014). Reason: CaseOps byte accounting for BPB-weighted is ~1 hour of implementation + validation, while BigramHash is ~30 LOC of straightforward code. Faster to get BigramHash into a run, which means faster signal on whether the overall "add orthogonal arch lever" approach is working.

## Cross-references

- Source: PR #1716 (himanshudongre).
- Companion training-time levers: `research/ideas/gradpower-muon.md`, `research/ideas/per-layer-qk-gain.md`, `research/ideas/bpb-weighted-loss.md`.
- Queued behind: `research/specs/011-training-bundle.md`.
