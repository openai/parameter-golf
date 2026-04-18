# BigramHash Embeddings

**Status:** candidate — **code already landed in our `train_gpt_sota.py` (commit `d529fe8`), but NEVER tested.** Spec 000 will intentionally disable it (`BIGRAM_VOCAB_SIZE=0`) so the baseline replicates the leaderboard SOTA without BigramHash. This idea becomes the follow-up candidate to test it on top.
**Expected Δ:** +0.003 to +0.005 bpb (prior submission's claim; highest expected Δ among our current candidates)
**Source:** One prior submission used a BigramHash 3072×112 table, claimed +0.005 bpb. Not in the current leaderboard SOTA (verified: `grep -i bigram` on `records/track_10min_16mb/2026-04-09_*/train_gpt.py` returns nothing).

## Idea
Small learned hash table mapping **bigrams** (pairs of adjacent tokens) to a low-dim vector, added to the token embedding. Size: 3072 hash buckets × 112 dims ≈ 344K params.

Rationale: the model already sees bigram statistics implicitly through the first attention layer, but a direct embedding gives it a shortcut for frequent pairs, freeing attention capacity for longer-range structure.

## Why it might help
- Prior submission claimed +0.005 bpb — the largest single Δ among our current candidate set.
- Parameter budget is modest: 344K params at INT6 ≈ 260 KB pre-Brotli (well under the 16MB budget even with the rest of the model).
- Simple mechanism: hash `(token_t, token_{t+1}) % buckets` → lookup → add to embeddings.
- **Already implemented in our code** — no new code to write, just enable.

## Current implementation in our code
- `Hyperparameters` (train_gpt_sota.py:96-98): `BIGRAM_VOCAB_SIZE` (default 3072), `BIGRAM_DIM` (default 112).
- `BigramHashEmbedding` class: L432-460. Hash function: `bigram_hash` at L450-452.
- Integration: L474 creates `self.bigram` if `bigram_vocab_size > 0`; L553-554 adds it to `x` in `forward_logits`.
- Env controls: set `BIGRAM_VOCAB_SIZE=0` to disable; defaults (3072, 112) match the prior-submission claim.

## Hotstart screening plan
Unlike Hessian-SDClip and per-group bit allocation, **BigramHash cannot be screened via hotstart** — it adds parameters to the model, so the weights from a no-BigramHash checkpoint can't be reused. Testing it requires a **full training run**.

However, we can still screen it cheaply before committing to an 8×H100 official:

- **Rung 1 (mini):** 2×H100 for ~30-40 min at reduced step count, same hyperparams except `BIGRAM_VOCAB_SIZE=3072`. Compare to a paired no-BigramHash baseline mini run. This is the Exp 24 methodology — not a submission, but signal on whether BigramHash moves pre-quant bpb in the right direction.
- **Rung 2 (official):** full 8×H100 10-min run with `BIGRAM_VOCAB_SIZE=3072`. Single seed to check, then 3 seeds if signal holds.
- **Cost per rung:** mini ≈ $4, official ≈ $3.50, full 3-seed ≈ $10.50.

**Alternative: skip the mini and go straight to 8×H100.** BigramHash has been on the leaderboard before (different submission lineage); the risk is low that it's outright broken. A single 8×H100 seed is $3.50 and gives a direct comparison against spec 000's 1.0810.

## Risks / open questions
- **16MB post-Brotli fit.** 344K extra params at INT6 = ~260KB pre-Brotli. The SOTA submission hits ~15.99MB after Brotli, leaving ~10KB of headroom. BigramHash would push the post-Brotli size up — **open question: by exactly how much?** Need to actually run the full quant + Brotli pipeline to measure. **This is the gating question** — if it doesn't fit, the candidate is dead.
- **Quantization behavior.** Embeddings are often kept at higher precision (SOTA uses GPTQ on embed already). Does BigramHash quantize cleanly at INT6? Our code quantizes it along with other matrices; worth verifying no artifact.
- **Interaction with SP8192 tokenizer.** Prior submission that used BigramHash was on a different tokenizer (likely SP1024 or SP4096). SP8192 has 2-8× larger vocab, so bigram combinatorics are ~64× more varied. 3072 hash buckets collides a lot more. Might need `BIGRAM_VOCAB_SIZE` larger (e.g. 4096 or 6144) for SP8192 — at the cost of budget.
- **Hash function quality.** Simple modular hash is likely fine. More sophisticated hashing isn't expected to matter.
- **Default was misleading.** Commit `d529fe8` message says "disabled by default" but the code default is 3072 (enabled). Easy to accidentally include BigramHash in a run that should have been a baseline. Spec 000 explicitly overrides this.

## Stacking with other candidates
- **Orthogonal to Hessian-SDClip** (Candidate 1, quant-time only). BigramHash + Hessian-SDClip should stack additively.
- **Orthogonal to Progressive Recurrence** (Candidate 2, training-dynamics). Stacks additively.
- **Interacts with Per-Group Bit Allocation** (Candidate 3). Both compete for the 16MB post-Brotli budget — if BigramHash adds, say, 200KB post-Brotli, bit-allocation has 200KB less headroom and may need to compensate by dropping some late blocks to INT5.

## Decision order for screening
Because BigramHash needs a full training run (no hotstart screen), screen it **after** the cheap-to-screen candidates (Hessian-SDClip, per-group bit allocation) have been evaluated. That way we know whether to run BigramHash in isolation or stacked with already-promoted winners.

Suggested order:
1. After spec 000 produces `ckpt_final_pre_ema`, hotstart-screen Hessian-SDClip and per-group bit allocation (cheap, ~$8 total).
2. Run BigramHash as its own spec on 8×H100 (seed 42). ~$3.50.
3. If BigramHash shows Δ ≥ 0.002 AND other candidates also promoted, run a final stacked record attempt (3 seeds, 8×H100).

## If this works
- Largest single contribution among our candidates (+0.003 to +0.005 bpb).
- Clears the path to a comfortable record: 1.0810 − 0.004 = ~1.077, well below any noise floor.
- The 260KB+ post-Brotli cost forces other candidates to be budget-conscious, which may constrain per-group bit allocation.
