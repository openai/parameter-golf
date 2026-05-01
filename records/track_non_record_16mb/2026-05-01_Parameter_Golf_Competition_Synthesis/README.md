# Non-record: Parameter Golf Closing Synthesis

**Track:** non-record / methodology
**Author:** Himanshu Dongre
**Date:** 2026-05-01
**Leaderboard claim:** none

This is not a record submission.  I am submitting it because my final record
attempt did not deserve to be dressed up as a win, but the work around it still
left useful evidence.

The goal is to leave behind the map I wish I had earlier in the competition:
which mechanisms actually moved the 10min/16MB frontier, which ideas failed
when moved from local probes to 8xH100, which results need legality review, and
which lessons seem useful outside this leaderboard.

The memo combines:

- my own final-day 8xH100 negative results,
- local CPU/A40/H100-side probes run during the final week,
- and public evidence from late PRs and issues through PR #2103.

If you only read one file for competition review, read `MAINTAINER_BRIEF.md`.
If you are interested in how these lessons translate to small model research
outside the leaderboard, read `OPENAI_RESEARCH_BRIEF.md`.
If you want the chronology of how my own view changed across record and
non-record PRs, read `PARTICIPATION_ARC.md`.

I am not trying to declare winners among disputed open PRs.  I am trying to
separate the evidence into review buckets: clean neural improvements,
tokenizer and byte-accounting-sensitive improvements, eval-time memory/PPM
methods, validation-adaptation methods, and infrastructure lessons.

## What I Think Is Useful Here

The parts that seem worth preserving are:

1. **A decomposition of BPB gains by evidence class.**  The late leaderboard
   mixed neural modeling gains, tokenizer/normalization gains, eval-time
   causal-memory gains, denominator bugs, and validation-adaptation leakage.
   Treating them as one scalar leaderboard obscures the science.  This memo
   separates them into mechanisms that can be reviewed independently.

2. **A practical "pre-quant kill gate" for frontier experiments.**  My final
   runs show that, on the late frontier, a branch that is already about
   `+0.01 BPB` worse before quantization is not rescued by GPTQ or TTT.  This
   gives future experimenters a cheap stop rule and reduces wasted 8xH100 time.

3. **A corrected interpretation of eval-time memory.**  Simple copy/memory
   overlays can look strong when the neural model is accidentally scored with
   shallow context.  Once prefix-depth is fixed, the neural model is already
   confident on many repeated spans, so the memory expert mostly pays miss
   penalties.  This is a useful warning for future "working memory" ideas.

4. **A tokenizer direction with a testable legality standard.**  Cross-whitespace
   BPE produced a stable `~5.15%` token-count reduction on train-proxy and
   val-derived samples.  I did not submit it because the legal burden is the
   full raw-byte exporter and sidecar harness, not the token-count proxy.  That
   distinction matters.

5. **A reviewer checklist that makes legality debate more concrete.**  The
   proposed checks for scored alphabet, byte denominator, score/update
   intervals, and tokenizer provenance are meant to turn "this feels legal" into
   evidence someone else can inspect.

6. **A bridge from competition hacks to general small-model research.**  The
   companion `OPENAI_RESEARCH_BRIEF.md` argues that the real transferable
   object is not any one trick, but the full compressed prediction system:
   representation + weights + quantizer + memory + evaluator + legality
   protocol.

7. **A chronology of updated beliefs.**  `PARTICIPATION_ARC.md` links my
   earlier record and non-record PRs, including closed early attempts, to the
   final conclusions here.  I include it so the research process is auditable.

## For Maintainers

My read of the late competition is:

1. **The clean neural frontier is real and incremental.**  The strongest
   obviously neural-only line improved through better 10-minute optimization,
   gated attention/XSA variants, LQER/AWQ/asymmetric quantization, longer
   context around 2048-2560, and score-first phased TTT.  Public examples
   include PR #1855, #1953, #2014, #2018, #2041, #2060, and #2101.

2. **Most large apparent jumps came from representation or evaluation
   semantics rather than larger neural models.**  CaseOps/casefold, byte
   sidecars, token-only n-gram tilt, and byte/PPM mixtures dominate the late PR
   feed.  These are the ideas with the largest apparent effect sizes, but also
   the ones where a
   one-line BPB table is least sufficient.

3. **Pre-quant BPB became the best early kill gate.**  On the final frontier,
   quantization and TTT can recover a few millibits when the trained model is
   already strong.  They do not rescue a branch that is +0.01 BPB worse before
   quantization.  My final Gate32 and BigramHash transfer attempts failed at
   this exact gate.

4. **Artifact compression mattered, but did not hide unlimited capacity.**
   Per-group compression, lrzip/brotli, small-code wrappers, and careful
   tensor routing were essential.  However, semantic-preserving repacking gave
   only about 10 KB on my frontier blobs; extra model capacity still had to
   earn its bytes.

5. **The review burden shifted from "what is the BPB?" to "what distribution
   was actually scored?"**  Issues #1017, #1604, #1719, #1872, #897, and #43
   collectively define the hard part of the leaderboard: causal state,
   normalized distributions over the scored alphabet, score-before-update,
   single-pass evaluation, tokenizer normalization, and byte accounting.

## Caveats

This is not an official ruling on any open PR.  It is a participant's synthesis
from public PR text, issue discussions, and my own logs.  Where a technique is
marked "high risk" or "needs ruling," that means exactly that: I think the
mechanism is interesting, but the leaderboard should not rely on my judgment
alone.

## One-page Map of the Frontier

| Evidence class | Typical gain size | Legality risk | My read |
|---|---:|---:|---|
| Neural/quant/TTT stack | millibits to ~0.02 BPB | low to medium | Most trustworthy, but increasingly saturated. |
| Custom tokenizer / normalization | up to several percent tokens-per-byte | medium to high | Highest clean longer-horizon lever if byte sidecars are exact. |
| Token-level closed-form n-gram tilt | small to medium | low if fully timed and normalized | Good legal eval-time use; gains shrink on strong bases. |
| Byte/PPM mixtures | very large in public claims | high / unresolved C2 | Mechanistically interesting, but needs maintainer ruling on scored alphabet. |
| PreQuantTTT on validation | very large | high / generally unsafe | Useful as an upper-bound signal, not as record evidence. |
| Artifact-only repacking | KB to tens of KB | low | Necessary craft, rarely a new modeling lever by itself. |

## A Useful Taxonomy of Techniques

### 1. Clean Neural / Quantization Frontier

These are ordinary model/training/quantization changes where the evaluated
predictor remains a standard full-vocabulary neural distribution plus legal
score-first TTT.

Representative public PRs:

| Area | Examples | Why it mattered |
|---|---|---|
| Gated attention / XSA / SmearGate | #1855, #2018 | Lets a small model selectively route attention signal without adding much eval cost. |
| LQER / AWQ-lite / asymmetric quantization | #1855, #1953, #2060, #2101 | Reduces quantization damage under the 16 MB cap. |
| Long-context eval around 2048-2560 | #1953, #2014, #2060 | Helps until RoPE/context mismatch and eval time erase the gain. |
| No-QV / local TTT retunes | #1953, #2060 | Score-first TTT gains survive when adaptation targets and LR are tuned. |
| In-timer token-only n-gram tilt | #2018, #2041 | Small but real if implemented as a closed-form normalized token distribution. |

My interpretation: this is the safest part of the leaderboard.  Gains are
usually small, but the legality story is straightforward when the code and logs
show the three budgets and C1-C4.

### 2. Tokenizer and Text Representation

Tokenizer-side work was one of the biggest levers because it changes tokens per
byte without necessarily changing the model artifact.

Public context:

- Issue #43 confirms the important practical fact that tokenizer files are not
  counted in the 16 MB artifact in the same way code/model bytes are.
- Issue #1604 shows that lossy normalization, casefolding, and tokenizer
  transform legality are unresolved policy questions.
- CaseOps-style sidecars became a major late-lineage idea.

My own finding:

- Cross-whitespace SentencePiece (`split_by_whitespace=False`) gave a stable
  token-count ratio around `0.9466-0.9484` versus default SP8192 on val-derived
  and train-proxy samples.
- The 10 MB train-proxy result was:

| tokenizer | tokens | tokens/byte | ratio |
|---|---:|---:|---:|
| default SP8192 training | 2,880,110 | 0.26126 | 1.00000 |
| cross-whitespace SP8192 | 2,731,553 | 0.24778 | 0.94842 |

Why I did not submit it as a record:

- a custom tokenizer must have a full raw-doc export pipeline,
- validation byte sidecars must be exact,
- byte-fallback pieces and U+2581 handling are easy to get wrong,
- and the deadline did not leave enough time to run the full data export and
  3-seed training package safely.

Future recommendation: custom tokenizers are worth serious research, but they
need a reference exporter and a byte-accounting test harness before any BPB is
trusted.

### 3. Eval-Time N-gram and PPM Methods

This was the most explosive and most legality-sensitive class of ideas.

There are at least three distinct mechanisms that should not be conflated:

1. **Token-level closed-form n-gram tilt.**  A prefix-only hint chooses a token
   `h`, the model distribution `p` is tilted by `exp(beta)` on that token, and
   the full SP vocabulary is renormalized:

   ```text
   p'(a) = exp(beta * 1[a = h]) * p(a) / Z
   Z = 1 + p(h) * (exp(beta) - 1)
   ```

   This has a clean C2 story because it remains a full normalized distribution
   over the official token vocabulary.

2. **Byte-level PPM mixtures.**  These can be strictly causal and
   score-before-update, but their C2 status depends on whether the competition
   accepts scoring a byte distribution rather than a token distribution.  Issue
   #1872 captures the open concern.

3. **Realized-token byte decompositions.**  Some methods spread token
   probability across the bytes of the realized token.  These need especially
   careful review because a scalar derived from the realized token probability
   is not automatically a normalized distribution over the scored alphabet.

Representative late PRs:

| PR | Claimed mechanism | Review posture |
|---|---|---|
| #2018 | strict token-only n-gram tilt, in timer | strongest clean token-level form I saw |
| #2041 | inside-timer n-gram on V21 base | useful reproduction of in-timer eval recipe |
| #1991 / #2083 / #2098 | byte/PPM mixtures with large gains | high-impact but C2/byte-alphabet questions remain |
| #2039 | conditional byte PPM with marginalization argument | more careful than naive byte spread; still needs maintainer ruling |
| #2103 | SP1024 value residual + PPM mix, single H100 note | not directly comparable without 8xH100 record evidence |

My own Memento/copy-memory result ended as a no-go.  After fixing a
sliding-window scoring bug, longer-prefix neural context already explained the
copy events, and the memory overlay turned negative at deeper context.

### 4. Validation Adaptation / PreQuantTTT

Validation adaptation produced some of the most impressive reported numbers,
but also the clearest rule danger.

The safe pattern is:

- score a token or chunk first,
- update only after that score is committed,
- let the update influence only future tokens/chunks.

The unsafe pattern is:

- adapt on validation tokens,
- then report scores for those same tokens after the adapted state has already
  been shaped by them.

Issue #1017 is the right lens here.  PRs such as #1972 explicitly describe
multi-epoch full-pass AdamW on validation tokens as part of the scored pipeline;
that is not the same thing as score-first TTT.

Recommendation: future rules would benefit from a tiny reference evaluator that
logs `(score_start, update_start, token_range)` intervals and mechanically
flags score-after-update overlap.

### 5. Byte Accounting and Denominator Bugs

This competition showed that denominator bugs can look like algorithmic
breakthroughs.

Important public threads:

- Issue #1719: double-counted leading-space bytes in `build_sentencepiece_luts`
  deflated BPB for a family of submissions.
- Issue #897: custom SP models without a standalone U+2581 token can overcount
  the SentencePiece space sentinel as three UTF-8 bytes instead of one space
  byte.
- CaseOps/custom-tokenizer sidecars must sum exactly to original UTF-8 bytes.

My recommendation is simple: every tokenizer or byte-sidecar PR should include:

```text
assert decode(encode(text)) == text
assert sum(byte_sidecar) == len(text.encode("utf-8"))
assert every validation token contributes exactly one score term
```

and the tests should cover byte fallback, NUL, U+2581, multi-byte Unicode,
empty documents, BOS boundaries, and document packing.

## What Consistently Worked

These are the ideas I would carry forward into future small-model work:

1. **Start with a strong SP8192/CaseOps-class text representation.**  Tokens per
   byte is not everything, but it sets the playing field.

2. **Use model capacity where it survives quantization.**  Gated attention,
   XSA, LQER, and carefully chosen recurrence/looping survived the 16 MB cap
   better than many exotic architectures.

3. **Optimize for wallclock, not steps alone.**  The best stacks used nearly all
   600 seconds of training and still left enough eval time for legal TTT.

4. **Treat quantization as part of training, not an afterthought.**  AWQ-lite,
   LQER, asymmetric logit rescale, and GPTQ settings often moved the final BPB
   more than small architecture changes.

5. **Use eval time causally.**  Score-first TTT and normalized token-level
   n-gram tilt are legitimate ways to spend the 600-second eval budget when
   implemented inside the timer.

6. **Kill weak runs at pre-quant.**  On the late frontier, a +0.01 pre-quant
   regression was already unrecoverable.

## What Often Failed

1. **Architecture novelty without frontier transfer evidence.**  Many ideas
   compiled and fit, but did not improve BPB under the real 8xH100 schedule.

2. **Longer context beyond the trained operating point.**  My context audit
   found 2048 best on a local slice; 4096 was worse, and 4096+TTT was not
   budget-safe.

3. **Runtime memory overlays after strong context.**  Copy/memory caches looked
   good when the neural model was artificially context-starved, then collapsed
   when prefix depth was fixed.

4. **Compression-only hopes.**  Repacking can save KBs, but not magically add a
   new large module under 16 MB.

5. **Validation adaptation that rescored adapted tokens.**  Large gains here
   are not evidence for a legal record unless score-before-update is enforced.

## Proposed Review Checklist

For a record PR, I would ask for the following table in every submission:

| Gate | Required evidence |
|---|---|
| Artifact <= 16,000,000 bytes | exact `code_bytes + compressed_model_bytes` per seed |
| Train <= 600s | wallclock per seed, not step count only |
| Eval <= 600s | all preprocessing inside the timed region unless rules say otherwise |
| C1 strict causal dependence | state at token t cannot depend on token >= t |
| C2 normalized distribution | explicit scored alphabet and normalization argument |
| C3 score-before-update | logs or code proving no overlap between scored and adapted ranges |
| C4 single pass | no best-of-k validation rescoring or post-hoc selection |
| Byte denominator | exact original UTF-8 byte accounting, especially for custom tokenizers |
| Tokenizer provenance | train/val split rule and sidecar-generation rule |
| Seed policy | fixed seed set, no replacement after seeing val BPB |

## My Own Final-Day Contribution in Context

My companion non-record folder,
`2026-05-01_LastDay_Frontier_Transfer_Autopsy`, contains the reproducible
negative evidence from my final paid attempts:

- Gate32 + q-aware token-only tilt on #2018: no-go.
- Gate32 + native #2018 n-gram: no-go at pre-quant.
- Exact #2018 gates + tiny BigramHash + Path-A-v3 small routing: no-go at
  pre-quant.
- CrossWS tokenizer: promising but not deadline-ready.
- Memento/copy memory: corrected no-go.

I include these not as an apology for missing the record, but as evidence for a
general lesson: when the leaderboard is this compressed, the most valuable
thing a participant can contribute may be a clean map of what *not* to spend
the next 8xH100 hour on.

## Closing Thoughts

Parameter Golf was less a single model-design contest than a stress test for
the boundary between modeling, compression, tokenization, evaluation, and
causality.  The strongest submissions did not win by one trick.  They won by
stacking many small, budget-aware decisions while keeping the legality story
coherent.

The next version of this competition would benefit from:

- a reference byte-accounting harness,
- a reference score-before-update harness,
- an explicit policy on tokenizer normalization and sidecars,
- a standard timing boundary for eval-time precomputation,
- and a public non-record lane that rewards well-measured negative results.

That last point matters.  A clean failed experiment can save the community more
compute than a shaky record claim consumes.
