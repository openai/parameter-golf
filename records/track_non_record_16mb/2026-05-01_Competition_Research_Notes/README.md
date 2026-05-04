# Non-record: Competition Research Notes

**Track:** non-record / methodology
**Author:** Himanshu Dongre
**Date:** 2026-05-01
**Leaderboard claim:** none

These are research notes from the 10min/16MB track.  They are separate from my
final-run evidence package in PR #2110.  This folder does not contain a scored
model, logs, or a leaderboard claim.

The aim is to describe what I think the competition taught us about small
models, tokenization, quantization, eval-time memory, and benchmark semantics.
I have tried to keep this grounded in public PRs/issues and in my own failed
experiments, without pretending to give an official ruling on any open PR.

## The Main Split

By the end, final BPB was no longer enough to understand a submission.  Similar
numbers could come from very different mechanisms:

```text
reported BPB =
  neural model quality
+ quantization damage or recovery
+ tokenizer / normalization choice
+ byte-denominator accounting
+ legal eval-time memory
+ validation-adaptation protocol
+ timing boundary choices
```

Mixing those effects into one leaderboard number made review hard.  The notes
below are organized around that split.

## 1. Clean Neural and Quantization Work

The clean neural frontier was incremental but real.  The repeated ingredients
were:

- gated attention / XSA / SmearGate,
- LQER, AWQ-lite, and asymmetric quantization,
- longer context around 2048-2560,
- score-first phased TTT,
- no-QV or retuned local TTT,
- per-group compression and artifact-aware tensor routing.

Public examples include PR #1855, #1953, #2014, #2018, #2041, #2060, and
#2101.

This was the easiest part of the leaderboard to reason about.  The evaluated
object stayed close to a standard causal neural model over the token vocabulary.
The gains were smaller than the PPM/representation jumps, but the legality
story was clearer.

The practical lesson I got from my own final runs is also simple:

```text
pre-quant BPB is the first serious kill gate on a mature stack
```

In PR #2110, my branches were about `+0.013 BPB` worse than the PR #2018
reference before quantization.  That was already enough to stop.

## 2. Tokenizer and Representation Work

Tokenizer-side work was one of the largest levers.  It also had the highest
burden of proof.

The public context:

- Issue #43 discusses tokenizer artifact accounting.
- Issue #1604 discusses tokenizer normalization, casefolding, and CaseOps-style
  transforms.
- Issue #897 and Issue #1719 show how byte-denominator bugs can create large
  phantom gains.

My strongest unfinished tokenizer result was CrossWS:

| tokenizer | tokens | tokens/byte | ratio |
|---|---:|---:|---:|
| default SP8192 training | 2,880,110 | 0.26126 | 1.00000 |
| cross-whitespace SP8192 | 2,731,553 | 0.24778 | 0.94842 |

That number came from a 10 MB train-proxy slice decoded from an official train
shard.  The effect was stable on val-derived samples around `0.9466-0.9484`.

I did not finish it as a record because a tokenizer result needs more than a
token-count table:

- raw-doc train/val split by row index,
- tokenizer trained only on train rows,
- exact validation byte sidecar,
- byte-fallback handling,
- U+2581 handling,
- adversarial Unicode round-trip tests,
- and full validation byte sums.

The research signal remains interesting.  Standard whitespace-splitting
assumptions appear to leave capacity unused for small compressed models.

## 3. Eval-Time N-gram and PPM Methods

Eval-time memory was the most sensitive part of the late competition.  I would
separate it into at least two categories.

### Token-level tilt

The cleanest form is a prefix-only token hint with closed-form renormalization:

```text
p'(a) = exp(beta * 1[a = h]) * p(a) / Z
Z = 1 + p(h) * (exp(beta) - 1)
```

This keeps a full normalized distribution over the SP token vocabulary.  PR
#2018 and #2041 are useful public references for this style of method.

### Byte-level PPM

Byte-level PPM can be strictly causal and score-first.  The open question is
C2: whether the scored alphabet can be bytes rather than the official token
vocabulary.  Issue #1872 is the main thread I would read here.

PR #1991, #2039, #2083, #2098, and #2103 all belong in this broader family,
with different arguments for how the byte distribution relates to the neural
token distribution.  The mechanisms are interesting.  The policy question is
separate from the engineering question.

## 4. Runtime Memory Needs a Better Gate

I spent a lot of time on the idea that eval-time working memory might be
underpriced: the artifact is capped, but cache/RAM at eval time is not.

The first copy-memory probe looked promising.  After fixing a sliding-window
prefix-depth bug, the gain collapsed or turned negative at deeper context.

The reason was instructive:

- repeated spans are high precision when they fire,
- a strong long-context neural model already predicts many of those spans,
- memory hits add little when the model already knows,
- memory misses still cost probability mass.

The principle I would carry forward:

```text
External memory should be gated by expected improvement over the model's own
distribution, not by memory confidence alone.
```

This applies beyond the competition.  Retrieval and caches for small assistants
need to know whether the base model is already confident.

## 5. Validation Adaptation

Score-first TTT is useful when the update affects only future tokens.  The
unsafe pattern is adapting on validation tokens and then reporting scores for
those same tokens after the adapted state has seen them.

For adaptive submissions, I would want score/update intervals in the logs:

```text
score token range:  [a, b)
update token range: [c, d)
assert updates affect only strictly future score ranges
```

That turns a vague legality argument into something inspectable.

## 6. Byte Accounting

Byte accounting was not a side detail.  It defined the metric.

For any custom tokenizer or sidecar method, the basic invariants should be:

```text
decode(encode(text)) == text
sum(byte_sidecar) == len(text.encode("utf-8"))
each original byte is counted exactly once
each scored token contributes exactly one score term
```

The tests should cover byte fallback, NUL, U+2581, multi-byte Unicode, empty
documents, BOS boundaries, and packed documents.

## 7. What Transfers Beyond Parameter Golf

The benchmark is artificial, but the pressure it creates is real.  It asks:

```text
How much prediction quality can be bought with:
  - small persistent weights,
  - limited training time,
  - bounded inference time,
  - quantized artifacts,
  - strict causal scoring?
```

That resembles small OSS models, local models, cheap specialist models, and
adaptive assistants.

The useful object is the full compressed prediction system:

```text
representation + weights + quantizer + memory + evaluator + update protocol
```

I would study those pieces together rather than treating the tokenizer,
quantizer, and evaluation state as afterthoughts.

## 8. What Does Not Transfer Directly

I would not build a production small language model by copying the final
competition stack unchanged.

Outside the competition, the best path would likely include tools the contest
mostly rules out or makes unattractive:

- longer training,
- larger training mixtures,
- supervised and preference tuning,
- distillation from a much larger teacher,
- synthetic data from stronger models,
- architecture search with more than one 10-minute shot,
- and latency/throughput constraints measured on real serving hardware.

Distillation is the clearest example.  Under the competition rules, a large
teacher is hard to use because all useful training has to fit inside the
600-second training budget or inside the submitted artifact.  In ordinary small
model work, a large teacher can supply soft targets, reasoning traces, data
selection, and curriculum.  I would expect that to dominate many of the tiny
last-day leaderboard knobs once the rules allow it.

So the claim here is narrower:

```text
Parameter Golf is not the best recipe for training a production small LM.
It is a useful stress test for compressed prediction systems.
```

The parts I think transfer are the systems lessons:

- tokenizer and representation matter,
- quantization has to be part of model design,
- pre-quant/quant/post-adaptation metrics should be logged separately,
- eval-time memory needs calibrated gating against the base model,
- and adaptive benchmarks need explicit score/update timing.

## 9. Model Memory vs Working Memory

One unusual feature of this competition is the split between persistent model
memory and eval-time working memory.

The persistent artifact is capped at 16 MB.  That strongly limits model weights,
code, and anything shipped as part of the predictor.  Eval-time working memory
is different.  During validation, the program can use H100 memory, KV cache,
temporary lookup tables, online statistics, and other prefix-derived state, as
long as it stays causal and finishes inside the eval-time budget.

That makes Parameter Golf different from a normal deployed LLM:

| Setting | Persistent model memory | Working memory / inference state |
|---|---|---|
| Parameter Golf | very expensive, capped at 16 MB | comparatively cheap until eval time runs out |
| Production serving | expensive, but amortized across many users | also expensive because it drives latency, KV cache, batch size, and serving cost |

This explains why eval-time n-gram caches, PPM-style memory, TTT state, and
large temporary statistics were so tempting in the competition.  They spend the
resource that the rules price least directly.

For production, that tradeoff changes.  A method that wins by using large
working memory may be unattractive if it increases latency or reduces batch
throughput.  Techniques that compress KV cache, reduce activation memory, or
speed up inference can be more valuable in production than they look in this
contest.

The research question I would take forward is:

```text
Given a fixed serving budget, what should live in persistent weights and what
should live in per-request working memory?
```

Parameter Golf put almost all pressure on persistent weights.  Production
systems need both sides to be efficient.

## 10. Claims I Would Test Next

The notes above can be turned into testable claims.  These are the ones I would
prioritize.

### Claim A: representation first

In this competition, tokenization and representation often moved the target
more than another small gate, rank, or learning-rate tweak.  My CrossWS result
is one example, not a proof.  I would test whether this remains true after
byte accounting is fully controlled.

Test:

```text
Fix the architecture, quantizer, training time, and eval protocol.
Train several byte-exact tokenizers on the same train rows.
Report tokens/byte, pre-quant BPB, quantized BPB, and eval latency.
```

The important part is to keep the byte denominator exact.  Otherwise the test
measures accounting, not modeling.

### Claim B: memory needs marginal pricing

The repeated-span cache looked good until I fixed prefix depth.  Then the base
model already knew many of the cache hits.

Test:

```text
For every memory event, log:
  memory confidence
  model probability on the memory hint
  realized hit/miss
  loss delta after normalized mixing
```

If a memory method cannot predict positive marginal gain before seeing the
token, it is not a memory policy.  It is a hopeful cache.

### Claim C: search the deployed model

A BF16 improvement that disappears after GPTQ is not useful for a 16 MB model.

Test:

```text
For each candidate:
  pre-quant BPB
  quantized BPB
  post-adaptation BPB
  artifact bytes
  eval seconds
```

Then rank by the deployed tuple, not by pre-quant loss alone.

### Claim D: distillation outside the rules

The competition mostly prevents a large teacher from being useful because the
teacher has to be trained or encoded within the budget.  In normal small-model
training, a teacher can shape data, targets, curriculum, and error correction.

Test:

```text
Train the same small quantized architecture with:
  CE only
  teacher soft targets
  teacher-selected data
  teacher-generated hard negatives
  teacher reasoning traces when applicable
Compare after quantization as well as before.
```

My expectation is that distillation would beat many of the last-day
hyperparameter tricks, while the competition's quantization and tokenizer
lessons would still matter.

### Claim E: serving cost decides memory placement

Parameter Golf made persistent memory scarce and working memory relatively
cheap.  Production makes both expensive, but in different units.

Test:

```text
For a target latency and batch size, compare:
  more weights
  longer context / larger KV cache
  retrieval or cache memory
  online adaptation state
  smaller weights plus better tokenizer
```

Report quality per dollar or quality per token-second, not BPB alone.

## My Research Arc

These notes also reflect how my own view changed during the competition.

| Date | PR | Type | Lesson |
|---|---:|---|---|
| 2026-03-26 | #826 / #846 | closed record attempts | Large eval-time memory numbers need strict scoring semantics first. |
| 2026-03-28 | #1012 / #1013 | non-record | Synthetic and SSM/JEPA-style successes did not transfer cleanly. |
| 2026-04-01 | #1227 | non-record | Small-scale tests lie when they miss the real bottleneck. |
| 2026-04-02 | #1259 | non-record | Retrieval that helps weak models can vanish on strong contextual models. |
| 2026-04-03 | #1301 | non-record | Mechanical novelty is cheap; frontier transfer is hard. |
| 2026-04-04 | #1341 | non-record | Adaptation and quantization have to be designed together. |
| 2026-04-15 | #1642 | non-record | Legal eval-time memory can still be a null result. |
| 2026-04-18 | #1716 | record attempt | Small causal input features can help in the right base. |
| 2026-04-18 | #1718 | non-record | Ablations are necessary to avoid copying ingredients blindly. |
| 2026-04-30 | #1965 | record candidate | Tail seeds matter; fixed seed policy matters. |
| 2026-05-01 | #2110 | non-record | Final frontier transfer failed at pre-quant. |

The through-line is that the work moved from tricks toward measurement:
mechanism, denominator, legality, quantization, and hardware budget.

## Source Map

| Item | Why it matters |
|---|---|
| Issue #1017 | C1-C4 framing: causal dependence, normalized distribution, score-before-update, single pass. |
| Issue #1604 | Custom tokenizer normalization and casefold/CaseOps policy. |
| Issue #43 | Tokenizer artifact accounting. |
| Issue #897 | U+2581 / byte-fallback denominator bug. |
| Issue #1719 | Leading-space byte double-count bug. |
| Issue #1872 | Byte-level PPM-D mixture legality question under C2. |
| PR #1855 | Merged late SOTA with LQER, SparseAttnGate, BOS-fixed SmearGate, per-group compression, phased TTT. |
| PR #1953 | Long-context 2560, no-QV TTT mask, local LR 0.75, QK_GAIN 5.25. |
| PR #2018 | Gated XSA + LQER top-1 + strict token-only in-timer n-gram TTT. |
| PR #2041 | V21 + inside-timer n-gram TTT without Gated XSA. |
| PR #2060 | LongCtx/no-QV/AsymLogit/LQER retune. |
| PR #2101 | AWQ-lite + AsymLogit + GradCentral + LabelSmooth. |
| PR #1991 / #2083 / #2098 | Byte/PPM mixture line with large claimed gains and C2 sensitivity. |
| PR #1972 | PreQuantTTT line, useful as a warning about validation adaptation. |

## Closing

The main thing I would keep from this competition is the systems view.  A tiny
language model is a representation, a set of weights, a quantizer, a memory
policy, an evaluator, and an update protocol.

Most confusing results came from mixing those pieces without saying which one
actually moved.  Most useful results made the split visible.
