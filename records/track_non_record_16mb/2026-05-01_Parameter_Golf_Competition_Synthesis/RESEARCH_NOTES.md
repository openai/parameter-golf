# Research Notes

These notes are the reasoning layer behind the synthesis.  I include them so
the PR is a reusable research map rather than only a narrative summary.

## 1. BPB is an endpoint, not an explanation

Many late submissions report one scalar: final BPB.  That scalar is necessary,
but insufficient.  Two runs with the same BPB can have very different meaning:

- a better neural model under the same scoring protocol,
- a different tokenization that changes token entropy,
- a byte-denominator bug,
- a legal causal cache,
- or an illegal score-after-adapt loop.

For future competitions, I would recommend storing a small decomposition table:

| Component | Diagnostic |
|---|---|
| Train quality | post-EMA / pre-quant BPB |
| Quantization damage | quantized no-TTT BPB minus pre-quant BPB |
| Eval adaptation gain | post-TTT BPB minus quantized no-TTT BPB |
| Tokenizer effect | tokens per original UTF-8 byte |
| Denominator integrity | exact byte sidecar sum |

This makes it much harder for a denominator effect to masquerade as modeling
progress.

## 2. Why pre-quant became the final-day kill gate

The late clean frontier had already optimized quantization and TTT heavily.  A
typical strong run could lose several millibits at quantization and regain a
few millibits through TTT.  But my final branches lost roughly 13 millibits
before quantization.  That is too large for the downstream machinery to erase.

The practical rule:

```text
If a frontier branch is >0.003 BPB worse pre-quant on the same seed, be wary.
If it is >0.010 BPB worse pre-quant, stop unless you have a new eval mechanism
with proven legal gain of that size.
```

This is not a theorem.  It is an expensive empirical prior.

## 3. Why runtime memory gave false positives

The Memento/copy-memory idea originally looked attractive because exact repeated
spans have high precision when they fire.  The bug was subtle: shallow-context
scoring made the neural model look weaker on those positions than it really was.

After fixing prefix-depth accounting, the repeated-span positions had high
neural probability already.  The memory expert then had asymmetric economics:

- on hits, it improved a token the neural model often already knew;
- on misses, it paid a full penalty;
- at deeper context, the miss penalty dominated.

The lesson is not "memory is useless."  It is that eval-time memory must be
measured against the same context horizon the final model actually uses.

## 4. Why token-level n-gram tilt is cleaner than byte PPM

Token-level tilt keeps the scored alphabet fixed:

```text
p'(a) = exp(beta * 1[a = h]) * p(a) / Z
```

Every official token receives probability mass, and the normalization constant
is explicit.  This gives a direct C2 argument.

Byte-level PPM may be strictly causal and useful, but it changes the scoring
alphabet.  That can be a perfectly interesting compression idea, but the
leaderboard needs a maintained policy on whether the metric is token-autoregres-
sive or byte-autoregressive.

## 5. Why CrossWS remains interesting

Cross-whitespace BPE changes how SentencePiece can merge common function-word
bridges such as "of the", "in the", and "to be".  In local probes it produced a
stable `~5.15%` token-count reduction without needing case normalization.

The reason I did not submit it is also the reason it is scientifically
interesting: the hard part is not training a tokenizer; the hard part is
proving that the transformed validation stream is scored against the exact
original UTF-8 denominator with no split leakage and no SentencePiece sentinel
bug.

This is a good future target because the hypothesis is simple, the effect size
is large enough to matter, and the legality standard can be made mechanical.

## 6. What I would do next with more compute

I would not start with another architecture tweak.  I would build a reference
custom-tokenizer harness:

1. stream raw docs once to define train/val by row index,
2. train tokenizer on train rows only,
3. export token shards and byte sidecars for val,
4. run adversarial round-trip tests,
5. run a small neural stack to measure whether token-count savings survive as
   BPB savings.

Only after that would I spend another 8xH100 run.
