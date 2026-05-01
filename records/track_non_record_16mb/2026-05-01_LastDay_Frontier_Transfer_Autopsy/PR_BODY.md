# Non-record: Last-Day Frontier Transfer Autopsy

This is a non-record methodology submission.  It does **not** claim a new
leaderboard score.

It documents my final-day attempt to transfer several plausible orthogonal
ideas onto the clean late frontier around PR #2018, plus the local probes that
informed those attempts.  The main purpose is to leave reviewers and future
participants with a precise negative-results package rather than a pile of
unexplained failed pod runs.

## Headline finding

On the current frontier, **pre-quant BPB is the first kill gate**.  The final
branches all failed before quantization/TTT could matter.

| Run | Base | Change | Seed | Pre-quant BPB | Quant BPB | Final BPB | Verdict |
|---|---|---|---:|---:|---:|---:|---|
| PR #2018 reference | #2018 | none | 1337 | **1.05124428** | 1.05990331 | **1.04826351** | target |
| Plan A | #2018 | Gate32 + q-aware token-only tilt | 1337 | 1.06385301 | 1.07199665 | 1.06057508 | no-go |
| Plan B | #2018 | Gate32 + native n-gram | 1337 | 1.06434971 | not run | not run | killed at pre-quant |
| Plan C | #2018 | native n-gram + BigramHash 512x4 + Path-A-v3 small | 1337 | 1.06471733 | not retained | not retained | killed at pre-quant |

The most important result is **where** these branches failed.  Gate32 damaged
the trained neural model itself.  Removing the q-aware patch did not fix it.
Reverting to exact #2018 gate settings and adding a tiny BigramHash branch also
failed to recover pre-quant quality.

## What is included

- full Plan A log and metrics,
- Plan B log through the pre-quant kill gate,
- terminal-observed Plan C pre-quant result,
- exact runner and patch scripts used for the final attempts,
- summary of earlier probes:
  - A40 structural smoke matrix,
  - CrossWS tokenizer train-proxy result,
  - fixed Memento/copy-memory no-go,
  - context horizon audit,
  - online bias/temperature probe,
  - artifact savings audit.

## Why this is useful

Several last-day ideas looked plausible from public PRs or local probes:

- gate widening,
- runtime n-gram/memory overlays,
- BigramHash-style input features,
- CrossWS tokenizer changes,
- small artifact-routing tricks.

This PR records which of those actually survived contact with the 8xH100
competition surface.  The answer was mostly "mechanically viable, not
BPB-transferable."

## Compliance posture

The executed runs use the CaseOps byte sidecar (`CASEOPS_ENABLED=1`), standard
causal transformer scoring, full-vocab normalized softmax / closed-form
renormalized token tilt where applicable, and inherited score-first TTT.  No
validation PreQuantTTT, no score-after-update, and no byte-only PPM scoring are
used.

## Cost note

After grant credits were exhausted, I used roughly `$150` of personal RunPod
spend across the final pushes.  I include this only as compute-accounting
context.  The real conclusion is methodological: when pre-quant is not
competitive, stop before quantization and TTT.

## Conclusion

This is not a record.  It is a record-shaped autopsy.

The clearest longer-horizon positive result remains CrossWS tokenization:
`original_crossws` gave a stable train-proxy token ratio of `0.94842` versus
default SP8192, with byte-denominator invariants passing.  It was not
operationally safe to finish before the deadline, but it remains the most
promising novel direction from this work.
