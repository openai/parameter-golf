---
title: Compute Grant Draft
read_when:
  - You are filling the OpenAI Parameter Golf compute-credit form.
---

# Recommended grant

Request the `Development grant (~$500 / ~160 compute hours)`.

Reason:

- concrete approach exists
- local experimentation completed
- not near the top leaderboard yet, so `Advanced competitor` would be overstated

## Form draft

### Brief description of your approach

I’m pursuing a compression-first Parameter Golf approach on the published SP-1024 setup. I’ve already reproduced the repo workflow locally, added tooling to optimize the scored post-quant roundtrip metric, and built a packaging flow that turns a finished run into a PR-ready `records/...` folder with logs, metadata, and the exact trainer snapshot. My next step is to start from the published 9-layer / 512-dim / KV4 baseline, run tightly scoped 1xH100 sweeps on `QK_GAIN_INIT`, `LOGIT_SOFTCAP`, learning-rate splits, and sequence length, then focus on reducing post-quant degradation with the quantization controls I exposed in the trainer. Winning configs will be promoted to 8xH100 confirmation runs and then packaged for submission.

### What have you tried so far?

Forked and instrumented the repo, built a local MLX smoke loop, downloaded the published SP-1024 data slice, added post-quant parsing and packaging tools, and validated one end-to-end local smoke run plus baseline-log parsing.

### Link(s) to your PR submission

Use your prep PR or compare link once you push this branch.

Suggested placeholder format:

```text
https://github.com/sanky369/parameter-golf/compare/main...YOUR_BRANCH
```

### What are you going to do with it?

Use the credits for a staged Parameter Golf sweep: 1xH100 baseline reproduction, 1xH100 low-risk sweeps on compression-sensitive knobs, then a small number of 8xH100 confirmation runs on the best candidates. The goal is a clean PR-ready submission with a better post-quant score under the 16MB cap, plus enough repeated evidence to justify a serious leaderboard attempt.
