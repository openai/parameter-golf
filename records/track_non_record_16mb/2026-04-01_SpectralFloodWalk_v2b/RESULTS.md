# V2b Results

`V2b` was the strongest version of the "coprocessor grows the model in VRAM during eval" idea that we implemented, but it stayed an exploratory branch rather than turning into a submission candidate.

## What Held Up

- Hidden-space online memory was the right seam. It let the host transformer stay fixed while the runtime state changed.
- Read gating mattered. Delaying reads until a slot had enough scored evidence reduced the early-chunk tax.
- The tooling around it became solid:
  - staged matrix generation
  - RunPod wrappers
  - queue scripts for multi-GPU pods
  - FLOP accounting for lookup, update, and maintenance

## What Failed

- Attention-style maintenance across slots was the wrong operator.
  - It blurred sharp corrections into neighboring slots.
  - Extra maintenance FLOPs made the memory smoother, not better.
- Even after shifting to replay-style sharpening, the broader line still looked like a mechanism probe more than a leaderboard path.
- The late gains were real but modest, and they did not justify how much engineering and eval budget the branch consumed.

## Why This Matters

`V2b` was still useful because it clarified what not to do:

- do not spend extra compute diffusing memory entries together
- do not assume "more runtime structure" automatically means better predictions
- do keep the host fixed and look for sharp, local corrections instead of global smoothing

That negative result helped motivate the later committee benchmark: use hardware to maintain multiple useful hypotheses in parallel, rather than forcing one online memory structure to carry the whole burden.
