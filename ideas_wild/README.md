# Wild Ideas

This folder saves a second-pass set of more aggressive, challenge-specific ideas for the Parameter Golf baseline.

These are not primarily baseline-safe improvements. They are the higher-variance ideas I would consider if the goal is a real record attempt rather than a tidy iteration on the current script.

Files:

- `01-recurrent-transformer-with-few-unique-blocks.md`
- `02-train-compressed-parameterization-directly.md`
- `03-eval-time-iterative-refinement.md`
- `04-stateful-recurrent-memory.md`
- `05-mixture-of-depth.md`
- `06-tiny-artifact-resident-cache.md`
- `07-hypernetwork-generated-layer-deltas.md`
- `08-shared-base-plus-low-rank-deltas.md`
- `09-joint-tokenizer-and-model-bpb-optimization.md`
- `10-learned-dequantization-at-load-time.md`
- `probably-too-dangerous.md`
- `single-best-wild-bet.md`

Summary of why these ideas are interesting:

- The current baseline still improves when it hits the 10-minute stop.
- The same architecture keeps improving far beyond that in the 4-hour run.
- The baseline export path gives back a meaningful amount of the gain.
- Evaluation time is tiny relative to the challenge cap, so test-time compute is underused.
