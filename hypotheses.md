# Parameter Golf Hypotheses

## Objective

Use AlphaEvolve to minimize `val_bpb` under the actual track constraint: less than 10 minutes of training on `8xH100`, with the artifact still fitting under `16MB`.

This folder now seeds from the Enigma Stage 5 single winner `H533_shape_beta2`, not from the original pure baseline. The reason is simple: in short-horizon runs, early optimizer adaptation matters more than late asymptotics, and `H533` was the cleanest robust Stage 5 single that improved the Muon path without needing a large stack.

## Seed Hypothesis

### H1. Shape-aware Muon beta2 warmup should help more in Parameter Golf than in NanoChat

- Mechanism: rectangular Muon matrices use a lower beta2 early, then warm back to the baseline beta2.
- Why it transferred from Enigma: Stage 5 showed that rectangular and square-ish matrices did not want the same variance schedule.
- Why it may matter even more here: the challenge is dominated by short training, aggressive width constraints, and lots of projection matrices with different aspect ratios. That is exactly the regime where a better early variance estimate can matter.

## Main Follow-On Hypotheses

### H2. H533 is probably stronger on MLP projection matrices than on attention projections

- The rectangular matrices in this trainer are mostly expansion and contraction paths.
- In a tiny model, those layers carry a large share of representational growth early.
- AlphaEvolve should test narrower scope variants that target only `fc`/`proj` MLP matrices or only attention output projections.

### H3. The best Stage 5 helper is still worth testing next: `H535`

- Mechanism in Enigma: apply the epsilon schedule only to embedding-like AdamW groups.
- Why it is relevant here: embeddings are a large fraction of the parameter budget in Parameter Golf, and tied embeddings make their optimizer behavior especially consequential.
- Priority: medium. It was weak solo in Enigma, but materially helpful in composition.

### H4. Short-run wins should come from denominator and variance-shaping, not large momentum stacks

- The 10-minute regime is mostly an early-training problem.
- Good candidates: beta2 warmups, epsilon schedules, first-step denominator seeding, and other changes that alter optimizer bootstrap.
- Lower priority: deeper late-phase schedules, multi-stage resets, or large composite stacks.

### H5. Any optimizer mutation that costs more than a small step-time penalty is suspect

- The wallclock cap is hard.
- A mutation that improves loss but slows step time enough to cut total steps will often lose on final `val_bpb`.
- AlphaEvolve should treat throughput regressions as first-class negative evidence.

## Negative Knowledge

- Do not assume large stacks add cleanly. Stage 5 repeatedly showed that bigger composites often got worse.
- Do not port `H64` by default into the new family. It was explicitly harmful once `H533/H535` entered the stack.
- Do not prioritize `H538` yet. It was unstable across Stage 5 batches.
- Do not spend early search budget on late-warmdown ideas. This challenge is dominated by the first several thousand steps, not the last phase of a 20k NanoChat run.

## AlphaEvolve Search Priorities

1. Keep `H533_shape_beta2` as the seed and search local variants first.
2. Prefer single-surface mutations over broad rewrites.
3. Score ideas by `val_bpb`, realized steps before the wallclock cap, and artifact size.
4. Retire any mutation family that slows training materially without a same-run validation gain.

## Concrete Variant Queue

1. Change the rectangular threshold from `2.0` to nearby values.
2. Change `H533` warmup length and start beta2.
3. Restrict `H533` to MLP rectangular matrices only.
4. Add an embedding-only epsilon schedule on top of `H533`.
5. Seed Adam second moments for embeddings, but only after the `H533` family is characterized.

## Caveat

This port is intentionally minimal. Parameter Golf uses a much simpler trainer than NanoChat, so the current implementation carries over the core Stage 5 idea: Enigma-style Muon variance reduction plus shape-aware beta2 warmup. It is not a byte-for-byte reproduction of the NanoChat optimizer stack.
