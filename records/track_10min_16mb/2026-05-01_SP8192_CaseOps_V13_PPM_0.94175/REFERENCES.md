# References and lineage

This submission builds on public Parameter Golf ideas rather than claiming a new standalone architecture.

- PR #1991: SP8192 + byte-PPM tuned order/gate, `0.94290` three-seed mean. v13 keeps the same core PPM direction and retunes the final gate to `H=0.999`, `L=0.18`, `T=0.80`.
- PR #2014: SmearGate BOS leak fix and per-group compression notes. v13 includes the BOS mask and per-group `lrzip` compression path.
- PR #1795 / PR #1959 family: causal byte-PPM mixer and SP8192 neural distribution lineage.
- modded-nanogpt / Parameter Golf community submissions: Muon, sliding eval, aggressive quantization, and compact GPT training patterns.

The main novelty here is the small but repeatable v13 consolidation and final gate retune over the CaseOps sidecar-aware PPM lane.
