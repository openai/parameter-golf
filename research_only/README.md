# Research-Only Boundary

This directory is intentionally outside the default submission launcher and preset path.

Rules for this lane:
- `research_only/run.py` is the only runner in this directory.
- Research-only runs never emit submission-readiness artifacts.
- Research-only presets never appear in `research/run.py`.
- Research-only results must always say `submission_safe=false`.

Keep anything here clearly labeled as speculative or uncertain with respect to the competition rules or spirit. In particular, do not let this directory become a hidden dependency of the stable or challenger workflow.

Default path excludes the following as core submission strategy:
- brute-force seed searches
- hidden offline meta-optimization whose main win comes from prior compute
- external checkpoints or assets fetched during evaluation
- saved-state generation from many prior runs as the main submission mechanism

Possible research-only topics, if explored transparently:
- warm-start studies with explicit provenance
- test-time adaptation variants that need careful rule review
- aggressive evaluation-time tricks that may be legal but need scrutiny
- hybrid attention + SSM / retention sequence mixers

If a direction graduates from here into the main workflow, document why it is self-contained, reproducible, and clearly within both the written rules and the spirit of the challenge.
