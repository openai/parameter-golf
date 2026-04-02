# BWX_Latest_2k — Hypothesis

Objective: run a focused validation pass with the latest strong crawler stack.

Chosen stack for BWX:
- Tap-off Nightcrawler
- `NUM_FLAT_LAYERS=9` (best depth that remains under a 16MB-ish artifact guardrail in current results)
- Post-window quant policy sweep on frozen checkpoint

Hypothesis:
- `9F + GPTQ` beats `9F naive int6` while keeping artifact size operational.

Decision policy:
- Quality winner by `int6_sw_bpb`
- Reject if size becomes operationally unacceptable for target track.
