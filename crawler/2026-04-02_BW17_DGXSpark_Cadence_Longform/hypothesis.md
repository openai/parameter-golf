# BW17_DGXSpark_Cadence_Longform — Hypothesis

Objective: exploit the now-stable `9F` crawler stack by testing cadence interactions quickly on local DGX-Spark, then replay only top candidates at full 600s.

Current stable base (from BWX full run):
- `NUM_FLAT_LAYERS=9`
- tap-off (`CRAWLER_TAP_DIM=0`)
- no anchor (`ANCHOR_DIM=0`)
- `NUM_CRAWLER_LAYERS=1`, `CRAWLER_LOOPS=3`, `INST_DIM=32`

Key assumptions:
1. Most remaining gains are in crawler cadence routing, not tap/anchor knobs.
2. Small-token rapid tests are enough to rank directionally good cadence variants.
3. Long-form replay is required before promotion decisions.

Test structure:

RAPID stage (WINDOW):
- `BW17DGX-00`: control cadence
- `BW17DGX-01`: loops=2
- `BW17DGX-02`: loops=4
- `BW17DGX-03`: loops=5
- `BW17DGX-04`: 2 crawler layers x 2 loops
- `BW17DGX-05`: rope cadence shift `(16,4,1)`
- `BW17DGX-06`: `INST_DIM=64`
- `BW17DGX-07`: loop smear on

LONGFORM stage (WINDOW):
- replay control + top-K rapid arms (default top-2) at 600s.

POST_WINDOW stage:
- quant-only bake-off on the best LONGFORM checkpoint:
  - naive int6
  - GPTQ standard
  - GPTQ-lite
  - optional loop-aware GPTQ

Promotion policy:
- Must beat LONGFORM control on `int6_sw_bpb`.
- Must remain legal on size for submission track.
- Quant policy promoted only if it improves the same LONGFORM checkpoint.
