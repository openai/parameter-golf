# BWX_Latest_2k — Hypothesis

Objective: run the primary BWX contender sequence using latest confirmed signals from BW12..BW16.

Assumptions from latest runs:
- BW12/BW13: tap-off architecture is better than tap-shared in this regime.
- BW13: anchor on tap-off regresses quality (`ANCHOR_DIM=32/64` both worse), so contender path keeps `ANCHOR_DIM=0`.
- BW12/BW13: post-window GPTQ gives consistent gain over naive int6 (about `-0.0018` to `-0.0020` `int6_sw_bpb`).
- BW14: depth floor shift (`NUM_FLAT_LAYERS=6`) was a high-amplitude win vs its control (around `-0.007`).
- BW16: deeper flats (6..11) improved bpb monotonically but raised model size quickly; 8F is the practical contender threshold, 9F+ is size-risky.

Arm structure:

WINDOW arms (full train, run first):
- `BWXLT-00`: viable contender control (`tap-off`, `ANCHOR_DIM=0`, `NUM_FLAT_LAYERS=8`, naive int6).
- `BWXLT-06`: big-swing retest (`NUM_FLAT_LAYERS=6`) to confirm large architecture effect is still present.
- `BWXLT-07`: sanity below contender.
- `BWXLT-09`: sanity above contender (kept late because size/time risk).

POST_WINDOW arms (no retrain, run second on best WINDOW checkpoint):
- `BWXLT-Q0`: naive int6 replay (`SKIP_GPTQ=1`).
- `BWXLT-Q1`: standard GPTQ (`128x2048`).
- `BWXLT-Q1L`: GPTQ-lite (`64x1024`).
- `BWXLT-Q2`: loop-aware GPTQ (optional; only when explicitly enabled).

Why this ordering:
- WINDOW first resolves architecture/depth branch decisions before spending time on quant policies.
- POST_WINDOW second cheaply ranks quant policies on the strongest available checkpoint.

8x full-run promotion policy:
- Pick best WINDOW arm by lowest `int6_sw_bpb`, with practical preference for 8F-class size/speed unless a lower bpb result clears risk guardrails.
- Promote quant policy only if it beats `Q0` on that same checkpoint by a clear margin (target at least `-0.0008`; expected `Q1` range ~`-0.0018..-0.0020`).
- Execute 8x full run as: `best WINDOW architecture + winning quant policy`; keep runner-up depth/quant combo as fallback only if the winner violates size or runtime constraints.
