# Hypothesis: RASCAL_SLOT_H2H_2K
Date: 2026-04-01
Track: neural
Parent: neural/2026-03-31_Rascal_III_SLOT/

Goal: settle whether SLOT quality gain is accompanied by artifact growth on the same trained weights.

Method:
- train once for 2000 steps
- export once
- log shared serialized/model artifact bytes once
- run sliding-window eval twice on the same checkpoint:
  - H2H_BASE: SLOT disabled
  - H2H_SLOT: SLOT enabled

Interpretation:
- If `h2h_sliding_window_slot8steps_exact` beats `h2h_sliding_window_base_exact` while shared bytes stay fixed, SLOT quality gain does not require extra artifact bytes on that checkpoint.
- If quality and bytes somehow both move inside this one-run H2H, that would indicate a real serialization-path bug.
