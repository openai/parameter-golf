# Ablation: RASCAL_WINDOWN_TESTING
Date: 2026-03-31
Track: neural
Parent: neural/2026-03-30_Rascal_II

## Suite Gate (1-GPU, ~120s, seed=444)
Status: [ ] pending  [ ] complete

### Results table (fill after run_suite.sh completes)

| Arm | int6_sw_bpb | delta_vs_ctrl | Verdict |
|-----|-------------|---------------|---------|
| CTRL-00 | | — | control |
| SLOT-01 | | | |
| SCALE-02 | | | |
| SLOT+SCALE-03 | | | |

SLOT-01 expected delta: ~−0.0057 (proxy prior). If wildly different, investigate.
SCALE-02 signal threshold: < −0.0005 to proceed to 8×GPU confirmation.

### Scale TTT failure modes to watch for
- SCALE-02 WORSE than CTRL: likely learning rate too high, lower to 1e-5 and retest
- SCALE-02 neutral (< 0.0002): try resid_mix params instead, or larger chunk
- SLOT+SCALE-03 worse than SLOT-01: interference — don't combine for full run

## 8×GPU Confirmation (if SCALE-02 passes)
Status: [ ] pending  [ ] pass  [ ] fail
int6_sw_bpb (seed 444):
int6_sw_bpb (seed 300):
artifact_bytes:
