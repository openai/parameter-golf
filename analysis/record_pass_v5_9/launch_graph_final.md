# Launch Graph Final (V5.9)

## Collapsed Graph
```text
preflight -> T0 -> export sweep -> decision -> (STOP | HOLD_RECHECK | T1 | optional donor)
```

## Triggered Transitions
- `preflight -> T0`
  - Only if strict preflight passes.
- `T0 -> export sweep`
  - Only if `final_model.pt` exists.
- `export sweep -> decision`
  - Only if `export_decision.json` exists.
- `decision -> STOP`
  - `stop_rule_outcome=STOP`.
- `decision -> HOLD_RECHECK`
  - `stop_rule_outcome=HOLD_RECHECK`.
- `decision -> T1`
  - `stop_rule_outcome=RUN_T1`.
- `decision -> optional donor`
  - Only after `RUN_T1` context and strict donor gate is explicitly satisfied.

## Operator Entry Point
- Use orchestrator full cycle:
  - `python3 records/track_10min_16mb/2026-03-23_mike_record_pass_v5_9/launch_orchestrator.py --mode full_t0_cycle`
