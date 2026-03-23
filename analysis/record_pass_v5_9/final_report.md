# Record Pass V5.9 Final Report

This pass is execution-hardening only for first-cycle Runpod spend.

## Plain Answers

### 1) What is the single canonical first paid run?
Use orchestrator full cycle (preflight -> T0 -> export -> decision):
```bash
python3 records/track_10min_16mb/2026-03-23_mike_record_pass_v5_9/launch_orchestrator.py --mode full_t0_cycle
```

### 2) What is the exact second run if needed?
If decision resolves to `RUN_T1`, run:
```bash
bash records/track_10min_16mb/2026-03-23_mike_record_pass_v5_9/launch_t1_if_needed.sh
```

### 3) What is optional only?
Donor path is optional-only and strict-gated:
```bash
bash records/track_10min_16mb/2026-03-23_mike_record_pass_v5_9/launch_donor_if_allowed.sh --allow-donor-run
```
Only use after donor gate approval.

### 4) What is now removed?
- Separate T0A/T0B/T0C paid trainings.
- Donor-first second spend.
- QAT-first second spend.

### 5) What file should execute first when credits arrive?
- `records/track_10min_16mb/2026-03-23_mike_record_pass_v5_9/launch_orchestrator.py`

## Launch Packet Delivered
- Preflight: `runpod_preflight.py`
- Canonical T0: `launch_t0.sh`
- Canonical export sweep: `launch_export_sweep.sh`
- Decision/orchestration: `launch_orchestrator.py`
- Conditional T1: `launch_t1_if_needed.sh`
- Optional donor-only: `launch_donor_if_allowed.sh`
- Canonical machine-readable config: `launch_config.json`

## Success Criterion For This Pass
- First paid cycle is now one command path with deterministic decision outputs and explicit failure handling.
- No new model-search claim is made.
