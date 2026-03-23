# Decision Logic Audit (V5.9)

## Canonical Source Of Truth
- Machine-readable thresholds and gates are centralized in:
  - `records/track_10min_16mb/2026-03-23_mike_record_pass_v5_9/launch_config.json`
- Runtime evaluator is:
  - `records/track_10min_16mb/2026-03-23_mike_record_pass_v5_9/launch_orchestrator.py`

## Parsed Thresholds (Unchanged From V5.7/V5.8)
```json
{
  "noise_floor_bpb": 0.000007850897,
  "ref_best_bpb": 3.734376345615,
  "success_cutoff_bpb": 3.734384196512,
  "t1_trigger_cutoff_bpb": 3.734392047409
}
```

Derived relationships (same as prior passes):
- `success_cutoff_bpb = ref_best_bpb + noise_floor_bpb`
- `t1_trigger_cutoff_bpb = ref_best_bpb + 2 * noise_floor_bpb`

## Canonical Stop/Hold/T1 Logic
Input field used: `decision_block.stop_rule_outcome`
- `STOP` => operator output `STOP`
- `HOLD_RECHECK` => operator output `HOLD_RECHECK`
- `RUN_T1` => operator output `RUN_T1` unless donor probe is explicitly requested and donor gate fails

No new thresholds were introduced.

## Canonical Donor Gate (Strict)
From `launch_config.json`:
```json
{
  "quality_margin_bpb": -0.000012,
  "required_reruns": 2,
  "max_bytes_delta": 8192,
  "require_no_broad_qat": true
}
```

Donor gate passes only if all are true:
1. `quality_margin_bpb <= -0.000012`
2. `reruns_with_same_sign >= 2`
3. `donor_cap_safe == 1`
4. `bytes_delta <= 8192`
5. `broad_qat_required == 0` (because no broad-QAT dependency is allowed)

If `donor_probe_requested=1` and any gate check fails, operator output is `DONOR_NOT_ALLOWED`.

## Determinism Notes
- Decision mode is deterministic for a given input JSON.
- No stochastic tie-breaking is used.
- Output token domain is fixed to:
  - `STOP`
  - `HOLD_RECHECK`
  - `RUN_T1`
  - `DONOR_NOT_ALLOWED`

## Audit Input Files
- `analysis/record_pass_v5_7/final_report.md`
- `analysis/record_pass_v5_7/consensus_export_decision.md`
- `analysis/record_pass_v5_7/launch_runbook.md`
- `analysis/record_pass_v5_8/decision_table.md`
- `analysis/record_pass_v5_8/decision_table.csv`
- `analysis/record_pass_v5_8/final_report.md`

Note: `analysis/record_pass_v5_7/decision_table.md` was requested but is not present in repo; V5.8 decision table artifacts were used as the direct machine-readable continuation.
