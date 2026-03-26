# Draft PR Scope: Now vs Later

## Safe To Include In Draft PR Now
- Launch-system hardening and portability patches that preserve behavior:
  - `records/track_10min_16mb/2026-03-23_mike_record_pass_v5_9/launch_orchestrator.py`
  - `records/track_10min_16mb/2026-03-23_mike_record_pass_v5_9/runpod_preflight.py`
  - `records/track_10min_16mb/2026-03-23_mike_record_pass_v5_9/launch_t0.sh`
  - `records/track_10min_16mb/2026-03-23_mike_record_pass_v5_9/launch_export_sweep.sh`
  - `records/track_10min_16mb/2026-03-23_mike_record_pass_v5_9/launch_t1_if_needed.sh`
  - `records/track_10min_16mb/2026-03-23_mike_record_pass_v5_9/launch_donor_if_allowed.sh`
- Existing V5.9 analysis docs with path hygiene cleanup:
  - `analysis/record_pass_v5_9/final_report.md`
  - `analysis/record_pass_v5_9/decision_logic_audit.md`
  - `analysis/record_pass_v5_9/launch_graph_final.md`
  - `analysis/runpod_cycle_1/preflight_summary.md`
  - `analysis/runpod_cycle_1/final_status.json`
- This PR packet (`analysis/pr_packet_v1/*`) as reviewer-facing framing.

## Must Wait Until After First Paid T0 Run
- Any "final" score/bytes statements intended for public comparison.
- Any `submission.json` metrics values.
- Any README or PR language implying paid validation success.
- Any T1-or-donor execution conclusions driven by paid evidence.
- Any artifact promotion based on real paid-run outputs.
