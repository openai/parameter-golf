# Reviewer Checklist (Draft PR)

- [ ] Confirm no threshold values changed (`noise_floor`, `ref_best`, `success_cutoff`, `t1_trigger_cutoff`).
- [ ] Confirm donor gate constants are unchanged.
- [ ] Confirm trunk context remains V5.2A-style non-QAT and donor path remains optional-only.
- [ ] Confirm launch-control token behavior remains deterministic and unchanged.
- [ ] Confirm portability patch only affects path/root resolution and not launch decision semantics.
- [ ] Confirm docs do not claim paid-run validation or leaderboard result.
- [ ] Confirm `runpod_cycle_1` status is represented accurately as preflight fail on CUDA runtime.
- [ ] Confirm no fake metrics were added to `submission.json` or README claims.
- [ ] Confirm PR scope is explicitly draft/WIP and pending paid T0.
