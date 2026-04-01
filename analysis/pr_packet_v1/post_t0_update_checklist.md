# Post-T0 Update Checklist

- [ ] Execute strict preflight on paid machine and capture output payload.
- [ ] Run canonical T0 full cycle and archive logs/artifacts.
- [ ] Record `export_decision.json` outcome token and recommended option fields.
- [ ] Update `analysis/pr_packet_v1/local_evidence_summary.csv` with paid T0 evidence rows.
- [ ] Update `analysis/pr_packet_v1/rules_audit.md` to move any confirmed items from pending to verified.
- [ ] Update PR body "Pending Paid-Run Validation" section with T0-resolved facts.
- [ ] If token resolves `RUN_T1`, leave PR draft and proceed with post-T1 checklist.
- [ ] If token resolves `STOP`, keep no-score-claim language until submission metrics are actually produced and validated.
