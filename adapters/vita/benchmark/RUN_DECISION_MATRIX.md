# Run Decision Matrix (Operator Guide)

Purpose:
- Make operator choices unambiguous.
- Keep claims guard strict.
- Ensure first real benchmark attempt is reproducible.

Claim policy:
- All paths remain NON-CLAIMING unless real benchmark evidence is produced and validated.

## Primary path selection

Primary sanity path:
- `local-smoke`
- Goal: packaging/integration sanity only.
- Claim status after run: still NON-CLAIMING.

First real benchmark candidate path:
- `RunPod/H100` template path (via `target-gpu-template` runbook).
- Goal: execute real benchmark pipeline and collect evidence.
- Claim status after run: still NON-CLAIMING until checklist/claims gate passes.

## Decision matrix

| Condition | Action | Output | Claim impact |
|---|---|---|---|
| Adapter changed / verify gate changed | Run `build_submission.py --verify-only` with expected values | VERIFY_OK/VERIFY_FAIL | None |
| Need quick environment sanity | Run `benchmark_commands.sh local-smoke` | Placeholder evidence files + env snapshot | None |
| Local smoke fails | Fix environment/integration before any target-GPU attempt | Updated local sanity logs | None |
| Local smoke passes | Prepare first real attempt using RunPod/H100 template | target-gpu runbook + command sheet | None |
| Real benchmark executed with logs + metric + artifact accounting | Run acceptance checklist | Evidence set complete | Potentially claim-eligible only after explicit gate |

## Canonical execution order

1) Verify gate
- `python3 adapters/vita/build_submission.py --verify-only ...expected values...`

2) Local sanity run
- `adapters/vita/benchmark/runs/090-c2-benchmark-attempt-01/benchmark_commands.sh local-smoke`

3) Prepare first real benchmark runbook
- `adapters/vita/benchmark/runs/090-c2-benchmark-attempt-01/benchmark_commands.sh target-gpu-template`
- Fill hardware-specific details in runbook before execution.

4) Real benchmark execution (RunPod/H100 candidate path)
- Execute runbook commands on target hardware.
- Collect required evidence files under run `evidence/`.

5) Claim gate
- Use `BENCHMARK_CHECKLIST.md` + `benchmark_claims_guard.json` unlock requirements.
- No claim language before all checks pass.
