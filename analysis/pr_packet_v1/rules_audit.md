# Rules / Evidence Audit

## Verified Locally
- V5.9 packet files exist and are runnable in non-destructive modes (`--print-only`, preflight `--dry-run`).
- Decision thresholds and donor-gate constants are present and unchanged in `launch_config.json`.
- Orchestrator decision token mapping remains deterministic (`STOP`, `HOLD_RECHECK`, `RUN_T1`, `DONOR_NOT_ALLOWED`).
- Strict local preflight failed on CUDA requirement, and cycle status records no paid-run completion.

## Pending Paid-Run Confirmation
- Any T0 quality/byte outcome under paid compute.
- Any T1 necessity determination based on paid T0 export decision.
- Any donor-optional eligibility based on paid evidence.
- Any final artifact bytes and roundtrip bpb for submission use.

## Still Speculative (Must Be Labeled As Such)
- That T0 will pass strict preflight in paid environment.
- That T0 will land in STOP/HOLD_RECHECK/RUN_T1 region as forecast.
- That donor path will ever be allowed in practice.

## PR Must NOT Imply Yet
- No leaderboard win claim.
- No paid-run validation claim.
- No confirmed final submission bytes or roundtrip bpb.
- No claim that T1 or donor outcomes are known.
