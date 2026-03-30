# H100 Dirichlet Full Run — 2026-03-27

## Summary
- Successful canonical run: `h100_ppm_dirichlet_full_run_8xh100`
- Preset: `sota_plus_ppm_dirichlet`
- Scale: `full_run`
- GPU profile: `8xh100`
- Commit: `c3a23b4c2c33e78e51229e0c9ff6bc6f2c6ab945`
- Completion: `completed`
- Legality: `legal`

## What Changed Technically
- Validated the distributed exact-eval path for cache-enabled post-train evaluation on a real 8xH100 full run.
- Confirmed distributed exact sliding-window eval and distributed legal TTT exact eval both complete on the same stabilized trainer path.
- Preserved canonical reporting so completed runs keep training-best validation separate from the official submission metric.

## Canonical Submission Result
- Official submission metric: `legal_ttt_exact`
- Official submission loss: `0.62355877`
- Official submission bpb: `0.36930761`

Other exact evals from the same successful run:
- `final_int6_roundtrip_exact`: loss `2.30528416`, bpb `1.36531913`
- `final_int6_sliding_window_exact`: loss `0.62378916`, bpb `0.36944405`

Training-best validation from the same run:
- `val_loss = 2.0886360661952943`
- `val_bpb = 1.2370079255856536`

These training-best values are historical training metrics only. They are not the official submission metric.

## Validation Evidence
- Run directory: `/workspace/parameter-golf/research/results/runs/20260327_065341_h100_ppm_dirichlet_full_run_8xh100`
- Distributed exact-eval breadcrumbs observed in the successful log:
  - `sliding_eval: distributed_cache_shards=1 world_size=8 chunk_tokens=32768 stride=64`
  - `ttt_sliding:distributed_cache_shards world_size=8 chunk_tokens=32768`
- Exported model bytes: `10044244`
- Code bytes: `132164`
- Artifact bytes: `10176408`
- Remaining headroom to 16 MB: `5823592`
- Byte budget satisfied: `true`
- Submission consistency: `true`

Legality summary from the successful run:
- Posterior predictive backoff uses only previously committed counts plus the current score-step model probability.
- No future-token signal or target-aware chooser is introduced by the recursive update.
- Cache updates and TTT remain strictly score-first.

## Known Limitation
- `wall_clock_seconds = 1831.174`
- `train_time_seconds = 600.261`
- `submission_readiness.wall_clock_constraint_appears_satisfied = false`

Training respected the configured 600s cap, but total end-to-end wall clock exceeded 10 minutes because post-train exact sliding-window eval and legal TTT remain expensive. This run is therefore a completed legal artifact with an unsatisfied total wall-clock readiness flag.

## Historical Note
- Earlier attempt `h100_ppm_dirichlet_full_run_8xh100_real` failed with an NCCL timeout after `final_int6_roundtrip_exact`.
- That failed run should remain historical only and must not be treated as the canonical Dirichlet submission result.
