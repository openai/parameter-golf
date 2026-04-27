# Vast Single-GPU Run Report (2026-03-27)

Run tag: `20260327_114840`
Test instance: `33667350` (RTX 4090)
Contract status: destroyed after run (confirmed)

## Outcome
- The strict canonical A/B path could not complete on this host/image combination without multiple runtime fixes (driver/library mismatch, package compatibility, and 24GB OOM constraints).
- Final completed signal was obtained via a **mini-data proxy A/B** (valid run, non-canonical for leaderboard).

## Final Proxy A/B Result (mini-data val_bpb)
- `control` (xsa_last_n=11): `2.8709`
- `a_xsa9` (xsa_last_n=9): `2.8623`
- Delta (`a_xsa9 - control`): `-0.0086`
- Promotion rule used: require `<= -0.0100`
- Decision: `PROMOTE: none (mini-data proxy)`

## Key Technical Notes
- Root incompatibility encountered initially: CUDA runtime/driver mismatch behavior until forcing host libcuda precedence.
- Needed runtime env for CUDA availability on this host:
  - `LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib:/usr/local/nvidia/lib64`
- `fla` packages were removed for non-GDN runs to avoid import-time assertion failures in CPU/unsupported paths.
- 24GB memory budget required reduced batch/sequence and proxy harnessing for stable completion.

## Local Artifacts
- Main remote orchestration logs and debugging:
  - `experiments/GreenRod_X_1/lab_protocol_20260327/vast_tests/20260327_114840/remote_run.log`
  - `experiments/GreenRod_X_1/lab_protocol_20260327/vast_tests/20260327_114840/remote_run_nongdn.log`
  - `experiments/GreenRod_X_1/lab_protocol_20260327/vast_tests/20260327_114840/remote_run_nongdn_v2.log`
  - `experiments/GreenRod_X_1/lab_protocol_20260327/vast_tests/20260327_114840/remote_run_nongdn_v3.log`
  - `experiments/GreenRod_X_1/lab_protocol_20260327/vast_tests/20260327_114840/remote_run_nongdn_v4.log`
  - `experiments/GreenRod_X_1/lab_protocol_20260327/vast_tests/20260327_114840/remote_run_nongdn_v5.log`
  - `experiments/GreenRod_X_1/lab_protocol_20260327/vast_tests/20260327_114840/remote_proxy_ab_v8_run2.log`
- Remote proxy root path reported during run (instance now destroyed):
  - `/workspace/parameter-golf-lab/experiments/GreenRod_X_1/lab_protocol_20260327/vast_tests/20260327_114840/proxy_ab_v8_20260327_174825`
