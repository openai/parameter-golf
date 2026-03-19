This record captures a Psionic-owned non-record submission package for the bounded local-reference Parameter Golf lane.

This package is intentionally **not** a record-track runtime claim. The shipped `train_gpt.py` now launches a prebuilt Psionic runtime payload that restores the included int8+zlib model artifact and re-runs the bounded local-reference validation path on the shipped fixture, writing its own runtime receipt inside the folder.

Configuration:
- Track: `non-record-unlimited-compute-16mb`
- Run ID: `parameter-golf-local-reference-run`
- Claim posture: `non_record_submission`
- Claim boundary: `current honest non-record submission package only; the shipped train_gpt.py launches a shipped Psionic runtime payload that restores the included int8+zlib artifact and re-runs the bounded local-reference validation path on the shipped fixture, not a record-track runtime claim`
- Wrapper entrypoint: `train_gpt.py`
- Runtime payload: `runtime/parameter_golf_submission_runtime`
- Runtime manifest: `runtime/parameter_golf_submission_runtime.json`
- Runtime receipt path: `parameter-golf-local-reference-run/benchmark/parameter_golf_submission_runtime_receipt.json`
- Validation oracle: `benchmark://openagents/psionic/parameter_golf/challenge_review`

Key metrics:
- Trained validation: `val_loss = 8.60598779`, `val_bpb = 9.93265272`
- Final int8+zlib roundtrip: `val_loss = 8.59898782`, `val_bpb = 9.92457367`
- Training wallclock: `25.779s`
- Counted bytes: `956676` total = `877718` code + `78958` compressed model

Artifact accounting:
- `entrypoint_code_bytes`: `390` bytes; the package ships one Python train_gpt.py launcher at the submission root that execs the shipped Psionic runtime payload
- `compressed_model_bytes`: `78958` bytes; the counted model artifact is the final int8+zlib roundtrip export emitted by the Psionic local-reference lane
- `shipped_runtime_code_bytes`: `877328` bytes; the package ships one prebuilt Psionic runtime payload that restores the included int8+zlib artifact and replays the bounded local-reference validation path
- `shipped_wrapper_code_bytes`: `0` bytes; no helper wrapper code beyond the top-level train_gpt.py launcher is shipped in this package
- `required_build_dependency_bytes`: `0` bytes; the package ships a prebuilt runtime payload and requires no build step or vendored dependency tree during execution
- Within 16,000,000-byte cap: `true`

Included files:
- `README.md`
- `submission.json`
- `train.log`
- `train_gpt.py`
- `runtime/parameter_golf_submission_runtime`
- `runtime/parameter_golf_submission_runtime.json`
- `runtime/parameter_golf_local_reference_fixture.json`
- `parameter-golf-local-reference-run/step-00002/final_model.int8.ptz`
- `parameter-golf-local-reference-run/benchmark/parameter_golf_benchmark_package.json`
- `parameter-golf-local-reference-run/benchmark/parameter_golf_challenge_score_report.json`
- `parameter-golf-local-reference-run/benchmark/parameter_golf_challenge_benchmark_receipt.json`
- `parameter-golf-local-reference-run/benchmark/parameter_golf_submission_accounting.json`
- `parameter-golf-local-reference-run/benchmark/parameter_golf_submission_runtime_receipt.json`
- `parameter-golf-local-reference-run/benchmark/run_bundle.json`

The package root for challenge-repo publication is `records/track_non_record_16mb/2026-03-18_psionic_local_reference_runtime_replay_v2`.
