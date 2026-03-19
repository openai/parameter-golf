# Psionic Parameter Golf PR Checklist

- Confirm the PR adds exactly one new folder under `records/track_non_record_16mb/2026-03-18_psionic_local_reference_runtime_replay_v2`.
- Confirm the folder remains self-contained with `README.md`, `submission.json`, `train.log`, `train_gpt.py`, and the shipped runtime payload.
- Review `psionic_parameter_golf_submission_run_evidence.json` for exact entrypoint, runtime, model, and receipt digests.
- Review `psionic_parameter_golf_record_folder_replay_verification.json` for metric, wallclock, and counted-byte replay facts.
- Review `psionic_parameter_golf_submission_promotion_receipt.json` before making any record or waiver claim.
- Re-run `scripts/check-parameter-golf-record-folder-compatibility.sh` and `scripts/check-parameter-golf-record-folder-replay.sh` against the staged folder in the live challenge repo.
- Preserve explicit claim language: this bundle targets track `non-record-unlimited-compute-16mb`, and the current promotion receipt disposition is `refused`.
