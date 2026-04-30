# RTX2070 Smoke: Auditable Turtle Baseline

This is a non-record local smoke run, not an official leaderboard score.

It verifies that a local RTX2070 setup can complete:

- training
- validation
- int8+zlib serialization
- roundtrip reload
- final eval_val

Caveats:

- RTX2070, not 8xH100
- shortened smoke dataset
- tiny model settings
- RTX2070 compatibility patch: flash SDPA disabled, math SDPA enabled, fp16 autocast
- TORCHDYNAMO_DISABLE=1

Result:

- final_int8_zlib_roundtrip val_loss: 8.40257616
- final_int8_zlib_roundtrip val_bpb: 4.91592263
- Total submission size int8+zlib: 504009 bytes

Causality:

Standard causal LM path. input_ids produce logits. target_ids are used only for cross entropy. eval_val uses inference_mode. no validation-time update is performed.

Submission intent:

This submission is intentionally small. It is not a leaderboard attempt or a new architecture claim.

It is a non-record local smoke artifact documenting an end-to-end Parameter Golf submission path from a first-time, non-ML participant using AI assistance, with emphasis on code/log/README/metadata alignment.

See:

- `SUBMISSION_INTENT.md` for why this non-record submission exists
- `PR_ARCHAEOLOGY.md` for the auditability / log-alignment note
