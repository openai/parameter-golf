# Budget Accounting Template

## Scope

This file is a placeholder for artifact-size accounting before formal measurement is scripted.

## Metrics to track

- `code bytes`
  - default assumption: counted bytes for the submission code snapshot, typically `train_gpt.py`
- `model bytes`
  - default assumption: compressed model artifact size, typically `final_model.int8.ptz`
- `total artifact bytes`
  - `code bytes + model bytes`

## Planned measurement method

- Code bytes
  - Measure the exact submission script copied into the candidate record directory.
  - Candidate command: `wc -c train_gpt.py`
- Model bytes
  - Measure the serialized compressed artifact after the run.
  - Candidate command: `wc -c final_model.int8.ptz`
- Total artifact bytes
  - Sum the two values above.
  - Cross-check against the `Total submission size int8+zlib` line printed by the training script.

## Current status

- Formal measurement has not been run yet in this workspace.
- No baseline artifact has been generated locally yet.
- This note exists so later branches keep size accounting explicit from the first experiment onward.

## Follow-up

- Convert the measurement steps into a small reproducible script once the first non-record candidate produces actual artifacts.
- Store regenerated size summaries under `runs/<exp_id>/` rather than committing large binaries.
