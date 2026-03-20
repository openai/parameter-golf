# Parameter Golf Research Program

You are working inside the OpenAI Parameter Golf repository.

## Objective

Improve the challenge score under these constraints:

- optimize `final_int8_ttt_lora val_bpb`
- optimize `final_int8_zlib_roundtrip_exact val_bpb`
- keep `Total submission size int8+zlib` under `16,000,000` bytes
- preserve reproducibility

Lower `val_bpb` is better.

## Primary Rules

1. Prefer small, ablation-friendly changes.
2. Keep changes concentrated in `train_gpt.py` unless there is a strong reason not to.
3. Reject changes that improve one metric but badly regress the other.
4. Reject changes that push artifact size toward the budget without a clear score win.
5. Do not change the validation set.
6. Treat tokenizer or dataset changes as higher-risk and require stronger evidence.

## Current Priors

- Sliding-window evaluation is high value.
- FP16 tied embedding export is high value.
- 10-layer small models are promising.
- Decoupled Muon weight decay is promising.
- `ATTN_TWICE_ALPHA=0.05` currently looks better than baseline.
- `Z_LOSS_COEF=0.0001` currently looks worse than baseline.

## Current Best Known Local Results

- `base10l`
  - `roundtrip_val_bpb = 1.40296458`
  - `ttt_val_bpb = 1.3976`
  - `artifact_bytes = 10831123`

- `twice_low`
  - `roundtrip_val_bpb = 1.40177526`
  - `ttt_val_bpb = 1.3969`
  - `artifact_bytes = 10836065`

## Experiment Order

1. `twice_eval2048`
2. best `twice_*` variant on more seeds
3. training-context and batch tradeoff ablations
4. tokenizer ablations on published docs cache

## Allowed Edit Zones

- architecture details in `train_gpt.py`
- training schedule and optimizer settings
- quantization/export logic
- evaluation logic
- remote profile scripts

## High-Risk Areas

- external datasets
- validation handling
- complex multi-file refactors
- changes that increase code size substantially

## Decision Policy

Keep a change only if at least one is true:

- `final_int8_ttt_lora` improves and `roundtrip_exact` does not materially regress
- `roundtrip_exact` improves and `ttt` does not materially regress
- artifact size drops meaningfully with near-flat score

Reject a change if:

- both `ttt` and `roundtrip_exact` regress
- artifact size grows with no score benefit
- it adds a lot of complexity without measurable value

## Logging And Packaging

- Use `scripts/run_remote_profile.sh` or `scripts/run_and_score.sh`
- Parse logs with `scripts/parse_run.py`
- Package strong candidates with `scripts/package_record.sh`
