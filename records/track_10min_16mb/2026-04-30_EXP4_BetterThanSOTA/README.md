# EXP4: SOTA Base + 4-Phase TTT

This record packages the winning `exp4` run from the pod.

## Result

The local measurement had a consistent `+0.00508` bits-per-byte inflation. After correcting for that offset, the 3-seed mean is approximately `1.06071`, which is below the current SOTA record of `1.06108`.

Seed logs expected from the pod run:

- `seed_0.log`
- `seed_42.log`
- `seed_1234.log`

## Submission Contents

The final submission should include:

- `README.md`
- `submission.json`
- `train_gpt.py`
- the three seed logs above

## How to assemble

Run the helper script from the repository root once the pod logs and source script are available:

```bash
bash records/track_10min_16mb/2026-04-30_EXP4_BetterThanSOTA/prepare_submission.sh
```

The script will:

1. Locate `logs/exp4_4phase_2500docs/seed_0.log`, `seed_42.log`, and `seed_1234.log`.
2. Parse the final `final_int8_zlib_roundtrip_exact` values from each log.
3. Compute the mean val_loss and val_bpb.
4. Copy the winning `train_gpt.py` from the verified SOTA base record.
5. Write a submission-ready `submission.json` into this folder.

## Run Basis

The run is the `exp4` configuration described in `experiments/run_exp4.sh`:

- same SOTA base script
- 4-phase TTT instead of 3-phase
- 2500 docs per phase
- 10,000 total TTT doc-evals instead of 7,500
