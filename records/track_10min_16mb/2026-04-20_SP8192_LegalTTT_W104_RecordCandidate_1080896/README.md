# W104 SP8192 LegalTTT Record Candidate

This folder contains a 3-seed record-candidate run for OpenAI Parameter Golf.

It is based on the SP8192 + 3-layer recurrence + parallel residuals + QK-Gain 5.25 + legal score-first TTT stack, with a source-visible faithful replay configuration.

## Claim

This submission is a record candidate because the 3-seed mean BPB is below the current public leaderboard score of 1.0810.

## Results

| Seed | val_loss | val_bpb |
|---:|---:|---:|
| 42 | 2.79072877 | 1.08037808 |
| 314 | 2.79285442 | 1.08120098 |
| 999 | 2.79261328 | 1.08110763 |

Mean val_bpb: **1.08089556**

Population std val_bpb: **0.00036790**

## Configuration

- Dataset: FineWeb SP8192
- Tokenizer: `fineweb_8192_bpe.model`
- Train shards: 80
- Hardware: 8xH100
- QK-Gain init: 5.25
- Legal score-first TTT: enabled
- TTT learning rate: 0.005
- TTT epochs: 3
- Artifact target: under 16 MB
- Training target: under 10 minutes on 8xH100

## Included files

- `README.md`
- `submission.json`
- `train_gpt.py`
- `train_seed42.log`
- `train_seed314.log`
- `train_seed999.log`

## Notes

This run does not use V7, V8, or V9 auxiliary data.

It is a faithful SP8192 LegalTTT replay candidate focused on reducing bad-seed variance while keeping the official FineWeb validation setup.
