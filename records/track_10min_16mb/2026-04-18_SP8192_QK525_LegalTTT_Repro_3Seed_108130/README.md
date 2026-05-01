# Near-SOTA Reproduction: SP8192 + QK-Gain 5.25 + Legal TTT

This submission is an independent 3-seed reproduction of the current SP8192 + 3-layer recurrence + parallel residuals + QK-Gain 5.25 + Legal TTT stack.

This does **not** claim a new SOTA record because the 3-seed mean does not beat the current 1.0810 record.

## Results (3 seeds)

- seed 42: val_loss 2.7982063, val_bpb 1.08041364
- seed 314: val_loss 2.7941035, val_bpb 1.08168719
- seed 999: val_loss 2.79443824, val_bpb 1.08181413
- mean val_bpb: 1.08130499
- population std val_bpb: 0.00063240

## Hardware

- 8xH100 80GB

## Notes

- All runs were under the 10-minute training target based on logs.
- Run from the official `openai/parameter-golf` code path.

## Included files

- `train_seed42.log`
- `train_seed314.log`
- `train_seed999.log`
- `train_gpt.py`
- `submission.json`
