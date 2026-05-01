# 9L1216 Int8 Insurance Artifact (Beast Box)

Non-record / fallback Parameter Golf submission from @texanfirst.

This folder preserves the best local insurance artifact we had before the April 30 deadline. Later Vast canaries on 1x H100 NVL and 4x H200 did not beat this score, so this is the safest artifact to submit.

## Result

- Final exact validation BPB: **1.26347117**
- Source log: `train.log`
- Model artifact: `final_model.int8.ptz`
- Training script: `train_gpt.py`
- Artifact SHA256 archive: `5320170738fe709cecac7260d8a7d47d13ed649cdbe30444a8ace5e3e7115ed7`

## Size

- `train_gpt.py`: 59198 bytes
- `final_model.int8.ptz`: 15687281 bytes
- Combined counted artifact bytes: **15746479**
- Cap: 16,000,000 bytes

## Notes

This is not a SOTA claim. It is submitted as a non-record/fallback artifact under the 16MB cap, with the full script, log, manifest, checksum file, and compressed model artifact included for reproducibility.
