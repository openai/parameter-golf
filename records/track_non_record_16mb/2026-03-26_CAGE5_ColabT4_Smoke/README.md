# CAGE5 Colab T4 smoke (non-record 16MB)

This folder captures a non-record smoke submission for Parameter Golf.

This run is **not** intended for the main 10-minute leaderboard. It is an in-progress tiny debug configuration used to validate a complete training -> quantization/export -> evaluation pipeline and a strictly causal hashed 5-gram mixer.

## Summary

- Hardware: 1x Tesla T4 (Google Colab GPU)
- Track: non-record-16mb
- Dataset/tokenizer: SP-1024, 1 training shard, validation limited for smoke testing
- Core idea: interpolate the neural model with a strictly causal hashed 5-gram cache during sliding-window evaluation

## Key result from `train.log`

- `final_int8_zlib_roundtrip_exact val_loss: 4.66523169`
- `final_int8_zlib_roundtrip_exact val_bpb: 2.69806373`
- `Total submission size int6+lzma: 656896 bytes`
- `Serialized model int6+lzma: 562620 bytes`
- `Code size: 94276 bytes`

## A/B signal seen during Colab smoke testing

- Baseline (no n-gram): `sliding_bpb = 3.57847716`
- Best 100-step alpha sweep (`alpha=0.30`): `sliding_bpb = 2.84614804`
- 300-step confirm run (`seed=2026`): `sliding_bpb = 2.69806373`

## Included files

- `train_gpt.py` — Colab-tested script used for the smoke run
- `flash_attn_interface.py` — fallback attention shim used by the Colab-tested script
- `train.log` — captured output of the confirm run
- `submission.json` — metadata for this non-record smoke submission
- `requirements.txt` — dependency snapshot
