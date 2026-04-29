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
