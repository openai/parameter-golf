This record uses the same valid `13x512` `GQA-4` training run (tied embeddings + Flash SDPA, `train_time: 568574ms`) and updates the submitted score to a later quantization-only improvement that fits the strict `< 32,000,000` byte cap.

Final quantization strategy used for the submitted artifact (strict-cap winner):
- `INT8_QUANT_MODE=mixed_rows_2d`
- `INT8_KEEP_FLOAT_MAX_NUMEL=512`
- `INT8_PER_ROW_SCALE_DTYPE=float16`
- `INT8_SCALE_ENCODE=linear_u8`
- `INT8_SCALE_ENCODE_MIN_NUMEL=32`
- `INT8_ROW_GROUP_SIZE=7`
- `INT8_CLIP_PERCENTILE=99.99984`
- `INT8_PER_ROW_2D_NAME_PATTERNS=tok_emb.weight,blocks.12.mlp.proj.weight,blocks.11.mlp.fc.weight,blocks.12.mlp.fc.weight`
- `INT8_PER_ROW_GROUP_2D_NAME_PATTERNS=blocks.4.attn.proj.weight,blocks.8.attn.proj.weight`

Track eligibility (per user clarification: `train_time` is the relevant timer):
- `train_time: 568574ms` (< 10 minutes)

Key metrics:
- End-of-training fp eval (`train.log`): `val_loss:2.3131`, `val_bpb:1.1413`
- Final submitted artifact size (strict-cap winner): `Total submission size int8+zlib: 31963090 bytes`
- Exact post-quant eval (strict-cap winner): `val_loss:2.32142687`, `val_bpb:1.14543695`

Included logs:
- `train.log` (full training log for the 7375-step run)
- `quant_eval_per_tensor.log` (per-tensor int8 baseline on the same checkpoint)
- `quant_build_candidate.log` (candidate quantization serialization/size log)
- `quant_eval_candidate_ptz.log` (exact eval of the candidate quantized `.ptz`)
- `train_gpt.py` (training code snapshot used for the run / evals)

Note:
- The bundled quantization logs in this folder are from an earlier candidate (`1.1470`). The final strict-cap winning quantization result (`1.14543695` at `31,963,090` bytes) was produced later from the same checkpoint during a follow-up quantization sweep; the exact trusted eval log referenced during that sweep was `/root/code/openai-parameter-challenge/logs/drop_a5a10_9999984.txt` on the remote eval box.
