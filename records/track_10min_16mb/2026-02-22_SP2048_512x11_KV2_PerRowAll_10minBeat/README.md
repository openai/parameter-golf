This record captures the `SP-2048 11x512` KV2 (`NUM_KV_HEADS=2`) model that beats the user baseline under the decimal `32,000,000` byte cap using a compiled checkpoint-reload quantization pass.

Why two logs are included:
- `train.log` is the exact 10-minute-track training run that produced the strong checkpoint (`raw val_bpb=1.1535`).
- `quant.log` is the compiled reload quant-eval pass on that saved checkpoint using tuned int8 settings that preserve quality under the size cap.

Stage 1 (training, exact-cap track run):
- Layout: `NUM_LAYERS=11 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=2 FFN_EXPAND=4 VOCAB_SIZE=2048`
- Runtime: `10000/10000` steps in `588414ms` (under 10 minutes)
- Pre-quant eval (from training run): `val_loss:2.3141`, `val_bpb:1.1535`
- Default final int8+zlib roundtrip on this checkpoint fails badly for KV2 (`val_bpb:6.5553`), so a quantization-tuned reload pass is required.

Stage 2 (winning compiled quant-reload eval):
- Quant settings:
  - `INT8_QUANT_MODE=per_row_2d` (all 2D tensors)
  - `INT8_KEEP_FLOAT_MAX_NUMEL=65536`
  - `INT8_KEEP_FLOAT_STORE_DTYPE=float16`
  - `INT8_PER_ROW_SCALE_DTYPE=float16`
  - `INT8_CLIP_PERCENTILE=99.999`
  - `COMPILE_MODEL=1` (required for trustworthy checkpoint reload eval in this setup)
- Final post-quant roundtrip eval: `val_loss:2.3173`, `val_bpb:1.1551`
- Exact printed metric: `final_int8_zlib_roundtrip_exact val_bpb:1.15506351`
- Submission size int8+zlib total: `31203908 bytes`

Comparison to user baseline:
- User baseline post-quant roundtrip `val_bpb`: `1.1556`
- This result post-quant roundtrip `val_bpb`: `1.1551` (better by `0.0005`)

Included files:
- `train_gpt.py` (local code snapshot containing flash-only SDPA and quantizer controls used)
- `train.log` (remote training log copied from `speedrunb1-0`)
- `quant.log` (remote compiled quant-reload log copied from `speedrunb1-0`)
- `params.env` (training + quant command parameters)
