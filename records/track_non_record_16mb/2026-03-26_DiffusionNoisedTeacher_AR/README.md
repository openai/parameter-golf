# Diffusion Noised Teacher Forcing (Smoke)

This is a non-record submission exploring a diffusion-inspired training objective while keeping the repository's standard autoregressive evaluation intact.

The core idea is simple:

- Keep the normal next-token loss and `val_bpb` computation unchanged.
- Add a denoising auxiliary loss during training by corrupting the input prefix tokens before predicting the next token.
- Ramp the corruption ratio over training, so the model sees progressively noisier contexts.

This is intentionally not a literal diffusion language model. The point of this run is to test an easier-to-integrate approximation first: "teach the autoregressive model to recover next-token predictions from partially corrupted history" without changing the tokenizer, dataset format, or `val_bpb` accounting.

## What Changed

The record-local `train_gpt.py` differs from the root baseline in three main ways:

1. It adds a diffusion-style noising path:
   - `diffusion_noise_ratio_for_step(...)` linearly interpolates the noise level from `0.05` to `0.35`.
   - `corrupt_input_ids(...)` preserves the first token in each sequence, then corrupts later tokens using an EOS-token sentinel (`mask_token_id=2`) plus `15%` random replacements inside the noisy subset.
   - Training minimizes a weighted interpolation of clean AR loss and noisy-context AR loss with `DIFFUSION_AUX_WEIGHT=0.35`.

2. It keeps validation honest:
   - Validation is still the repository's standard autoregressive `eval_val(...)`.
   - No tokenizer edits, no dataset edits, no custom scoring conversion from denoising steps back into next-token probabilities.

3. It is made portable for local smoke runs:
   - `COMPILE_ENABLED=0` by default to avoid Triton/Inductor requirements on this machine.
   - Safe math SDP is enabled by default instead of flash-only kernels.
   - LoRA TTT evaluation is gated behind `TTT_EVAL_ENABLED=0` for this submission.

## Smoke Run

This run is a real end-to-end smoke test on a local Windows workstation with `1x NVIDIA GeForce RTX 4080`, using:

- Dataset: published `fineweb10B_sp1024`
- Training shards: `1`
- Validation: full `fineweb_val_*` split
- Model: `4` layers, `256` dim, `4` attention heads, `2` KV heads
- Sequence length: `512`
- Batch: `65536` train tokens/step
- Steps: `4` train steps after `1` warmup step

Command:

```bash
RUN_ID=diffusion_smoke_clean_20260326 \
DATA_PATH=D:/Development/parameter-golf/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=D:/Development/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
NUM_LAYERS=4 \
MODEL_DIM=256 \
NUM_HEADS=4 \
NUM_KV_HEADS=2 \
MLP_MULT=2 \
TIE_EMBEDDINGS=1 \
ITERATIONS=4 \
WARMUP_STEPS=1 \
MAX_WALLCLOCK_SECONDS=0 \
TRAIN_BATCH_TOKENS=65536 \
TRAIN_SEQ_LEN=512 \
TRAIN_LOG_EVERY=1 \
VAL_LOSS_EVERY=0 \
VAL_BATCH_SIZE=524288 \
DIFFUSION_ENABLED=1 \
DIFFUSION_AUX_WEIGHT=0.35 \
DIFFUSION_NOISE_MIN_RATIO=0.05 \
DIFFUSION_NOISE_MAX_RATIO=0.35 \
DIFFUSION_RANDOM_REPLACE_PROB=0.15 \
DIFFUSION_MASK_TOKEN_ID=2 \
TTT_EVAL_ENABLED=0 \
COMPILE_ENABLED=0 \
python train_gpt.py
```

## Results

From `train.log`:

- Final pre-quant validation: `val_loss=6.9113`, `val_bpb=4.0933`
- Final int8+zlib roundtrip: `val_loss=6.91404936`, `val_bpb=4.09488948`
- Training time to step 4: `1448ms`
- Roundtrip eval time: `76638ms`
- Peak memory: `1731 MiB allocated`, `2978 MiB reserved`
- Model parameters: `2,101,776`
- Serialized model int8+zlib: `1,673,079 bytes`
- Code size: `64,832 bytes`
- Total submission size int8+zlib: `1,737,911 bytes`

## Takeaway

This particular smoke run is a negative-result-style submission, not a competitive one. The value here is the scaffold:

- It demonstrates a clean way to inject diffusion-like corruption into the existing Parameter Golf training loop.
- It preserves the challenge's standard autoregressive metric, making results easy to interpret.
- It gives a concrete stepping stone toward a later, more literal diffusion submission that would need a different scoring story.

Included files:

- `train_gpt.py`
- `train.log`
- `submission.json`
