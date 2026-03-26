# Prefix-Conditioned Suffix Diffusion (Smoke)

This is a true discrete diffusion non-record submission.

Unlike the earlier hybrid run in this branch, this model is not trained with autoregressive teacher forcing plus a denoising auxiliary term. The training objective here is literal denoising over token sequences:

- keep an observed prefix clean,
- choose a future suffix,
- corrupt that suffix with an absorbing mask process at a sampled diffusion timestep,
- predict the original clean suffix tokens from the noisy sequence.

## Why This Version

The goal was a clear scientific prototype, not a leaderboard play.

I chose prefix-conditioned suffix diffusion because it is both more distinctive and more defensible than full-sequence denoising for this challenge setting. The model is asked to reverse corruption only on the unknown future while conditioning on an exact observed history. That gives the submission a real diffusion identity while still preserving a plausible left-to-right scoring story.

## Model and Objective

The implementation starts from the existing compact GPT-style backbone and adds diffusion-specific conditioning:

- learned timestep embeddings,
- learned role embeddings for `prefix` vs `diffused suffix`,
- absorbing-mask corruption over suffix tokens only,
- denoising loss computed only on corrupted suffix positions.

Training uses the token stream directly as a denoising target sequence. For each sampled sequence:

1. sample a prefix length,
2. sample a diffusion timestep,
3. mask suffix tokens according to the timestep,
4. predict clean tokens at every position,
5. backprop only on the masked suffix positions.

This makes the training objective genuinely diffusion-based rather than autoregressive.

## Approximate Scoring

Exact challenge-style compression from a diffusion model is nontrivial, so this submission reports an explicit approximation instead of pretending otherwise.

The evaluator computes a `diffusion_pll_bpb` metric:

- walk left-to-right through validation sequences,
- treat the prefix as observed,
- mask the entire remaining suffix at the maximum diffusion step,
- run the denoiser,
- score only the first masked token,
- accumulate bytes with the same SentencePiece byte LUT used elsewhere in the repo.

This is an approximate prefix-conditioned pseudo-log-likelihood metric, not exact autoregressive BPB. The log labels and metadata call that out directly.

## Smoke Run

This smoke run used:

- Dataset: `fineweb10B_sp1024`
- Training shards: `1`
- Validation subset: first `4` sequences of the fixed validation split
- Model: `4` layers, `256` dim, `4` heads, `2` KV heads
- Sequence length: `512`
- Global train tokens/step: `65536`
- Diffusion steps: `8`
- Minimum clean prefix: `16`
- Diffusion eval batch size: `64`
- Steps: `4` train steps after `1` warmup step
- Hardware: `1x NVIDIA GeForce RTX 4080`

Command:

```bash
RUN_ID=literal_diffusion_smoke_20260326 \
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
DIFFUSION_NUM_STEPS=8 \
DIFFUSION_MIN_PREFIX=16 \
DIFFUSION_MASK_TOKEN_ID=2 \
DIFFUSION_EVAL_MAX_SEQS=4 \
DIFFUSION_EVAL_BATCH_SIZE=64 \
TTT_EVAL_ENABLED=0 \
COMPILE_ENABLED=0 \
python records/track_non_record_16mb/2026-03-26_PrefixConditionedSuffixDiffusion/train_gpt.py
```

## Results

From `train.log`:

- Final training loss after 4 steps: `6.8679`
- Final approximate validation metric: `diffusion_pll_loss=6.8622`, `diffusion_pll_bpb=3.7157`
- Final roundtrip approximate metric: `diffusion_pll_loss=6.86997051`, `diffusion_pll_bpb=3.71991200`
- Scored tokens in eval: `2044`
- Training time to step 4: `1150ms`
- Roundtrip eval time: `1684ms`
- Peak memory: `962 MiB allocated`, `1602 MiB reserved`
- Model parameters: `2,104,592`
- Serialized model int8+zlib: `1,604,850 bytes`
- Code size: `74,826 bytes`
- Total submission size int8+zlib: `1,679,676 bytes`

## What Makes It Distinct

This submission stands apart from the existing records because it does not optimize a standard autoregressive objective at all. It is a real discrete diffusion denoiser over token sequences, with an explicit prefix/suffix split and an explicit approximate sequential scoring construction.

It is still an early prototype. The next obvious upgrades are:

- a stronger reverse process than pure absorbing-mask denoising,
- better timestep parameterization,
- longer or adaptive diffusion evaluation,
- a tighter coding story than first-token PLL approximation.

Included files:

- `train_gpt.py`
- `train.log`
- `submission.json`
