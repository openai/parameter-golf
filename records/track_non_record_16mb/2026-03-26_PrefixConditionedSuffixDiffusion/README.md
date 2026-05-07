# Prefix-Conditioned Suffix Diffusion

True discrete diffusion non-record submission. Unlike autoregressive approaches, this model is trained with literal denoising over token sequences:

- Keep an observed prefix clean
- Choose a future suffix
- Corrupt that suffix with an absorbing mask process at a sampled diffusion timestep
- Predict the original clean suffix tokens from the noisy sequence

## Why This Approach

Prefix-conditioned suffix diffusion is both more distinctive and more defensible than full-sequence denoising for this challenge. The model reverses corruption only on the unknown future while conditioning on an exact observed history, giving the submission a real diffusion identity with a plausible left-to-right scoring story.

## Model and Objective

Compact GPT-style backbone with diffusion-specific conditioning:

- Learned timestep embeddings
- Learned role embeddings for `prefix` vs `diffused suffix`
- Absorbing-mask corruption over suffix tokens only
- Denoising loss computed only on corrupted suffix positions

Training for each sampled sequence: sample prefix length, sample timestep, mask suffix, predict clean tokens, backprop on masked positions only.

## Approximate Scoring

Reports `diffusion_pll_bpb` — an explicit approximation, not exact autoregressive BPB:

- Walk left-to-right through validation sequences
- Treat the prefix as observed, mask entire remaining suffix at max diffusion step
- Run the denoiser, score only the first masked token
- Accumulate bytes with the standard SentencePiece byte LUT

## Results (8xH100 SXM, 600s, seed=1337)

From `train_8xH100.log`:

- **diffusion_pll_bpb: 1.8587** (int8+zlib roundtrip)
- diffusion_pll_loss: 3.2109
- Steps: 2,398 at 250ms/step
- Training time: 600s (wallclock cap)
- Artifact: 13.7MB (under 16MB)
- Scored tokens: 32,736

### Training progression

| Step | diffusion_pll_bpb | diffusion_pll_loss |
|------|------------------:|-------------------:|
| 1000 | 2.1727 | 3.7532 |
| 2000 | 1.8994 | 3.2811 |
| 2398 | 1.8576 | 3.2089 |

## Reproduction

```bash
DIFFUSION_NUM_STEPS=8 \
DIFFUSION_MIN_PREFIX=16 \
DIFFUSION_MASK_TOKEN_ID=2 \
DIFFUSION_EVAL_MAX_SEQS=4 \
DIFFUSION_EVAL_BATCH_SIZE=64 \
SEED=1337 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Included files

- `train_gpt.py` — training script
- `train_8xH100.log` — full 8xH100 training log (seed=1337)
- `train.log` — local smoke test log (4-step, 1xGPU)
- `submission.json`

## Credits

- Baseline architecture: OpenAI Parameter Golf starter code
- Discrete diffusion concepts: MDLM, SEDD literature
