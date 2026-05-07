# Diffusion Noised Teacher Forcing

Non-record submission exploring a diffusion-inspired training objective while keeping the standard autoregressive evaluation intact. Answers OpenAI's explicit request for text diffusion submissions.

## Approach

Standard AR transformer with a diffusion-inspired auxiliary loss: during training, a fraction of input tokens are corrupted (masked or randomly replaced), and the model trains on both clean and noisy forward passes with interpolated loss:

```
loss = lerp(clean_loss, noisy_loss, 0.35)
```

Noise ratio ramps from 5% to 35% over training, forcing the model to learn robust representations.

### Key design choices

1. **Diffusion-style noising path**:
   - `diffusion_noise_ratio_for_step(...)` linearly interpolates noise from `0.05` to `0.35`.
   - `corrupt_input_ids(...)` preserves the first token in each sequence, then corrupts later tokens using an EOS sentinel (`mask_token_id=2`) plus 15% random replacements.
   - Training minimizes `DIFFUSION_AUX_WEIGHT=0.35` weighted interpolation of clean and noisy AR loss.

2. **Honest validation**: standard autoregressive `eval_val(...)` — no tokenizer edits, no dataset edits, no custom scoring.

## Results (8xH100 SXM, 600s, seed=1337)

From `train_8xH100.log`:

- **val_bpb: 1.2734** (int8+zlib roundtrip)
- val_loss: 2.1500
- Steps: 3,227 at 186ms/step
- Training time: 600s (wallclock cap)
- Artifact: 15.8MB (under 16MB)
- Baseline 9L architecture (no SmearGate/BigramHash/XSA/FA3)

### Key observations

- Diffusion auxiliary loss trains stably with no instability
- Clean loss (2.15) consistently lower than noisy loss (2.50) — model learns to distinguish
- Gap widens as noise ratio increases — expected behavior
- Not competitive with SOTA but demonstrates the concept works

## Reproduction

```bash
DIFFUSION_ENABLED=1 \
DIFFUSION_AUX_WEIGHT=0.35 \
DIFFUSION_NOISE_MIN_RATIO=0.05 \
DIFFUSION_NOISE_MAX_RATIO=0.35 \
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
- Diffusion concept: inspired by MDLM, SEDD discrete diffusion literature
