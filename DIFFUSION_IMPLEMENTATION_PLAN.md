# Diffusion LLM Implementation Plan

This document turns the initial roadmap into a concrete implementation checklist for building a language diffusion model in this repository, starting from a very small local prototype on the existing FineWeb subset and then scaling toward stronger architectures inspired by the current diffusion LLM literature.

## Grounding Constraints

- The challenge target is a self-contained artifact under `16,000,000` bytes, with leaderboard runs training in under 10 minutes on `8xH100` and evaluation also bounded in runtime.
- Non-record submissions can exceed the training-time limit if they still satisfy the artifact limit and are interesting or novel.
- This repo already contains the baseline data loader, tokenizer path conventions, validation pipeline, and artifact-size accounting in `train_gpt.py`.
- In the current workspace, `sp1024` tokenizer assets are present and there is already a local FineWeb subset available:
  - `data/tokenizers/fineweb_1024_bpe.model`
  - `data/datasets/fineweb10B_sp1024/`
  - 10 train shards plus the validation shard

## Guiding Approach

- Start with a discrete masked diffusion language model, not continuous latent diffusion.
- Reuse as much of the repo's data, validation, and artifact plumbing as possible.
- Treat evaluation correctness as a first-class task, not a cleanup item.
- Only add architecture complexity after the simple version is stable and measurable.
- Keep "research correctness" and "parameter-golf optimization" as separate phases.

## Week 1

Goal: get a minimal diffusion LM training locally on the existing FineWeb subset.

### Implementation Checklist

- Create `train_diffusion.py` beside `train_gpt.py`.
- Reuse the existing env-var pattern for:
  - `DATA_PATH`
  - `TOKENIZER_PATH`
  - `VOCAB_SIZE`
  - `TRAIN_SEQ_LEN`
  - `TRAIN_BATCH_TOKENS`
  - `ITERATIONS`
  - `MAX_WALLCLOCK_SECONDS`
- Reuse the current shard-loading and validation-token-loading path from `train_gpt.py`.
- Start with `sp1024`, 1 to 2 train shards, and `seq_len=256` or `512`.
- Build a small bidirectional denoiser:
  - token embedding
  - timestep embedding
  - 6 layers
  - model dim `256` or `384`
  - full self-attention, not causal masking
  - tied output head
- Implement absorbing-mask corruption:
  - sample timestep `t`
  - mask a fraction of tokens according to a schedule
  - predict original tokens only on masked positions
- Use plain masked-position cross-entropy loss first.
- Add simple iterative unmasking or ancestral-style sampling for sanity checks.
- Log:
  - train loss
  - fraction masked
  - tokens/sec
  - sample text every N steps on a tiny prompt or empty start
- Add a smoke-test config for laptop or single GPU.
- Add a tiny synthetic-data mode for debugging:
  - repeated patterns
  - short vocab
  - overfit in a few hundred steps

### Exit Criteria

- Training runs end-to-end on the local subset.
- Loss falls on synthetic data and on at least one FineWeb shard.
- Sampling produces non-degenerate text.
- No artifact-budget optimization work yet beyond basic serialization.

### Read This Week

- `train_gpt.py`
  - Focus on data loading, logging, validation structure, and artifact accounting.
- D3PM: Structured Denoising Diffusion Models in Discrete State-Spaces
  - Focus on absorbing-state corruption, discrete forward process, and the auxiliary CE view.
- MDLM: Simple and Effective Masked Diffusion Language Models
  - Focus on why masked diffusion is the simplest useful setup for text.

## Week 2

Goal: make evaluation mathematically respectable.

### Implementation Checklist

- Define the exact probabilistic objective you will report:
  - masked-denoising CE proxy first
  - then ELBO or lower-bound estimate
- Implement validation for the diffusion model over the full `fineweb_val_*` split.
- Reuse tokenizer-byte accounting logic from `train_gpt.py`.
- Add a validation mode that reports:
  - proxy loss
  - estimated bits/token
  - estimated BPB
- Build correctness tests on tiny toy sequences:
  - enumerate short vocab/state spaces where possible
  - verify the bound is sensible
  - verify lower-noise predictions improve the likelihood estimate
- Compare against a tiny AR baseline on the same toy corpus so the metric behavior is easier to sanity-check.

### Exit Criteria

- The code produces a repeatable validation number for diffusion runs.
- The BPB pipeline is connected to the repo's tokenizer-byte accounting.
- The likelihood estimate is trustworthy enough to compare ablations.

### Read This Week

- D3PM
  - Re-read the objective and likelihood sections carefully.
- SEDD
  - Focus on why discrete diffusion likelihoods are tricky and how score-entropy or ratio ideas help.
- `train_gpt.py`
  - Focus on the validation and BPB path.

## Week 3

Goal: make the baseline actually good, not just correct.

### Implementation Checklist

- Ablate noise schedules:
  - uniform mask rate
  - cosine
  - log-linear or high-noise-biased
- Add self-conditioning.
- Add timestep importance weighting or loss reweighting.
- Try fixed-step versus random-step training.
- Try parameterization variants:
  - predict `x0`
  - predict masked-token logits directly
- Increase to 2 to 4 shards and `seq_len=512` or `1024` if stable.
- Improve logging:
  - validation curve
  - performance by mask-rate bucket
  - sample quality at several denoising lengths
- Save the best checkpoint by validation estimate.

### Exit Criteria

- One recipe clearly beats the naive masked baseline.
- Training is stable across at least 2 to 3 seeds on the small setup.
- The most important knobs are identified.

### Read This Week

- MDLM
  - Focus on the simplifications that matter in practice.
- SEDD
  - Focus on which ideas are worth porting only after the baseline works.
- LLaDA
  - Focus on architecture and training choices that seem to scale with standard Transformer components.

## Week 4

Goal: scale the architecture modestly and choose a serious research direction.

### Implementation Checklist

- Increase model size to something like:
  - 8 to 12 layers
  - model dim `384` to `768`
- Try `sp4096` if embedding cost is manageable.
- Benchmark a plain bidirectional Transformer denoiser against a lightweight U-Net-style denoiser.
- Test whether skip connections help iterative denoising.
- Measure step time carefully.
- Add wallclock-aware config behavior similar to `train_gpt.py`.
- Run longer unconstrained experiments if needed.

### Exit Criteria

- There is a clear direction among:
  - plain masked diffusion
  - SEDD-style stronger objective
  - U-Net-like denoiser
  - tokenizer or vocabulary changes

### Read This Week

- LLaDA
  - Focus on scaling behavior and training setup.
- DiffuSeq
  - Read selectively; mostly useful if conditional generation or infilling becomes relevant.
- Repo examples:
  - `records/track_10min_16mb/2026-03-24_74M_Ternary_UNet_FP8_10L_8192BPE_YaRN_NeoMuon/README.md`
  - `records/track_non_record_16mb/2026-03-24_106M_Binary_Asymmetric_UNet_FP8_15L_8192BPE_YaRN_NeoMuon_Smear/README.md`

## Week 5

Goal: start adapting the research model to Parameter Golf constraints.

### Implementation Checklist

- Add artifact accounting from `train_gpt.py`.
- Measure compressed model size after every serious run.
- Try factorized embeddings.
- Keep tied embeddings unless there is a clear gain from untying.
- Explore smaller denoiser depth with more denoising steps at inference.
- Test whether iterative generation quality can compensate for smaller model size.
- Add quantization experiments only after the core model is stable.

### Exit Criteria

- A diffusion model fits comfortably under projected artifact limits or has a clear path there.
- The main budget drivers are understood:
  - embeddings
  - block weights
  - code size

### Read This Week

- Repo README artifact rules and submission requirements.
- Quantized and U-Net submission READMEs for compression tactics, not as a full architecture template.

## Week 6

Goal: prepare either a credible non-record submission or a narrower record-oriented sprint.

### Implementation Checklist

- Freeze one main architecture.
- Run 3-seed comparisons.
- Write up:
  - what worked
  - what failed
  - where diffusion helps or hurts versus the AR baseline
- Decide one of two tracks:
  - non-record research submission if quality is promising but slow
  - record-oriented optimization if quality and speed both look viable

### Exit Criteria

- There is a clear decision on whether the current direction belongs in:
  - a non-record research submission
  - or a more aggressive record-oriented optimization phase

### Read This Week

- Revisit whichever of `MDLM`, `SEDD`, or `LLaDA` most resembles the winning direction.
- Re-read the repo submission requirements carefully.

## Cross-Cutting Checklist

Do these throughout the project:

- Keep one `tiny` config, one `local-dev` config, and one `scale-up` config.
- Add unit tests for:
  - corruption
  - masking
  - timestep embedding
  - likelihood math
- Keep synthetic overfit tests passing before each architecture change.
- Track both quality and step time from the start.
- Separate "research correctness" from "artifact-budget optimization".
- Save enough metadata per run to reproduce decisions later.

## Suggested File and Module Breakdown

Recommended implementation order:

1. `train_diffusion.py`
2. `diffusion_model.py`
3. `diffusion_objectives.py`
4. `diffusion_eval.py`
5. `configs/diffusion_tiny.env`
6. `configs/diffusion_local.env`
7. `configs/diffusion_scale.env`

Possible responsibilities:

- `train_diffusion.py`
  - argument/env plumbing
  - training loop
  - logging
  - checkpointing
  - artifact-size accounting
- `diffusion_model.py`
  - denoiser architecture
  - timestep embeddings
  - forward pass
- `diffusion_objectives.py`
  - corruption process
  - masked loss
  - optional self-conditioning
  - schedule definitions
- `diffusion_eval.py`
  - validation loop
  - ELBO or lower-bound estimate
  - BPB conversion using tokenizer-byte accounting

## Best Reading Order

1. `train_gpt.py`
2. D3PM
3. MDLM
4. SEDD
5. LLaDA
6. DiffuSeq, only if conditional-generation ideas become relevant

## Immediate Next Step

The next concrete build step should be:

- implement `train_diffusion.py`
- keep the architecture tiny
- reuse the current data and tokenizer plumbing
- get synthetic-data overfitting and one-shard FineWeb training working before touching advanced likelihood estimation or compression
