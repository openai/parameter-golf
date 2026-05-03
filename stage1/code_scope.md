# Stage 1 Code Scope Deep Pass

## Core Claim

The real editable surface is still concentrated in [train_gpt.py]( nanoevolve/pgolf/parameter-golf/train_gpt.py), but Stage 1 should not treat every exposed env var as a distinct search family.

The right split is:

- `2` config families to anchor baseline geometry
- `8` code or data families that mutate distinct mechanisms

## Why The Earlier Framing Was Too Narrow

The earlier Stage 1 pass gave too much weight to:

- width/depth/KV sweeps,
- sequence-length and batch-token sweeps,
- isolated schedule retunes.

Those are useful, but they mostly live in one part of the codebase: `Hyperparameters`.
That is not enough surface coverage for Enigma.

## Exact Stage 1 Neighborhoods

### M01: Matched-Budget Architecture Reallocation

- type: config
- main edit region: `Hyperparameters`, `GPT.__init__`
- knobs involved: `NUM_LAYERS`, `MODEL_DIM`, `NUM_KV_HEADS`, `MLP_MULT`
- why it is real: one necessary allocation family
- why it is not enough: all of these are still budget retunes inside the same region

### M02: Token/Update Geometry

- type: config
- main edit region: `Hyperparameters`, loader calls in the main loop
- knobs involved: `TRAIN_SEQ_LEN`, `TRAIN_BATCH_TOKENS`
- why it is real: one necessary throughput family
- hidden coupling: this changes optimization geometry, attention cost, and validation cost at once

### M03: Parameter Reuse

- type: code
- main edit region: `GPT.blocks`, `GPT.block_map`, skip reuse in `GPT.forward`
- concrete forms:
  - alternate-layer sharing
  - light recurrence
  - tied encoder/decoder blocks
- imported abstract idea:
  - ALBERT and universal-transformer style reuse

### M04: Nonuniform Stack Roles

- type: code
- main edit region: `GPT.__init__`, `Block`, `CausalSelfAttention`, `MLP`
- concrete forms:
  - asymmetric encoder and decoder halves
  - different MLP multipliers or control policies by depth band
  - different block definitions in early and late stack regions
- imported abstract idea:
  - allocate different functions to different depth bands instead of forcing a uniform stack

### M05: Control-Tensor Simplification

- type: code
- main edit region:
  - `CausalSelfAttention.q_gain`
  - `Block.attn_scale`
  - `Block.mlp_scale`
  - `Block.resid_mix`
  - `GPT.skip_weights`
  - `CONTROL_TENSOR_NAME_PATTERNS`
  - keep-float logic in quantization
- concrete forms:
  - per-block scalars instead of per-dimension vectors
  - grouped controls
  - shared controls across layers
- imported abstract idea:
  - mobile and edge models often win by simplifying what must stay high precision

### M06: Selective Optimizer Partitioning

- type: code
- main edit region:
  - `block_named_params`
  - `matrix_params`
  - `scalar_params`
  - optimizer construction in `main`
- concrete forms:
  - separate attention and MLP matrix families
  - move selected tensors off Muon
  - special-case the most fragile tensor families
- imported abstract idea:
  - different tensor families often want different update laws in short-horizon regimes

### M07: Adaptive Muon Compute

- type: code
- main edit region:
  - `Muon.step`
  - `get_muon_backend_steps`
  - `apply_muon_schedule`
- concrete forms:
  - lower early orthogonalization cost
  - shape-aware backend depth
  - phase-aware momentum or backend scheduling
- important subtlety:
  - here Muon is both an optimizer surface and a systems surface

### M08: Wallclock-Aware Schedule

- type: code
- main edit region:
  - `lr_mul`
  - warmup/reset block
  - validation cadence around `should_validate`
  - stop logic around the wallclock cap
- concrete forms:
  - spend less of the 600s budget on intermediate evaluation
  - reshape warmdown against actual stop time
  - make warmup and main-run budgeting less wasteful
- imported abstract idea:
  - speedrun training systems often win by removing non-learning tax, not by inventing new math

### M09: Export-Aware Quantization

- type: code
- main edit region:
  - `keep_float_tensor`
  - `quantize_float_tensor`
  - `quantize_state_dict_int8`
  - `dequantize_state_dict_int8`
- concrete forms:
  - different keep-float policy
  - better scale granularity
  - different clipping logic
  - architecture-specific export treatment
- important subtlety:
  - this is not a post-processing detail; it changes the actual leaderboard metric

### M10: Tokenizer/Data Frontier

- type: code/data
- main edit region:
  - `data/tokenizer_specs.json`
  - `data/download_hf_docs_and_tokenize.py`
  - sentencepiece validation LUT path in `train_gpt.py`
- concrete forms:
  - different vocab size
  - different tokenizer family
  - changed tokenizer normalization on the fixed docs cache
- important subtlety:
  - tokenizer changes affect BPB, embedding budget, and sequence statistics simultaneously

## Stage 1 Surface Summary By File Region

### `train_gpt.py:39-97`

- `Hyperparameters`
- mostly config surface
- should only account for `M01` and `M02`, not most of Stage 1

### `train_gpt.py:122-183`

- `Muon`
- core of `M06` and `M07`

### `train_gpt.py:186-288`

- tokenizer-aware evaluation
- critical for `M10`
- also where fake wins can appear if byte accounting is wrong

### `train_gpt.py:302-420`

- quantization and export
- core of `M05` and `M09`

### `train_gpt.py:565-726`

- attention, block internals, stack construction, skip behavior
- core of `M03`, `M04`, and `M05`

### `train_gpt.py:840-1151`

- optimizer split, warmup, schedule, wallclock stop, validation cadence
- core of `M06`, `M07`, and `M08`

### `data/`

- tokenizer specs and export pipeline
- core of `M10`

## Hidden Couplings That Matter For The Recut

- pre-quant wins can disappear after export
- step-time regressions can erase optimizer gains by reducing total updates
- small float-preserved control tensors can silently consume the remaining byte slack
- `TRAIN_SEQ_LEN` is not a pure data knob because it changes both throughput and optimization geometry
- tokenizer changes are expensive because they require regeneration and careful metric validation
- warmup and intermediate validation are part of the actual 600-second budget

## Bottom Line

The right Stage 1 direction is:

- treat `Hyperparameters` as only a small part of the search
- spend most of the budget on distinct code neighborhoods
- use architecture, systems, optimizer, export, and tokenizer surfaces together

That is the right search-space shape even if not every family is implemented immediately.
