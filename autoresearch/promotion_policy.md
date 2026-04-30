# Autoresearch Promotion Policy

## Purpose
This policy defines how to run `autoresearch.py` across multiple research lanes without mixing incompatible objectives or histories.

The main idea is:
- use cheap runs to rank ideas inside a lane
- apply lane-specific gates before accepting a result
- promote only the strongest candidates to more expensive runs
- combine ideas only after they win in isolation

## Lanes

### `core`
Use for:
- shape changes
- optimizer and LR tuning
- schedule changes
- attention layout
- recurrence and parameter sharing close to the current stack

Primary discovery metric:
- post-quantization `val_bpb`

Required gates:
- artifact size under budget
- lower `val_bpb` than current lane best

Secondary diagnostics:
- steps reached
- pre-quantization `val_bpb`
- quantization gap
- eval time

### `eval_time`
Use for:
- cache and pointer/copy logic
- online adaptation
- dynamic evaluation
- context mixing at evaluation time

Primary discovery metric:
- post-quantization `val_bpb`

Required gates:
- artifact size under budget
- lower `val_bpb` than current lane best
- evaluation time under `MAX_EVAL_TIME_MS`
- optional memory ceiling via `MAX_EVAL_MEMORY_MIB`

Secondary diagnostics:
- eval time
- peak memory
- quantization gap

### `representation`
Use for:
- tokenizer changes
- segmentation changes
- byte/latent hybrids
- retokenization pipeline work

Primary discovery metric:
- correctness first, then `val_bpb`

Required gates:
- artifact size under budget
- exact accounting and reproducibility verified
- lower `val_bpb` than current lane best

Notes:
- by default this lane will not keep a result unless `REPRESENTATION_VERIFIED=1`
- this is intentional; representation wins are too easy to fake accidentally

### `storage`
Use for:
- quantization-aware training
- codebooks
- zlib-aware parameterization
- tying and export-aware compression work

Primary discovery metric:
- post-export quality, not raw pre-quant loss

Required gates:
- artifact size under budget
- acceptable quantization gap
- either lower `val_bpb` than current lane best
- or materially smaller artifact with only a small `val_bpb` regression

Default storage tradeoff:
- keep if `val_bpb` regresses by at most `STORAGE_MAX_REGRESSION`
- and compressed artifact shrinks by at least `STORAGE_MIN_SIZE_IMPROVEMENT`

## Stages

### `discovery`
Use:
- `1xH100`
- `180s`

Purpose:
- broad ranking inside a lane
- reject weak ideas cheaply

### `promotion`
Use:
- `1xH100`
- `600s`

Purpose:
- confirm that a discovery win survives a longer run
- filter out short-horizon artifacts

### `final_validation`
Use:
- `8xH100`
- `600s`

Purpose:
- estimate competition-like behavior
- verify runtime and artifact constraints before submission work

## Default Funnel

### `core`
`discovery -> promotion -> final_validation`

### `eval_time`
`discovery -> promotion -> final_validation`

Additional rule:
- promotion requires passing the eval-time gate first

### `representation`
`correctness verification -> discovery -> promotion -> final_validation`

Additional rule:
- do not promote unverified tokenizer/segmentation changes

### `storage`
`discovery -> promotion -> final_validation`

Additional rule:
- promotion decision can be driven by size wins as well as `val_bpb` wins

## Combination Policy
Do not combine ideas just because they both look promising.

Combine only when:
- each idea already passed its own lane gate
- the two ideas consume different resources
- you can explain why they should be additive

Good examples:
- `core` recurrence + `eval_time` cache
- `core` shape win + `storage` export win

Bad examples:
- tokenizer change + new backbone + eval-time adaptation all at once
- multiple unverified changes from the same lane

## Namespaces
Each lane should usually have its own namespace so pods do not overwrite each other.

Recommended pattern:
- `AUTORESEARCH_NAMESPACE=core_discovery`
- `AUTORESEARCH_NAMESPACE=core_promotion`
- `AUTORESEARCH_NAMESPACE=eval_time_discovery`
- `AUTORESEARCH_NAMESPACE=storage_discovery`

This creates separate state under:
- `autoresearch/<namespace>/history.jsonl`
- `autoresearch/<namespace>/experiments/`
- `autoresearch/<namespace>/logs/`
- `autoresearch/<namespace>/train_gpt.best.py`

## Current Script Support
`autoresearch.py` now supports:
- `AUTORESEARCH_LANE`
- `AUTORESEARCH_STAGE`
- `AUTORESEARCH_NAMESPACE`
- `MAX_ARTIFACT_BYTES`
- `MAX_EVAL_TIME_MS`
- `MAX_EVAL_MEMORY_MIB`
- `MAX_QUANTIZATION_GAP`
- `STORAGE_MAX_REGRESSION`
- `STORAGE_MIN_SIZE_IMPROVEMENT`
- `REPRESENTATION_VERIFIED`

## Example Commands

Core discovery:

```bash
AUTORESEARCH_LANE=core \
AUTORESEARCH_STAGE=discovery \
AUTORESEARCH_NAMESPACE=core_discovery \
EXPERIMENT_SECONDS=180 \
GPUS=1 \
python3 autoresearch.py
```

Core promotion:

```bash
AUTORESEARCH_LANE=core \
AUTORESEARCH_STAGE=promotion \
AUTORESEARCH_NAMESPACE=core_promotion \
EXPERIMENT_SECONDS=600 \
GPUS=1 \
python3 autoresearch.py
```

Eval-time discovery:

```bash
AUTORESEARCH_LANE=eval_time \
AUTORESEARCH_STAGE=discovery \
AUTORESEARCH_NAMESPACE=eval_time_discovery \
EXPERIMENT_SECONDS=180 \
MAX_EVAL_TIME_MS=120000 \
GPUS=1 \
python3 autoresearch.py
```

Storage discovery:

```bash
AUTORESEARCH_LANE=storage \
AUTORESEARCH_STAGE=discovery \
AUTORESEARCH_NAMESPACE=storage_discovery \
EXPERIMENT_SECONDS=180 \
MAX_QUANTIZATION_GAP=0.08 \
STORAGE_MAX_REGRESSION=0.003 \
STORAGE_MIN_SIZE_IMPROVEMENT=250000 \
GPUS=1 \
python3 autoresearch.py
```

Representation discovery after correctness checks:

```bash
AUTORESEARCH_LANE=representation \
AUTORESEARCH_STAGE=discovery \
AUTORESEARCH_NAMESPACE=representation_discovery \
REPRESENTATION_VERIFIED=1 \
EXPERIMENT_SECONDS=180 \
GPUS=1 \
python3 autoresearch.py
```
