# Activation-Space CSA on SP1024 (8xH100)

This folder contains a **non-record submission** for **Activation-Space Compressed-Sensing Adapters** derived from the `2026-03-17_LoRA_TTT` reference and executed on `8xH100 SXM 80GB`.

## Scope

- submission type: `non-record`, still under the `16,000,000` byte artifact cap
- current tracked run: `8xH100 SXM 80GB`
- historical lineage: local-first pilot on `GeForce RTX 3060 12GB`
- `sp1024` tokenizer family
- score-before-update evaluation loop preserved
- LoRA replacement by activation-space adapters with:
  - shared structured sensing map
  - per-layer tiny gates
  - `top-k` sparse reconstruction in measurement space

This is still an experimental `SP1024` stack. It is packaged as a non-record submission because it does not make a leaderboard or statistical-significance claim, but it does include a full `8xH100` run, the compressed artifact, and the evaluation result for the current ACSA mechanism.

## Summary

This experiment starts from the `2026-03-17_LoRA_TTT` record and replaces LoRA-based evaluation-time adaptation with **Activation-Space Compressed-Sensing Adapters (ACSA)**. The main idea is to adapt the model through a tiny sparse code in activation space instead of attaching trainable low-rank matrices to the model weights.

The implementation keeps the existing score-before-update evaluation discipline and document-level reset behavior. Adaptation is injected into the post-block residual stream, with optional support for a `prehead` target through an environment variable.

## Main Result

From the committed `8xH100` run in [training.log](/home/lus/git/ia/parameter-golf/records/track_non_record_16mb/2026-04-28_ActivationSpaceCSA_SP1024_8xH100/training.log):

- pre-ACSA quantized roundtrip: `val_bpb 1.2875`
- final ACSA evaluation: `val_bpb 1.2580`
- compressed artifact size: `15,829,207` bytes
- total ACSA eval time: `144,491 ms`

This demonstrates that the activation-space adapter materially improves the final compressed model without breaking the artifact-size budget.

## Gain / Loss Versus the Starting Baseline

The initial code baseline for this experiment lineage is the `2026-03-17_LoRA_TTT` record, which reports a mean result of `1.1928 bpb` on the full validation set.

Relative to that starting baseline:

- `LoRA_TTT` baseline: `1.1928 bpb`
- current ACSA non-record result: `1.2580 bpb`
- delta versus baseline: `+0.0652 bpb`

So this ACSA implementation is currently **worse than the original LoRA-TTT baseline** in absolute compression quality.

However, within the current ACSA run itself, the eval-time adapter is still contributing a real gain:

- quantized roundtrip without ACSA: `1.2875 bpb`
- final quantized eval with ACSA: `1.2580 bpb`
- ACSA gain over quantized roundtrip: `-0.0296 bpb`

This is the main positive result of the submission: even though the full ACSA stack does not yet beat the original LoRA-TTT baseline, the activation-space adapter materially improves the compressed model relative to its own no-adaptation quantized evaluation.

## Why Non-Record

- this folder is being submitted as `non-record`, not as a leaderboard claim
- no multi-seed package or `p < 0.01` significance claim is included
- the stack remains `SP1024`, well behind the current leaderboard family
- the value here is the mechanism: replacing LoRA-style eval-time adaptation with sparse activation-space adaptation

## Method

The underlying hypothesis is that document-specific adaptation does not necessarily need to modify the model weights directly. Instead, a useful portion of the adaptation can be expressed as a small corrective signal in the hidden-state trajectory of the network.

ACSA implements this by freezing the base model and learning a compact per-document adapter during evaluation. For each selected target location, the hidden activation is:

1. projected into a lower-dimensional measurement space,
2. sparsified in that measurement space,
3. reconstructed back to model dimension, and
4. added back to the original activation as a residual delta.

The measurement operator is a structured random map built from sign flips, a random permutation, and a Fast Walsh-Hadamard Transform. The reconstruction path uses the inverse permutation and the same transform structure. This gives a cheap compressed-sensing-style bottleneck without introducing a full learned projection matrix.

The trainable adapter state is intentionally small. Instead of learning dense matrices, ACSA learns:

- per-measurement gates,
- a per-target scaling factor `alpha`, and
- a per-target sparsity threshold `tau`

Sparsification is controlled either by `top-k` support selection or by threshold shrinkage, with `top-k` used by default in the current pilot.

Conceptually, this changes the adaptation object from a weight update to an activation update:

- LoRA-style TTT: learn a low-rank change in the model parameters
- ACSA-style TTT: learn a sparse corrective code that edits hidden activations online

In the current pilot, these activation deltas are applied after transformer blocks (`postblock`) and may also be applied just before the output head (`prehead`).

During evaluation, the script preserves the same legal score-before-update protocol used in the LoRA TTT reference:

1. score the current chunk causally,
2. accumulate the compression metric on already-scored tokens,
3. update only the ACSA parameters using those scored tokens, and
4. carry the updated adapter state forward to later chunks from the same document

The adapter state is reset between documents, so no adaptation information leaks across document boundaries.

## What Was Implemented

- a new ACSA run folder under `records/track_non_record_16mb/`
- a derived `train_gpt.py` specialized for ACSA evaluation
- dataset/tokenizer verification before launching runs
- local knobs for adapter dimension, sparsity, target locations, initialization, and learning rate

The current default ACSA path uses:

- a shared structured sensing map
- tiny per-layer gates
- `top-k` sparse reconstruction in measurement space
- `postblock` residual injection

## Numerical Validation

This record includes a stabilized localhost smoke configuration with:

- `ACSA_LR=0.001`
- `ACSA_ALPHA_INIT=0.01`
- `ACSA_TAU_INIT=0.05`
- `DEBUG_VAL_DOCS=32`
- `ITERATIONS=2`
- `TRAIN_SEQ_LEN=512`
- `TRAIN_BATCH_TOKENS=8192`
- `TTT_BATCH_SIZE=4`
- `TTT_EVAL_SEQ_LEN=512`
- `TTT_CHUNK_SIZE=128`

Observed smoke metrics:

- `step:2/2 val_bpb:4.1369`
- `final_int8_zlib_roundtrip val_bpb:4.1401`
- `final_int8_ttt_acsa val_bpb:4.1415`

Historical local development included longer localhost runs before the final `8xH100` package. Those traces were manually interrupted before final evaluation and should be treated as partial development logs rather than the current submission artifact. One representative intermediate validation point from that phase is:

- `step:13800/20000 val_bpb:1.4124`

The current submission artifact is instead based on the committed `8xH100` run in `training.log`, whose final lines show:

- `final_int8_zlib_roundtrip val_bpb:1.2875`
- `final_int8_ttt_acsa val_bpb:1.2580`

## Why This Is Interesting

LoRA TTT adapts model weights at evaluation time. ACSA instead adapts a sparse latent code over activation-space measurements. This changes the adaptation surface substantially:

- fewer trainable adaptation degrees of freedom per update step
- a natural sparsity bottleneck
- the possibility of sharing the sensing structure across layers
- a cleaner bridge to future compressed or token-budget-aware eval-time adaptation schemes

The project started as a local-first pilot to answer “does this mechanism train and evaluate stably at all?” and then moved to an `8xH100` execution path to test challenge-aligned wallclock and evaluation budgets.

## Local Setup

Use `uv` on localhost.

```bash
uv venv
uv pip install -r records/track_non_record_16mb/2026-04-28_ActivationSpaceCSA_SP1024_8xH100/requirements.txt
uv run python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1
```

## Smoke Run

```bash
RUN_ID=acsa_local_smoke \
COMPILE_MODEL=0 \
AUTOCAST_DTYPE=fp16 \
DEBUG_VAL_DOCS=128 \
TRAIN_SEQ_LEN=512 \
TRAIN_BATCH_TOKENS=8192 \
VAL_BATCH_SIZE=8192 \
TTT_BATCH_SIZE=4 \
TTT_EVAL_SEQ_LEN=512 \
TTT_CHUNK_SIZE=128 \
ACSA_DIM=64 \
ACSA_TOPK=16 \
ACSA_LR=0.001 \
ACSA_ALPHA_INIT=0.01 \
ACSA_TAU_INIT=0.05 \
ACSA_TARGETS=postblock \
uv run torchrun --standalone --nproc_per_node=1 \
records/track_non_record_16mb/2026-04-28_ActivationSpaceCSA_SP1024_8xH100/train_gpt.py
```

## Reproduction

The folder is self-contained around `train_gpt.py`, `requirements.txt`, `submission.json`, `training.log`, and the compressed artifact.

Example localhost smoke command:

```bash
RUN_ID=acsa_local_smoke \
COMPILE_MODEL=0 \
AUTOCAST_DTYPE=fp16 \
DEBUG_VAL_DOCS=128 \
TRAIN_SEQ_LEN=512 \
TRAIN_BATCH_TOKENS=8192 \
VAL_BATCH_SIZE=8192 \
TTT_BATCH_SIZE=4 \
TTT_EVAL_SEQ_LEN=512 \
TTT_CHUNK_SIZE=128 \
ACSA_DIM=64 \
ACSA_TOPK=16 \
ACSA_LR=0.001 \
ACSA_ALPHA_INIT=0.01 \
ACSA_TAU_INIT=0.05 \
ACSA_TARGETS=postblock \
uv run torchrun --standalone --nproc_per_node=1 \
records/track_non_record_16mb/2026-04-28_ActivationSpaceCSA_SP1024_8xH100/train_gpt.py
```

Example `8xH100` command in the style used for the committed run:

```bash
SUBMISSION_AUTHOR="Your Name" \
SUBMISSION_GITHUB_ID="your_github" \
SUBMISSION_NAME="Activation-Space Compressed-Sensing Adapters (SP1024, 8xH100)" \
SUBMISSION_BLURB="Non-record ACSA submission on SP1024 with score-before-update eval-time adaptation." \
SUBMISSION_TRACK="non_record_16mb" \
SUBMISSION_HARDWARE="8xH100 SXM 80GB" \
WRITE_SUBMISSION_JSON=1 \
LOG_FILENAME=training.log \
SUBMISSION_FILENAME=submission.json \
OUTPUT_DIR=records/track_non_record_16mb/2026-04-28_ActivationSpaceCSA_SP1024_8xH100 \
COMPILE_MODEL=1 \
AUTOCAST_DTYPE=bf16 \
DEBUG_VAL_DOCS=0 \
TRAIN_SHARDS=10 \
TRAIN_SEQ_LEN=1024 \
TRAIN_BATCH_TOKENS=65536 \
VAL_BATCH_SIZE=65536 \
VAL_LOSS_EVERY=0 \
ITERATIONS=50000 \
WARMUP_STEPS=20 \
TRAIN_LOG_EVERY=50 \
MAX_WALLCLOCK_SECONDS=590 \
TTT_BATCH_SIZE=64 \
TTT_EVAL_SEQ_LEN=1024 \
TTT_CHUNK_SIZE=256 \
ACSA_DIM=96 \
ACSA_TOPK=24 \
ACSA_LR=0.001 \
ACSA_ALPHA_INIT=0.01 \
ACSA_TAU_INIT=0.05 \
ACSA_TARGETS=postblock \
ENFORCE_ARTIFACT_LIMIT=1 \
SUBMISSION_MAX_BYTES=16000000 \
ENFORCE_EVAL_LIMIT=1 \
EVAL_MAX_SECONDS=600 \
ENABLE_ACSA_EVAL=1 \
uv run python -m torch.distributed.run --standalone --nproc_per_node=8 \
records/track_non_record_16mb/2026-04-28_ActivationSpaceCSA_SP1024_8xH100/train_gpt.py
```

If this line of work is ever pushed toward a record-track attempt, the next missing piece is a proper multi-seed package with separate logs and statistical analysis. That is intentionally not claimed in the current non-record packaging.

## Notable Knobs

- `COMPILE_MODEL=0|1`
- `AUTOCAST_DTYPE=bf16|fp16`
- `DEBUG_VAL_DOCS=<int>`
- `ACSA_DIM=<int>`
- `ACSA_TOPK=<int>`
- `ACSA_LR=<float>`
- `ACSA_ALPHA_INIT=<float>`
- `ACSA_TAU_INIT=<float>`
- `ACSA_TARGETS=postblock[,prehead]`
- `ACSA_SHRINK_MODE=topk|threshold`

## Current Limitations

- no `SP4096` or `SP8192` port yet
- no multi-seed statistical package yet
- no multi-seed statistical claim
- one preserved `training.log` path line still references the earlier `RTX3060_Pilot` folder label from before the final packaging rename

## Provenance

The pilot was introduced and stabilized across commits `189f0ae` through `045db69`, which added the ACSA experiment folder, the dedicated runner, smoke/full profiles, dataset verification, and committed localhost logs.

## Next Steps

- consolidate the best `8xH100` run and its compliance details
- compare `postblock` versus `postblock,prehead` ACSA targets under the 10-minute budget
- decide whether the next bridge should be `SP4096` first or a direct `SP8192` path
- build a proper multi-seed package if this line is pushed toward record candidacy

## Status

Implementation, logs, submission metadata, and a record-local `requirements.txt` are present.

Note: this folder was renamed after the original run organization changed from the local `RTX3060_Pilot` naming to the current `8xH100` packaging. The committed `training.log` is preserved as generated, so one internal `wrote_submission_json:` line still points to the older path label.
