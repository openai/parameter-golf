# Non-Record Submission: Sophonics LoRA Int5 DeepMLP

This is an unlimited-compute non-record submission for a two-stage Sophonics workflow:

1. train a compact base language model once
2. quantize that base aggressively
3. freeze it
4. train tiny localized LoRA repair modules
5. merge the repair back into ordinary weights
6. quantize the merged model to `int8` and compress it for the final artifact

The key result is that a heavily quantized `int5` base can be repaired back toward `int8` quality with a very small amount of learned capacity concentrated in the deepest MLP blocks.

Longer-form writeup:

- full white paper placeholder: `https://arxiv.org/abs/XXXX.XXXXX`

## Main Result

Winning repair configuration:

- Base checkpoint: 1xRTX5090 600s base run documented in `base_train.log`
- Repair target: `blocks.7-8.mlp.(fc|proj)`
- Base precision during repair: `int5`
- Reference precision during repair: `int8`
- LoRA rank: `16`
- Trainable parameters: `98,304`
- Repair steps: `600`

Validation results:

- `1024`-sequence slice:
  - `int8` baseline: `1.3327 BPB`
  - `int5` base: `1.5084 BPB`
  - repaired merged checkpoint: `1.3461 BPB`
  - recovered `92.4%` of the `int5 -> int8` gap
- Full validation:
  - merged repaired checkpoint: `1.3390 BPB`
  - merged repaired checkpoint after uniform `int8` roundtrip: `1.3411 BPB`

Artifact size:

- compressed model (`int8+zlib`): `10,924,302 bytes`
- required code snapshot:
  - `train_gpt.py`
  - `sophonic_lora_repair.py`
  - `sophonic_eval.py`
- conservative code bytes: `91,821`
- conservative total bytes: `11,016,123`

This leaves `4,983,877` bytes of headroom under the `16,000,000` byte cap.

## What Sophonics Means Here

The original static Sophonics idea was:

- low-bit base weights
- plus compressed higher-precision residual patches

That failed. Low-rank residuals recovered only about `1%` to `3%` of the lost `int5/int6 -> int8` performance.

The working formulation is different:

- low-bit base weights
- plus tiny learned repair modules attached only where quantization damage matters most

In this submission, the “sophons” are those small learned repair modules.

Important implementation detail:

- the current submission artifact is **not** yet a runtime package of `int5 base + explicit live Sophonic modules`
- the repair modules are trained on top of the frozen quantized base and then **merged back into ordinary weights**
- the merged checkpoint is then uniformly quantized to `int8` for the final artifact

So this result proves:

- a strongly compressed base can be behaviorally restored by small localized learned expansions
- the repaired model still fits comfortably under the `16MB` cap

It does **not** yet prove a final runtime that keeps an `int5` substrate and separate conditional Sophonic modules alive during inference.

## Why “Deepest MLPs”

This model has transformer blocks `blocks.0` through `blocks.8`.

Each block contains:

- attention projections
- an MLP with two large matrices:
  - `mlp.fc`
  - `mlp.proj`

The winning target set was:

- `blocks.7.mlp.fc`
- `blocks.7.mlp.proj`
- `blocks.8.mlp.fc`
- `blocks.8.mlp.proj`

These are the deepest two MLP blocks, meaning the feed-forward layers closest to the model output.

## Localization Evidence

The same parameter budget was tested in shallow MLP blocks as a negative control.

Control configuration:

- target: `blocks.0-1.mlp.(fc|proj)`
- same base precision: `int5`
- same LoRA rank: `16`
- same trainable parameters: `98,304`
- same repair length: `600` steps

Control result on the `1024`-sequence slice:

- repaired merged checkpoint: `1.4092 BPB`
- recovered only `56.5%` of the `int5 -> int8` gap

Interpretation:

- learned repair helps broadly
- but it is much more effective in the deepest MLP blocks

## Key Experiments

| Experiment | Eval set | Repaired BPB | Recovery | Notes |
| --- | --- | ---: | ---: | --- |
| Static residual Sophonics, `int5`, rank-4 | trained-base full-val | `1.4988` | `1%` | negative result |
| LoRA repair, `int6`, `blocks.7-8.mlp`, rank-16 | `1024`-seq slice | `1.3367` | `90.6%` | best `int6` result |
| LoRA repair, `int5`, `blocks.7-8.mlp`, rank-16, 600 steps | `1024`-seq slice | `1.3461` | `92.4%` | main result |
| LoRA repair, `int5`, `blocks.0-1.mlp`, rank-16, 600 steps | `1024`-seq slice | `1.4092` | `56.5%` | negative control |
| Winning merged checkpoint, pre-quant | full validation | `1.3390` | — | full-val gate |
| Winning merged checkpoint, `int8+zlib` | full validation | `1.3411` | — | final artifact score |

## Why This Is Non-Record

This submission is not intended for the 10-minute record track.

Reasons:

- the method is two-stage rather than a single integrated training script
- the final artifact depends on a post-training repair phase
- the current goal is to validate the Sophonics idea, not to beat leaderboard SOTA

The natural next step is to integrate this repair mechanism into a stronger base run and eventually test whether a true runtime `int5 + Sophonic repair` packaging can compete on the main track.

## Setup Notes

The included scripts assume a standard PyTorch environment with the packages listed in `requirements.txt`. They also expect the cached FineWeb SP-1024 dataset and tokenizer paths used by the main competition repo.

## Included Files

- `base_train.log` — exact 1xRTX5090 base-model training log
- `repair_train.log` — exact LoRA repair training log for the winning `int5` run
- `submission_check.log` — full-validation and artifact-size verification log
- `requirements.txt` — minimal Python package requirements for the included scripts
- `train_gpt.py` — base model training snapshot
- `sophonic_eval.py` — evaluation and quantization helpers
- `sophonic_lora_repair.py` — working repair implementation
- `sophonic_submission_check.py` — full-val and size verification helper
- `submission.json` — leaderboard metadata
