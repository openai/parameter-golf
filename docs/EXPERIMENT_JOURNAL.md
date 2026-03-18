# Parameter Golf Experiment Journal

Last updated: 2026-03-18
Current phase: challenge orientation and baseline capture

## Purpose of this file

This is the living lab notebook for this repo.
Every meaningful run, failed idea, parameter sweep, architectural change, and compression observation should be logged here so we do not learn the same lesson twice.

This file should help us answer:

- What did we try?
- Why did we try it?
- Under which exact settings did we run it?
- What happened before and after quantization?
- Did bytes go up or down?
- What should we try next?

## Logging rules

Every new run entry should include, at minimum:

1. Date and local timestamp
2. Run name / run id
3. Track
4. Goal or hypothesis
5. Exact code snapshot
6. Exact command
7. Dataset and tokenizer
8. Key changed parameters
9. Relevant environment notes
10. Result metrics
11. Artifact sizes
12. Decision: keep, discard, or follow up
13. Next experiment idea
14. Run trust fields: run state, evidence tier, planner eligibility, parse warnings
15. Funnel fields: funnel stage, follow-up reason, follow-up parent, promotion target stage

## Comparison discipline

When comparing two runs, explicitly note whether these stayed the same:

- dataset export
- tokenizer
- validation path
- training time cap
- world size / gradient accumulation
- sequence length
- batch tokens
- quantization path
- code size

If any of those changed, say so in plain language.

## Run entry template

Copy this block for each new run:

```md
## YYYY-MM-DD HH:MM - <run name>

- Status: planned | running | completed | invalidated
- Run state:
- Evidence tier:
- Planner eligible:
- Parse warnings:
- Funnel stage:
- Follow-up of experiment:
- Follow-up reason:
- Promotion target stage:
- Track:
- Hypothesis:
- Thesis:
- Hypothesis family:
- Lineage id:
- Expected upside:
- Risk level:
- Kill criteria:
- Promotion rule:
- Code snapshot:
- Branch:
- Commit:
- Source files changed:
- Dataset:
- Tokenizer:
- Command:
- Important env vars:
- Hardware:
- Wallclock target:
- Notes during run:
- Pre-quant metrics:
- Final roundtrip metrics:
- Artifact sizes:
- Outcome:
- Follow-up:
```

## Baseline snapshot table

| Date | Run | Track | Dataset | Tokenizer | Final `val_bpb` | Pre-quant `val_bpb` | Total bytes | Notes |
|---|---|---|---|---|---:|---:|---:|---|
| 2026-03-18 | Naive Baseline | `track_10min_16mb` | `fineweb10B_sp1024` | `sp1024` | `1.22436570` | `1.2172` | `15,863,489` | Current official baseline in local repo |
| 2026-03-18 | 4-Hour Baseline | `track_non_record_16mb` | `quasi10Bfrom50B_50keval_sp1024_v0` | `sp1024` | `1.20737944` | `1.1749` | `15,810,161` | Unlimited-compute reference |

## Captured baseline runs

## 2026-03-18 00:00 - Naive Baseline

- Status: completed
- Track: `track_10min_16mb`
- Hypothesis: establish a simple official reference run under the record-track constraints
- Expected upside: not an upside run; this is the anchor for all future comparisons
- Risk: none, other than mistaking it for an optimized baseline
- Code snapshot: local root `train_gpt.py` copied into the record folder
- Branch: repository snapshot from official repo
- Commit: local history shows `0c0ea98` as latest visible commit in this clone
- Source files changed: none by us; recorded run lives in `records/track_10min_16mb/2026-03-17_NaiveBaseline/`
- Dataset: published `fineweb10B_sp1024`
- Tokenizer: `data/tokenizers/fineweb_1024_bpe.model`
- Command:

```bash
NCCL_IB_DISABLE=1 \
RUN_ID=hf_verify_sp1024_8gpu \
DATA_PATH=/root/code/parameter-golf/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/root/code/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=600 \
TRAIN_LOG_EVERY=50 \
VAL_LOSS_EVERY=200 \
torchrun --standalone --nproc_per_node=8 /root/code/parameter-golf/train_gpt.py
```

- Important env vars:
  - `VOCAB_SIZE=1024`
  - `MAX_WALLCLOCK_SECONDS=600`
  - `VAL_LOSS_EVERY=200`
- Hardware: `8xH100`
- Wallclock target: `600s`
- Notes during run:
  - timed loop stopped at `13780/20000` because of the wallclock cap
  - validation ran periodically on the full fixed validation split
- Pre-quant metrics:
  - `val_loss=2.0606`
  - `val_bpb=1.2172`
- Final roundtrip metrics:
  - `val_loss=2.07269931`
  - `val_bpb=1.22436570`
- Artifact sizes:
  - int8+zlib model: `15,815,847`
  - code: `47,642`
  - total: `15,863,489`
- Outcome: this is the current official baseline target to beat on the record track
- Follow-up: every future main-track experiment should state whether it is directly comparable to this run

## 2026-03-18 00:00 - 4-Hour Baseline

- Status: completed
- Track: `track_non_record_16mb`
- Hypothesis: test how far the same compact architecture can go with far more training time
- Expected upside: identify the headroom between the 10-minute regime and longer-training regime
- Risk: results are not main-track comparable on wallclock
- Code snapshot: current root `train_gpt.py` at time of record capture
- Branch: repository snapshot from official repo
- Commit: exact commit not recorded in the run README; local repo head during review is `0c0ea98`
- Source files changed: none by us; recorded run lives in `records/track_non_record_16mb/2026-03-18_Quasi10Bfrom50B_SP1024_9x512_KV4_4h_pgut3/`
- Dataset: `quasi10Bfrom50B_50keval_sp1024_v0`
- Tokenizer: `fineweb_1024_bpe.model`
- Command:

```bash
OMP_NUM_THREADS=1 \
TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
RUN_ID=train_gpt_pgut3_quasi10b_sp1024_4h_20260318_075102 \
DATA_PATH=/tmp/fineweb_quasi10Bfrom50B_50keval_sp1024_v0/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/tmp/fineweb_quasi10Bfrom50B_50keval_sp1024_v0/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
NUM_LAYERS=9 \
MODEL_DIM=512 \
NUM_HEADS=8 \
NUM_KV_HEADS=4 \
MLP_MULT=2 \
TIE_EMBEDDINGS=1 \
TIED_EMBED_LR=0.05 \
ITERATIONS=500000 \
WARMUP_STEPS=20 \
MAX_WALLCLOCK_SECONDS=14400 \
TRAIN_BATCH_TOKENS=524288 \
TRAIN_SEQ_LEN=1024 \
TRAIN_LOG_EVERY=200 \
VAL_LOSS_EVERY=20000 \
torchrun --standalone --nproc_per_node=8 /root/code/parameter-golf/train_gpt.py
```

- Important env vars:
  - `ITERATIONS=500000`
  - `MAX_WALLCLOCK_SECONDS=14400`
  - `VAL_LOSS_EVERY=20000`
- Hardware: `8xH100`
- Wallclock target: `4h`
- Notes during run:
  - same compact 9x512 SP-1024 baseline layout
  - used as an unlimited-compute reference, not as a leaderboard-legal record attempt
- Pre-quant metrics:
  - `val_loss=1.9837`
  - `val_bpb=1.1749`
- Final roundtrip metrics:
  - `val_loss=2.03860961`
  - `val_bpb=1.20737944`
- Artifact sizes:
  - int8+zlib model: `15,762,519`
  - code: `47,642`
  - total: `15,810,161`
- Outcome: proves there is real headroom in the architecture, but also proves that post-quant degradation remains a central bottleneck
- Follow-up: future experiments should track both raw-model quality and roundtrip degradation explicitly

## Current starting assumptions

These are the assumptions we will use until a later experiment disproves them:

1. The post-quant roundtrip score matters more than the raw validation score.
2. Code-size growth is a real resource cost, not a cosmetic issue.
3. Unlimited-compute runs are useful for idea discovery but must be translated back into the 10-minute regime.
4. We should preserve apples-to-apples comparability whenever possible.
5. Every serious experiment should document both its intended gain and its failure mode.

## Current local workspace state

Verified on 2026-03-18:

- local dataset folder: `data/datasets/fineweb10B_sp1024/`
- present train shards: `1`
- present validation shards: `1`
- checked-in tokenizer artifacts: `.model` and `.vocab` for `sp1024`
- `docs_selected.jsonl` is not present locally
- source sidecar for document reconstruction is not present locally

Why this matters:

- Right now the repo is ready for orientation and smoke tests.
- It is not yet in a “full baseline reproduction” data state.
- Any run we do before downloading more shards must be labeled clearly as subset-only or smoke-only.

## Harness status

Harness v1 now exists in `harness/`.

Current capabilities:

- structured history in `lab/experiments.jsonl`
- isolated per-run workdirs under `lab/runs/`
- profile-based planning
- heuristic next-run proposal
- trainer-log parsing for both PyTorch and MLX shapes
- automatic journal append into the Harness Runs section below
- historical record bootstrap from `records/`
- comparability labels such as `smoke-only` and `subset-only`
- preflight plus challenge-readiness classification
- hard timeout plus no-progress timeout
- challenge-profile gate that refuses non-ready environments
- consistent torch launching through the same preferred Python used by preflight
- built-in `doctor` and `selfcheck` commands for operational status
- deterministic run-local code mutations for trainer-policy experiments
- trust-layer fields that mark whether a run is actually safe for autonomous planner reuse
- hypothesis metadata on planned runs so promising lines can be followed intentionally

Still intentionally missing in v1:

- free-form repo-editing autonomous code changes
- automatic promotion into `records/`
- resume/restart logic
- parallel multi-run scheduling
- architecture-mutation strategies beyond env-vars and a small deterministic code-mutation registry

Important trust rule:

- failed, blocked, dry-run, or invalid-metric entries should still be logged
- only planner-eligible runs with validated metrics should influence autonomous next-step planning

## Harness Runs

Automatically generated entries from the local experiment harness live here.
The structured source of truth is `lab/experiments.jsonl`; this section is the human-readable mirror.

<!-- HARNESS_RUNS_START -->
## 2026-03-18T19:29:47Z - harness::20260318_202742_mlx_smoke_0003_baseline

- Status: failed
- Profile: `mlx_smoke`
- Track: `local-smoke`
- Objective: establish comparable baseline
- Hypothesis: Run the profile baseline end-to-end so the harness has a clean local anchor before it starts mutating anything.
- Rationale: There is no comparable completed history for this profile yet, so the planner starts with a baseline capture.
- Parent run: `none`
- Mutation tag: `baseline`
- Comparability: `subset-only`
- Script: `/Users/kevin/Code/ParameterGolf_OAI/train_gpt_mlx.py`
- Run directory: `/Users/kevin/Code/ParameterGolf_OAI/lab/runs/20260318_202742_mlx_smoke_0003_baseline`
- Command: `/Users/kevin/Code/ParameterGolf_OAI/.venv/bin/python /Users/kevin/Code/ParameterGolf_OAI/train_gpt_mlx.py`
- Important env vars:
```bash
DATA_PATH=/Users/kevin/Code/ParameterGolf_OAI/data/datasets/fineweb10B_sp1024
GRAD_ACCUM_STEPS=1
ITERATIONS=3
MAX_WALLCLOCK_SECONDS=0
OUT_DIR=.
TOKENIZER_PATH=/Users/kevin/Code/ParameterGolf_OAI/data/tokenizers/fineweb_1024_bpe.model
TRAIN_BATCH_TOKENS=8192
TRAIN_LOG_EVERY=1
TRAIN_SEQ_LEN=1024
VAL_BATCH_SIZE=8192
VAL_LOSS_EVERY=1
VOCAB_SIZE=1024
WARMUP_STEPS=0
```
- Log file: `/Users/kevin/Code/ParameterGolf_OAI/lab/runs/20260318_202742_mlx_smoke_0003_baseline/20260318_202742_mlx_smoke_0003_baseline.txt`
- Stdout capture: `/Users/kevin/Code/ParameterGolf_OAI/lab/runs/20260318_202742_mlx_smoke_0003_baseline/stdout.txt`
- Dataset: `fineweb10B_sp1024`
- Train shards: `1` / `195`
- Best pre-quant `val_bpb`: `None`
- Last pre-quant `val_bpb`: `None`
- Final roundtrip `val_bpb`: `None`
- Quantization gap `val_bpb`: `None`
- Serialized int8+zlib bytes: `None`
- Total submission bytes: `None`
- Raw model bytes: `None`
- Stop step: `None` / `None`
- Stopped on wallclock: `False`
- Model params: `17059912`
- Notes: local subset warning was present in the trainer log, so this run is not comparable to a full-data baseline.

## 2026-03-18T20:03:39Z - harness::20260318_210339_torch_record_8gpu_0006_baseline

- Status: blocked
- Profile: `torch_record_8gpu`
- Track: `record-10min-16mb`
- Objective: establish comparable baseline
- Hypothesis: Run the profile baseline end-to-end so the harness has a clean local anchor before it starts mutating anything.
- Rationale: There is no comparable completed history for this profile yet, so the planner starts with a baseline capture.
- Parent run: `none`
- Mutation tag: `baseline`
- Comparability: `subset-only`
- Script: `/Users/kevin/Code/ParameterGolf_OAI/train_gpt.py`
- Run directory: `/Users/kevin/Code/ParameterGolf_OAI/lab/runs/20260318_210339_torch_record_8gpu_0006_baseline`
- Command: `torchrun --standalone --nproc_per_node=1 /Users/kevin/Code/ParameterGolf_OAI/train_gpt.py`
- Important env vars:
```bash
DATA_PATH=/Users/kevin/Code/ParameterGolf_OAI/data/datasets/fineweb10B_sp1024
MAX_WALLCLOCK_SECONDS=600
TOKENIZER_PATH=/Users/kevin/Code/ParameterGolf_OAI/data/tokenizers/fineweb_1024_bpe.model
TRAIN_BATCH_TOKENS=524288
TRAIN_LOG_EVERY=50
TRAIN_SEQ_LEN=1024
VAL_LOSS_EVERY=200
VOCAB_SIZE=1024
```
- Log file: `missing`
- Stdout capture: `/Users/kevin/Code/ParameterGolf_OAI/lab/runs/20260318_210339_torch_record_8gpu_0006_baseline/stdout.txt`
- Dataset: `unknown`
- Train shards: `None` / `None`
- Best pre-quant `val_bpb`: `None`
- Last pre-quant `val_bpb`: `None`
- Final roundtrip `val_bpb`: `None`
- Quantization gap `val_bpb`: `None`
- Serialized int8+zlib bytes: `None`
- Total submission bytes: `None`
- Raw model bytes: `None`
- Stop step: `None` / `None`
- Stopped on wallclock: `None`
- Model params: `None`
- Preflight launchable: `False`
- Challenge ready: `False`
- Preflight fatal issues: `required module missing: torch; unable to determine available CUDA GPU count`
- Preflight warnings: `dataset subset detected: train shards 1/195`
- Failure reason: `preflight_failed`

## 2026-03-18T21:49:47Z - harness::20260318_224640_mlx_smoke_0009_baseline

- Status: failed
- Profile: `mlx_smoke`
- Track: `local-smoke`
- Objective: establish comparable baseline
- Hypothesis: Run the profile baseline end-to-end so the harness has a clean local anchor before it starts mutating anything.
- Thesis: Run the profile baseline end-to-end so the harness has a clean local anchor before it starts mutating anything.
- Hypothesis family: `baseline_capture`
- Hypothesis id: `hyp::mlx_smoke::baseline_capture::baseline`
- Lineage id: `hyp::mlx_smoke::baseline_capture::baseline`
- Parent hypothesis: `none`
- Expected upside: establish a trusted local anchor for later comparisons
- Risk level: `low`
- Kill criteria: invalidate only if the run fails to produce planner-eligible metrics
- Promotion rule: promote only after a completed run establishes a clean local baseline
- Rationale: There is no comparable completed history for this profile yet, so the planner starts with a baseline capture.
- Parent run: `none`
- Mutation tag: `baseline`
- Mutation kind: `baseline`
- Comparability: `subset-only`
- Run state: `failed_pre_train`
- Evidence tier: `invalid`
- Planner eligible: `False`
- Source script: `/Users/kevin/Code/ParameterGolf_OAI/train_gpt_mlx.py`
- Materialized script: `/Users/kevin/Code/ParameterGolf_OAI/lab/runs/20260318_224640_mlx_smoke_0009_baseline/train_gpt_mlx.py`
- Run directory: `/Users/kevin/Code/ParameterGolf_OAI/lab/runs/20260318_224640_mlx_smoke_0009_baseline`
- Command: `/Users/kevin/Code/ParameterGolf_OAI/.venv/bin/python /Users/kevin/Code/ParameterGolf_OAI/lab/runs/20260318_224640_mlx_smoke_0009_baseline/train_gpt_mlx.py`
- Important env vars:
```bash
DATA_PATH=/Users/kevin/Code/ParameterGolf_OAI/data/datasets/fineweb10B_sp1024
GRAD_ACCUM_STEPS=8
GRAD_CLIP_NORM=0.0
ITERATIONS=5
MATRIX_LR=0.04
MAX_WALLCLOCK_SECONDS=0
OUT_DIR=.
QK_GAIN_INIT=1.5
SCALAR_LR=0.04
TIED_EMBED_LR=0.05
TOKENIZER_PATH=/Users/kevin/Code/ParameterGolf_OAI/data/tokenizers/fineweb_1024_bpe.model
TRAIN_BATCH_TOKENS=8192
TRAIN_LOG_EVERY=1
TRAIN_SEQ_LEN=1024
VAL_BATCH_SIZE=8192
VAL_LOSS_EVERY=1
VOCAB_SIZE=1024
WARMDOWN_ITERS=1200
WARMUP_STEPS=0
```
- Log file: `/Users/kevin/Code/ParameterGolf_OAI/lab/runs/20260318_224640_mlx_smoke_0009_baseline/20260318_224640_mlx_smoke_0009_baseline.txt`
- Stdout capture: `/Users/kevin/Code/ParameterGolf_OAI/lab/runs/20260318_224640_mlx_smoke_0009_baseline/stdout.txt`
- Dataset: `fineweb10B_sp1024`
- Train shards: `1` / `195`
- Best pre-quant `val_bpb`: `None`
- Last pre-quant `val_bpb`: `None`
- Final roundtrip `val_bpb`: `None`
- Quantization gap `val_bpb`: `None`
- Serialized int8+zlib bytes: `None`
- Total submission bytes: `None`
- Raw model bytes: `None`
- Code bytes: `49177`
- Code bytes delta: `0`
- Stop step: `None` / `None`
- Stopped on wallclock: `False`
- Model params: `17059912`
- Preflight launchable: `True`
- Challenge ready: `False`
- Ready for execution: `True`
- Preflight warnings: `dataset subset detected: train shards 1/195`
- Parse warnings: `subset_dataset_detected; no_final_roundtrip_metric_detected; no_train_or_val_lines_detected; trainer_output_signature_weak`
- Failure reason: `no_progress_timeout`
- Notes: local subset warning was present in the trainer log, so this run is not comparable to a full-data baseline.

## 2026-03-18T21:57:22Z - harness::20260318_225416_mlx_smoke_0010_baseline

- Status: failed
- Profile: `mlx_smoke`
- Track: `local-smoke`
- Objective: establish comparable baseline
- Hypothesis: Run the profile baseline end-to-end so the harness has a clean local anchor before it starts mutating anything.
- Thesis: Run the profile baseline end-to-end so the harness has a clean local anchor before it starts mutating anything.
- Hypothesis family: `baseline_capture`
- Hypothesis id: `hyp::mlx_smoke::baseline_capture::baseline`
- Lineage id: `hyp::mlx_smoke::baseline_capture::baseline`
- Parent hypothesis: `none`
- Expected upside: establish a trusted local anchor for later comparisons
- Risk level: `low`
- Kill criteria: invalidate only if the run fails to produce planner-eligible metrics
- Promotion rule: promote only after a completed run establishes a clean local baseline
- Rationale: There is no comparable completed history for this profile yet, so the planner starts with a baseline capture.
- Parent run: `none`
- Mutation tag: `baseline`
- Mutation kind: `baseline`
- Comparability: `subset-only`
- Run state: `failed_mid_run`
- Evidence tier: `subset`
- Planner eligible: `False`
- Source script: `/Users/kevin/Code/ParameterGolf_OAI/train_gpt_mlx.py`
- Materialized script: `/Users/kevin/Code/ParameterGolf_OAI/lab/runs/20260318_225416_mlx_smoke_0010_baseline/train_gpt_mlx.py`
- Run directory: `/Users/kevin/Code/ParameterGolf_OAI/lab/runs/20260318_225416_mlx_smoke_0010_baseline`
- Command: `/Users/kevin/Code/ParameterGolf_OAI/.venv/bin/python /Users/kevin/Code/ParameterGolf_OAI/lab/runs/20260318_225416_mlx_smoke_0010_baseline/train_gpt_mlx.py`
- Important env vars:
```bash
DATA_PATH=/Users/kevin/Code/ParameterGolf_OAI/data/datasets/fineweb10B_sp1024
GRAD_ACCUM_STEPS=1
GRAD_CLIP_NORM=0.0
ITERATIONS=2
MATRIX_LR=0.04
MAX_WALLCLOCK_SECONDS=0
OUT_DIR=.
QK_GAIN_INIT=1.5
SCALAR_LR=0.04
TIED_EMBED_LR=0.05
TOKENIZER_PATH=/Users/kevin/Code/ParameterGolf_OAI/data/tokenizers/fineweb_1024_bpe.model
TRAIN_BATCH_TOKENS=2048
TRAIN_LOG_EVERY=1
TRAIN_SEQ_LEN=1024
VAL_BATCH_SIZE=8192
VAL_LOSS_EVERY=0
VOCAB_SIZE=1024
WARMDOWN_ITERS=1200
WARMUP_STEPS=0
```
- Log file: `/Users/kevin/Code/ParameterGolf_OAI/lab/runs/20260318_225416_mlx_smoke_0010_baseline/20260318_225416_mlx_smoke_0010_baseline.txt`
- Stdout capture: `/Users/kevin/Code/ParameterGolf_OAI/lab/runs/20260318_225416_mlx_smoke_0010_baseline/stdout.txt`
- Dataset: `fineweb10B_sp1024`
- Train shards: `1` / `195`
- Best pre-quant `val_bpb`: `None`
- Last pre-quant `val_bpb`: `None`
- Final roundtrip `val_bpb`: `None`
- Quantization gap `val_bpb`: `None`
- Serialized int8+zlib bytes: `None`
- Total submission bytes: `None`
- Raw model bytes: `None`
- Code bytes: `49177`
- Code bytes delta: `0`
- Stop step: `2` / `2`
- Stopped on wallclock: `False`
- Model params: `17059912`
- Preflight launchable: `True`
- Challenge ready: `False`
- Ready for execution: `True`
- Preflight warnings: `dataset subset detected: train shards 1/195`
- Parse warnings: `subset_dataset_detected; no_final_roundtrip_metric_detected`
- Failure reason: `no_progress_timeout`
- Notes: local subset warning was present in the trainer log, so this run is not comparable to a full-data baseline.

<!-- HARNESS_RUNS_END -->

## Next planned work

Immediate next phase after the first harness bring-up:

1. keep the harness stable while we stay in the no-experiments phase
2. switch from subset-only local validation to full comparable data only when we intentionally move into challenge execution
3. improve planner quality later, but only after the real challenge environment is available
