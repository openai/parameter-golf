# Parameter Golf Experiment Tracker

This file is the working memory for this repo. We update it whenever we change architecture, hyperparameters, evaluation, environment, branch state, or run results.

## Objective

Track experiments, evaluations, infra state, and conclusions so we stay synchronized between:

- local development in `/Users/kevin/Code/OAI_Paramgolf`
- remote execution on RunPod
- future submission candidates under `records/`

## Working Rules

- Every meaningful run gets logged, including failed runs.
- Every run entry should include branch, commit, command, machine, data slice, and result.
- Every conclusion should produce a next action or decision.
- If local and RunPod diverge, the git branch/commit is the source of truth.

## Current Repo State

As of 2026-03-20 14:50 CET:

- Local repo path: `/Users/kevin/Code/OAI_Paramgolf`
- `origin`: `git@github.com:Kevxn97/parameter-golf.git`
- `upstream`: `https://github.com/openai/parameter-golf.git`
- Local Python env: `.venv` installed and verified
- RunPod access: working over SSH
- RunPod bootstrap caveat: install `requirements.txt` manually on pod before running downloader or training

## Current Baseline Context

Upstream baseline shape:

- `VOCAB_SIZE=1024`
- `NUM_LAYERS=9`
- `MODEL_DIM=512`
- `NUM_HEADS=8`
- `NUM_KV_HEADS=4`
- `MLP_MULT=2`
- tied embeddings enabled

Important code entry points:

- Training path: `train_gpt.py`
- Local Apple Silicon path: `train_gpt_mlx.py`
- Model topology: `GPT` / `Block` / attention / MLP in `train_gpt.py`
- Submission examples: `records/track_10min_16mb/`

## Sync Protocol

### Local -> GitHub -> RunPod

```bash
git checkout -b codex/my-experiment
git add .
git commit -m "Describe the experiment"
git push -u origin codex/my-experiment
```

On RunPod:

```bash
cd /workspace/parameter-golf
git remote set-url origin https://github.com/Kevxn97/parameter-golf.git
git fetch origin
git checkout -B codex/my-experiment origin/codex/my-experiment
git rev-parse --short HEAD
```

Before launching a run, verify the expected feature marker for that branch is present:

```bash
grep -n "FEATURE_MARKER_HERE" train_gpt.py
```

### Upstream refresh

```bash
git checkout main
git fetch upstream
git pull upstream main
git push origin main
```

## Run Log

| Date | ID | Branch | Commit | Machine | Data | Change | Command | Result | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-03-20 | infra-001 | `main` | n/a | local macOS | none | cloned repo, created `.venv`, installed deps, verified MLX/PyTorch env | `./scripts/verify_env.sh` | success | local development environment ready |
| 2026-03-20 | infra-002 | `main` | n/a | RunPod 1xH100 pod | none | first pod bootstrap attempt from clean clone | `python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1` and baseline `torchrun` | failed | missing `huggingface_hub` and `sentencepiece`; pod required manual `pip install -r requirements.txt` |
| 2026-03-20 | infra-003 | `main` | n/a | RunPod 1xH100 pod | `fineweb10B_sp1024` val + 1 train shard | first baseline launch after installing deps | baseline `torchrun --standalone --nproc_per_node=1 train_gpt.py` | failed | pod image has `torch 2.4.1+cu124`; this build does not support `enable_gqa` in SDPA, so baseline code needed compatibility fallback |
| 2026-03-20 | infra-004 | `main` | n/a | RunPod 1xH100 pod | none | tried pulling experiment branch from fork over SSH | `git fetch origin` after `origin=git@github.com:Kevxn97/parameter-golf.git` | failed | pod had no GitHub SSH key loaded; for public forks use HTTPS remotes for fetch/pull |
| 2026-03-20 | exp-002-attempt-001 | `codex/sliding-window-eval-v1` | unverified on pod | RunPod 1xH100 pod | `fineweb10B_sp1024` | attempted isolated sliding-window eval run | `RUN_ID=baseline_slide64_v1 ... EVAL_STRIDE=64 EVAL_BATCH_SEQS=128 torchrun --standalone --nproc_per_node=1 train_gpt.py` | ambiguous | run finished, but the expected `final_eval_mode:sliding_window` log line was missing and final eval time was only `11771ms`, so this result should not be treated as valid sliding-window evidence |
| 2026-03-20 | exp-002-valid-001 | `codex/sliding-window-eval-v1` | `a3798b2` verified on pod | RunPod 1xH100 pod | `fineweb10B_sp1024` | verified sliding-window post-quant eval run | `RUN_ID=baseline_slide64_v1 ... EVAL_STRIDE=64 EVAL_BATCH_SEQS=128 torchrun --standalone --nproc_per_node=1 train_gpt.py` | success | startup marker and source checks matched; final stride-64 eval produced `final_int8_zlib_roundtrip_exact val_bpb:1.32756718` |

## Experiment Scoreboard

This is the fast comparison view. One row per experiment, updated as soon as a run finishes.

| ID | Status | Branch | Commit | Main change | Final val_bpb | Size int8+zlib | Decision | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `exp-001` | complete | `codex/looped-shared-depth-v1` | `848f31f` | shared-depth looping (`NUM_LAYERS=6`, `NUM_LOOPS=2`) | `1.39779315` | `8567061` bytes | discard | very parameter-efficient, but quality and speed both poor |
| `exp-002` | complete | `codex/sliding-window-eval-v1` | `a3798b2` | sliding-window post-quant eval (`EVAL_STRIDE=64`) | `1.32756718` | `13385763` bytes | keep | valid run; strong gain, but final eval cost is very high (`834917ms`) |

## Hypothesis Backlog

Ideas to test, in rough priority order:

1. Increase depth while preserving artifact budget, e.g. `10-12` layers with adjusted width.
2. Reduce KV heads further and measure quality/size tradeoff.
3. Improve evaluation first, not training first, e.g. sliding-window eval.
4. Try weight sharing / looped blocks to buy effective depth under the 16 MB budget.
5. Explore alternative MLP/gating choices inside `Block`.

## Research Notes

Online research that informed the first experiment:

- `ALBERT: A Lite BERT for Self-supervised Learning of Language Representations` argues that cross-layer parameter sharing is a practical way to cut parameter growth while preserving or improving quality under memory constraints.
  Source: [arXiv:1909.11942](https://arxiv.org/abs/1909.11942)
- `Universal Transformers` argues that recurrent application of shared self-attention/transition blocks adds useful recurrent inductive bias while keeping the model parallel over sequence positions.
  Source: [arXiv:1807.03819](https://arxiv.org/abs/1807.03819)
- `Learning to Skip for Language Modeling` suggests that deeper effective computation can help, but fully dynamic routing/conditional execution adds complexity and tuning burden.
  Source: [arXiv:2311.15436](https://arxiv.org/abs/2311.15436)

First synthesis from those papers:

- For this challenge, a simple shared-block recurrent depth experiment is a better first move than a complex conditional-compute design.
- It directly targets the 16 MB artifact constraint by increasing effective depth without introducing a full second stack of unique block parameters.
- It keeps the diff small, easy to reason about, and reversible if it hurts optimization.

## Active Experiment Template

Copy this block when we start a new run:

```md
### exp-XXX
- Date:
- Owner:
- Branch:
- Commit:
- Goal:
- Hypothesis:
- Code changes:
- Command:
- Machine:
- Dataset/tokenizer:
- Metrics:
- Artifact size:
- Outcome:
- Next action:
```

### exp-001
- Date: 2026-03-20
- Owner: Kevin + Codex
- Branch: `codex/looped-shared-depth-v1`
- Commit: `848f31f`
- Goal: test whether shared-block recurrent depth improves the baseline under tight parameter budget
- Hypothesis: reusing the same `NUM_LAYERS` blocks for multiple `NUM_LOOPS` will increase effective depth and expressivity faster than widening the model, while keeping unique parameter bytes close to baseline
- Code changes: add `NUM_LOOPS` to `train_gpt.py` and reuse the same block stack across loops during forward pass; log unique/effective depth at startup
- Command:
  `RUN_ID=looped_shared_depth_v1 DATA_PATH=./data/datasets/fineweb10B_sp1024/ TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model VOCAB_SIZE=1024 NUM_LAYERS=6 NUM_LOOPS=2 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2 TRAIN_LOG_EVERY=50 VAL_LOSS_EVERY=200 torchrun --standalone --nproc_per_node=1 train_gpt.py`
- Machine: RunPod 1xH100 first, then scale only if promising
- Dataset/tokenizer: `fineweb10B_sp1024`, initial run can use `--train-shards 1` for smoke/bootstrap
- Metrics:
  - `sdpa_enable_gqa_supported:False`
  - `model_params:11549744 unique_layers:6 loops:2 effective_depth:12`
  - `step:200 val_bpb:1.6602`
  - `step:400 val_bpb:1.5148`
  - `step:600 val_bpb:1.4561`
  - `step:800 val_bpb:1.4212`
  - `step:1000 val_bpb:1.3988`
  - final `final_int8_zlib_roundtrip_exact val_bpb:1.39779315`
- Artifact size:
  - serialized model int8+zlib: `8506653` bytes
  - total submission size int8+zlib: `8567061` bytes
- Outcome: successful run, but decisively worse than the upstream baseline quality-wise while also being much slower per step
- Next action: stop investing in naive shared-depth looping; next experiment should prioritize quality-per-minute, likely via better evaluation or a stronger non-shared 10-layer baseline variant

### exp-002
- Date: 2026-03-20
- Owner: Kevin + Codex
- Branch: `codex/sliding-window-eval-v1`
- Commit: `a3798b2`
- Goal: test the highest-confidence isolated improvement path after exp-001 failed
- Hypothesis: sliding-window post-quant evaluation will improve reported `val_bpb` materially without changing training dynamics, because each scored validation token sees near-max context instead of the low-context chunk boundaries used by standard evaluation
- Why this next: repo evidence shows `SlidingWindowEval` improved post-quant `val_bpb` by about `0.032` with training otherwise unchanged, while stronger 10-layer records combine multiple simultaneous changes and are harder to attribute cleanly
- Code changes: add opt-in `EVAL_STRIDE` / `EVAL_BATCH_SEQS`, `forward_logits`, and `eval_val_sliding`; keep training-time validation and optimization unchanged
- Command:
  `RUN_ID=baseline_slide64_v1 DATA_PATH=./data/datasets/fineweb10B_sp1024/ TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model VOCAB_SIZE=1024 NUM_LOOPS=1 EVAL_STRIDE=64 EVAL_BATCH_SEQS=128 TRAIN_LOG_EVERY=50 VAL_LOSS_EVERY=200 torchrun --standalone --nproc_per_node=1 train_gpt.py`
- Machine: RunPod 1xH100
- Dataset/tokenizer: `fineweb10B_sp1024` + `fineweb_1024_bpe.model`
- Metrics:
  - verified on pod: `sdpa_enable_gqa_supported:False`
  - verified on pod: `final_eval_mode:sliding_window stride:64 batch_seqs:128`
  - verified on pod: `model_params:17059912 unique_layers:9 loops:1 effective_depth:9`
  - verified on pod: `step:1200 val_bpb:1.3615`
  - verified on pod: `step:1265 val_bpb:1.3590`
  - verified on pod: `final_int8_zlib_roundtrip_exact val_bpb:1.32756718`
  - verified on pod: `final_int8_ttt_lora val_bpb:1.3276`
  - verified on pod: `step_avg:474.34ms`
  - verified on pod: `final eval_time:834917ms`
  - comparison vs earlier baseline-like observed run: about `-0.0262 val_bpb` (`1.35379936 -> 1.32756718`)
- Artifact size:
  - verified on pod: serialized model int8+zlib `13320624` bytes
  - verified on pod: total submission size int8+zlib `13385763` bytes
- Outcome: valid and successful confirmation that sliding-window post-quant evaluation is a real win on this setup; the quality gain is strong enough to keep, but the eval-time cost is substantial
- Next action: keep sliding eval in the measurement stack and move to the next high-probability training-side lever, most likely the tuned non-shared 10-layer family

## Decisions and Learnings

- 2026-03-20: Git-based sync is the default workflow between local and RunPod; avoid manual file copying unless it is a one-off emergency.
- 2026-03-20: Environment/bootstrap failures belong in this tracker because they affect iteration speed and reproducibility.
- 2026-03-20: We will use this file to decide which architectural or evaluation changes are worth promoting into real submission candidates.
- 2026-03-20: Every experiment should be visible in two places: a detailed per-experiment block and a one-row scoreboard summary for quick comparison.
- 2026-03-20: Public-branch sync from RunPod should use HTTPS remotes unless the pod has an authorized GitHub SSH key.
- 2026-03-20: A branch name alone is not enough to trust remote experiment attribution; for every pod run we need the checked-out commit hash and at least one feature-specific source/log marker before treating results as valid evidence.
- 2026-03-20: Hosted GPU images may lag on PyTorch features; challenge code should not assume `scaled_dot_product_attention(enable_gqa=...)` exists.
- 2026-03-20: Simple block reuse (`6` unique layers, `2` loops) is very parameter-efficient on disk, but the quality hit is too large for this challenge in its current form.
- 2026-03-20: On 1xH100, the looped shared-depth variant ran at roughly `570ms/step`, far slower than the upstream 8xH100 baseline step budget, so it is not a good candidate for scale-up without further optimization.
- 2026-03-20: After a failed speculative architecture change, the next experiment should maximize causal clarity; isolated evaluation changes are a better follow-up than stacking several new training tricks at once.
- 2026-03-20: Verified sliding-window post-quant eval is a real gain on this setup, but it shifts a lot of wallclock into evaluation; we should keep it for scoring-quality experiments, while being deliberate about the slower turnaround time.

## Next Actions

- Finish pod bootstrap with `pip install -r requirements.txt`.
- Use `exp-002` as the new measurement baseline for future experiments.
- Build `exp-003` around the strongest next training-side lever, likely a tuned non-shared 10-layer variant combined with the now-validated sliding eval.
