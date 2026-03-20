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
git remote set-url origin git@github.com:Kevxn97/parameter-golf.git
git remote add upstream https://github.com/openai/parameter-golf.git 2>/dev/null || true
git fetch origin
git checkout codex/my-experiment
git pull
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
- Commit: `1c25913` initially, then compatibility follow-up pending
- Goal: test whether shared-block recurrent depth improves the baseline under tight parameter budget
- Hypothesis: reusing the same `NUM_LAYERS` blocks for multiple `NUM_LOOPS` will increase effective depth and expressivity faster than widening the model, while keeping unique parameter bytes close to baseline
- Code changes: add `NUM_LOOPS` to `train_gpt.py` and reuse the same block stack across loops during forward pass; log unique/effective depth at startup
- Command:
  `RUN_ID=looped_shared_depth_v1 DATA_PATH=./data/datasets/fineweb10B_sp1024/ TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model VOCAB_SIZE=1024 NUM_LAYERS=6 NUM_LOOPS=2 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2 TRAIN_LOG_EVERY=50 VAL_LOSS_EVERY=200 torchrun --standalone --nproc_per_node=1 train_gpt.py`
- Machine: RunPod 1xH100 first, then scale only if promising
- Dataset/tokenizer: `fineweb10B_sp1024`, initial run can use `--train-shards 1` for smoke/bootstrap
- Metrics: pending
- Artifact size: pending
- Outcome: pending
- Next action: rerun from the compatibility-patched branch on RunPod and compare stability/speed/quantized `val_bpb` to standard 9-layer baseline

## Decisions and Learnings

- 2026-03-20: Git-based sync is the default workflow between local and RunPod; avoid manual file copying unless it is a one-off emergency.
- 2026-03-20: Environment/bootstrap failures belong in this tracker because they affect iteration speed and reproducibility.
- 2026-03-20: We will use this file to decide which architectural or evaluation changes are worth promoting into real submission candidates.
- 2026-03-20: Public-branch sync from RunPod should use HTTPS remotes unless the pod has an authorized GitHub SSH key.
- 2026-03-20: Hosted GPU images may lag on PyTorch features; challenge code should not assume `scaled_dot_product_attention(enable_gqa=...)` exists.

## Next Actions

- Finish pod bootstrap with `pip install -r requirements.txt`.
- Run a clean 1xH100 baseline with `--train-shards 1`.
- Choose the first real experiment branch and log it here before launching.
