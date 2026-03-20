# Parameter Golf Setup

This checkout is set up for two loops:

1. Local iteration on Apple Silicon with `train_gpt_mlx.py`
2. Remote training on RunPod with `train_gpt.py`

## Current local status

As of 2026-03-20 on this machine:

- Repo cloned at `/Users/kevin/Code/OAI_Paramgolf`
- `.venv` created
- `requirements.txt` installed
- `mlx` installed
- `gh auth status` passed for GitHub user `Kevxn97`
- Existing SSH key found at `~/.ssh/id_ed25519.pub`

## Local Mac bootstrap

If you need to recreate the environment:

```bash
cd /Users/kevin/Code/OAI_Paramgolf
./scripts/bootstrap_mac.sh
source .venv/bin/activate
./scripts/verify_env.sh
```

## First local smoke run

The smallest useful local loop is:

```bash
cd /Users/kevin/Code/OAI_Paramgolf
source .venv/bin/activate
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1
RUN_ID=mlx_smoke \
ITERATIONS=200 \
TRAIN_BATCH_TOKENS=8192 \
VAL_LOSS_EVERY=0 \
VAL_BATCH_SIZE=8192 \
python3 train_gpt_mlx.py
```

Notes:

- `--train-shards 1` keeps the first download smaller for smoke testing.
- Validation still uses the full fixed validation split.
- Once that works, bump `--train-shards` and training settings gradually.

## RunPod checklist

Before launching a pod:

```bash
cat ~/.ssh/id_ed25519.pub
```

Copy that public key into RunPod Settings so SSH access works.

Then launch the official Parameter Golf template linked from the upstream README.
Inside the pod:

```bash
cd /workspace
git clone https://github.com/openai/parameter-golf.git
cd parameter-golf
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1
RUN_ID=baseline_sp1024 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

Current observed behavior:

- On 2026-03-20 the RunPod template did not have Python package dependencies preinstalled, despite upstream README wording suggesting they should be available.
- Installing `requirements.txt` on the pod is therefore part of the practical bootstrap flow for this workspace.

Useful knobs:

- Set `MAX_WALLCLOCK_SECONDS=0` for longer-than-10-minute exploratory runs.
- Use `VAL_LOSS_EVERY=200` if you want periodic validation logging.
- Stay on cheaper 1x GPU pods while iterating; save 8xH100 SXM for serious leaderboard attempts.

## GitHub participation flow

This checkout currently points `origin` at the upstream repo. Before pushing your own work, create a fork and switch `origin` to it, keeping the OpenAI repo as `upstream`.

Suggested flow:

```bash
gh repo fork openai/parameter-golf --clone=false
git remote rename origin upstream
git remote add origin git@github.com:Kevxn97/parameter-golf.git
git fetch origin
git fetch upstream
```

Verify with:

```bash
git remote -v
```

Current repo state for this workspace:

- `origin` -> `git@github.com:Kevxn97/parameter-golf.git`
- `upstream` -> `https://github.com/openai/parameter-golf.git`

Recommended daily sync flow:

```bash
git checkout main
git fetch upstream
git pull upstream main
git push origin main
```

Experiment branch flow:

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
git remote add upstream https://github.com/openai/parameter-golf.git 2>/dev/null || true
git fetch origin
git checkout codex/my-experiment
git pull
```

Track all of this in `EXPERIMENT_TRACKER.md`.

If the pod does not have your GitHub SSH key loaded, prefer HTTPS remotes for fetch/pull on public branches. SSH is only needed if you plan to push from inside the pod.

## Submission shape

When you have a result worth sharing, add a new folder under the appropriate `records/` track with:

- `README.md`
- `submission.json`
- training log(s)
- a runnable `train_gpt.py` and any local dependencies for that record folder

For new SOTA claims, upstream currently requires:

- at least `0.005` nats better than the current record
- enough runs to support `p < 0.01`
- reproducibility under 10 minutes on `8xH100 SXM`
