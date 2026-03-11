# Parameter Golf

**OpenAI ModelCraft Challenge: Parameter Golf** is a challenge to train the best language model that fits in a 16MB artifact and trains in under 10 minutes on 8xH100s, evaluated by compression on the FineWeb validation set (tokenizer-agnostic, bits per byte).

This challenge is heavily inspired by the [NanoGPT Speedrunning](https://github.com/KellerJordan/modded-nanogpt) challenge, where participants compete to train a model that reaches 3.28 FineWeb validation loss as quickly as possible. We're excited to see how optimizing for a parameter-constrained setting pushes people toward unique architectures, compression schemes, and creative submissions.

## Participant Form

If you enjoy solving very difficult technical problems, please introduce yourself via the [Challenge Participant Form](https://jobs.ashbyhq.com/openai/form/open-ai-challenge-parameter-golf). It helps us attribute challenge submissions and reach out about opportunities with OpenAI. _Completing the form is not required to participate._

Many researchers at OpenAI first distinguished themselves through elite mathematics and programming competitions. The ModelCraft Challenge is designed in that spirit: testing the ability to tackle unfamiliar problems with creativity and rigor, qualities we believe are essential for frontier AI research.

In June, we plan to hire a small cohort of early-career researchers, targeting current undergraduate students and recent graduates, including Olympiad medalists and engineers who demonstrate unusual technical ability. For exceptional participants, the challenge may also serve as a way to stand out to OpenAI researchers and recruiters.

The challenge runs from March 18th to April 30th. 

Happy training!

## Leaderboard


| Rank | Run              | Score  | Author         | Summary                              | Date       | Code              | Description      |
|-----:|------------------|-------:|----------------|--------------------------------------|------------|-------------------|------------------|
| 1    | GQA-4 Mixed-Rows Quant (Strict <16MB) | 1.1454 | Codex          | 13x512 GQA-4, tied embeds; stricter mixed row/group int8 tuning, clipping, and scale encoding | 2026-02-21 | [code](records/track_10min/2026-02-21_GQA4_PartialPerRowMLPProj/train_gpt.py) | [info](records/track_10min/2026-02-21_GQA4_PartialPerRowMLPProj/README.md) |

## Getting Started

### Training Your First Model (Mac with Apple Silicon)

If you have an Apple laptop or desktop with Apple Silicon, we've set up a simple MLX training script to help you start iterating locally.

If you don't have a Mac with Apple Silicon, you can run an adapted version of this script without MLX support. Just ask [Codex](https://openai.com/codex/) to refactor it; the change is straightforward. It may still be fairly slow, so we recommend jumping straight to cloud GPUs with RunPod.

First, clone the repository, create a fresh Python environment, and install the packages needed for the MLX path plus dataset download:

```bash
git clone https://github.com/openai/parameter-golf.git
cd parameter-golf
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install mlx numpy sentencepiece huggingface-hub datasets tqdm
```

Download our cached version of FineWeb with the 1024-token vocabulary:

```bash
python3 data/cached_challenge_fineweb.py --variant sp1024 1
```

Then run a small MLX training job:

```bash
RUN_ID=mlx_smoke \
DATA_PATH=./data/challenge_fineweb/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/challenge_fineweb/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
NUM_LAYERS=8 \
MODEL_DIM=384 \
NUM_HEADS=6 \
MLP_MULT=3 \
TRAIN_MAX_SEQ_LEN=512 \
ITERATIONS=200 \
TRAIN_BATCH_TOKENS=8192 \
GRAD_ACCUM_STEPS=8 \
VAL_LOSS_EVERY=10 \
VAL_TOKENS=8192 \
VAL_BATCH_TOKENS=1024 \
python3 train_gpt_mlx.py
```

You should see printed `val_loss` and `val_bpb` values, along with a compressed model size.

### Scaling Up to a Remote Machine

Once you're happy with your local tests, or you want more compute, switch to a remote CUDA machine.

You can rent GPUs from anywhere, but OpenAI is partnering with RunPod to make setup as easy as possible.

We also know compute is expensive, so OpenAI is sponsoring $1,000,000 in compute credits to help people get started training their models. To request a compute grant of up to $500, use this form: [Request a Compute Grant](url).

#### Launching a 1xH100 Pod

1. First, [create a RunPod account](https://console.runpod.io/deploy). You should also set up an SSH key in the Settings tab on the left so you can connect to your remote machine. If you're new to this, ask Codex to help you set it up.

2. Once you've set up your account, create a new GPU Cloud Pod. You can choose whichever GPU SKU you'd like. Final leaderboard submissions must run in under 10 minutes on 8xH100s, but we strongly recommend testing and running experiments on cheaper SKUs first, since an 8xH100 box can cost around $20/hour.

3. Let's start with a 1xH100 pod. Configure your pod to use the RunPod PyTorch 2.1 template and enable SSH terminal access, leaving the other settings at their defaults. Deploy your pod and SSH into it once it's up.

On your remote machine, clone the repo and start a fresh environment for the PyTorch trainer:

```bash
git clone https://github.com/openai/parameter-golf.git /workspace/parameter-golf
cd /workspace/parameter-golf
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Download our cached version of FineWeb. We'll use the 1024-token vocabulary for now.

```bash
python3 data/cached_challenge_fineweb.py --variant sp1024 1
```

Launch your first training run. Note that we're passing `nproc_per_node=1` because we're running on a single H100 GPU in this case.

```bash
RUN_ID=baseline_sp1024 \
DATA_PATH=./data/challenge_fineweb/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/challenge_fineweb/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

Double-check that you see printed `val_loss` and `val_bpb` values around ~1.2, along with a compressed model size under 16MB.

## OPENAI ONLY

1. Assume you already have a live `8xH100` box. Pull locally before syncing so the pod gets the repo state you actually want, then push your local checkout to the box and SSH to pod `0`.

```bash
git pull
brix git push <pool>
brix ssh <pool>-0
```

2. On the pod, use the pyenv Python and make a repo-local environment. Do not use `/usr/bin/python3`.

```bash
cd /root/code/parameter-golf
/root/.pyenv/versions/3.12.9/bin/python -m venv .venv-openai
. .venv-openai/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

3. Set explicit Triton and Inductor cache dirs inside the repo. Reuse these env vars for compile checks and training reruns.

```bash
export TRITON_CACHE_DIR=/root/code/parameter-golf/.cache/triton
export TORCHINDUCTOR_CACHE_DIR=/root/code/parameter-golf/.cache/inductor
export XDG_CACHE_HOME=/root/code/parameter-golf/.cache
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
```

4. Stage the cached SP1024 tokenizer and dataset shards with `bbb`:

```bash
mkdir -p data/matched_10B_docs2m_seed1337/tokenizers
mkdir -p data/matched_10B_docs2m_seed1337/datasets/fineweb10B_sp1024
bbb cp az://oaidatasets2/speedrunkits/matched_10B_docs2m_seed1337/tokenizers/fineweb_1024_bpe.model data/matched_10B_docs2m_seed1337/tokenizers/fineweb_1024_bpe.model
bbb cp az://oaidatasets2/speedrunkits/matched_10B_docs2m_seed1337/datasets/fineweb10B_sp1024/fineweb_train_000001.bin data/matched_10B_docs2m_seed1337/datasets/fineweb10B_sp1024/fineweb_train_000001.bin
bbb cp az://oaidatasets2/speedrunkits/matched_10B_docs2m_seed1337/datasets/fineweb10B_sp1024/fineweb_val_000000.bin data/matched_10B_docs2m_seed1337/datasets/fineweb10B_sp1024/fineweb_val_000000.bin
```

5. Launch the normal trainer:

```bash
cd /root/code/parameter-golf
. .venv-openai/bin/activate
export TRITON_CACHE_DIR=/root/code/parameter-golf/.cache/triton
export TORCHINDUCTOR_CACHE_DIR=/root/code/parameter-golf/.cache/inductor
export XDG_CACHE_HOME=/root/code/parameter-golf/.cache
RUN_ID=openai_smoke_sp1024 \
DATA_PATH=/root/code/parameter-golf/data/matched_10B_docs2m_seed1337/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/root/code/parameter-golf/data/matched_10B_docs2m_seed1337/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
TRAIN_BATCH_TOKENS=262144 \
WARMUP_STEPS=2 \
ITERATIONS=8 \
VAL_LOSS_EVERY=4 \
VAL_TOKENS=131072 \
VAL_BATCH_SIZE=65536 \
MAX_WALLCLOCK_SECONDS=120 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

This path was smoke-tested on `pgolf-zebra-openai-0` on March 11, 2026: compile stayed enabled, all `8` ranks came up, warmup completed, and the run finished `8` real training steps with `step:8/8 val_bpb:4.1021`. Peak memory was about `6.9 GiB` per GPU. On the immediate rerun with the same cache dirs, the first measured train step dropped from `2911ms` to `721ms`, with later train steps around `30-35ms`.

Reminders:

```bash
git pull
brix git push <pool>
```

## FAQ

**What exactly counts toward the 16MB artifact size?**

The submission artifact is computed as code bytes plus compressed model bytes. All counted code should live in the `train_gpt.py` script.
The cap is decimal 16MB, i.e. 16,000,000 total bytes, not 16 MiB / 16,777,216 bytes.
No external downloads, dataset access, or network calls are allowed during evaluation. The artifact must be fully self-contained and reproducible.

**Are scores independently verified by OpenAI?**

We're not automatically verifying every submission, but we will verify the top leaderboard entries over time. Any non-reproducible results can be disqualified, and issues reproducing submissions should be raised on the PR.

**What counts as 'external compute'? For example, is it fair to tune my hyperparameters offline?**

There's no perfectly clear answer here, and it's hard to draw a clean line around what does or does not count as external compute. For now, we're reserving the right to disqualify runs that are not in the spirit of the challenge. Tuning your Adam hyperparameters across a bunch of runs is fine, but if there's evidence that you're sneaking in additional compute unfairly, such as brute-forcing a seed, we won't allow it. Use your best judgment, and there's no penalty for asking questions.

(etc)

## Submission Process

New SOTA records must fulfill the following criteria:

1. They must beat the existing SOTA by at least 0.005 nats. As in modded-nanogpt, because of inter-run variance all submissions must provide enough run logs to show at `p < 0.01` that they achieved the required 0.005-nat improvement. For submissions that improve speed through systems optimization without changing the ML, this requirement is waived.

2. If changes are made to the tokenizer or dataset, prove with certainty that the val_bpb is correctly calculated. Submissions that edit the tokenizer will be examined much more carefully, since bugs may unjustly improve your score.

3. Reproducibly run in under 10 minutes on 8xH100s.

All submissions should be made as a pull request that only adds a new folder to the appropriate `/records` subfolder and includes the following files. Submissions without the full set of requirements will not be accepted.

1. A README.md file that explains the submission in reasonable detail.

2. A `submission.json` file (see the example runs) that includes your name, GitHub ID, `val_bpb`, and related metadata.

3. A train log, automatically produced by your script.

4. A `train_gpt.py` script and any other dependencies. Note: this must successfully compile and run within the records folder. Broken scripts will not be accepted.

### Non-record Submissions

Submissions are also open to unique and interesting approaches that might not beat the existing SOTA, but still satisfy the 16MB artifact limit. We strongly encourage participants to submit implementations for weird or out-of-the-box ideas, in-progress or unoptimized solutions, so long as they run successfully, or even interesting negative results. We're excited to see what you come up with. We'll still maintain a high bar for non-record submissions, so be sure to justify your ideas and results in detail when submitting.

We also accept non-record submissions to an unlimited compute track for runs that are not intended to meet the 10-minute cutoff. Just note as such in your README file.

Non-record submissions should be made in the same fashion as SOTA records, as described above.

#### PRs on Core Code

The `train_gpt.py` and `train_gpt_mlx.py` scripts are intended as good launching-off points for new participants, not SOTA configs. We'll accept PRs that tune, improve, or simplify these scripts without significantly increasing complexity, but the best models should stay in the `/records` folder.

## Support

Reach out to parametergolf@openai.com for any other questions.

Join the [Discord server](url).
