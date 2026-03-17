<img width="3840" height="1280" alt="1920x640-discord" src="https://github.com/user-attachments/assets/90607b26-171f-476a-90ae-69b9dbb7cb30" />

<br>
<br>

**OpenAI Model Craft Challenge: Parameter Golf** is a challenge to train the best language model that fits in a 16MB artifact and trains in under 10 minutes on 8xH100s, evaluated by compression on the FineWeb validation set (tokenizer-agnostic, bits per byte).

This challenge is heavily inspired by the [NanoGPT Speedrunning](https://github.com/KellerJordan/modded-nanogpt) challenge, where participants compete to train a model that reaches 3.28 FineWeb validation loss as quickly as possible. We're excited to see how optimizing for a parameter-constrained setting pushes people toward unique architectures (test-time compute, aggressive parameter tying, depth recurrence, low-rank training, ...), compression schemes (low precision, QAT, bitnets, novel tokenizers, ...), and other creative submissions (test-time training, long context, ...). 

If you're familiar with [neural scaling laws](https://arxiv.org/abs/2001.08361), you can consider this challenge a form of L(N) optimization, where the objective is to optimize the lowest loss given a fixed number of parameters (N) unconstrained by data, compute, steps, or architecture. Challenges like the [NanoGPT Speedrun](https://github.com/KellerJordan/modded-nanogpt), which optimizes for a form of L(T) (~lowest loss given constrained time) or the [NanoGPT Slowrun](qlabs-eng/slowrun), which optimizes for L(D) (lowest lost given constrained dataset size), can be thought of as equivalent challenges in this family.

Ideally, we'd allow for submissions to use arbitrary computational resources. But in order to make the challenge not inaccessibly expensive, we're limiting *leaderboard submissions* to 10 minutes on 8xH100s. However, we'd still love to see submissions that don't meet the compute limitation requirements in our 'Non-record Submissions' section: We're excited to see people push the infinite frontier of parameter limited performance as well.

## Participant Form

If you enjoy solving very difficult technical problems, please introduce yourself via the [Challenge Participant Form](https://jobs.ashbyhq.com/openai/form/open-ai-challenge-parameter-golf). It helps us attribute challenge submissions and reach out about opportunities with OpenAI. _Completing the form is not required to participate._

Many researchers at OpenAI first distinguished themselves through elite mathematics and programming competitions. The Model Craft Challenge is designed in that spirit: testing the ability to tackle unfamiliar problems with creativity and rigor, qualities we believe are essential for frontier AI research.

In June, we plan to hire a small cohort of early-career researchers, targeting current undergraduate students and recent graduates, including Olympiad medalists and elite competitors. For exceptional participants, the challenge may also serve as a way to stand out to OpenAI researchers and recruiters.

The challenge runs from March 18th to April 30th. 

Happy training!

## Leaderboard


| Rank | Run              | Score  | Author         | Summary                              | Date       | Info              |
|-----:|------------------|-------:|----------------|--------------------------------------|------------|-------------------|
| 1    | 2048vocab d11 448dim |  1.2166 | Codex | 11layer 448dim 2048vocab tied embed 4 KV heads & row-wise `int8+zlib` | 2026-03-12 | [info](records/track_10min_16mb/2026-03-11_SP2048_11x448_KV4_Tied_pgze1/README.md) |
| 2    | Very Simple Baseline| 1.2499 | Codex 5.4          | 9layer 512dim 1024vocab tied embed 4 KV heads & row-wise `int8+zlib` | 2026-03-12 | [info](records/track_10min_16mb/2026-03-11_Baseline/README.md) |

## Getting Started

### Training Your First Model (Mac with Apple Silicon)

If you have an Apple laptop or desktop with Apple Silicon, we've set up a simple MLX training script to help you start iterating locally.

If you don't have a Mac with Apple Silicon, you can run an adapted version of this script without MLX support. Just ask [Codex](https://openai.com/codex/) to refactor it; the change is straightforward. It may still be fairly slow, so we recommend jumping straight to cloud GPUs with Runpod.

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
python3 data/cached_challenge_fineweb.py --variant sp1024
```

This populates `./data/datasets/fineweb10B_sp1024/` and `./data/tokenizers/`.
By default this downloads all training shards for the selected variant. If you only want a quick smoke-test download, pass a smaller shard count instead, for example `python3 data/cached_challenge_fineweb.py --variant sp1024 1`.

Then run a small MLX training job:

```bash
RUN_ID=mlx_smoke \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
NUM_LAYERS=8 \
MODEL_DIM=384 \
NUM_HEADS=6 \
NUM_KV_HEADS=2 \
MLP_MULT=3 \
TRAIN_SEQ_LEN=512 \
ITERATIONS=200 \
TRAIN_BATCH_TOKENS=8192 \
GRAD_ACCUM_STEPS=8 \
VAL_LOSS_EVERY=10 \
VAL_TOKENS=8192 \
VAL_BATCH_SIZE=8192 \
python3 train_gpt_mlx.py
```

You should see printed `val_loss` and `val_bpb` values, along with a compressed model size.

### Scaling Up to a Remote Machine

Once you're happy with your local tests, or you want more compute, switch to a remote CUDA machine.

You can rent GPUs from anywhere, but OpenAI is partnering with Runpod to make setup as easy as possible.

We also know compute is expensive, so OpenAI is sponsoring $1,000,000 in compute credits to help people get started training their models. To request a compute grant of up to $500, use this form: [Request a Compute Grant](url).

#### Launching a 1xH100 Pod

1. First, [create a Runpod account](https://console.runpod.io/deploy). You should also set up an SSH key in the Settings tab on the left so you can connect to your remote machine. If you're new to this, ask Codex to help you set it up.

2. Once you've set up your account, create a new GPU Cloud Pod. You can choose whichever GPU SKU you'd like. Final leaderboard submissions must run in under 10 minutes on 8xH100s, but we strongly recommend testing and running experiments on cheaper SKUs first, since an 8xH100 box can cost around $20/hour.

3. Let's start with a 1xH100 pod. Deploy using the official Parameter Golf template: [Launch Template](https://console.runpod.io/deploy?template=y5cejece4j&ref=nl2r56th). Enable SSH terminal access, leaving the other settings at their defaults. Deploy your pod and SSH into it once it's up.

On your remote machine, clone the repo onto local disk. All Python dependencies are already pre-installed in the image.

```bash
mkdir -p /root/code
git clone https://github.com/openai/parameter-golf.git /root/code/parameter-golf
cd /root/code/parameter-golf
```

Download our cached version of FineWeb. We'll use the 1024-token vocabulary for now.

```bash
python3 data/cached_challenge_fineweb.py --variant sp1024
```

Launch your first training run. Note that we're passing `nproc_per_node=1` because we're running on a single H100 GPU in this case.

```bash
RUN_ID=baseline_sp1024 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

By default, `train_gpt.py` keeps its ~10 minute wallclock cap. If you want a longer run, override it explicitly, for example `MAX_WALLCLOCK_SECONDS=0`.

By default, this command prints `train_loss` step logs during training and prints `val_loss`, `val_bpb`, and compressed model size in the final `final_int8_zlib_roundtrip` lines at the end. If you want periodic validation logs during the run, set `VAL_LOSS_EVERY`, for example `VAL_LOSS_EVERY=200`. For the baseline config, the final `val_bpb` should land around ~1.2 with a compressed model size under 16MB.


## FAQ

**What exactly counts toward the 16MB artifact size?**

The submission artifact is computed as code bytes plus compressed model bytes. All counted code should live in the `train_gpt.py` script.
The cap is decimal 16MB, i.e. 16,000,000 total bytes, not 16 MiB / 16,777,216 bytes.
No external downloads, training dataset access, or network calls are allowed during evaluation. The artifact must be fully self-contained and reproducible.

**Are scores independently verified by OpenAI?**

We're not automatically verifying every submission, but we will verify the top leaderboard entries over time. Any non-reproducible results can be disqualified, and issues reproducing submissions should be raised on the PR.

**What counts as 'external compute'? For example, is it fair to tune my hyperparameters offline?**

There's no perfectly clear answer here and it's hard to draw a clean line around what does or does not count as external compute. For now, we're reserving the right to disqualify runs that are not in the spirit of the challenge. Tuning your Adam hyperparameters across a bunch of runs is fine, but if there's evidence that you're sneaking in additional compute unfairly, such as brute-forcing ridiculous seeds, we won't allow it. Use your best judgment and there's no penalty for asking questions.

**What are the restrictions on evaluation?**

We won't accept submissions that take more than 10 minutes on 8xH100 to evaluate, but otherwise you're free to evaluate however. As with modded-nanogpt, we allow evaluation at any sequence length. And, obviously, you aren't allowed to access any training data during evaluation, unless you pay for those bits in the <16MB limit. We encourage competitors to push the bounds of evaluation methods as aggressively as with training methods.

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
