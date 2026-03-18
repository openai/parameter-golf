<img width="3840" height="1280" alt="1920x640-discord" src="https://github.com/user-attachments/assets/90607b26-171f-476a-90ae-69b9dbb7cb30" />

<br>
<br>

**OpenAI Model Craft Challenge: Parameter Golf** is a challenge to train the best language model that fits in a 16MB artifact and trains in under 10 minutes on 8xH100s, evaluated by compression on the FineWeb validation set (tokenizer-agnostic, bits per byte).

This challenge is heavily inspired by the [NanoGPT Speedrunning](https://github.com/KellerJordan/modded-nanogpt) challenge, where participants compete to train a model that reaches 3.28 FineWeb validation loss as quickly as possible. We're excited to see how optimizing for a parameter-constrained setting pushes people toward unique architectures (test-time compute, aggressive parameter tying, depth recurrence, low-rank training, ...), compression schemes (low precision, QAT, bitnets, novel tokenizers, ...), and other creative submissions (test-time training, long context, megakernels ...). 

If you're familiar with [neural scaling laws](https://arxiv.org/abs/2001.08361), you can consider this challenge a form of L(N) optimization, where the objective is to optimize the lowest loss given a fixed number of parameters (N) unconstrained by data, compute, steps, or architecture. Challenges like the [NanoGPT Speedrun](https://github.com/KellerJordan/modded-nanogpt), which optimizes for a form of L(T) (~lowest loss given constrained time) or the [NanoGPT Slowrun](https://github.com/qlabs-eng/slowrun), which optimizes for L(D) (lowest lost given constrained dataset size), can be thought of as equivalent challenges in this family.

Ideally, we'd allow for submissions to use arbitrary computational resources. But in order to make the challenge not inaccessibly expensive, we're limiting *leaderboard submissions* to 10 minutes on 8xH100s. However, we'd still love to see submissions that don't meet the compute limitation requirements in our 'Non-record Submissions' section: We're excited to see people push the infinite frontier of parameter limited performance as well.

## Participant Form

If you enjoy solving very difficult technical problems, please introduce yourself via the [Challenge Participant Form](https://jobs.ashbyhq.com/openai/form/open-ai-challenge-parameter-golf). It helps us attribute challenge submissions and reach out about opportunities with OpenAI. _Completing the form is not required to participate._

Many researchers at OpenAI first distinguished themselves through elite mathematics and programming competitions. The Model Craft Challenge is designed in that spirit: testing the ability to tackle unfamiliar problems with creativity and rigor, qualities we believe are essential for frontier AI research.

In June, we plan to hire a small cohort of early-career researchers, targeting current undergraduate students and recent graduates, including Olympiad medalists and elite competitors. For exceptional participants, the challenge may also serve as a way to stand out to OpenAI researchers and recruiters.

The challenge runs from March 18th to April 30th. 

Happy training!

## Leaderboard


| Rank | Run | Score | Author | Summary | Date | Info |
|-----:|-----|------:|--------|---------|------|------|
| 1 | Naive Baseline | 1.2169 | Baseline | 9layer 512dim 1024vocab TiedEmbeddings 4 KV heads | 2026-03-18 | [info](records/track_10min_16mb/2026-03-17_SP1024_9x512_KV4_VectorQK_Q150_FP32Clip_pgut3/README.md) |

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
By default this downloads only the first training shard for the selected variant so short local runs do not need the full dataset. To fetch more shards, pass `--train-shards`, for example `python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 4`.

Then run a small MLX training job:

```bash
RUN_ID=mlx_smoke \
TRAIN_SEQ_LEN=512 \
ITERATIONS=200 \
TRAIN_BATCH_TOKENS=8192 \
VAL_LOSS_EVERY=0 \
VAL_BATCH_SIZE=8192 \
python3 train_gpt_mlx.py
```

Validation always runs on the full `fineweb_val_*` split, which is the fixed first-50k-document set. The smoke command above skips periodic validation and just prints the final `val_loss` and `val_bpb` once at the end.

### Scaling Up to a Remote Machine

Once you're happy with your local tests, or you want more compute, switch to a remote CUDA machine.

You can rent GPUs from anywhere, but OpenAI is partnering with Runpod to make setup as easy as possible.

We also know compute is expensive, so OpenAI is sponsoring $1,000,000 in compute credits to help people get started training their models. To request a compute grant of up to $500, use this form: [Request a Compute Grant](url).

#### Launching a 1xH100 Pod

1. First, [create a Runpod account](https://console.runpod.io/deploy). You should also set up an SSH key in the Settings tab on the left so you can connect to your remote machine. If you're new to this, ask Codex to help you set it up.

2. Once you've set up your account, create a new GPU Cloud Pod. You can choose whichever GPU SKU you'd like. Final leaderboard submissions must run in under 10 minutes on 8xH100s, but we strongly recommend testing and running experiments on cheaper SKUs first, since an 8xH100 box can cost around $20/hour.

3. Let's start with a 1xH100 pod. Deploy using the official Parameter Golf template: [Launch Template](https://console.runpod.io/deploy?template=y5cejece4j&ref=nl2r56th). Enable SSH terminal access, leaving the other settings at their defaults. Deploy your pod and SSH into it once it's up. You should be in `/workspace/parameter-golf/`.

On your remote machine, clone the repo onto local disk. All Python dependencies are already pre-installed in the image.

```bash
git clone https://github.com/openai/parameter-golf.git .
cd parameter-golf
```

Download our cached version of FineWeb. We'll use the 1024-token vocabulary for now.

```bash
python3 data/cached_challenge_fineweb.py --variant sp1024
```

This defaults to the first training shard only. For a larger local subset, pass `--train-shards N`.

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

### Rebuilding the Large Export

To rebuild the larger blobstore-backed export locally or on a remote box, use:

```bash
python3 data/export_blobstore_fineweb100B_tokenizer_datasets.py \
  --output_root /tmp/matched_100B_train30Btok_even_seed1337 \
  --selection_mode even \
  --selection_seed 1337 \
  --target_train_tokens 30000000000 \
  --sp_vocab_sizes 512,1024,2048,4096 \
  --tokenizer_train_docs 5000000 \
  --skip_byte
```

This writes a shared `docs_selected.jsonl`, a `docs_selected.source_manifest.json` sidecar with source-shard metadata, tokenizers, dataset shards, and a final `manifest.json`. Copying the whole export root uploads the docs cache too, so others can retrain tokenizers or rebuild matching shards from the same selected document stream. The downloader reads shard counts from that manifest, and by default only fetches the first train shard so short runs do not need the full export.

For the current blobstore-canonical `10B` export with a fixed `50k`-doc validation prefix and byte plus `512/1024/2048` SentencePiece variants, use:

```bash
python3 data/export_blobstore_fineweb100B_tokenizer_datasets.py \
  --output_root /tmp/fineweb_blobstore100B_train10B_val50k \
  --selection_mode even \
  --selection_seed 1337 \
  --target_train_tokens 10000000000 \
  --num_val_docs 50000 \
  --sp_vocab_sizes 512,1024,2048 \
  --tokenizer_train_docs 5000000
```

This keeps the blobstore as the canonical source, takes the first `50k` documents from the blobstore val stream, then exports shuffled selected train runs until the raw GPT-2 token budget is met.

### Exporting From FineWeb Samples

To export a smaller family directly from Hugging Face FineWeb samples, for example `10B` raw GPT-2 tokens from `sample-350BT` with a fixed `50k`-doc validation prefix and byte plus `512/1024/2048` SentencePiece variants:

```bash
python3 data/export_hf_fineweb_sample_tokenizer_datasets.py \
  --dataset_name sample-350BT \
  --output_root /tmp/fineweb_sample350BT_train10B \
  --target_train_tokens 10000000000 \
  --num_val_docs 50000 \
  --sp_vocab_sizes 512,1024,2048 \
  --tokenizer_train_docs 5000000
```

This keeps the val split simple: the first `50k` streamed docs become validation, and training continues on the same sample stream until the raw GPT-2 token budget is reached.

For CPU-heavy local exports, two useful knobs are:

```bash
MATCHED_FINEWEB_GPT2_COUNT_BATCH_SIZE=512
MATCHED_FINEWEB_SP_BATCH_SIZE=2048
```

The first controls exact batched GPT-2 token counting while building the docs cache, and the second controls batched tokenizer encoding during shard export.


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

This repository adapts code from `modded-nanogpt`, see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for attribution.