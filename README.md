<img width="384" height="256" alt="Parameter Golf Logo" src="https://github.com/user-attachments/assets/5afa2832-f306-45cf-b819-eeb971ee560b" />

[Placeholder Readme]

**OpenAI ModelCraft Challenge: Parameter Golf** is a challenge to train the best language model that fits in a 16MB (16,000,000-byte, not 16 MiB) artifact + trains in <10 minutes on 8xH100, evaluated by their FineWeb validation set compression (tokenizer-agnostic, bits per byte).

This challenge takes heavy inspiration from the [NanoGPT Speedrunning](https://github.com/KellerJordan/modded-nanogpt) challenge, where individuals compete to train a model that reaches 3.28 FineWeb validation loss as fast as possible. We're excited to see how optimizing for a parameter-constrained setting pushes people towards unique architectures, compression schemes, and creative submission.

The challenge runs from March 18th to April [PLACEHOLDER]. 

If you enjoy solving very difficult technical problems, please introduce yourself via the [Challenge Participant Form](https://you.ashbyhq.com/meeting/3187917e-c244-4aae-9b92-5399048f0677/)
, which allows us to attribute challenge submissions and reach out about opportunities with OpenAI. _Completing the form is not required to participate._

Many researchers at OpenAI first distinguished themselves through elite mathematics and programming competitions. The ModelCraft Challenge is designed in that spirit: testing the ability to tackle unfamiliar problems with creativity and rigor, qualities we believe are essential for frontier AI research.

In June, we plan to hire a small cohort of early-career researchers, targeted at current undergraduate students and recent graduates, including Olympiad medalists and engineers who demonstrate unusual technical ability. For exceptional participants, the challenge may also serve as a way to stand out to OpenAI researchers and recruiting.

Happy training!

### Training your first model (Mac with Apple Silicon)

If you have a Apple laptop or desktop with Apple Silicon, we've setup a simple MLX training script that makes it simple to start iterating locally. 

If you don't have a Mac with Apple Silicon, you can run an adapted version of this script without MLX support (Just ask [Codex](codex) to refactor! It's pretty simple) but it may be fairly slow. We'd recommend jumping straight to working on cloud GPUs with RunPod (see below).

First, clone the repository, create a fresh Python environment, and install the packages needed for the MLX path plus dataset download:

```bash
git clone https://github.com/openai/parameter-golf.git
cd /parameter-golf
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install mlx numpy sentencepiece huggingface-hub datasets tqdm
```

Download our cached version of Fineweb with the 1024 vocab tokenizer:

```bash
python3 data/cached_challenge_fineweb.py --variant sp1024 1
```

Then run a small MLX training job:

```bash
RUN_ID=mlx_smoke \
DATA_PATH=./data/fineweb10B_sp4k \
TOKENIZER_PATH=./data/tokenizers/fineweb_4k_bpe.model \
VOCAB_SIZE=4096 \
NUM_LAYERS=8 \
MODEL_DIM=384 \
NUM_HEADS=6 \
MLP_MULT=3 \
TRAIN_MAX_SEQ_LEN=512 \
ITERATIONS=50 \
TRAIN_BATCH_TOKENS=8192 \
GRAD_ACCUM_STEPS=8 \
VAL_LOSS_EVERY=10 \
VAL_TOKENS=8192 \
VAL_BATCH_TOKENS=1024 \
python3 train_gpt_mlx.py
```

You should see a printed `val_loss` and `val_bpb`, as well as a compressed model size. 

[explainer]

### Scaling up to a remote machine

Once you're happy with your local tests, or want to move on to a setup with a bit more juice, switch to a remote CUDA machine. 

You can rent GPUs from anywhere, but OpenAI is partnering with RunPod to make setup as easy as possible.

We also know compute is expensive, so OpenAI is sponsoring $1,000,000 in compute credits for people to get started training their models. To request a credit grant (up to $500), request at this form here: [Request a Compute Grant](url).

#### Launching a 1xH100 pod

1) First, you'll need to [create a RunPod account](https://console.runpod.io/deploy). You'll also want to setup an SSH key in the Settings tab on the left side [list more steps for people] so you can connect to your remote machine.

2) Once you've setup your account, create a new GPU Cloud Pod. You can choose whichever GPU SKU you'd like! Note that all final leaderboard submissions should run in under 10 minutes on 8xH100s, but we'd strongly recommend testing and running experiments on cheaper SKUs given a 8xH100 box can cost ~$20/hour. 

3) Let's start with a 1xH100 pod. Configure your pod to use (1) the Runpod Pytorch 2.1 template and (2) enable SSH terminal access, otherwise keeping default settings. Deploy your pod and SHH into it once it's up. We're ready to start training!

On your remote machine, clone the repo and start a fresh environment for the PyTorch trainer:

```bash
git clone https://github.com/openai/parameter-golf.git /root/parameter-golf
cd /root/parameter-golf
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Download our cached version of Fineweb. We'll use the 1024 vocab tokenizer for now.

```bash
python3 data/cached_challenge_fineweb.py --variant sp1024 1
```

Launch your first training run! Note that we're passing nproc_per_node==1 since we're running on a single H100 GPU.

```bash
RUN_ID=baseline_sp1024 VOCAB_SIZE=1024 torchrun --standalone --nproc_per_node=1 train_gpt.py
```

Double check that you see a printed `val_loss` and `val_bpb` around [x value], as well as a compressed model size under 16MB. 

### Leaderboard

*Track source:* `records/track_10min`  
*Score metric shown below:* `submission.json.loss` (lower is better). Most rows are `final_int8_zlib_roundtrip val_bpb`; rows that differ are called out in the summary.

| Rank | Run              | Score  | Author         | Summary                              | Date       | Code              | Description      |
|-----:|------------------|-------:|----------------|--------------------------------------|------------|-------------------|------------------|
| 1    | GQA-4 Mixed-Rows Quant (Strict <16MB) | 1.1454 | Codex          | 13x512 GQA-4, tied embeds; same valid 7375-step checkpoint with stricter mixed row/group int8 tuning, clipping, and scale encoding to optimize post-quant `val_bpb` under 16,000,000 bytes | 2026-02-21 | [code](records/track_10min/2026-02-21_GQA4_PartialPerRowMLPProj/train_gpt.py) | [info](records/track_10min/2026-02-21_GQA4_PartialPerRowMLPProj/README.md) |
| 2    | Baseline (SP-2048 11x512, flash+untied beat) | 1.1539 | Codex          | Flash-only + untied baseline rerun; beats user baseline score, but train_time was contaminated by background jobs (>10m) | 2026-02-22 | [code](records/track_10min/2026-02-22_Baseline_SP2048_512x11_FlashUntied_Beat/train_gpt.py) | [info](records/track_10min/2026-02-22_Baseline_SP2048_512x11_FlashUntied_Beat/README.md) |
| 3    | SP-2048 11x512 KV2 (10min + per-row int8 beat) | 1.1551 | Codex          | Exact 10-minute KV2 train run (`NUM_KV_HEADS=2`) plus compiled checkpoint-reload int8 tuning (global per-row 2D) to beat the user baseline under 16,000,000 bytes | 2026-02-22 | [code](records/track_10min/2026-02-22_SP2048_512x11_KV2_PerRowAll_10minBeat/train_gpt.py) | [info](records/track_10min/2026-02-22_SP2048_512x11_KV2_PerRowAll_10minBeat/README.md) |
| 4    | Baseline (SP-2048 11x512, rerun) | 1.1819 | Codex          | Reproducible baseline rerun (full log + code snapshot; this rerun is >10m train_time) | 2026-02-22 | [code](records/track_10min/2026-02-21_Baseline_SP2048_512x11_Rerun/train_gpt.py) | [info](records/track_10min/2026-02-21_Baseline_SP2048_512x11_Rerun/README.md) |
| 5    | GPT-Simple No-Tie (SP-1024 9x256, 1-LR tuned) | 1.3319 | Codex          | Untied `train_gpt_simple` baseline with one-global-LR sweep + 8x DDP confirmation; clean 596.6s run. Score is direct final `val_bpb` (no int8/zlib). | 2026-02-22 | [code](records/track_10min/2026-02-22_GPTSimpleNoTie_SP1024_256x9_OneLR_DDP8_10min/train_gpt_simple_no_tied_embeddings.py) | [info](records/track_10min/2026-02-22_GPTSimpleNoTie_SP1024_256x9_OneLR_DDP8_10min/README.md) |

### FAQ

**What exactly counts toward the 16MB artifact size?**

Submission artifact is computed as code bytes + compressed model bytes. All code should live in the train.py script to be counted.
The cap is decimal 16MB, i.e. 16,000,000 total bytes, not 16 MiB / 16,777,216 bytes.
No external downloads, dataset access, or network calls are allowed during evaluation. The artifact must be fully self-contained and reproducible.

**Are scores independently verified by OpenAI?**

We're not automatically verifying every submission, but we will verify the top leaderboard entries over time. Any non-reproducible results can be disqualified, and issues with reproducing submissions should be brought up on the PR.

**What counts as 'external compute'? For ex, is it fair to tune my hyperparameters offline?**

There's no clear answer unfortunately, and it's hard to delineate what does or does not count as external compute. For now, we're reserving the right to disqualify runs that we find are 'not in the spirit of the challenge.' Tuning your Adam hyperparameters across a bunch of runs is fine, but if there's evidence you're sneaking in additional compute unfairly (brute forcing some seed, etc.) we won't allow that. Use your best judgement, and there's no penalty for asking questions.

(etc)

### Submission Process

New SOTA records must fulfill the following criteria:

1. They must beat the existing SOTA by at least 0.005 nats. (Same as modded-nanogpt, due to inter-run variance, submissions must provide enough run logs to attain a statistical significance level of p<0.01 that they achieved the sufficient 0.005 nat win. For submissions which improve speed by optimizing the systems performance, without touching the ML, this requirement is waived.) 

2. If changes are made to the tokenizer or dataset, prove with certainty that the val_bpb is correctly calculated. Submissions that edit the tokenizer will be examined much more carefully, since bugs may unjustly improve your score.

3. Reproducibly run in under 10 minutes on 8xH100s.

All submissions should be made by a Pull Request that purely adds a new folder to the appropriate /records sub-folder and includes the following files. Submissions without the complete set of requirements will not be accepted.

1. A README.md file that explains the submission in reasonable detail.

2. A submission.json (see example runs) that includes your name, Github ID, val_bpb, etc. 

3. A train log, automatically produced by your script.

4. A train_gpt.py script and any other dependencies. Note: This must sucessfully compile and run within the records folder! Broken scripts will not be accepted.

### Non-record submissions

Submissions are also open to unique and interesting approaches that don't beat the existing SOTA, given review by OpenAI judges. We strongly encourage participants to submit implementations for weird or out-of-the-box ideas, in-progress or unoptimized solutions (so long as they succesfully run), or even interesting negative results. We're excited to see what you come up with!

Non-record submissions should be made in the same fashion as SOTA records, see above. 
Note: We'll be maintaining a high bar for non-record submissions, so be sure to justify your ideas and results in detail when submitting. 

#### PRs on core code

The train_gpt.py and train_gpt_mlx.py scripts are intended as good launching off points for new participants, not SOTA configs. We'll accept PRs that make tunings, improvements, or simplifications to these scripts without majorly increasing complexity, but the best models should stay in the /records folder.

### Support

Reach out to parametergolf@openai.com for any other questions.

Join the [Discord server](url).
