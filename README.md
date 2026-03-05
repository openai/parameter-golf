<img width="384" height="256" alt="Parameter Golf Logo" src="https://github.com/user-attachments/assets/5afa2832-f306-45cf-b819-eeb971ee560b" />

[Placeholder Readme]

**OpenAI ModelCraft Challenge: Parameter Golf** is a challenge to train the best language model that fits in a 16MB file + trains in <10 minutes on 8xH100, evaluated by their FineWeb validation set compression (tokenizer-agnostic, bits per byte).

This challenge takes heavy inspiration from the [NanoGPT Speedrunning](https://github.com/KellerJordan/modded-nanogpt) challenge, where individuals compete to train a model that reaches 3.28 FineWeb validation loss as fast as possible. We're excited to see how optimizing for a parameter-constrained setting pushes people towards unique architectures, compression schemes, and creative submission.

The challenge runs from March 18th to April [PLACEHOLDER]. 

If you enjoy solving very difficult technical problems, please introduce yourself via the [Challenge Participant Form](https://you.ashbyhq.com/meeting/3187917e-c244-4aae-9b92-5399048f0677/)
, which allows us to attribute challenge submissions and reach out about opportunities with OpenAI. _Completing the form is not required to participate._

Many researchers at OpenAI first distinguished themselves through elite mathematics and programming competitions. The ModelCraft Challenge is designed in that spirit: testing the ability to tackle unfamiliar problems with creativity and rigor, qualities we believe are essential for frontier AI research.

In June, we plan to hire a small cohort of early-career researchers, targeted at current undergraduate students and recent graduates, including Olympiad medalists and engineers who demonstrate unusual technical ability. For exceptional participants, the challenge may also serve as a way to stand out to OpenAI researchers and recruiting.

Happy training!

### Training your first model

First, let's get started training something on your laptop. 

```bash
insert setup guide

train .py
```

This is pretty slow though, so let's move to GPUs to iterate faster.
You can rent H100s from a lot of different places, but OpenAI is working with RunPod to make setup as easy as possible. [Click here to launch a pod.](url)

We know compute is expensive, so OpenAI is sponsoring $1,000,000 in compute credits for people to get started training their models. To request a credit grant ($500), request at this form here: [Request a Compute Grant](url).

```bash
running training on 8xh100s

```

### Leaderboard

*Track source:* `records/track_10min`  
*Score metric shown below:* `submission.json.loss` (lower is better). Most rows are `final_int8_zlib_roundtrip val_bpb`; rows that differ are called out in the summary.

| Rank | Run              | Score  | Author         | Summary                              | Date       | Code              | Description      |
|-----:|------------------|-------:|----------------|--------------------------------------|------------|-------------------|------------------|
| 1    | GQA-4 Mixed-Rows Quant (Strict <32M) | 1.1454 | Codex          | 13x512 GQA-4, tied embeds; same valid 7375-step checkpoint with stricter mixed row/group int8 tuning, clipping, and scale encoding to optimize post-quant `val_bpb` under 32,000,000 bytes | 2026-02-21 | [code](records/track_10min/2026-02-21_GQA4_PartialPerRowMLPProj/train_gpt.py) | [info](records/track_10min/2026-02-21_GQA4_PartialPerRowMLPProj/README.md) |
| 2    | Baseline (SP-2048 11x512, flash+untied beat) | 1.1539 | Codex          | Flash-only + untied baseline rerun; beats user baseline score, but train_time was contaminated by background jobs (>10m) | 2026-02-22 | [code](records/track_10min/2026-02-22_Baseline_SP2048_512x11_FlashUntied_Beat/train_gpt.py) | [info](records/track_10min/2026-02-22_Baseline_SP2048_512x11_FlashUntied_Beat/README.md) |
| 3    | SP-2048 11x512 KV2 (10min + per-row int8 beat) | 1.1551 | Codex          | Exact 10-minute KV2 train run (`NUM_KV_HEADS=2`) plus compiled checkpoint-reload int8 tuning (global per-row 2D) to beat the user baseline under 32,000,000 bytes | 2026-02-22 | [code](records/track_10min/2026-02-22_SP2048_512x11_KV2_PerRowAll_10minBeat/train_gpt.py) | [info](records/track_10min/2026-02-22_SP2048_512x11_KV2_PerRowAll_10minBeat/README.md) |
| 4    | Baseline (SP-2048 11x512, rerun) | 1.1819 | Codex          | Reproducible baseline rerun (full log + code snapshot; this rerun is >10m train_time) | 2026-02-22 | [code](records/track_10min/2026-02-21_Baseline_SP2048_512x11_Rerun/train_gpt.py) | [info](records/track_10min/2026-02-21_Baseline_SP2048_512x11_Rerun/README.md) |
| 5    | GPT-Simple No-Tie (SP-1024 9x256, 1-LR tuned) | 1.3319 | Codex          | Untied `train_gpt_simple` baseline with one-global-LR sweep + 8x DDP confirmation; clean 596.6s run. Score is direct final `val_bpb` (no int8/zlib). | 2026-02-22 | [code](records/track_10min/2026-02-22_GPTSimpleNoTie_SP1024_256x9_OneLR_DDP8_10min/train_gpt_simple_no_tied_embeddings.py) | [info](records/track_10min/2026-02-22_GPTSimpleNoTie_SP1024_256x9_OneLR_DDP8_10min/README.md) |

### FAQ

**What exactly counts toward the 32 MiB artifact size?**

Submission artifact is computed as code bytes + compressed model bytes. All code should live in the train.py script to be counted.
No external downloads, dataset access, or network calls are allowed during evaluation. The artifact must be fully self-contained and reproducible.

**Are scores independently verified by OpenAI?**

We're not automatically verifying every submission, but we will verify the top leaderboard entries over time. Any non-reproducible results can be disqualified, and issues with reproducing submissions should be brought up on the PR.

**What counts as 'external compute'? For ex, is it fair to tune my hyperparameters offline?**

This is hard to set a super clear line on. We reserve the right to disqualify runs that we find are 'not in the spirit of the challenge.' Tuning your Adam hyperparameters across a bunch of runs is fine, but if there's evidence you're sneaking in additional compute unfairly (brute forcing some seed, etc.) we won't allow that. 

(etc)

### Submission Process

_I'll fill this in soon. Pretty much people create a PR with their code and a training log and a description to the /records/ folder and we add it there. And the leaderboard can auto update on that._

### Support

Reach out to parametergolf@openai.com for any other questions.

Join the [Discord server](url).
