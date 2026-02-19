<img width="384" height="256" alt="Parameter Golf Logo" src="https://github.com/user-attachments/assets/5afa2832-f306-45cf-b819-eeb971ee560b" />

**OpenAI Parameter Golf** is a challenge to train the best language model that fits in a 30MB file + trains in <10 minutes on 8xH100, evaluated by their FineWeb validation set compression (tokenizer-agnostic, bits per byte).

This challenge takes heavy inspiration from the [NanoGPT Speedrunning](https://github.com/KellerJordan/modded-nanogpt) challenge, where individuals compete to train a model that reaches 3.28 FineWeb validation loss as fast as possible. We're excited to see how optimizing for a parameter-constrained setting pushes people towards unique architectures, compression schemes, and creative submission.

The chalenge runs from March 4th to March 31st, after which we'll select the top 10 most creative submissions for a small prize.

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

| Rank | Run              | Score  | Author         | Summary                              | Date       | Code              | Description      |
|-----:|------------------|-------:|----------------|--------------------------------------|------------|-------------------|------------------|
| 1    | Baseline         | 3.2783 | Will DePue     | Reference transformer configuration   | 2026-02-10 | [log](run-folder) | [info](url)      |
| 2    | Tied Embeddings  | 3.2792 | Bill DePue     | Input/output embedding weights tied  | 2026-02-11 | [log](run-folder) | [info](url)      |
| 3    | LayerNorm Variant| 3.2801 | Ada Chen       | Modified LayerNorm placement/style   | 2026-02-11 | [log](run-folder) | [info](url)      |
| 4    | Residual Scaling | 3.2814 | Marcus Lee     | Applied scaling to residual branches  | 2026-02-12 | [log](run-folder) | [info](url)      |
| 5    | Wider MLP        | 3.2830 | Sofia Patel    | Increased feedforward hidden width    | 2026-02-12 | [log](run-folder) | [info](url)      |
| 6    | Rotary Embeddings| 3.2847 | Daniel Kim     | Switched to RoPE positional encoding  | 2026-02-13 | [log](run-folder) | [info](url)      |
| 7    | Deep Stack v1    | 3.2862 | Elena Garcia   | Increased number of transformer layers| 2026-02-13 | [log](run-folder) | [info](url)      |
| 8    | Dropout Sweep    | 3.2889 | Noah Smith     | Tuned dropout probabilities           | 2026-02-14 | [log](run-folder) | [info](url)      |
| 9    | Attention Bias Fix| 3.2905| Priya Rao      | Corrected attention bias handling     | 2026-02-14 | [log](run-folder) | [info](url)      |
| 10   | GELU Approx      | 3.2931 | Liam Johnson   | Used approximate GELU activation      | 2026-02-15 | [log](run-folder) | [info](url)      |
| 11   | Init Scale Tuning| 3.2958 | Olivia Brown   | Adjusted parameter initialization scale| 2026-02-15| [log](run-folder) | [info](url)      |
| 12   | PosEnc Shift     | 3.2980 | Ethan Davis    | Shifted positional encoding indices   | 2026-02-16 | [log](run-folder) | [info](url)      |
| 13   | NormFormer Lite  | 3.3004 | Mia Wilson     | Lightweight NormFormer-style changes  | 2026-02-16 | [log](run-folder) | [info](url)      |
| 14   | Sparse Attention | 3.3042 | Lucas Martin   | Introduced sparse attention pattern   | 2026-02-17 | [log](run-folder) | [info](url)      |
| 15   | FlashAttn Patch  | 3.3075 | Harper Clark   | Integrated FlashAttention optimization| 2026-02-17 | [log](run-folder) | [info](url)      |

### FAQ

**What exactly counts toward the 32 MiB artifact size?**

Submission artifact is computed as code bytes + compressed model bytes. All code should live in the train.py script to be counted.
No external downloads, dataset access, or network calls are allowed during evaluation. The artifact must be fully self-contained and reproducible.

**Are scores independently verified by OpenAI?**

We're not automatically verifying every submission, but we will verify the top leaderboard entries over time. Any non-reproducible results can be disqualified, and issues with reproducing submissions should be brought up on the PR.

**What counts as 'external compute'? For ex, is it fair to tune my hyperparameters offline?**

This is hard to set a super clear line on. We reserve the right to disqualify runs that we find are 'not in the spirit of the challenge.' Tuning your Adam hyperparameters across a bunch of runs is fine, but if there's evidence you're sneaking in additional compute unfairly (brute forcing some seed, etc.) we won't allow that. 

(etc)

### Support

Reach out to parametergolf@openai.com for any other questions.
