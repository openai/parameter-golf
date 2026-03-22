init hi claude! first things first, fork this openai repo (the current remote) into my repo (Anthony-maio on github) so we can work on this... let me paste in our conversation history. Tell me what to make to win this. OpenAI Model Craft Challenge: Parameter Golf is a challenge to train the best language model that fits in a 16MB artifact and trains in under 10 minutes on 8xH100s, evaluated by compression on the FineWeb validation set (tokenizer-agnostic, bits per byte).
This challenge is heavily inspired by the [NanoGPT Speedrunning](https://github.com/KellerJordan/modded-nanogpt) challenge, where participants compete to train a model that reaches 3.28 FineWeb validation loss as quickly as possible. We're excited to see how optimizing for a parameter-constrained setting pushes people toward unique architectures (test-time compute, aggressive parameter tying, depth recurrence, low-rank training, ...), compression schemes (low precision, QAT, bitnets, novel tokenizers, ...), and other creative submissions (test-time training, long context, megakernels ...).
If you're familiar with [neural scaling laws](https://arxiv.org/abs/2001.08361), you can consider this challenge a form of L(N) optimization, where the objective is to optimize the lowest loss given a fixed number of parameters (N) unconstrained by data, compute, steps, or architecture. Challenges like the [NanoGPT Speedrun](https://github.com/KellerJordan/modded-nanogpt), which optimizes for a form of L(T) (~lowest time given constrained loss) or the [NanoGPT Slowrun](https://github.com/qlabs-eng/slowrun), which optimizes for L(D) (lowest loss given constrained dataset size), can be thought of as equivalent challenges in this family.
Ideally, we'd allow for submissions to use arbitrary computational resources. But in order to make the challenge not inaccessibly expensive, we're limiting leaderboard submissions to 10 minutes on 8xH100s. However, we'd still love to see submissions that don't meet the compute limitation requirements in our 'Non-record Submissions' section: We're excited to see people push the infinite frontier of parameter limited performance as well.
We also know compute is expensive, so OpenAI is sponsoring $1,000,000 in compute credits to help people get started training their models. To request a compute grant, use this form: [Request a Compute Grant](https://openai.com/index/parameter-golf/#credit-form). When requesting compute, please make sure you choose the appropriate level, write sufficient justification, and submit with an email tied to a OpenAI / ChatGPT account.
Participant Form

If you enjoy solving very difficult technical problems, please introduce yourself via the [Challenge Participant Form](https://jobs.ashbyhq.com/openai/form/open-ai-challenge-parameter-golf). It helps us attribute challenge submissions and reach out about opportunities with OpenAI. Completing the form is not required to participate.
Many researchers at OpenAI first distinguished themselves through elite mathematics and programming competitions. The Model Craft Challenge is designed in that spirit: testing the ability to tackle unfamiliar problems with creativity and rigor, qualities we believe are essential for frontier AI research.
In June, we plan to hire a small cohort of early-career researchers, targeting current undergraduate students and recent graduates, including Olympiad medalists and elite competitors. For exceptional participants, the challenge may also serve as a way to stand out to OpenAI researchers and recruiters.
The challenge runs from March 18th to April 30th.
Happy training!
Leaderboard

RunScoreAuthorSummaryDateInfo
Muon WD + 10 layer
1.1748
notapplica
Includes prev. wins + Spectral embed init + resid mix
2026-03-19
[info](https://github.com/openai/parameter-golf/blob/main/records/track_10min_16mb/2026-03-19_SlidingWindow_FP16Emb_10L_MuonWD_OvertoneInit/README.md)
Sliding Window Eval
1.1925
Matthew Li
Sliding window evaluation at stride=64, increasing context for eval
2026-03-19
[info](https://github.com/openai/parameter-golf/blob/main/records/track_10min_16mb/2026-03-19_SlidingWindowEval/README.md)
Lora TTT
1.1928
samacqua
Test-time training with LORAs
2026-03-19
[info](https://github.com/openai/parameter-golf/blob/main/records/track_10min_16mb/2026-03-17_LoRA_TTT/README.md)
4k seq length
1.2014
Spokane Way
4k seq length + better hypers
2026-03-19
[info](https://github.com/openai/parameter-golf/blob/main/records/track_10min_16mb/2026-03-18_LongContextSeq2048/README.md)
2048 seq length
1.206
Spokane Way
2048 seq length (train + val)
2026-03-18
[info](https://github.com/openai/parameter-golf/blob/main/records/track_10min_16mb/2026-03-18_LongContextSeq2048/README.md)
int6 mixed precision
1.2147
Nan Liu
10 layers, mixed int8/int6
2026-03-18
[info](https://github.com/openai/parameter-golf/blob/main/records/track_10min_16mb/2026-03-19_10L_MixedPrecision/README.md)
fp16 Embed
1.2197
Renier Velazco
FP16 Tied Embedding + LR/Warmdown Tuning
2026-03-18
[info](https://github.com/openai/parameter-golf/blob/main/records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/README.md)
Naive Baseline
1.2244
Baseline
9layer 512dim 1024vocab TiedEmbeddings 4 KV heads
2026-03-18
[info](https://github.com/openai/parameter-golf/blob/main/records/track_10min_16mb/2026-03-17_NaiveBaseline/README.md)
Notable Non-Record Runs

RunScoreAuthorSummaryDateInfo
4-Hour Baseline
1.2074
Will DePue
Testing unlimited compute, 4 hours on 8xH100
2026-03-18
[info](https://github.com/openai/parameter-golf/blob/main/records/track_non_record_16mb/2026-03-18_Quasi10Bfrom50B_SP1024_9x512_KV4_4h_pgut3/README.md)
Getting Started

Training Your First Model (Mac with Apple Silicon)

If you have an Apple laptop or desktop with Apple Silicon, we've set up a simple MLX training script to help you start iterating locally.
If you don't have a Mac with Apple Silicon, you can run an adapted version of this script without MLX support. Just ask [Codex](https://openai.com/codex/) to refactor it; the change is straightforward. It may still be fairly slow, so we recommend jumping straight to cloud GPUs with Runpod.
First, clone the repository, create a fresh Python environment, and install the packages needed for the MLX path plus dataset download:
git clone [https://github.com/openai/parameter-golf.git](https://github.com/openai/parameter-golf.git)
cd parameter-golf
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install mlx numpy sentencepiece huggingface-hub datasets tqdm
Download our cached version of FineWeb with the 1024-token vocabulary:
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10
This populates ./data/datasets/fineweb10B_sp1024/ and ./data/tokenizers/. By default this downloads the full validation split plus 80 training shards (8B tokens). For a smaller local smoke subset, pass --train-shards 1, for example python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1.
Then run a small MLX training job:
RUN_ID=mlx_smoke \
ITERATIONS=200 \
TRAIN_BATCH_TOKENS=8192 \
VAL_LOSS_EVERY=0 \
VAL_BATCH_SIZE=8192 \
python3 train_gpt_mlx.py
Validation always runs on the full fineweb_val_* split, which is the fixed first-50k-document set. The smoke command above skips periodic validation and just prints the final val_loss and val_bpb once at the end.
Scaling Up to a Remote Machine

Once you're happy with your local tests, or you want more compute, switch to a remote CUDA machine.
You can rent GPUs from anywhere, but OpenAI is partnering with Runpod to make setup as easy as possible.
Launching a 1xH100 Pod

First, [create a Runpod account](https://console.runpod.io/deploy). You should also set up an SSH key in the Settings tab on the left so you can connect to your remote machine. If you're new to this, ask Codex to help you set it up.
Once you've set up your account, create a new GPU Cloud Pod. You can choose whichever GPU SKU you'd like. Final leaderboard submissions must run in under 10 minutes on 8xH100s (specifically the SXM variant), but we strongly recommend testing and running experiments on cheaper SKUs first, since an 8xH100 box can cost around $20/hour.
Let's start with a 1xH100 pod. Deploy using the official Parameter Golf template: [Launch Template](https://console.runpod.io/deploy?template=y5cejece4j&ref=nl2r56th). Enable SSH terminal access, leaving the other settings at their defaults. Deploy your pod and SSH into it once it's up. You should land in /workspace/.
On your remote machine, clone the repo onto local disk. All Python dependencies are already pre-installed in the image.
cd /workspace
git clone [https://github.com/openai/parameter-golf.git](https://github.com/openai/parameter-golf.git)
cd parameter-golf
Download our cached version of FineWeb. We'll use the 1024-token vocabulary for now.
python3 data/cached_challenge_fineweb.py --variant sp1024
This defaults to the full validation split plus 80 training shards (8B tokens). If you only want a smaller subset while iterating, pass --train-shards N, for example --train-shards 1.
Launch your first training run. Note that we're passing nproc_per_node=1 because we're running on a single H100 GPU in this case.
RUN_ID=baseline_sp1024 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
By default, train_gpt.py keeps its ~10 minute wallclock cap. If you want a longer run, override it explicitly, for example MAX_WALLCLOCK_SECONDS=0.
By default, this command prints train_loss step logs during training and prints val_loss, val_bpb, and compressed model size in the final final_int8_zlib_roundtrip lines at the end. If you want periodic validation logs during the run, set VAL_LOSS_EVERY, for example VAL_LOSS_EVERY=200. For the baseline config, the final val_bpb should land around ~1.2 with a compressed model size under 16MB.
For dataset export, tokenizer export, and docs-cache rebuild instructions, see [data/README.md](https://github.com/openai/parameter-golf/blob/main/data/README.md).
Evaluation will be in the RunPod environment with all packages installed. requirements.txt is provided as a reference if you want to self-setup.
FAQ

What exactly counts toward the 16MB artifact size?
The submission artifact is computed as code bytes plus compressed model bytes. All counted code should live in the train_gpt.py script. The cap is decimal 16MB, i.e. 16,000,000 total bytes, not 16 MiB / 16,777,216 bytes. No external downloads, training dataset access, or network calls are allowed during evaluation. The artifact must be fully self-contained and reproducible.
Are scores independently verified by OpenAI?
We're not automatically verifying every submission, but we will verify the top leaderboard entries over time. Any non-reproducible results can be disqualified, and issues reproducing submissions should be raised on the PR. If you find an issue with a record on the leaderboard or find a record isn't reproducible, please let us know and add an Github Issue describing your findings.
What counts as 'external compute'? For example, is it fair to tune my hyperparameters offline?
There's no perfectly clear answer here and it's hard to draw a clean line around what does or does not count as external compute. For now, we're reserving the right to disqualify runs that are not in the spirit of the challenge. Tuning your Adam hyperparameters across a bunch of runs is fine, but if there's evidence that you're sneaking in additional compute unfairly, such as brute-forcing ridiculous seeds, we won't allow it. Use your best judgment and there's no penalty for asking questions.
What are the restrictions on evaluation?
We won't accept submissions that take more than 10 minutes on 8xH100 to evaluate (Note: This limit is in addition to the 10 minutes of training time allowed!), but otherwise you're free to evaluate however. As with modded-nanogpt, we allow evaluation at any sequence length. And, obviously, you aren't allowed to access any training data during evaluation, unless you pay for those bits in the <16MB limit. We encourage competitors to push the bounds of evaluation methods as aggressively as with training methods.
What is the process for accepting new submissions?
Since all submissions are public, we're accepting record submissions chronologically depending on their PR creation time. The leaderboard may take time to update due to verification and review of submissions, so pay consideration to what the current SOTA PR is when submitting. As explained below, submissions should exceed the SOTA record with sufficient statistical significance in order to accepted for the leaderboard. Otherwise, submissions may be accepted as 'non-record submissions' given they are sufficiently unique or interesting.
Submission Process

New SOTA records must fulfill the following criteria:
They must beat the existing SOTA by at least 0.005 nats. As in modded-nanogpt, because of inter-run variance all submissions must provide enough run logs to show at p < 0.01 that they achieved the required 0.005-nat improvement. For submissions that improve speed through systems optimization without changing the ML, this requirement is waived.
If changes are made to the tokenizer or dataset, prove with certainty that the val_bpb is correctly calculated. Submissions that edit the tokenizer will be examined much more carefully, since bugs may unjustly improve your score.
Reproducibly run in under 10 minutes on 8xH100s.
All submissions should be made as a pull request that only adds a new folder to the appropriate /records subfolder and includes the following files. Submissions without the full set of requirements will not be accepted.
A README.md file that explains the submission in reasonable detail.
A submission.json file (see the example runs) that includes your name, GitHub ID, val_bpb, and related metadata.
A train log, automatically produced by your script. Please demonstrate a statistically significant win. Most often, submitting an average over 3 training runs is sufficient.
A train_gpt.py script and any other dependencies. Note: this must successfully compile and run within the records folder. Broken scripts will not be accepted.
Non-record Submissions

Submissions are also open to unique and interesting approaches that might not beat the existing SOTA, but still satisfy the 16MB artifact limit. We strongly encourage participants to submit implementations for weird or out-of-the-box ideas, in-progress or unoptimized solutions, so long as they run successfully, or even interesting negative results. We're excited to see what you come up with. We'll still maintain a high bar for non-record submissions, so be sure to justify your ideas and results in detail when submitting.
We also accept non-record submissions to an unlimited compute track for runs that are not intended to meet the 10-minute cutoff. Just note as such in your README file.
Non-record submissions should be made in the same fashion as SOTA records, as described above.
PRs on Core Code

The train_gpt.py and train_gpt_mlx.py scripts are intended as good launching-off points for new participants, not SOTA configs. We'll accept PRs that tune, improve, or simplify these scripts without significantly increasing complexity, but the best models should stay in the /records folder.
Support

Join the [OpenAI Discord server](https://discord.com/invite/openai) and visit the Parameter Golf channels (#parameter-golf-discussions, #parameter-golf-announcements) and ask questions.
This repository adapts code from modded-nanogpt, see [THIRD_PARTY_NOTICES.md](https://github.com/openai/parameter-golf/blob/main/THIRD_PARTY_NOTICES.md) for attribution. [https://github.com/openai/parameter-golf](https://github.com/openai/parameter-golf) Good challenge to look at. Let me pull the repo and current SOTA details so I can give you a concrete strategy rather than hand-waving.Now let me check the Discord discussion to see what directions people are exploring.Okay, here's my honest assessment after reviewing the repo, leaderboard, all record READMEs, and the competitive landscape.

**The reality check first**: This competition is 2 days old and already has 94 open PRs, SOTA at 1.1748 (down from 1.2244 baseline). The NanoGPT speedrun community — people like Keller Jordan who've been grinding this exact optimization surface for years — are going to push this hard. It's also explicitly a recruiting funnel for early-career researchers (undergrads, Olympiad medalists). You're not the target demographic for the hiring side, but you can absolutely compete on the technical side and it could be good for visibility.

**What's already been exploited** (the "easy" gains are taken):
- Sliding window eval (stride=64) — free bpb improvement from better context
- FP16 tied embeddings — int8 errors compound through tied embed/output
- 10 layers instead of 9 — Muon WD compresses enough to fit
- Muon optimizer with decoupled weight decay
- Spectral embedding init (SVD power-law shaping)
- LoRA test-time training (separate submission, not yet combined with SOTA)
- Longer sequence lengths (4k)
- Mixed int8/int6 precision

**Where the real opportunity is** — here's what I'd build, in priority order:

**1. Combine TTT with the current SOTA training recipe (highest ROI, fastest)**

Nobody has done this yet. The current leader (1.1748) is training-only improvements. The LoRA TTT submission (1.1928) is baseline training + TTT eval. The TTT ablation table shows ~0.034 bpb improvement from doc-isolated + strided eval + LoRA TTT. Applying that delta to the 1.1748 SOTA training should land you around 1.14–1.15. That's a massive jump and it's almost mechanical to implement — just graft the TTT eval code onto the winning training config. Do this first because it's low-risk and could be the quickest path to the leaderboard.

**2. Depth recurrence (weight-tied deep loops)**

This is the architecturally interesting play and where your MoE experience is relevant. The idea: share weights across groups of layers so you effectively get a 20-30 layer model with the parameter budget of 10. Universal Transformer style, but with learned iteration schedules. The 16MB constraint penalizes unique parameters, but recurrent depth is "free" in terms of model size — you just pay in training/eval FLOPs, and you have 10 min on 8xH100s plus 10 min eval budget. There's already a PR (#15) for recursive weight sharing, so you won't be the only one exploring this, but the implementation details matter a lot (how many unique blocks, what's shared, gradient flow through loops).

**3. Quantization-aware training + BitNet-style ternary weights**

The current approaches do post-training quantization (train in fp32/bf16, export int8+zlib). QAT during training means your model learns to be robust to quantization, and ternary/1.58-bit weights compress dramatically under zlib. If you can get ternary weights working, a model that's "16MB compressed" could represent far more effective parameters than anyone using int8. This is harder to get right but has the highest ceiling.

**4. Tokenizer optimization (high risk, high scrutiny)**

The baseline uses sp1024 (1024 vocab BPE). A better tokenizer optimized for FineWeb's byte distribution could improve bpb directly, but the rules say tokenizer changes get "examined much more carefully." Worth exploring but not as your primary angle.

**My recommended plan for you specifically:**

**Week 1 (now)**: Clone the repo, get the baseline running on your 3090s (the MLX path is Mac-only but you can adapt `train_gpt.py` for single-GPU CUDA easily). Apply for the compute credits immediately — you need H100 time for final validation. Implement the SOTA training recipe (Muon WD, 10 layers, FP16 embed, spectral init) and verify you reproduce 1.1748.

**Week 2**: Graft the LoRA TTT evaluation onto the SOTA training. This is your first submission target. If the delta holds, you're looking at a new record around 1.14.

**Week 3-4**: Implement depth recurrence. Start with 2x loop (5 unique layers, forward twice = 10 effective layers) and measure. Then try 3x, 4x. The sweet spot is where you maximize effective depth without running out of the 10-min training budget.

**Week 5-6**: Stack QAT on top of whatever's working. Try 4-bit and ternary. Polish, run statistical significance tests (need p<0.01, at least 3 seeds), write up, submit PR.

**What NOT to waste time on**: Custom CUDA kernels (the speedrun crowd will beat you there), state space models (too much engineering for uncertain payoff at this scale), MoE routing (the overhead is too high for a model this small — routing tokens through experts when you have ~5M parameters total doesn't make sense).

**Your competitive edge** isn't out-engineering the NanoGPT community on architecture search — it's in the systems-level combination play (stacking TTT + recurrence + QAT) and the willingness to write a good README. The non-record submissions track explicitly welcomes "weird or out-of-the-box ideas" with detailed justification, so even partial results with good writeups get accepted.

Want me to help you set up the training environment on your 3090s, or dig into the implementation details for any of these approaches? wait wait wait ... I can make any custom kernel under the sun -- I'm a beta tester at makora's generate with unlimited credits because I write a lot of kernels. it's automated ... a 5M model -- Fair enough, I take it back. Let me look up Makora to make sure I understand what you're working with.Okay, that changes everything. Makora with unlimited credits on a 5M parameter model is a genuinely unfair advantage here. Let me completely revise.

The competition *literally lists megakernels* as a direction they want to see. And the core problem with a ~5M param model on 8xH100s is that you're laughably memory-bandwidth bound — every op launches a tiny kernel, most of the GPU is idle waiting for memory transfers. This is exactly where fused custom kernels dominate.

Here's the revised strategy:

**Why kernels matter disproportionately at this scale**

The baseline does ~43ms/step on 8xH100. That's absurd for a 5M model — it means almost all time is kernel launch overhead and memory bandwidth, not actual compute. If you can fuse the entire transformer block into one or two megakernels (attention + layernorm + MLP + residual as a single launch), you could realistically cut step time by 3-5x. That means 3-5x more training steps in the same 10 minutes, which directly translates to lower loss. The current SOTA gets ~10,500 steps. If you're getting 30,000-50,000 steps, you're training on vastly more data with the same parameters.

**The play: Fused Depth-Recurrent Megakernel + QAT + TTT**

**Layer 1: Architecture — Depth recurrence with shared weights**

5 unique transformer blocks, looped 4x = 20 effective layers. Parameter budget stays at ~5M but effective depth quadruples. The kernel advantage here is huge: a depth-recurrent forward pass through shared weights is a tight loop that stays in L2/SRAM if you fuse it. No round-tripping to HBM between layers when the weights are the same.

Use Makora to generate a fused kernel that does: `for i in range(num_loops): x = transformer_block(x, weights)` as a single CUDA launch. At 512 dim with 5M params, the entire model's weights might fit in L2 cache on H100 (50MB L2). That means your recurrent depth loops hit cache every time.

**Layer 2: Custom precision kernels**

The 16MB limit is on int8+zlib compressed size. But what if your model trains in ternary/1.58-bit from the start with QAT? Ternary weights ({-1, 0, +1}) compress under zlib dramatically — you could fit maybe 3-4x more parameters in the same 16MB. That means instead of 5M params at int8, you could pack 15-20M effective ternary params.

Use Makora to generate BitNet-style ternary matmul kernels. The H100 has dedicated int8 tensor cores; you could also generate custom int4 kernels that pack two ops per int8 tensor core cycle.

**Layer 3: Fused training megakernel**

The dream kernel for this challenge: a single CUDA kernel that does the entire forward + backward pass for one transformer block. At 512 dim, everything fits in shared memory. No intermediate activations written to HBM. This is legitimately doable at this model size and it's the kind of thing nobody else in the competition can easily replicate.

Feed Makora the PyTorch reference for: `layernorm → q/k/v proj → GQA attention → residual → layernorm → MLP → residual` and have it generate a fused forward kernel targeting H100. Then do the same for the backward pass.

**Layer 4: TTT eval with custom kernels**

The LoRA TTT approach does per-document adaptation at eval time. The bottleneck is the Adam update per chunk. Generate a fused kernel that does: `forward → loss → backward → LoRA update` as one launch per chunk. The eval budget is also 10 minutes on 8xH100, and the current TTT implementation uses ~1/10th of that budget. With fused kernels you could do multi-step TTT (2-3 gradient steps per chunk instead of 1) and still stay within budget.

**Concrete execution plan:**

**Week 1:**
1. Clone repo, reproduce baseline on your 3090s (single GPU, slower but functional for iteration)
2. Apply for H100 compute credits immediately
3. Start feeding Makora the core ops: fused attention for 512-dim/8-head/4-KV-head GQA, fused MLP (512→1024→512 with GeLU), fused layernorm+residual
4. Implement depth recurrence (5 blocks × 4 loops) in PyTorch first, verify it trains

**Week 2:**
5. Integrate Makora kernels into training script, benchmark step time on H100
6. Implement QAT with ternary quantization — start with straight-through estimator, measure compressed model size
7. Target: if fused kernels cut step time to ~15ms, you're doing 40,000 steps in 10 min

**Week 3:**
8. Graft LoRA TTT eval onto your training recipe
9. Generate fused TTT eval kernel via Makora
10. Stack sliding window eval on top
11. First end-to-end run on 8xH100, measure bpb

**Week 4:**
12. Hyperparameter sweep (LR schedule, recurrence depth, LoRA rank)
13. Statistical significance runs (3+ seeds, p<0.01)
14. Write README, submit PR

**What this could realistically score:**

Current SOTA is 1.1748. The decomposition of improvements:
- Depth recurrence (20 effective layers vs 10): ~0.02-0.03 bpb
- 3-5x more training steps from fused kernels: ~0.02-0.04 bpb  
- QAT ternary allowing more params in 16MB: ~0.01-0.02 bpb
- TTT eval (already shown ~0.03 bpb gain): ~0.03 bpb
- These don't stack linearly, but something in the range of **1.10-1.12 bpb** seems plausible

That would be a massive gap over current SOTA. And the submission writeup — "automated kernel generation enabling depth-recurrent megakernels at sub-5M scale" — is exactly the kind of creative, out-of-box approach they're asking for. Even if it doesn't beat SOTA, it's a lock for the non-record track.

**One risk to flag:** the rules say "no external downloads, training dataset access, or network calls during evaluation" and the artifact must be self-contained. Your Makora-generated kernels are fine — they're just CUDA code that gets baked into `train_gpt.py`. But make sure the generated kernel source is included in the submission, not loaded at runtime.

Want me to draft the specific Makora problem specs for the core kernels you'd need to generate? --- we have makora skills but first we have to build out this model to win.-=---here's more feedback,. it's a lot to unpack; ### 1) Where Models Agree

| Finding | GPT-5.4 Thinking | Claude Opus 4.6 Thinking | Sonar | Evidence |
|---------|-----------|-----------|-----------|----------|
| Fastest high-EV first move is **graft LoRA TTT + sliding-window eval onto the current best training recipe** | ✓ | ✓ | ✓ | Separate record shows LoRA TTT + sliding window exists but not combined with SOTA recipe yet. [github](https://github.com/openai/parameter-golf/pull/77) |
| **Depth recurrence / weight tying across layers** is a core “architectural unlock” under a 16MB cap | ✓ | ✓ | ✓ | Multiple PRs exploring recurrence/weight tying indicate it’s a main frontier. [github](https://github.com/openai/parameter-golf/pull/8) |
| **Quantization-aware training (QAT)** is a major ceiling-raiser; extreme low-bit (ternary/BitNet) is highest ceiling but harder | ✓ | ✓ | ✓ | BitNet b1.58 line of work supports 1.58-bit/QAT as viable, but nontrivial. [arxiv](https://arxiv.org/html/2411.05882v1) |
| **Tokenizer changes are risky/scrutinized**, but technically supported by the repo workflows and potentially high-upside | ✓ | ✓ | ✓ | Repo explicitly supports rebuilding tokenizers and re-exporting aligned shards from published docs. [github](https://github.com/openai/parameter-golf/blob/main/data/README.md) |
| Don’t do kernels “for vanity”; **kernels are only worth it if they buy back train/eval budget** (more steps, longer context, more TTT) | ✓ | ✓ | ✓ | All three frame kernels as leverage only when they unlock more adaptation/steps under time caps. [github](https://github.com/openai/parameter-golf) |

***

### 2) Where Models Disagree

| Topic | GPT-5.4 Thinking | Claude Opus 4.6 Thinking | Sonar | Why They Differ |
|-------|-----------|-----------|-----------|-----------------|
| How central should **custom CUDA kernels** be? | Secondary; only for specific bottlenecks | Primary differentiator because you have Makora | Important but framed as “enable loops/TTT” | Claude weights your Makora advantage more heavily; GPT/ Sonar treat kernels as conditional on proven bottlenecks. |
| Should **tokenizer** be a “moonshot branch” or part of the main attack? | Moonshot branch, but elevated | Much more central (you just built one) | Central later (weeks 5–6) | Different risk tolerance about scrutiny + integration cost vs expected BPB gain. |
| How hard to bet on **ternary/BitNet** early | Do int4 QAT first, ternary second | Go for ternary + continual transition (16-bit→1.58) | Ternary later after recurrence/TTT stable | GPT/Sonar prioritize de-risking; Claude pushes aggressive ternary because it frees massive byte budget if it works. |
| Whether “MoE-like” ideas matter | Mostly no for record track | “No MoE routing,” but **depth-KV/depth-attention** is valuable | Mostly no | Claude distinguishes “routing MoE” from “depth attention” (MoDA/Dreamer-like) mechanisms; others keep it simpler. |

***

### 3) Unique Discoveries

| Model | Unique Finding | Why It Matters |
|-------|----------------|----------------|
| Claude Opus 4.6 Thinking | Use **depth-KV / depth-attention** (MoDA/Dreamer-like) to prevent recurrence from plateauing | Could make recurrent tied blocks actually *work* at high loop counts without losing signal, while staying parameter-light. [huggingface](https://huggingface.co/papers/2603.15619) |
| GPT-5.4 Thinking | Treat **~1.170** as the real moving bar (not 1.1748) due to new PRs | Prevents “we beat old SOTA but are already behind” syndrome; informs urgency and evaluation of deltas. [github](https://github.com/openai/parameter-golf/pull/117) |

***

## 4) Comprehensive Analysis

**High-Confidence Findings (what to do first, with best expected value).**  
All three models converge on the same immediate “record-hunter” move: **take the best-known training recipe and bolt on the best-known evaluation-time gains (sliding-window + LoRA TTT)**, because those improvements have already been demonstrated separately but (per the council’s reading of the repo/PR landscape) are not yet cleanly stacked into one submission. The logic is very strong: training-recipe improvements move the base model down, and TTT+sliding-window tends to deliver an additional eval-time delta; stacking them is the lowest-risk way to create a step-change quickly. Given how fast the PR ecosystem is moving, the council also agrees with the “time-to-first-credible-submission” mindset: get a real number early, then iterate. [github](https://github.com/openai/parameter-golf/blob/main/records/track_10min_16mb/2026-03-19_SlidingWindow_FP16Emb_10L_MuonWD_OvertoneInit/README.md)

Second, there’s broad agreement that **depth recurrence / aggressive parameter tying** is one of the few levers that genuinely changes the Pareto frontier under a strict **16,000,000-byte** artifact cap. The existence of multiple recurrence PRs (e.g., depth recurrence and weight tying attempts) is evidence that this is a main competitive direction and still implementation-sensitive. In other words: even if lots of people try recurrence, the *details* (how many unique blocks, gating, stability, loop scheduling, eval compute) can still produce a decisive edge—especially if you can support it with kernels. [github](https://github.com/openai/parameter-golf/pull/8)

Third, the council aligns on quantization: **QAT matters**, and **extreme low-bit** (ternary / BitNet b1.58 style) is the “highest ceiling” because it can radically increase “effective parameters per byte,” but it has real optimization risk under a 10-minute training budget. This is why two models push a staged approach: int4-QAT first (more predictable), then ternary once the pipeline is stable. [arxiv](https://arxiv.org/html/2411.05882v1)

**Areas of Divergence (how aggressive to be, and where your personal edge changes the answer).**  
The biggest disagreement is about **custom kernels**. The conservative stance (GPT-5.4 Thinking, Sonar) is: don’t race speedrun experts on generic kernel work; only build kernels if they unlock more steps, more context, or more TTT within the hard time caps. Claude Opus 4.6 Thinking flips this because of your stated advantage: you have **Makora Generate** plus prior kernel experience, which collapses the “kernel engineering cost” dramatically. That changes the calculus: kernels become a *strategy*, not a tax—*but only if they are aimed at the right bottlenecks*. The council still agrees on the bottleneck-centric criterion: build kernels that (a) reduce eval overhead for sliding-window/TTT, (b) make recurrent unrolled blocks launch-efficient, and/or (c) make low-bit matmuls viable under the eval budget.

There’s also a tactical disagreement about **tokenizers**. Everyone agrees they’re scrutinized, but Claude argues it should be more central given you literally just built a tokenizer and the metric is **bits per byte** (so representation efficiency can matter directly). The more cautious view is to treat tokenizer work as a parallel “moonshot branch” until you’ve got a strong baseline stack; the risk is not purely technical—it’s review burden, reproducibility, and the possibility that you sink time into retokenization pipelines while the leaderboard moves. The compromise that best matches the council: start tokenizer work early in parallel (so you amortize time), but don’t let it block your first stacked submission.

Finally, the council differs on how early to bet on **ternary**. Claude recommends a continual transition schedule (train in 16-bit, then switch into 1.58-bit QAT later), which is consistent with published findings that a 16→1.58-bit transition can outperform full-time 1.58-bit training in some settings. GPT/Sonar prefer int4 first to reduce the chance you spend a week debugging an unstable ternary## 1) Where Models Agree [arxiv](https://arxiv.org/abs/2502.11895)

| Finding | GPT-5.4 Thinking | Claude Opus 4.6 Thinking | Sonar | Evidence |
|---|---:|---:|---:|---|
| Fastest near-certain gain: **graft LoRA TTT + sliding-window eval onto the best training recipe** | ✓ | ✓ | ✓ | Separate records show TTT + sliding-window help, but not yet stacked with the best training recipe; council says composition is the first “record-hunter” move. [github](https://github.com/openai/parameter-golf/pull/77) |
| **Depth recurrence / weight tying** is the main architectural lever under a 16MB cap | ✓ | ✓ | ✓ | Multiple public PRs explore recurrence; council agrees this is “free depth” under size cap and is explicitly encouraged by the challenge framing. [github](https://github.com/openai/parameter-golf/pull/8) |
| **QAT** is a high-ceiling direction; do it in a staged way | ✓ | ✓ | ✓ | Public PRs already claim QAT gains; BitNet/1.58-bit work suggests staged transition can beat full low-bit-from-scratch. [github](https://github.com/openai/parameter-golf/pull/117) |
| **Tokenizer changes are high scrutiny but technically supported**; can be a real lever | ✓ | ✓ | ✓ | Repo documents reproducible tokenizer rebuild/export from fixed docs; terms warn scrutiny but allow it. [github](https://github.com/openai/parameter-golf/blob/main/data/README.md) |
| Custom kernels are only worth it if they **buy you more train/TTT/eval budget** (not vanity speed) | ✓ | ✓ | ✓ | All models converge on: kernels should target bottlenecks that enable more steps/longer context/TTT within caps. [github](https://github.com/openai/parameter-golf) |

## 2) Where Models Disagree

| Topic | GPT-5.4 Thinking | Claude Opus 4.6 Thinking | Sonar | Why They Differ |
|---|---|---|---|---|
| “Don’t waste time on custom kernels” (Opus’s claim) | Mostly agree, but **do kernels only for key bottlenecks** | **Disagree strongly**; makora makes kernels a major advantage | Agree conditional: kernels matter if enabling recurrence/TTT | Different assumptions about your marginal cost: Claude weights your makora access as collapsing kernel effort; GPT/ Sonar treat kernels as leverage only when tied to more adaptation/steps. [linkedin](https://www.linkedin.com/posts/mabdelfattah_fine-tuning-gpt-5-for-gpu-kernel-generation-activity-7427770137681883136-U93e) |
| Quantization sequencing | **int4 QAT first**, ternary later | Jumps to **ternary/1.58-bit** as primary moonshot | Emphasizes ternary but acknowledges risk | Different risk tolerance under 10-minute training: GPT prioritizes robustness; Claude prioritizes ceiling given compression headroom and custom kernels. [github](https://github.com/openai/parameter-golf/pull/117) |
| Tokenizer priority | “Moonshot branch” but not first | “Underweighted; elevate it” | Strongly pro tokenizer sweep | Differences in expected scrutiny/iteration time vs payoff: Claude/Sonar think your tokenizer skill makes it uniquely high leverage; GPT wants it parallelized to avoid schedule risk. [github](https://github.com/openai/parameter-golf/blob/main/data/README.md) |
| MoE / sparsity | “Not primary record branch” | Suggests **depth attention** / MoDA-style depth-KV, not classic MoE | Avoid MoE; focus recurrence | Claude distinguishes “MoE routing” from “depth attention”; others lump sparsity together as overhead risk at this scale. [arxiv](https://arxiv.org/pdf/2601.21582.pdf) |

## 3) Unique Discoveries

| Model | Unique Finding | Why It Matters |
|---|---|---|
| Claude Opus 4.6 Thinking | Use **depth-KV / MoDA-style attention across recurrence iterations** (depth attention) rather than classic MoE routing | Solves the “recurrent plateau” problem: recurrence gets more expressive without adding many parameters. [arxiv](https://arxiv.org/abs/2603.15619) |
| GPT-5.4 Thinking | Treat **~1.170** as the practical bar (mentions a public PR claiming 1.1702 with QAT + sliding window) | Changes your “must beat” target; implies the window for a simple graft-only PR is shrinking fast. [github](https://github.com/openai/parameter-golf/pull/117) |

## 4) Comprehensive Analysis

### High-Confidence Findings
All three models converge on the same core meta-strategy: **the win condition is stacking orthogonal levers**—(a) evaluation-time adaptation (TTT), (b) parameter-efficiency tricks (depth recurrence / tying), (c) compression/quantization (QAT → lower bit), and (d) possibly tokenizer improvements—because the challenge is explicitly constrained by **artifact bytes and wallclock**, not by “purity” of architecture. The agreement that your first move should be “mechanical composition” (SOTA training recipe + LoRA-TTT eval + sliding-window eval) is especially actionable: it’s the shortest path to a credible leaderboard jump while you build the higher-upside system. [github](https://github.com/openai/parameter-golf)

The second strong consensus is that **depth recurrence is the architectural unlock** under a 16MB cap. The public repo already has multiple recurrence PRs, which signals both that it’s promising and that execution quality (stability, gating, loop count, training recipe) is the differentiator. Since recurrence trades parameters for FLOPs, it aligns well with a setting where you have a hard size cap but a generous 8×H100 budget for short bursts. [github](https://github.com/openai/parameter-golf/pull/29)

Finally, all models agree QAT is a major ceiling-raiser, but it needs sequencing. The evidence base they cite is that staged transitions (train higher precision, then transition into extreme low-bit) can be easier to optimize than “low-bit from step 1,” which matters under a 10-minute budget. The council also notes that the repo/records ecosystem is already trending toward mixed precision and QAT-style “snap” phases, so this is not speculative. [github](https://github.com/openai/parameter-golf/pull/117)

### Areas of Divergence
The biggest practical disagreement is **how much to lean into kernels**. Claude Opus 4.6 argues that with makora, kernels stop being a time sink and become a decisive advantage—because fused recurrent blocks and fused LoRA-TTT steps can convert directly into more steps/loops/adaptation inside the caps. GPT-5.4 and Sonar don’t reject kernels, but they’re stricter: only write kernels when you can point to a specific bottleneck that unlocks *new* behavior (more TTT steps, longer context sliding-window, recurrent unrolling) rather than incremental throughput that the speedrun crowd may match. Given your stated background (custom kernels, custom attention, tokenizer) and makora access, Claude’s weighting is plausibly correct *for you*, but GPT’s caution is still operationally important: kernels can sprawl unless tightly scoped. [github](https://github.com/openai/parameter-golf/pull/77)

Quantization is the other key divergence: GPT-5.4 recommends **int4 QAT first, ternary later**, while Claude pushes harder toward **1.58-bit / ternary** as the central moonshot. The underlying difference is risk management: int4 has a more reliable optimization path; ternary has a bigger compression/param-density payoff but higher instability risk in short training. The right reconciliation is to branch: keep a robust int4-QAT branch producing incremental record attempts while a ternary branch explores the ceiling. [arxiv](https://arxiv.org/html/2502.11895v1)

Tokenizer emphasis differs mostly on scheduling. All models acknowledge the repo supports reproducible tokenizer rebuild and shard export, but GPT frames tokenizer as a “moonshot branch” whereas Claude/Sonar elevate it because you personally can do it quickly and because the metric is bits-per-byte, making tokenizer efficiency directly relevant. Practically: tokenizer work is high leverage only if you can (1) keep the artifact under 16,000,000 bytes, (2) keep evaluation byte accounting indisputable, and (3) avoid spending your entire iteration budget on retokenization pipelines. [github](https://github.com/openai/parameter-golf/blob/main/data/README.md)

### Unique Insights Worth Noting
Claude’s most interesting unique contribution is **depth attention / MoDA-style depth-KV** layered on top of recurrence: recurrence alone can become “same layer repeated,” but allowing attention heads to attend to KV from previous recurrence iterations gives the model a cheap mechanism to reuse intermediate computations across depth, increasing expressivity without ballooning unique parameters. This is unusually well-matched to your “custom attention” comfort. [arxiv](https://arxiv.org/abs/2603.15619)

GPT’s unique warning—treat the bar as already ~1.170—matters strategically: if the leaderboard is moving daily, a “TTT + SOTA” graft that might have topped yesterday could be “nice but not enough” tomorrow, so you should **parallelize** the record-hunter graft with at least one deeper architectural bet (recurrence and/or low-bit QAT). [github](https://github.com/openai/parameter-golf/pull/117)

### Recommendations (actionable “highest upside” plan)
Do this as **three parallel branches**, with explicit stop/go gates:

1) **Branch A (48–72h): Record-hunter graft**
   - Reproduce current best training recipe, then add **sliding-window eval + LoRA TTT** as-is. [github](https://github.com/openai/parameter-golf/blob/main/records/track_10min_16mb/2026-03-19_SlidingWindow_FP16Emb_10L_MuonWD_OvertoneInit/README.md)
   - Goal: quick submission, establishes credibility + gives you a working eval/packaging pipeline.

2) **Branch B (1–2 weeks): Recurrence + depth-attention**
   - Implement **2–4 unique blocks looped** (with per-iteration gating), then add **depth-KV (MoDA-like) attention** across loop iterations. [github](https://github.com/openai/parameter-golf/pull/8)
   - Only after it trains stably: fuse the “recurrent step” with makora if profiling shows launch/memory overhead is limiting.

3) **Branch C (runs alongside B): QAT ladder**
   - Start with **int4 QAT** (more stable), then attempt **ternary/1.58-bit** as a ceiling branch once the recurrence model is stable. [arxiv](https://arxiv.org/html/2502.11895v1)
   - Use staged transition (higher precision → low-bit) to reduce optimization pain under short wallclock. [arxiv](https://arxiv.org/html/2502.11895v1)

If you answer two questions, I can turn this into a concrete “implementation spec” (exact module targets, LoRA ranks/where, loop counts, and 3 makora kernel problem statements):
1) Are you optimizing for **record track only**, or also happy to land a “non-record weird idea” PR?
2) Do you have **Runpod H100 time now**, or are you constrained to 2×3090 until credits land?