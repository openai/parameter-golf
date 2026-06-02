![](https://cdn.prod.website-files.com/67d20fb9f56ff2ec6a7a657d/67ffe107b1b4ece2dae583b9_background-stars-hollow.svg)

![](https://cdn.prod.website-files.com/67d20fb9f56ff2ec6a7a657d/680bdd6f4e2537aa58f58c70_glow-medium.avif)

[Back](https://www.runpod.io/blog)

News

March 25th, 2026

5

minute read

# OpenAI's Parameter Golf: Train the Best Language Model That Fits in 16MB on Runpod

[![](https://cdn.prod.website-files.com/plugins/Basic/assets/placeholder.60f9b1840c.svg)](https://www.runpod.io/blog/openais-parameter-golf-train-the-best-language-model-that-fits-in-16mb-on-runpod#)

![](https://www.runpod.io/blog/openais-parameter-golf-train-the-best-language-model-that-fits-in-16mb-on-runpod)

# Heading 1

## Heading 2

### Heading 3

#### Heading 4

##### Heading 5

###### Heading 6

Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.

> Block quote

Ordered list

1. Item 1
2. Item 2
3. Item 3

Unordered list

- Item A
- Item B
- Item C

[Text link](https://university.webflow.com/lesson/add-and-nest-text-links-in-webflow)

**Bold text**

_Emphasis_

Superscript

Subscript

## What is Parameter Golf?

Parameter Golf is OpenAI's Model Craft Challenge: train the best language model you can that fits inside a 16MB artifact and trains in under 10 minutes on 8×H100 GPUs on Runpod. The metric is simple: lowest bits per byte (BPB) on the FineWeb validation set wins. No tricks with tokenizer scoring; it's tokenizer-agnostic compression performance, full stop.

If you're familiar with neural scaling laws, the challenge is essentially L(N) optimization: push validation loss as low as possible given a fixed parameter budget, with no constraints on architecture, data volume, or training steps (beyond the 10-minute wall clock). It's the parameter-constrained cousin of the [NanoGPT Speedrun](https://github.com/KellerJordan/modded-nanogpt) (which optimizes training time) and the [NanoGPT Slowrun](https://github.com/qlabs-eng/slowrun) (which constrains dataset size).

The challenge runs from **March 18 to April 30, 2026**, and OpenAI is backing it with **$1M in compute credits** to help participants get started. The leaderboard already has 18 record submissions, with the top score dropping from the naive baseline of 1.2244 BPB down to 1.1228 BPB in just five days.

## Why this matters

Parameter Golf isn't just a leaderboard game. It's pushing the community toward ideas that matter for the future of efficient AI: depth recurrence, aggressive parameter tying, quantization-aware training, novel tokenizers, test-time compute, and low-rank factorizations. The constraint forces creativity in a way that "throw more H100s at it" never does. The search for AI excellence is over breadth (more GPU compute) and depth (more efficient, targeted use of that compute.) This contest focuses on digging as deep as possible to eke out performance on a small GPU.

For OpenAI, it's also a talent pipeline. The challenge is designed in the spirit of competitive mathematics and programming olympiads, and strong participants may catch the attention of OpenAI's research team for early-career hiring.

## How scoring works

The 16MB cap is measured as **code bytes plus compressed model bytes**. All of your training code lives in a single `train_gpt.py` script, and the model is int8-quantized and zlib-compressed after training. No external downloads, network calls, or dataset access are allowed during evaluation. The artifact must be fully self-contained.

Validation performance is measured in **bits per byte (BPB)** on the FineWeb validation set, which is a fixed first-50k-document slice. Because the metric is BPB rather than cross-entropy over a fixed vocabulary, changing your tokenizer doesn't give you a free scoring advantage; the metric normalizes for it.

New leaderboard records must beat the current SOTA by at least 0.005 nats, with enough run logs to demonstrate statistical significance at p < 0.01. Typically, averaging over three runs is sufficient.

## Getting started on Runpod

OpenAI is partnering directly with Runpod to make the setup as frictionless as possible. There's even a [pre-built Runpod template](https://console.runpod.io/deploy?template=y5cejece4j&ref=nl2r56th) with all Python dependencies pre-installed.

### Step 1: Launch a GPU pod

[Create a Runpod account](https://console.runpod.io/deploy) if you don't already have one, and set up an SSH key in Settings so you can connect to your pod (or just use Jupyter Notebook if you prefer.). Then deploy a GPU Cloud Pod using the official Parameter Golf template linked above.

Start with a **1×H100** for development and experimentation. An 8×H100 SXM box (required for final leaderboard submissions) costs roughly $20/hr, so you'll want to validate your ideas on cheaper hardware first. A single H100 or even an A100 works well for iteration.

### Step 2: Clone and download data

Once your pod is running and you've SSH'd in:

```javascript
git clone https://github.com/openai/parameter-golf.git
cd parameter-golf



      Copy

```

Download the cached FineWeb dataset with the 1024-token BPE vocabulary:

```javascript
python3 data/cached_challenge_fineweb.py --variant sp1024



      Copy

```

This pulls the full validation split plus 80 training shards (around 8B tokens). For faster iteration, use `--train-shards 1` to grab just a single shard.

### Step 3: Run the baseline

Launch a training run on your single GPU:

```javascript
RUN_ID=baseline_sp1024 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=1 train_gpt.py




      Copy

```

The script enforces a 10-minute wall clock by default. When the run finishes, it prints `val_loss`, `val_bpb`, and the compressed model size. The baseline config — 9 layers, 512 hidden dim, 1024 vocab, tied embeddings, 4 KV heads — should land around **1.2244 BPB** with a compressed artifact well under 16MB.

To enable periodic validation logging during training, set `VAL_LOSS_EVERY=200`. To remove the wall clock cap for unlimited experimentation, set `MAX_WALLCLOCK_SECONDS=0`.

### Step 4: Scale to 8×H100

When you're ready to submit, spin up an 8×H100 SXM pod on Runpod using the same template, and change the launch command:

```javascript
torchrun --standalone --nproc_per_node=8 train_gpt.py




      Copy

```

Everything else stays the same. The training script handles distributed data loading and gradient synchronization internally.

## Running your first experiments

Here are a few practical experiments to run on a single-GPU Runpod pod as you get your bearings.

### Test 1: Baseline sanity check

Run the stock baseline to confirm your environment is working and you can reproduce the published 1.2244 score:

```javascript
RUN_ID=test_baseline \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
VAL_LOSS_EVERY=100 \
torchrun --standalone --nproc_per_node=1 train_gpt.py



      Copy

```

Watch the validation loss curve — it should plateau around the expected value. If your compressed model is over 16MB, something is wrong with quantization or compression.

### Test 2: Longer context

One of the earliest wins on the leaderboard was simply increasing sequence length from the default to 2048 or 4096. Try it:

```javascript
RUN_ID=test_seq2048 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
SEQ_LEN=2048 \
torchrun --standalone --nproc_per_node=1 train_gpt.py



      Copy

```

This alone moved the score from 1.2244 to 1.206 in the original leaderboard submissions.

### Test 3: Layer depth vs. width

The baseline uses 9 layers with a 512 hidden dimension. Try 10 or 11 layers at the same dimension — the top submissions all use 10–11 layers, which suggests that depth is more parameter-efficient than width for this task:

```javascript
RUN_ID=test_11layers \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
NUM_LAYERS=11 \
torchrun --standalone --nproc_per_node=1 train_gpt.py



      Copy

```

## What the top submissions are doing

The leaderboard has evolved fast. Here's a look at the key techniques that have driven scores from 1.2244 down to 1.1228 BPB in less than a week.

### Quantization-aware training (QAT)

Nearly every competitive submission uses int6 or mixed int5/int6 quantization with straight-through estimator (STE) gradients. Since the artifact size is measured post-compression, training the model to be robust to low-precision weights is one of the highest-leverage moves. The current leader uses GPTQ-lite clip search combined with QAT at 0.15 to squeeze maximum information into the 16MB budget.

### Wider MLPs

Several top runs use a 3× MLP expansion ratio instead of the standard 4×, which keeps the parameter count balanced differently between attention and feed-forward layers. Combined with int6 quantization, this allows more expressive capacity per compressed byte.

### Sliding window evaluation

A surprisingly impactful evaluation technique: instead of evaluating the model at a fixed context length, top submissions use a sliding window with a stride of 64 tokens, effectively increasing the evaluation context. This doesn't change the model itself, but it better captures the model's actual compression ability. The first submission to use this technique jumped from ~1.20 to ~1.19 BPB.

### Exponential moving average (EMA) and stochastic weight averaging (SWA)

Model averaging during training produces smoother final weights, which compress better and generalize better on the validation set. The transition from SWA (used in earlier submissions) to EMA in the March 21 submissions coincided with another significant score drop.

### Cross-sequence attention (XSA)

One of the more creative architectural changes: applying attention across sequence boundaries in the deepest layers of the model, allowing the model to borrow context from adjacent sequences during evaluation. The partial variant (applying XSA to only the last 3–4 layers) keeps the computational overhead manageable.

### Test-time training (TTT)

The LoRA TTT submission takes a different approach entirely: the model adapts its own weights at evaluation time using low-rank updates on already-evaluated tokens. This doesn't violate the rules because it only trains on tokens that have already been scored. It achieved 1.1928 BPB, competitive with much more traditional approaches.

## Key rules and constraints to know

**Artifact size**: 16,000,000 bytes (decimal, not MiB). Code bytes plus compressed model bytes.

**Training compute**: 10 minutes wall clock on 8×H100 SXM GPUs.

**Evaluation compute**: An additional 10 minutes on 8×H100s. You can evaluate at any sequence length.

**External packages**: You can import any Python library (FlashAttention, custom CUDA kernels, etc.) as long as it doesn't sneak in additional data or violate the spirit of the challenge. Package bytes don't count toward the 16MB limit.

**No validation data during training**: You cannot train on the validation set. Test-time training is allowed only on tokens you've already evaluated.

**Statistical significance**: New records must beat the current SOTA by ≥0.005 nats at p < 0.01.

## How to submit

All submissions are made as pull requests to the [parameter-golf repo](https://github.com/openai/parameter-golf). Your PR should add a new folder under `records/track_10min_16mb/` containing:

1. A `README.md` explaining your approach
2. A `submission.json` with your name, GitHub handle, `val_bpb`, and metadata
3. A training log demonstrating statistical significance (usually 3 runs)
4. A working `train_gpt.py` and any dependencies

Non-record submissions, likeinteresting approaches that don't beat SOTA are welcome too, especially novel or unconventional ideas.

## Resources

- **Repository**: [github.com/openai/parameter-golf](https://github.com/openai/parameter-golf)
- **Runpod template**: [Launch on Runpod](https://console.runpod.io/deploy?template=y5cejece4j&ref=nl2r56th)
- **Runpod × OpenAI landing page**: [modelcraft.runpod.io](https://modelcraft.runpod.io/)
- **Compute credits**: [Request a grant from OpenAI](https://openai.com/index/parameter-golf/#credit-form)
- **Discord**: Join the OpenAI Discord server, channels `#parameter-golf-discussions` and `#parameter-golf-announcements`
- **Participant form**: [Sign up with OpenAI](https://jobs.ashbyhq.com/openai/form/open-ai-challenge-parameter-golf) (optional, but helps with attribution)

The challenge runs through April 30. The baseline is right there waiting to be beaten, the Runpod template makes setup a five-minute task, and OpenAI is handing out compute credits. Fore!

‍

Newly  Features

We've cooked up a bunch of improvements designed to reduce friction and make the.

[Create ->](https://console.runpod.io/deploy)

![](https://cdn.prod.website-files.com/67d20fb9f56ff2ec6a7a657d/68af0d45c8fbb0aed73b9670_Use%20Case%20Hero%20Visual.webp)

![](https://cdn.prod.website-files.com/67d20fb9f56ff2ec6a7a657d/680bdc63c870b659e486cd49_glow.avif)

Newly  Features

We've cooked up a bunch of improvements designed to reduce friction and make the.

[Create ->](https://console.runpod.io/deploy)

![](https://cdn.prod.website-files.com/67d20fb9f56ff2ec6a7a657d/68af0d457971d5aab88f6b3a_Thumbnail%20Image.webp)

![](https://cdn.prod.website-files.com/67d20fb9f56ff2ec6a7a657d/680bdc63c870b659e486cd49_glow.avif)

Newly  Features

We've cooked up a bunch of improvements designed to reduce friction and make the.

[Create ->](https://console.runpod.io/deploy)

![](https://cdn.prod.website-files.com/67d20fb9f56ff2ec6a7a657d/68af10f74787b8d6ca5ba48c_image%202%20(1).webp)

![](https://cdn.prod.website-files.com/67d20fb9f56ff2ec6a7a657d/680bdc63c870b659e486cd49_glow.avif)

Newly  Features

We've cooked up a bunch of improvements designed to reduce friction and make the.

[Create ->](https://console.runpod.io/deploy)

![](https://cdn.prod.website-files.com/67d20fb9f56ff2ec6a7a657d/68af10f74787b8d6ca5ba48c_image%202%20(1).webp)

![](https://cdn.prod.website-files.com/67d20fb9f56ff2ec6a7a657d/680bdc63c870b659e486cd49_glow.avif)

[![](https://cdn.prod.website-files.com/67d20fb9f56ff2ec6a7a657d/680d5323c5490e51002e6016_news-icon.svg)\\
\\
Runpod × OpenAI: Parameter Golf challenge is live \\
\\
![](https://cdn.prod.website-files.com/67d20fb9f56ff2ec6a7a657d/680d539311ae712d28f09376_cf84e4ef41ab0c4d3eb665bea53e4036_right-arrow-icon.svg)](https://modelcraft.runpod.io/)

![](https://cdn.prod.website-files.com/67d20fb9f56ff2ec6a7a657d/68c7c7a44ac3836441861f21_gift-1.svg)

You've unlocked a referral bonus! Sign up today and you'll get a random credit bonus between $5 and $500

You've unlocked a referral bonus!

[Claim Your Bonus\\
\\
Claim Bonus](https://console.runpod.io/signup)

[![Runpod logo](https://cdn.prod.website-files.com/67d20fb9f56ff2ec6a7a657d/683cd0ee11462aef4a016ef6_runpod%20lowercase.webp)](https://www.runpod.io/)

Product

[Cloud GPUs\\
\\
On-demand GPUs, deployed across 31 global regions.](https://www.runpod.io/product/cloud-gpus) [Serverless\\
\\
Instant AI workloads—no setup, scaling, or idle costs.](https://www.runpod.io/product/serverless)

[Clusters\\
\\
Deploy multi-node GPU clusters in minutes.](https://www.runpod.io/product/clusters) [RunPod Hub\\
\\
The fastest way to deploy open-source AI.](https://www.runpod.io/product/runpod-hub)

Use Cases

[Inference\\
\\
Serve models in real-time with low-latency GPUs.](https://www.runpod.io/use-cases/inference) [Fine-Tuning\\
\\
Train models faster with efficient, scalable compute.](https://www.runpod.io/use-cases/fine-tuning)

[Agents\\
\\
Deploy AI agents that run, react, and scale instantly.](https://www.runpod.io/use-cases/agents) [Compute-Heavy Tasks\\
\\
Process massive workloads with zero bottlenecks.](https://www.runpod.io/use-cases/compute-heavy-tasks)

Resources

[Blog\\
\\
Our team’s insights on building better and scaling smarter.](https://www.runpod.io/blog) [Case Studies\\
\\
Loved by leaders. But don’t just take it from us.](https://www.runpod.io/case-studies) [Articles\\
\\
In-depth perspectives, ideas, and updates from our team.](https://www.runpod.io/articles) [Press\\
\\
Company news and media coverage.](https://www.runpod.io/press)

Company

[About\\
\\
Redefining cloud compute with speed, scale, and innovation.](https://www.runpod.io/about)

[Careers\\
\\
Join our mission to build the launchpad for AI apps.](https://job-boards.greenhouse.io/runpod)

[Docs](https://docs.runpod.io/) [Pricing](https://www.runpod.io/pricing)

[Sign in](https://console.runpod.io/login) [Get started](https://console.runpod.io/signup)

[Pods\\
\\
On-demand GPUs, deployed across 31 global regions.](https://www.runpod.io/product/cloud-gpus) [Clusters\\
\\
Deploy multi-node GPU clusters in minutes.](https://www.runpod.io/product/clusters)

[Serverless\\
\\
Instant AI workloads—no setup, scaling, or idle costs.](https://www.runpod.io/product/serverless) [Hub\\
\\
The fastest way to deploy open-source AI.](https://www.runpod.io/product/runpod-hub)

[Inference\\
\\
Serve models in real-time with low-latency GPUs.](https://www.runpod.io/use-cases/inference) [Fine-Tuning\\
\\
Train models faster with efficient, scalable compute.](https://www.runpod.io/use-cases/fine-tuning)

[Agents\\
\\
Deploy AI agents that run, react, and scale instantly.](https://www.runpod.io/use-cases/agents) [Compute-Heavy Tasks\\
\\
Process massive workloads with zero bottlenecks.](https://www.runpod.io/use-cases/compute-heavy-tasks)

[Case Studies\\
\\
Loved by leaders. But don’t just take it from us.](https://www.runpod.io/case-studies) [Blog\\
\\
Our team’s insights on building better and scaling smarter.](https://www.runpod.io/blog)

[Articles\\
\\
In-depth perspectives, ideas, and updates from our team.](https://www.runpod.io/articles) [Press\\
\\
Company news and media coverage.](https://www.runpod.io/press)

[About\\
\\
Redefining cloud compute with speed, scale, and innovation.](https://www.runpod.io/about)

[Careers\\
\\
Join our mission to build the launchpad for AI apps.](https://job-boards.greenhouse.io/runpod)

Contact Sales

[Sign In](https://console.runpod.io/login) [Sign Up](https://console.runpod.io/deploy)

![](https://cdn.prod.website-files.com/67d20fb9f56ff2ec6a7a657d/680ed342430b928d63ce876a_menu-icon.svg)![](https://cdn.prod.website-files.com/67d20fb9f56ff2ec6a7a657d/68116f1fc6525d51ba45db55_e28207cf9410298cfeb52805ff63e2f6_x-icon.svg)

Form

# Tell Us About Your Needs

Please fill out the form below and a member of our team will be in touch shortly to schedule a conversation.

First Name\*

Last Name\*

Phone Number\*

🇺🇸

- 🇦🇫 Afghanistan (‫افغانستان‬‎) +93
- 🇦🇱 Albania (Shqipëri) +355
- 🇩🇿 Algeria (‫الجزائر‬‎) +213
- 🇦🇸 American Samoa +1684
- 🇦🇩 Andorra +376
- 🇦🇴 Angola +244
- 🇦🇮 Anguilla +1264
- 🇦🇬 Antigua and Barbuda +1268
- 🇦🇷 Argentina +54
- 🇦🇲 Armenia (Հայաստան) +374
- 🇦🇼 Aruba +297
- 🇦🇺 Australia +61
- 🇦🇹 Austria (Österreich) +43
- 🇦🇿 Azerbaijan (Azərbaycan) +994
- 🇧🇸 Bahamas +1242
- 🇧🇭 Bahrain (‫البحرين‬‎) +973
- 🇧🇩 Bangladesh (বাংলাদেশ) +880
- 🇧🇧 Barbados +1246
- 🇧🇾 Belarus (Беларусь) +375
- 🇧🇪 Belgium (België) +32
- 🇧🇿 Belize +501
- 🇧🇯 Benin (Bénin) +229
- 🇧🇲 Bermuda +1441
- 🇧🇹 Bhutan (འབྲུག) +975
- 🇧🇴 Bolivia +591
- 🇧🇦 Bosnia and Herzegovina (Босна и Херцеговина) +387
- 🇧🇼 Botswana +267
- 🇧🇷 Brazil (Brasil) +55
- 🇮🇴 British Indian Ocean Territory +246
- 🇻🇬 British Virgin Islands +1284
- 🇧🇳 Brunei +673
- 🇧🇬 Bulgaria (България) +359
- 🇧🇫 Burkina Faso +226
- 🇧🇮 Burundi (Uburundi) +257
- 🇰🇭 Cambodia (កម្ពុជា) +855
- 🇨🇲 Cameroon (Cameroun) +237
- 🇨🇦 Canada +1
- 🇨🇻 Cape Verde (Kabu Verdi) +238
- 🇧🇶 Caribbean Netherlands +599
- 🇰🇾 Cayman Islands +1345
- 🇨🇫 Central African Republic (République centrafricaine) +236
- 🇹🇩 Chad (Tchad) +235
- 🇨🇱 Chile +56
- 🇨🇳 China (中国) +86
- 🇨🇴 Colombia +57
- 🇰🇲 Comoros (‫جزر القمر‬‎) +269
- 🇨🇩 Congo (DRC) (Jamhuri ya Kidemokrasia ya Kongo) +243
- 🇨🇬 Congo (Republic) (Congo-Brazzaville) +242
- 🇨🇰 Cook Islands +682
- 🇨🇷 Costa Rica +506
- 🇨🇮 Côte d’Ivoire +225
- 🇭🇷 Croatia (Hrvatska) +385
- 🇨🇺 Cuba +53
- 🇨🇼 Curaçao +599
- 🇨🇾 Cyprus (Κύπρος) +357
- 🇨🇿 Czech Republic (Česká republika) +420
- 🇩🇰 Denmark (Danmark) +45
- 🇩🇯 Djibouti +253
- 🇩🇲 Dominica +1767
- 🇩🇴 Dominican Republic (República Dominicana) +1
- 🇪🇨 Ecuador +593
- 🇪🇬 Egypt (‫مصر‬‎) +20
- 🇸🇻 El Salvador +503
- 🇬🇶 Equatorial Guinea (Guinea Ecuatorial) +240
- 🇪🇷 Eritrea +291
- 🇪🇪 Estonia (Eesti) +372
- 🇪🇹 Ethiopia +251
- 🇫🇰 Falkland Islands (Islas Malvinas) +500
- 🇫🇴 Faroe Islands (Føroyar) +298
- 🇫🇯 Fiji +679
- 🇫🇮 Finland (Suomi) +358
- 🇫🇷 France +33
- 🇬🇫 French Guiana (Guyane française) +594
- 🇵🇫 French Polynesia (Polynésie française) +689
- 🇬🇦 Gabon +241
- 🇬🇲 Gambia +220
- 🇬🇪 Georgia (საქართველო) +995
- 🇩🇪 Germany (Deutschland) +49
- 🇬🇭 Ghana (Gaana) +233
- 🇬🇮 Gibraltar +350
- 🇬🇷 Greece (Ελλάδα) +30
- 🇬🇱 Greenland (Kalaallit Nunaat) +299
- 🇬🇩 Grenada +1473
- 🇬🇵 Guadeloupe +590
- 🇬🇺 Guam +1671
- 🇬🇹 Guatemala +502
- 🇬🇳 Guinea (Guinée) +224
- 🇬🇼 Guinea-Bissau (Guiné Bissau) +245
- 🇬🇾 Guyana +592
- 🇭🇹 Haiti +509
- 🇭🇳 Honduras +504
- 🇭🇰 Hong Kong (香港) +852
- 🇭🇺 Hungary (Magyarország) +36
- 🇮🇸 Iceland (Ísland) +354
- 🇮🇳 India (भारत) +91
- 🇮🇩 Indonesia +62
- 🇮🇷 Iran (‫ایران‬‎) +98
- 🇮🇶 Iraq (‫العراق‬‎) +964
- 🇮🇪 Ireland +353
- 🇮🇱 Israel (‫ישראל‬‎) +972
- 🇮🇹 Italy (Italia) +39
- 🇯🇲 Jamaica +1876
- 🇯🇵 Japan (日本) +81
- 🇯🇴 Jordan (‫الأردن‬‎) +962
- 🇰🇿 Kazakhstan (Казахстан) +7
- 🇰🇪 Kenya +254
- 🇰🇮 Kiribati +686
- 🇽🇰 Kosovo +383
- 🇰🇼 Kuwait (‫الكويت‬‎) +965
- 🇰🇬 Kyrgyzstan (Кыргызстан) +996
- 🇱🇦 Laos (ລາວ) +856
- 🇱🇻 Latvia (Latvija) +371
- 🇱🇧 Lebanon (‫لبنان‬‎) +961
- 🇱🇸 Lesotho +266
- 🇱🇷 Liberia +231
- 🇱🇾 Libya (‫ليبيا‬‎) +218
- 🇱🇮 Liechtenstein +423
- 🇱🇹 Lithuania (Lietuva) +370
- 🇱🇺 Luxembourg +352
- 🇲🇴 Macau (澳門) +853
- 🇲🇰 Macedonia (FYROM) (Македонија) +389
- 🇲🇬 Madagascar (Madagasikara) +261
- 🇲🇼 Malawi +265
- 🇲🇾 Malaysia +60
- 🇲🇻 Maldives +960
- 🇲🇱 Mali +223
- 🇲🇹 Malta +356
- 🇲🇭 Marshall Islands +692
- 🇲🇶 Martinique +596
- 🇲🇷 Mauritania (‫موريتانيا‬‎) +222
- 🇲🇺 Mauritius (Moris) +230
- 🇲🇽 Mexico (México) +52
- 🇫🇲 Micronesia +691
- 🇲🇩 Moldova (Republica Moldova) +373
- 🇲🇨 Monaco +377
- 🇲🇳 Mongolia (Монгол) +976
- 🇲🇪 Montenegro (Crna Gora) +382
- 🇲🇸 Montserrat +1664
- 🇲🇦 Morocco (‫المغرب‬‎) +212
- 🇲🇿 Mozambique (Moçambique) +258
- 🇲🇲 Myanmar (Burma) (မြန်မာ) +95
- 🇳🇦 Namibia (Namibië) +264
- 🇳🇷 Nauru +674
- 🇳🇵 Nepal (नेपाल) +977
- 🇳🇱 Netherlands (Nederland) +31
- 🇳🇨 New Caledonia (Nouvelle-Calédonie) +687
- 🇳🇿 New Zealand +64
- 🇳🇮 Nicaragua +505
- 🇳🇪 Niger (Nijar) +227
- 🇳🇬 Nigeria +234
- 🇳🇺 Niue +683
- 🇳🇫 Norfolk Island +672
- 🇰🇵 North Korea (조선 민주주의 인민 공화국) +850
- 🇲🇵 Northern Mariana Islands +1670
- 🇳🇴 Norway (Norge) +47
- 🇴🇲 Oman (‫عُمان‬‎) +968
- 🇵🇰 Pakistan (‫پاکستان‬‎) +92
- 🇵🇼 Palau +680
- 🇵🇸 Palestine (‫فلسطين‬‎) +970
- 🇵🇦 Panama (Panamá) +507
- 🇵🇬 Papua New Guinea +675
- 🇵🇾 Paraguay +595
- 🇵🇪 Peru (Perú) +51
- 🇵🇭 Philippines +63
- 🇵🇱 Poland (Polska) +48
- 🇵🇹 Portugal +351
- 🇵🇷 Puerto Rico +1
- 🇶🇦 Qatar (‫قطر‬‎) +974
- 🇷🇪 Réunion (La Réunion) +262
- 🇷🇴 Romania (România) +40
- 🇷🇺 Russia (Россия) +7
- 🇷🇼 Rwanda +250
- 🇧🇱 Saint Barthélemy (Saint-Barthélemy) +590
- 🇸🇭 Saint Helena +290
- 🇰🇳 Saint Kitts and Nevis +1869
- 🇱🇨 Saint Lucia +1758
- 🇲🇫 Saint Martin (Saint-Martin (partie française)) +590
- 🇵🇲 Saint Pierre and Miquelon (Saint-Pierre-et-Miquelon) +508
- 🇻🇨 Saint Vincent and the Grenadines +1784
- 🇼🇸 Samoa +685
- 🇸🇲 San Marino +378
- 🇸🇹 São Tomé and Príncipe (São Tomé e Príncipe) +239
- 🇸🇦 Saudi Arabia (‫المملكة العربية السعودية‬‎) +966
- 🇸🇳 Senegal (Sénégal) +221
- 🇷🇸 Serbia (Србија) +381
- 🇸🇨 Seychelles +248
- 🇸🇱 Sierra Leone +232
- 🇸🇬 Singapore +65
- 🇸🇽 Sint Maarten +1721
- 🇸🇰 Slovakia (Slovensko) +421
- 🇸🇮 Slovenia (Slovenija) +386
- 🇸🇧 Solomon Islands +677
- 🇸🇴 Somalia (Soomaaliya) +252
- 🇿🇦 South Africa +27
- 🇰🇷 South Korea (대한민국) +82
- 🇸🇸 South Sudan (‫جنوب السودان‬‎) +211
- 🇪🇸 Spain (España) +34
- 🇱🇰 Sri Lanka (ශ්‍රී ලංකාව) +94
- 🇸🇩 Sudan (‫السودان‬‎) +249
- 🇸🇷 Suriname +597
- 🇸🇿 Swaziland +268
- 🇸🇪 Sweden (Sverige) +46
- 🇨🇭 Switzerland (Schweiz) +41
- 🇸🇾 Syria (‫سوريا‬‎) +963
- 🇹🇼 Taiwan (台灣) +886
- 🇹🇯 Tajikistan +992
- 🇹🇿 Tanzania +255
- 🇹🇭 Thailand (ไทย) +66
- 🇹🇱 Timor-Leste +670
- 🇹🇬 Togo +228
- 🇹🇰 Tokelau +690
- 🇹🇴 Tonga +676
- 🇹🇹 Trinidad and Tobago +1868
- 🇹🇳 Tunisia (‫تونس‬‎) +216
- 🇹🇷 Turkey (Türkiye) +90
- 🇹🇲 Turkmenistan +993
- 🇹🇨 Turks and Caicos Islands +1649
- 🇹🇻 Tuvalu +688
- 🇻🇮 U.S. Virgin Islands +1340
- 🇺🇬 Uganda +256
- 🇺🇦 Ukraine (Україна) +380
- 🇦🇪 United Arab Emirates (‫الإمارات العربية المتحدة‬‎) +971
- 🇬🇧 United Kingdom +44
- 🇺🇸 United States +1
- 🇺🇾 Uruguay +598
- 🇺🇿 Uzbekistan (Oʻzbekiston) +998
- 🇻🇺 Vanuatu +678
- 🇻🇦 Vatican City (Città del Vaticano) +39
- 🇻🇪 Venezuela +58
- 🇻🇳 Vietnam (Việt Nam) +84
- 🇼🇫 Wallis and Futuna +681
- 🇾🇪 Yemen (‫اليمن‬‎) +967
- 🇿🇲 Zambia +260
- 🇿🇼 Zimbabwe +263

What's your work email?\*

Company Name\*

In which country is your company primarily based?\*

Website URL\*

33%

Next

What is your company's current monthly GPU infrastructure spend? (USD)\*

- $0
- $0 - $250
- $250 - $1000
- $1000 - $2500
- $2500 - $10,000
- $10,000 - $25,000
- $25,000+

Which GPU models are you currently using?\*

Who is your current GPU infrastructure provider?\*

What is your primary use case for GPU infrastructure?\*

What are your projected GPU infrastructure needs over the next 12 months?\*

How can we assist you right now, and what do you need from us?\*

33%

PreviousNext

How did you hear about Runpod?

Anything else we should know?

33%

PreviousSubmit

![](https://cdn.prod.website-files.com/67d20fb9f56ff2ec6a7a657d/67ffe107b1b4ece2dae583b9_background-stars-hollow.svg)

![](https://cdn.prod.website-files.com/67d20fb9f56ff2ec6a7a657d/68ec2a0a0107a1f150a0a835_Glow.avif)

[Blog](https://www.runpod.io/blog)

# OpenAI's Parameter Golf: Train the Best Language Model That Fits in 16MB on Runpod

OpenAI just launched one of the most exciting ML competitions in years, and Runpod is the official compute partner. Here's everything you need to know to get started.

Author

![](https://cdn.prod.website-files.com/plugins/Basic/assets/placeholder.60f9b1840c.svg)

Date

March 25, 2026

Table of contents

[What is Parameter Golf?](https://www.runpod.io/blog/openais-parameter-golf-train-the-best-language-model-that-fits-in-16mb-on-runpod#what-is-parameter-golf)

[Why this matters](https://www.runpod.io/blog/openais-parameter-golf-train-the-best-language-model-that-fits-in-16mb-on-runpod#why-this-matters)

[How scoring works](https://www.runpod.io/blog/openais-parameter-golf-train-the-best-language-model-that-fits-in-16mb-on-runpod#how-scoring-works)

[Getting started on Runpod](https://www.runpod.io/blog/openais-parameter-golf-train-the-best-language-model-that-fits-in-16mb-on-runpod#getting-started-on-runpod)

[Step 1: Launch a GPU pod](https://www.runpod.io/blog/openais-parameter-golf-train-the-best-language-model-that-fits-in-16mb-on-runpod#step-1-launch-a-gpu-pod)

[Step 2: Clone and download data](https://www.runpod.io/blog/openais-parameter-golf-train-the-best-language-model-that-fits-in-16mb-on-runpod#step-2-clone-and-download-data)

[Step 3: Run the baseline](https://www.runpod.io/blog/openais-parameter-golf-train-the-best-language-model-that-fits-in-16mb-on-runpod#step-3-run-the-baseline)

[Step 4: Scale to 8×H100](https://www.runpod.io/blog/openais-parameter-golf-train-the-best-language-model-that-fits-in-16mb-on-runpod#step-4-scale-to-8h100)

[Running your first experiments](https://www.runpod.io/blog/openais-parameter-golf-train-the-best-language-model-that-fits-in-16mb-on-runpod#running-your-first-experiments)

[Test 1: Baseline sanity check](https://www.runpod.io/blog/openais-parameter-golf-train-the-best-language-model-that-fits-in-16mb-on-runpod#test-1-baseline-sanity-check)

[Test 2: Longer context](https://www.runpod.io/blog/openais-parameter-golf-train-the-best-language-model-that-fits-in-16mb-on-runpod#test-2-longer-context)

[Test 3: Layer depth vs. width](https://www.runpod.io/blog/openais-parameter-golf-train-the-best-language-model-that-fits-in-16mb-on-runpod#test-3-layer-depth-vs-width)

[What the top submissions are doing](https://www.runpod.io/blog/openais-parameter-golf-train-the-best-language-model-that-fits-in-16mb-on-runpod#what-the-top-submissions-are-doing)

[Quantization-aware training (QAT)](https://www.runpod.io/blog/openais-parameter-golf-train-the-best-language-model-that-fits-in-16mb-on-runpod#quantization-aware-training-qat)

[Wider MLPs](https://www.runpod.io/blog/openais-parameter-golf-train-the-best-language-model-that-fits-in-16mb-on-runpod#wider-mlps)

[Sliding window evaluation](https://www.runpod.io/blog/openais-parameter-golf-train-the-best-language-model-that-fits-in-16mb-on-runpod#sliding-window-evaluation)

[Exponential moving average (EMA) and stochastic weight averaging (SWA)](https://www.runpod.io/blog/openais-parameter-golf-train-the-best-language-model-that-fits-in-16mb-on-runpod#exponential-moving-average-ema-and-stochastic-weight-averaging-swa)

[Cross-sequence attention (XSA)](https://www.runpod.io/blog/openais-parameter-golf-train-the-best-language-model-that-fits-in-16mb-on-runpod#cross-sequence-attention-xsa)

[Test-time training (TTT)](https://www.runpod.io/blog/openais-parameter-golf-train-the-best-language-model-that-fits-in-16mb-on-runpod#test-time-training-ttt)

[Key rules and constraints to know](https://www.runpod.io/blog/openais-parameter-golf-train-the-best-language-model-that-fits-in-16mb-on-runpod#key-rules-and-constraints-to-know)

[How to submit](https://www.runpod.io/blog/openais-parameter-golf-train-the-best-language-model-that-fits-in-16mb-on-runpod#how-to-submit)

[Resources](https://www.runpod.io/blog/openais-parameter-golf-train-the-best-language-model-that-fits-in-16mb-on-runpod#resources)

Share

![facebook](https://cdn.prod.website-files.com/67d20fb9f56ff2ec6a7a657d/68ec2fda7a747e57d7aa1242_Vector.svg)

![x](https://cdn.prod.website-files.com/67d20fb9f56ff2ec6a7a657d/68ec2fdf247c37995404144b_Vector.svg)

![linkedin](https://cdn.prod.website-files.com/67d20fb9f56ff2ec6a7a657d/68ec2fd6dfeb716fdbc21809_Vector.svg)

[Get started](https://www.runpod.io/console/signup)

![OpenAI's Parameter Golf: Train the Best Language Model That Fits in 16MB on Runpod](https://cdn.prod.website-files.com/67e1e36d3551f5a66e4095f5/69c459403c1d40bd08c24632_0_1%20(4).jpeg)

## What is Parameter Golf?

Parameter Golf is OpenAI's Model Craft Challenge: train the best language model you can that fits inside a 16MB artifact and trains in under 10 minutes on 8×H100 GPUs on Runpod. The metric is simple: lowest bits per byte (BPB) on the FineWeb validation set wins. No tricks with tokenizer scoring; it's tokenizer-agnostic compression performance, full stop.

If you're familiar with neural scaling laws, the challenge is essentially L(N) optimization: push validation loss as low as possible given a fixed parameter budget, with no constraints on architecture, data volume, or training steps (beyond the 10-minute wall clock). It's the parameter-constrained cousin of the [NanoGPT Speedrun](https://github.com/KellerJordan/modded-nanogpt) (which optimizes training time) and the [NanoGPT Slowrun](https://github.com/qlabs-eng/slowrun) (which constrains dataset size).

The challenge runs from **March 18 to April 30, 2026**, and OpenAI is backing it with **$1M in compute credits** to help participants get started. The leaderboard already has 18 record submissions, with the top score dropping from the naive baseline of 1.2244 BPB down to 1.1228 BPB in just five days.

## Why this matters

Parameter Golf isn't just a leaderboard game. It's pushing the community toward ideas that matter for the future of efficient AI: depth recurrence, aggressive parameter tying, quantization-aware training, novel tokenizers, test-time compute, and low-rank factorizations. The constraint forces creativity in a way that "throw more H100s at it" never does. The search for AI excellence is over breadth (more GPU compute) and depth (more efficient, targeted use of that compute.) This contest focuses on digging as deep as possible to eke out performance on a small GPU.

For OpenAI, it's also a talent pipeline. The challenge is designed in the spirit of competitive mathematics and programming olympiads, and strong participants may catch the attention of OpenAI's research team for early-career hiring.

## How scoring works

The 16MB cap is measured as **code bytes plus compressed model bytes**. All of your training code lives in a single `train_gpt.py` script, and the model is int8-quantized and zlib-compressed after training. No external downloads, network calls, or dataset access are allowed during evaluation. The artifact must be fully self-contained.

Validation performance is measured in **bits per byte (BPB)** on the FineWeb validation set, which is a fixed first-50k-document slice. Because the metric is BPB rather than cross-entropy over a fixed vocabulary, changing your tokenizer doesn't give you a free scoring advantage; the metric normalizes for it.

New leaderboard records must beat the current SOTA by at least 0.005 nats, with enough run logs to demonstrate statistical significance at p < 0.01. Typically, averaging over three runs is sufficient.

## Getting started on Runpod

OpenAI is partnering directly with Runpod to make the setup as frictionless as possible. There's even a [pre-built Runpod template](https://console.runpod.io/deploy?template=y5cejece4j&ref=nl2r56th) with all Python dependencies pre-installed.

### Step 1: Launch a GPU pod

[Create a Runpod account](https://console.runpod.io/deploy) if you don't already have one, and set up an SSH key in Settings so you can connect to your pod (or just use Jupyter Notebook if you prefer.). Then deploy a GPU Cloud Pod using the official Parameter Golf template linked above.

Start with a **1×H100** for development and experimentation. An 8×H100 SXM box (required for final leaderboard submissions) costs roughly $20/hr, so you'll want to validate your ideas on cheaper hardware first. A single H100 or even an A100 works well for iteration.

### Step 2: Clone and download data

Once your pod is running and you've SSH'd in:

```javascript
git clone https://github.com/openai/parameter-golf.git
cd parameter-golf



      Copy

```

Download the cached FineWeb dataset with the 1024-token BPE vocabulary:

```javascript
python3 data/cached_challenge_fineweb.py --variant sp1024



      Copy

```

This pulls the full validation split plus 80 training shards (around 8B tokens). For faster iteration, use `--train-shards 1` to grab just a single shard.

### Step 3: Run the baseline

Launch a training run on your single GPU:

```javascript
RUN_ID=baseline_sp1024 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=1 train_gpt.py




      Copy

```

The script enforces a 10-minute wall clock by default. When the run finishes, it prints `val_loss`, `val_bpb`, and the compressed model size. The baseline config — 9 layers, 512 hidden dim, 1024 vocab, tied embeddings, 4 KV heads — should land around **1.2244 BPB** with a compressed artifact well under 16MB.

To enable periodic validation logging during training, set `VAL_LOSS_EVERY=200`. To remove the wall clock cap for unlimited experimentation, set `MAX_WALLCLOCK_SECONDS=0`.

### Step 4: Scale to 8×H100

When you're ready to submit, spin up an 8×H100 SXM pod on Runpod using the same template, and change the launch command:

```javascript
torchrun --standalone --nproc_per_node=8 train_gpt.py




      Copy

```

Everything else stays the same. The training script handles distributed data loading and gradient synchronization internally.

## Running your first experiments

Here are a few practical experiments to run on a single-GPU Runpod pod as you get your bearings.

### Test 1: Baseline sanity check

Run the stock baseline to confirm your environment is working and you can reproduce the published 1.2244 score:

```javascript
RUN_ID=test_baseline \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
VAL_LOSS_EVERY=100 \
torchrun --standalone --nproc_per_node=1 train_gpt.py



      Copy

```

Watch the validation loss curve — it should plateau around the expected value. If your compressed model is over 16MB, something is wrong with quantization or compression.

### Test 2: Longer context

One of the earliest wins on the leaderboard was simply increasing sequence length from the default to 2048 or 4096. Try it:

```javascript
RUN_ID=test_seq2048 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
SEQ_LEN=2048 \
torchrun --standalone --nproc_per_node=1 train_gpt.py



      Copy

```

This alone moved the score from 1.2244 to 1.206 in the original leaderboard submissions.

### Test 3: Layer depth vs. width

The baseline uses 9 layers with a 512 hidden dimension. Try 10 or 11 layers at the same dimension — the top submissions all use 10–11 layers, which suggests that depth is more parameter-efficient than width for this task:

```javascript
RUN_ID=test_11layers \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
NUM_LAYERS=11 \
torchrun --standalone --nproc_per_node=1 train_gpt.py



      Copy

```

## What the top submissions are doing

The leaderboard has evolved fast. Here's a look at the key techniques that have driven scores from 1.2244 down to 1.1228 BPB in less than a week.

### Quantization-aware training (QAT)

Nearly every competitive submission uses int6 or mixed int5/int6 quantization with straight-through estimator (STE) gradients. Since the artifact size is measured post-compression, training the model to be robust to low-precision weights is one of the highest-leverage moves. The current leader uses GPTQ-lite clip search combined with QAT at 0.15 to squeeze maximum information into the 16MB budget.

### Wider MLPs

Several top runs use a 3× MLP expansion ratio instead of the standard 4×, which keeps the parameter count balanced differently between attention and feed-forward layers. Combined with int6 quantization, this allows more expressive capacity per compressed byte.

### Sliding window evaluation

A surprisingly impactful evaluation technique: instead of evaluating the model at a fixed context length, top submissions use a sliding window with a stride of 64 tokens, effectively increasing the evaluation context. This doesn't change the model itself, but it better captures the model's actual compression ability. The first submission to use this technique jumped from ~1.20 to ~1.19 BPB.

### Exponential moving average (EMA) and stochastic weight averaging (SWA)

Model averaging during training produces smoother final weights, which compress better and generalize better on the validation set. The transition from SWA (used in earlier submissions) to EMA in the March 21 submissions coincided with another significant score drop.

### Cross-sequence attention (XSA)

One of the more creative architectural changes: applying attention across sequence boundaries in the deepest layers of the model, allowing the model to borrow context from adjacent sequences during evaluation. The partial variant (applying XSA to only the last 3–4 layers) keeps the computational overhead manageable.

### Test-time training (TTT)

The LoRA TTT submission takes a different approach entirely: the model adapts its own weights at evaluation time using low-rank updates on already-evaluated tokens. This doesn't violate the rules because it only trains on tokens that have already been scored. It achieved 1.1928 BPB, competitive with much more traditional approaches.

## Key rules and constraints to know

**Artifact size**: 16,000,000 bytes (decimal, not MiB). Code bytes plus compressed model bytes.

**Training compute**: 10 minutes wall clock on 8×H100 SXM GPUs.

**Evaluation compute**: An additional 10 minutes on 8×H100s. You can evaluate at any sequence length.

**External packages**: You can import any Python library (FlashAttention, custom CUDA kernels, etc.) as long as it doesn't sneak in additional data or violate the spirit of the challenge. Package bytes don't count toward the 16MB limit.

**No validation data during training**: You cannot train on the validation set. Test-time training is allowed only on tokens you've already evaluated.

**Statistical significance**: New records must beat the current SOTA by ≥0.005 nats at p < 0.01.

## How to submit

All submissions are made as pull requests to the [parameter-golf repo](https://github.com/openai/parameter-golf). Your PR should add a new folder under `records/track_10min_16mb/` containing:

1. A `README.md` explaining your approach
2. A `submission.json` with your name, GitHub handle, `val_bpb`, and metadata
3. A training log demonstrating statistical significance (usually 3 runs)
4. A working `train_gpt.py` and any dependencies

Non-record submissions, likeinteresting approaches that don't beat SOTA are welcome too, especially novel or unconventional ideas.

## Resources

- **Repository**: [github.com/openai/parameter-golf](https://github.com/openai/parameter-golf)
- **Runpod template**: [Launch on Runpod](https://console.runpod.io/deploy?template=y5cejece4j&ref=nl2r56th)
- **Runpod × OpenAI landing page**: [modelcraft.runpod.io](https://modelcraft.runpod.io/)
- **Compute credits**: [Request a grant from OpenAI](https://openai.com/index/parameter-golf/#credit-form)
- **Discord**: Join the OpenAI Discord server, channels `#parameter-golf-discussions` and `#parameter-golf-announcements`
- **Participant form**: [Sign up with OpenAI](https://jobs.ashbyhq.com/openai/form/open-ai-challenge-parameter-golf) (optional, but helps with attribution)

The challenge runs through April 30. The baseline is right there waiting to be beaten, the Runpod template makes setup a five-minute task, and OpenAI is handing out compute credits. Fore!

‍

Blog Posts

## Related articles.

[View all](https://www.runpod.io/blog)

[![](https://cdn.prod.website-files.com/67e1e36d3551f5a66e4095f5/6852b91abd0cf637517f93b5_685232ca19a05b610043fedc_SSH%2520Key.webp)\\
\\
**How to Achieve True SSH in Runpod**\\
\\
This tutorial guides users through setting up a true SSH daemon on Runpod, enabling functionalities like SCP and IDE connections. It covers selecting a compatible pod, configuring OpenSSH, and obtaining the correct SSH connection command.](https://www.runpod.io/blog/how-to-achieve-true-ssh-in-runpod)

[![](https://cdn.prod.website-files.com/67e1e36d3551f5a66e4095f5/6852b91519322b9fec901b4d_6851e0ece4f60eee993822b8_GPU%2520Clusters-%2520Powering%2520High-Performance%2520AI%2520(When%2520You%2520Need%2520It).webp)\\
\\
**GPU Clusters: Powering High-Performance AI (When You Need It)**\\
\\
Different stages of AI development call for different infrastructure. This post breaks down when GPU clusters shine—and how to scale up only when it counts.](https://www.runpod.io/blog/gpu-clusters-high-performance-ai)

[![](https://cdn.prod.website-files.com/67e1e36d3551f5a66e4095f5/6852b932d22b916f070c86bf_6852715cc850562caf0cae12_Pygmalion%2520AI.webp)\\
\\
**Pygmalion-7b from PygmalionAI has been released, and it's amazing**\\
\\
Pygmalion 7b and Metharme 7b significantly improve on the creative writing capabilities of Pygmalion 6b. This post walks through model comparisons and how to deploy them on Runpod with the Oobabooga template.](https://www.runpod.io/blog/pygmalion-7b-release)

## Build what’s next.

The most cost-effective platform for building, training, and scaling machine learning models—ready when you are.

[Get started](https://console.runpod.io/deploy) [Request a demo](https://www.runpod.io/blog/openais-parameter-golf-train-the-best-language-model-that-fits-in-16mb-on-runpod#)

![](https://cdn.prod.website-files.com/67d20fb9f56ff2ec6a7a657d/67ffe107ba0b4bd0fbeb35b2_background-stars.svg)

Form

# Tell Us About Your Needs

Please fill out the form below and a member of our team will be in touch shortly to schedule a conversation.

First Name\*

Last Name\*

Phone Number\*

🇺🇸

- 🇦🇫 Afghanistan (‫افغانستان‬‎) +93
- 🇦🇱 Albania (Shqipëri) +355
- 🇩🇿 Algeria (‫الجزائر‬‎) +213
- 🇦🇸 American Samoa +1684
- 🇦🇩 Andorra +376
- 🇦🇴 Angola +244
- 🇦🇮 Anguilla +1264
- 🇦🇬 Antigua and Barbuda +1268
- 🇦🇷 Argentina +54
- 🇦🇲 Armenia (Հայաստան) +374
- 🇦🇼 Aruba +297
- 🇦🇺 Australia +61
- 🇦🇹 Austria (Österreich) +43
- 🇦🇿 Azerbaijan (Azərbaycan) +994
- 🇧🇸 Bahamas +1242
- 🇧🇭 Bahrain (‫البحرين‬‎) +973
- 🇧🇩 Bangladesh (বাংলাদেশ) +880
- 🇧🇧 Barbados +1246
- 🇧🇾 Belarus (Беларусь) +375
- 🇧🇪 Belgium (België) +32
- 🇧🇿 Belize +501
- 🇧🇯 Benin (Bénin) +229
- 🇧🇲 Bermuda +1441
- 🇧🇹 Bhutan (འབྲུག) +975
- 🇧🇴 Bolivia +591
- 🇧🇦 Bosnia and Herzegovina (Босна и Херцеговина) +387
- 🇧🇼 Botswana +267
- 🇧🇷 Brazil (Brasil) +55
- 🇮🇴 British Indian Ocean Territory +246
- 🇻🇬 British Virgin Islands +1284
- 🇧🇳 Brunei +673
- 🇧🇬 Bulgaria (България) +359
- 🇧🇫 Burkina Faso +226
- 🇧🇮 Burundi (Uburundi) +257
- 🇰🇭 Cambodia (កម្ពុជា) +855
- 🇨🇲 Cameroon (Cameroun) +237
- 🇨🇦 Canada +1
- 🇨🇻 Cape Verde (Kabu Verdi) +238
- 🇧🇶 Caribbean Netherlands +599
- 🇰🇾 Cayman Islands +1345
- 🇨🇫 Central African Republic (République centrafricaine) +236
- 🇹🇩 Chad (Tchad) +235
- 🇨🇱 Chile +56
- 🇨🇳 China (中国) +86
- 🇨🇴 Colombia +57
- 🇰🇲 Comoros (‫جزر القمر‬‎) +269
- 🇨🇩 Congo (DRC) (Jamhuri ya Kidemokrasia ya Kongo) +243
- 🇨🇬 Congo (Republic) (Congo-Brazzaville) +242
- 🇨🇰 Cook Islands +682
- 🇨🇷 Costa Rica +506
- 🇨🇮 Côte d’Ivoire +225
- 🇭🇷 Croatia (Hrvatska) +385
- 🇨🇺 Cuba +53
- 🇨🇼 Curaçao +599
- 🇨🇾 Cyprus (Κύπρος) +357
- 🇨🇿 Czech Republic (Česká republika) +420
- 🇩🇰 Denmark (Danmark) +45
- 🇩🇯 Djibouti +253
- 🇩🇲 Dominica +1767
- 🇩🇴 Dominican Republic (República Dominicana) +1
- 🇪🇨 Ecuador +593
- 🇪🇬 Egypt (‫مصر‬‎) +20
- 🇸🇻 El Salvador +503
- 🇬🇶 Equatorial Guinea (Guinea Ecuatorial) +240
- 🇪🇷 Eritrea +291
- 🇪🇪 Estonia (Eesti) +372
- 🇪🇹 Ethiopia +251
- 🇫🇰 Falkland Islands (Islas Malvinas) +500
- 🇫🇴 Faroe Islands (Føroyar) +298
- 🇫🇯 Fiji +679
- 🇫🇮 Finland (Suomi) +358
- 🇫🇷 France +33
- 🇬🇫 French Guiana (Guyane française) +594
- 🇵🇫 French Polynesia (Polynésie française) +689
- 🇬🇦 Gabon +241
- 🇬🇲 Gambia +220
- 🇬🇪 Georgia (საქართველო) +995
- 🇩🇪 Germany (Deutschland) +49
- 🇬🇭 Ghana (Gaana) +233
- 🇬🇮 Gibraltar +350
- 🇬🇷 Greece (Ελλάδα) +30
- 🇬🇱 Greenland (Kalaallit Nunaat) +299
- 🇬🇩 Grenada +1473
- 🇬🇵 Guadeloupe +590
- 🇬🇺 Guam +1671
- 🇬🇹 Guatemala +502
- 🇬🇳 Guinea (Guinée) +224
- 🇬🇼 Guinea-Bissau (Guiné Bissau) +245
- 🇬🇾 Guyana +592
- 🇭🇹 Haiti +509
- 🇭🇳 Honduras +504
- 🇭🇰 Hong Kong (香港) +852
- 🇭🇺 Hungary (Magyarország) +36
- 🇮🇸 Iceland (Ísland) +354
- 🇮🇳 India (भारत) +91
- 🇮🇩 Indonesia +62
- 🇮🇷 Iran (‫ایران‬‎) +98
- 🇮🇶 Iraq (‫العراق‬‎) +964
- 🇮🇪 Ireland +353
- 🇮🇱 Israel (‫ישראל‬‎) +972
- 🇮🇹 Italy (Italia) +39
- 🇯🇲 Jamaica +1876
- 🇯🇵 Japan (日本) +81
- 🇯🇴 Jordan (‫الأردن‬‎) +962
- 🇰🇿 Kazakhstan (Казахстан) +7
- 🇰🇪 Kenya +254
- 🇰🇮 Kiribati +686
- 🇽🇰 Kosovo +383
- 🇰🇼 Kuwait (‫الكويت‬‎) +965
- 🇰🇬 Kyrgyzstan (Кыргызстан) +996
- 🇱🇦 Laos (ລາວ) +856
- 🇱🇻 Latvia (Latvija) +371
- 🇱🇧 Lebanon (‫لبنان‬‎) +961
- 🇱🇸 Lesotho +266
- 🇱🇷 Liberia +231
- 🇱🇾 Libya (‫ليبيا‬‎) +218
- 🇱🇮 Liechtenstein +423
- 🇱🇹 Lithuania (Lietuva) +370
- 🇱🇺 Luxembourg +352
- 🇲🇴 Macau (澳門) +853
- 🇲🇰 Macedonia (FYROM) (Македонија) +389
- 🇲🇬 Madagascar (Madagasikara) +261
- 🇲🇼 Malawi +265
- 🇲🇾 Malaysia +60
- 🇲🇻 Maldives +960
- 🇲🇱 Mali +223
- 🇲🇹 Malta +356
- 🇲🇭 Marshall Islands +692
- 🇲🇶 Martinique +596
- 🇲🇷 Mauritania (‫موريتانيا‬‎) +222
- 🇲🇺 Mauritius (Moris) +230
- 🇲🇽 Mexico (México) +52
- 🇫🇲 Micronesia +691
- 🇲🇩 Moldova (Republica Moldova) +373
- 🇲🇨 Monaco +377
- 🇲🇳 Mongolia (Монгол) +976
- 🇲🇪 Montenegro (Crna Gora) +382
- 🇲🇸 Montserrat +1664
- 🇲🇦 Morocco (‫المغرب‬‎) +212
- 🇲🇿 Mozambique (Moçambique) +258
- 🇲🇲 Myanmar (Burma) (မြန်မာ) +95
- 🇳🇦 Namibia (Namibië) +264
- 🇳🇷 Nauru +674
- 🇳🇵 Nepal (नेपाल) +977
- 🇳🇱 Netherlands (Nederland) +31
- 🇳🇨 New Caledonia (Nouvelle-Calédonie) +687
- 🇳🇿 New Zealand +64
- 🇳🇮 Nicaragua +505
- 🇳🇪 Niger (Nijar) +227
- 🇳🇬 Nigeria +234
- 🇳🇺 Niue +683
- 🇳🇫 Norfolk Island +672
- 🇰🇵 North Korea (조선 민주주의 인민 공화국) +850
- 🇲🇵 Northern Mariana Islands +1670
- 🇳🇴 Norway (Norge) +47
- 🇴🇲 Oman (‫عُمان‬‎) +968
- 🇵🇰 Pakistan (‫پاکستان‬‎) +92
- 🇵🇼 Palau +680
- 🇵🇸 Palestine (‫فلسطين‬‎) +970
- 🇵🇦 Panama (Panamá) +507
- 🇵🇬 Papua New Guinea +675
- 🇵🇾 Paraguay +595
- 🇵🇪 Peru (Perú) +51
- 🇵🇭 Philippines +63
- 🇵🇱 Poland (Polska) +48
- 🇵🇹 Portugal +351
- 🇵🇷 Puerto Rico +1
- 🇶🇦 Qatar (‫قطر‬‎) +974
- 🇷🇪 Réunion (La Réunion) +262
- 🇷🇴 Romania (România) +40
- 🇷🇺 Russia (Россия) +7
- 🇷🇼 Rwanda +250
- 🇧🇱 Saint Barthélemy (Saint-Barthélemy) +590
- 🇸🇭 Saint Helena +290
- 🇰🇳 Saint Kitts and Nevis +1869
- 🇱🇨 Saint Lucia +1758
- 🇲🇫 Saint Martin (Saint-Martin (partie française)) +590
- 🇵🇲 Saint Pierre and Miquelon (Saint-Pierre-et-Miquelon) +508
- 🇻🇨 Saint Vincent and the Grenadines +1784
- 🇼🇸 Samoa +685
- 🇸🇲 San Marino +378
- 🇸🇹 São Tomé and Príncipe (São Tomé e Príncipe) +239
- 🇸🇦 Saudi Arabia (‫المملكة العربية السعودية‬‎) +966
- 🇸🇳 Senegal (Sénégal) +221
- 🇷🇸 Serbia (Србија) +381
- 🇸🇨 Seychelles +248
- 🇸🇱 Sierra Leone +232
- 🇸🇬 Singapore +65
- 🇸🇽 Sint Maarten +1721
- 🇸🇰 Slovakia (Slovensko) +421
- 🇸🇮 Slovenia (Slovenija) +386
- 🇸🇧 Solomon Islands +677
- 🇸🇴 Somalia (Soomaaliya) +252
- 🇿🇦 South Africa +27
- 🇰🇷 South Korea (대한민국) +82
- 🇸🇸 South Sudan (‫جنوب السودان‬‎) +211
- 🇪🇸 Spain (España) +34
- 🇱🇰 Sri Lanka (ශ්‍රී ලංකාව) +94
- 🇸🇩 Sudan (‫السودان‬‎) +249
- 🇸🇷 Suriname +597
- 🇸🇿 Swaziland +268
- 🇸🇪 Sweden (Sverige) +46
- 🇨🇭 Switzerland (Schweiz) +41
- 🇸🇾 Syria (‫سوريا‬‎) +963
- 🇹🇼 Taiwan (台灣) +886
- 🇹🇯 Tajikistan +992
- 🇹🇿 Tanzania +255
- 🇹🇭 Thailand (ไทย) +66
- 🇹🇱 Timor-Leste +670
- 🇹🇬 Togo +228
- 🇹🇰 Tokelau +690
- 🇹🇴 Tonga +676
- 🇹🇹 Trinidad and Tobago +1868
- 🇹🇳 Tunisia (‫تونس‬‎) +216
- 🇹🇷 Turkey (Türkiye) +90
- 🇹🇲 Turkmenistan +993
- 🇹🇨 Turks and Caicos Islands +1649
- 🇹🇻 Tuvalu +688
- 🇻🇮 U.S. Virgin Islands +1340
- 🇺🇬 Uganda +256
- 🇺🇦 Ukraine (Україна) +380
- 🇦🇪 United Arab Emirates (‫الإمارات العربية المتحدة‬‎) +971
- 🇬🇧 United Kingdom +44
- 🇺🇸 United States +1
- 🇺🇾 Uruguay +598
- 🇺🇿 Uzbekistan (Oʻzbekiston) +998
- 🇻🇺 Vanuatu +678
- 🇻🇦 Vatican City (Città del Vaticano) +39
- 🇻🇪 Venezuela +58
- 🇻🇳 Vietnam (Việt Nam) +84
- 🇼🇫 Wallis and Futuna +681
- 🇾🇪 Yemen (‫اليمن‬‎) +967
- 🇿🇲 Zambia +260
- 🇿🇼 Zimbabwe +263

What's your work email?\*

Company Name\*

In which country is your company primarily based?\*

Website URL\*

33%

Next

What is your company's current monthly GPU infrastructure spend? (USD)\*

- $0
- $0 - $250
- $250 - $1000
- $1000 - $2500
- $2500 - $10,000
- $10,000 - $25,000
- $25,000+

Which GPU models are you currently using?\*

Who is your current GPU infrastructure provider?\*

What is your primary use case for GPU infrastructure?\*

What are your projected GPU infrastructure needs over the next 12 months?\*

How can we assist you right now, and what do you need from us?\*

33%

PreviousNext

How did you hear about Runpod?

Anything else we should know?

33%

PreviousSubmit

[![Runpod Icon](https://cdn.prod.website-files.com/67d20fb9f56ff2ec6a7a657d/68153a499ce86d40a735491e_5278b5a9c2fd13508f888d8ea1988b8b_runpod-icon-white.svg)\\
1181 Nixon Dr. #1158\\
\\
Moorestown NJ 08057](https://www.runpod.io/)

Product

[Cloud GPUs](https://www.runpod.io/product/cloud-gpus) [Serverless\\
\\
Popular](https://www.runpod.io/product/serverless) [Instant Clusters](https://www.runpod.io/product/clusters) [Hub](https://www.runpod.io/product/runpod-hub)

Use Cases

[Inference](https://www.runpod.io/use-cases/inference) [Fine-Tuning](https://www.runpod.io/use-cases/fine-tuning) [Agents](https://www.runpod.io/use-cases/agents) [Compute-Heavy Tasks](https://www.runpod.io/use-cases/compute-heavy-tasks)

Resources

[Blog](https://www.runpod.io/blog) [Articles](https://www.runpod.io/articles) [Press](https://www.runpod.io/press) [Pricing](https://www.runpod.io/pricing) [GPU Benchmarks](https://www.runpod.io/gpu-compare) [Case Studies](https://www.runpod.io/case-studies) [Models Directory](https://www.runpod.io/models) [Startup Program](https://www.runpod.io/startup-program) [Referral Program](https://www.runpod.io/referral-and-affiliate-program) [Docs](https://docs.runpod.io/) [Brandkit](https://www.runpod.io/brandkit)

Company

[About](https://www.runpod.io/about) [Contact](https://contact.runpod.io/hc/en-us) [Careers](https://job-boards.greenhouse.io/runpod)

Make infrastructure our job, so it doesn't have to be yours.

Checking your Browser…

Verifying...

Stuck? [Troubleshoot](https://challenges.cloudflare.com/cdn-cgi/challenge-platform/h/g/turnstile/f/ov2/av0/rch/ali9i/0x4AAAAAAAQTptj2So4dx43e/auto/fbE/new/normal?lang=auto#refresh)

Success!

Verification failed

[Troubleshoot](https://challenges.cloudflare.com/cdn-cgi/challenge-platform/h/g/turnstile/f/ov2/av0/rch/ali9i/0x4AAAAAAAQTptj2So4dx43e/auto/fbE/new/normal?lang=auto#refresh)

Verification expired

[Refresh](https://challenges.cloudflare.com/cdn-cgi/challenge-platform/h/g/turnstile/f/ov2/av0/rch/ali9i/0x4AAAAAAAQTptj2So4dx43e/auto/fbE/new/normal?lang=auto#refresh)

Verification expired

[Refresh](https://challenges.cloudflare.com/cdn-cgi/challenge-platform/h/g/turnstile/f/ov2/av0/rch/ali9i/0x4AAAAAAAQTptj2So4dx43e/auto/fbE/new/normal?lang=auto#refresh)

[Troubleshoot](https://challenges.cloudflare.com/cdn-cgi/challenge-platform/h/g/turnstile/f/ov2/av0/rch/ali9i/0x4AAAAAAAQTptj2So4dx43e/auto/fbE/new/normal?lang=auto#refresh)

[Privacy](https://www.cloudflare.com/privacypolicy/) • [Help](https://challenges.cloudflare.com/cdn-cgi/challenge-platform/help)

Checking your Browser…

Verifying...

Stuck? [Troubleshoot](https://challenges.cloudflare.com/cdn-cgi/challenge-platform/h/g/turnstile/f/ov2/av0/rch/sbjh4/0x4AAAAAAAQTptj2So4dx43e/auto/fbE/new/normal?lang=auto#refresh)

Success!

Verification failed

[Troubleshoot](https://challenges.cloudflare.com/cdn-cgi/challenge-platform/h/g/turnstile/f/ov2/av0/rch/sbjh4/0x4AAAAAAAQTptj2So4dx43e/auto/fbE/new/normal?lang=auto#refresh)

Verification expired

[Refresh](https://challenges.cloudflare.com/cdn-cgi/challenge-platform/h/g/turnstile/f/ov2/av0/rch/sbjh4/0x4AAAAAAAQTptj2So4dx43e/auto/fbE/new/normal?lang=auto#refresh)

Verification expired

[Refresh](https://challenges.cloudflare.com/cdn-cgi/challenge-platform/h/g/turnstile/f/ov2/av0/rch/sbjh4/0x4AAAAAAAQTptj2So4dx43e/auto/fbE/new/normal?lang=auto#refresh)

[Troubleshoot](https://challenges.cloudflare.com/cdn-cgi/challenge-platform/h/g/turnstile/f/ov2/av0/rch/sbjh4/0x4AAAAAAAQTptj2So4dx43e/auto/fbE/new/normal?lang=auto#refresh)

[Privacy](https://www.cloudflare.com/privacypolicy/) • [Help](https://challenges.cloudflare.com/cdn-cgi/challenge-platform/help)

![](https://challenges.cloudflare.com/cdn-cgi/challenge-platform/h/g/d/9ec05eeb0f1fc94c/1776145174570/M05Gw7gYXB5UBgh)

Checking your Browser…

Verifying...

Stuck? [Troubleshoot](https://challenges.cloudflare.com/cdn-cgi/challenge-platform/h/g/turnstile/f/ov2/av0/rch/wkewr/0x4AAAAAAAQTptj2So4dx43e/auto/fbE/new/normal?lang=auto#refresh)

Success!

Verification failed

[Troubleshoot](https://challenges.cloudflare.com/cdn-cgi/challenge-platform/h/g/turnstile/f/ov2/av0/rch/wkewr/0x4AAAAAAAQTptj2So4dx43e/auto/fbE/new/normal?lang=auto#refresh)

Verification expired

[Refresh](https://challenges.cloudflare.com/cdn-cgi/challenge-platform/h/g/turnstile/f/ov2/av0/rch/wkewr/0x4AAAAAAAQTptj2So4dx43e/auto/fbE/new/normal?lang=auto#refresh)

Verification expired

[Refresh](https://challenges.cloudflare.com/cdn-cgi/challenge-platform/h/g/turnstile/f/ov2/av0/rch/wkewr/0x4AAAAAAAQTptj2So4dx43e/auto/fbE/new/normal?lang=auto#refresh)

[Troubleshoot](https://challenges.cloudflare.com/cdn-cgi/challenge-platform/h/g/turnstile/f/ov2/av0/rch/wkewr/0x4AAAAAAAQTptj2So4dx43e/auto/fbE/new/normal?lang=auto#refresh)

[Privacy](https://www.cloudflare.com/privacypolicy/) • [Help](https://challenges.cloudflare.com/cdn-cgi/challenge-platform/help)

Checking your Browser…

Verifying...

Stuck? [Troubleshoot](https://challenges.cloudflare.com/cdn-cgi/challenge-platform/h/g/turnstile/f/ov2/av0/rch/xg4gj/0x4AAAAAAAQTptj2So4dx43e/auto/fbE/new/normal?lang=auto#refresh)

Success!

Verification failed

[Troubleshoot](https://challenges.cloudflare.com/cdn-cgi/challenge-platform/h/g/turnstile/f/ov2/av0/rch/xg4gj/0x4AAAAAAAQTptj2So4dx43e/auto/fbE/new/normal?lang=auto#refresh)

Verification expired

[Refresh](https://challenges.cloudflare.com/cdn-cgi/challenge-platform/h/g/turnstile/f/ov2/av0/rch/xg4gj/0x4AAAAAAAQTptj2So4dx43e/auto/fbE/new/normal?lang=auto#refresh)

Verification expired

[Refresh](https://challenges.cloudflare.com/cdn-cgi/challenge-platform/h/g/turnstile/f/ov2/av0/rch/xg4gj/0x4AAAAAAAQTptj2So4dx43e/auto/fbE/new/normal?lang=auto#refresh)

[Troubleshoot](https://challenges.cloudflare.com/cdn-cgi/challenge-platform/h/g/turnstile/f/ov2/av0/rch/xg4gj/0x4AAAAAAAQTptj2So4dx43e/auto/fbE/new/normal?lang=auto#refresh)

[Privacy](https://www.cloudflare.com/privacypolicy/) • [Help](https://challenges.cloudflare.com/cdn-cgi/challenge-platform/help)

[All systems operational](https://uptime.runpod.io/)

© 2026 Runpod Inc.

[Visit Runpod's GitHub (opens in a new tab)](https://github.com/runpod)[Join The Runpod Discord (opens in a new tab)](https://discord.com/invite/cUpRmau42V)[Visit Runpod's X (opens in a new tab)](https://x.com/runpod)[Visit Runpod's Linkedin (opens in a new tab)](https://www.linkedin.com/company/runpod-io/)

[Terms of Service](https://www.runpod.io/legal/terms-of-service) [Privacy Policy](https://www.runpod.io/legal/privacy-policy) [Cookie Policy](https://www.runpod.io/legal/cookie-policy) [Compliance](https://www.runpod.io/legal/compliance)

![](https://cdn.prod.website-files.com/67d20fb9f56ff2ec6a7a657d/6880dfd8bed266019a64522b_refferal-icon.svg)

### You’ve unlocked a  referral bonus!

Sign up today and you’ll get a random credit bonus between $5 and $500 when you spend your first $10 on Runpod.

[Claim Your Bonus](https://console.runpod.io/signup)

![](https://cdn.prod.website-files.com/67d20fb9f56ff2ec6a7a657d/6880e141e9c52dbb2a3eaf24_iconamoon_close-bold.svg)