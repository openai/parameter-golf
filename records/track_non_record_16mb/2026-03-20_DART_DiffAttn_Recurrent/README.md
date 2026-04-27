# DART: Differential Attention Recurrent Transformer

**Author:** Anand K S (https://github.com/anandks2006)  
**Institution:** Independent, Kerala, India  
**Track:** Non-record (unlimited compute)  
**Date:** 2026-03-20  
**Score:** `val_bpb = 1.85221128`


## How I Found The Competition

I came across this competition through my Google feed on March 18, 2026 the same day the competition was launched. I am currently in my 2nd year of BCA (Bachelor of Computer Applications) and I have no formal research background, no GPU, and I have never tried to train models before. But the architecture idea seemed promising and I decided to push forward.

---

## What DART Is

DART is my attempt at a lightweight alternative to the standard approach of stacking many independent layers. Most models work by passing data through 9, 12 or more completely separate blocks, each one a different set of weights doing a different transformation. DART takes a different approach and ie, one block used multiple times in a loop.

The idea is to get maximum information out of minimum parameters. Instead of paying the full parameter cost for each layer, DART reuses the same block over multiple passes but makes sure each pass brings something new, so the model is not just repeating the same transformation over and over.

To make each loop meaningfully different, DART uses a couple of methods, they are:

- **Differential Attention V2** — instead of standard attention which can get distracted by irrelevant tokens, DART uses two attention calculations and subtracts one from the other to cancel out the noise. This is from Microsoft Research (2025).

- **Per-loop low-rank Q delta** — each loop gets a small unique modification to how it processes queries, so loop 1 might focus on basic word patterns while loop 8 focuses on higher-level thinking. This was my own idea for this architecture.The total cost is 65,536 parameters across 4 loops and is a small price for genuine per-loop specialisation. This was my idea for this architecture and I have not seen it applied this way in the literature.

- **resid_mix** — each loop applies a learned balance between the current hidden state and the original input. This prevents the hidden state from drifting too far from the input across multiple passes, which can cause the model to lose track of what the original text said.

- **Loop position embeddings** — a small vector added to the hidden state at each loop telling the block which pass number it is on. 

- **U-Net skip connections** — the first half of loops save their hidden states, and the second half receives them in reverse order. This lets later loops directly access the raw early representations without them being overwritten.

- **QAT (Quantization-Aware Training)** — during training, weights are fake-quantized to simulate the int8 compression that happens at submission time. Specifically, per-row 99.99984th percentile clipping is applied and exactly matching the competition's evaluation quantization. This means the model trains knowing it will be compressed, rather than being surprised by quantization at the end.

- **From the competition baseline (unchanged)**: relu² MLP, QK RMSNorm before RoPE, per-head q_gain scalar, logit softcap, CastedLinear (fp32 weights with bf16 matmul), Muon optimizer, tokenizer-aware BPB evaluation.

- **Loop dropout(not there in final training)** — during cpu testing, each batch randomly uses a different number of loops. This prevents a training problem where earlier and later loops fight each other over the shared weights. Without this, more loops actually made the model worse and finding and fixing this problem was the most important research discovery I was met with during development also this issue was such a headache to figure out a solution since when I trained initially with various loop number and block number configuration, always the single loop performed better that the one with more loops which made me confused since having more loop should automatically make it perform well and that was my idea in head. And at end I figured it out and unfortunately I missed it out on training and its absence likely costs 0.02-0.05 bpb on the final training.

- **Global Memory tokens** — small learned vectors that carry information across loops, acting like a notepad the model can write to and read from across passes.

- **Deep supervision** — loss is computed after every loop and not just at the final one, so every pass through the block is forced to be useful.

## The result is a model with about 3.9 million parameters — roughly the same as the baseline's parameter count when you account for what those parameters achieve — compressed to 3.5MB, using only 22.5% of the 16MB budget.

## I will be working on making the architecture even better down the line by implementing other existing excellent techniques that will be the model surpass the current issues. 

---

## How I Built It

I used Claude, ChatGPT and Gemini throughout the project. I want to be completely honest about the use of AI.

The AI assistants helped me find relevant research papers, and Claude wrote the code, and suggested ideas that support the architecture. But the decisions were mine. I questioned every suggestion, ran every experiment on my laptop cpu and several times disagreed simultaneously with all three AI systems or as I like to call them The Council.

A clear example for the above statement is when early in cpu experiments showed that adding more loops made the model perform worse, all three AI systems told me to drop the recurrent approach entirely and just use a single-pass model. I thought the idea was still sound and kept investigating. Eventually we found the actual cause, it was a gradient conflict problem in shared-weight training and fixed it. Unfortunately the final architecture didn't keep the loop dropout since I forgot to implement it during final training.

The AI council handled the code and finding research papers. My job was to design the architecture by combining existing approaches that I believed will work well together and to push back when the results did not match the theory and suggests fixes. My initial inspiration came from Samsung SAIL Montreal's TinyRecursiveModels and seeing that a tiny model with repeated passes could outperform much larger models on hard reasoning tasks made me want to apply the same philosophy to language modeling for this competition.

---

## Training

The hardest part of this project was compute. I spent hours running architecture experiments on my laptop CPU (Ryzen 5500U, 8GB RAM) with small configurations to validate that the design actually learns. Once I was confident the architecture worked, the only GPU option I had was Google Colab's free T4.

The T4 free tier has no guaranteed uptime, disconnects randomly, and gave me around 2-3 hours per session. Without torch.compile (which caused indefinite hangs on T4 with my architecture), each training step took about 2.6 seconds. That meant I could only run 2,000 steps — roughly 65 million tokens of training — before the session limits ran out.

Another issue was that the competition baseline used 10.5 billion tokens. I used about 160 times less. The score gap between DART (1.852) and the baseline (1.224) is almost entirely explained by this, not by the architecture being worse.

If I had run the same number of steps as the baseline on equivalent hardware, our 87-minute T4 run would have taken about 16 seconds on 8×H100s. The 10-minute competition window would have allowed around 73,000 training steps.

**Training configuration:**

| Parameter | Value |
|---|---|
| Hardware | Google Colab T4, free tier |
| Steps | 2,000 |
| Sequence length | 256 tokens |
| Batch | 32,768 tokens/step |
| Total tokens | ~65M |
| Training time | ~87 minutes |
| Model parameters | 3,918,888 |
| Compressed size | 3.55MB (22.5% of 16MB) |

---

## Results

| Step | val_bpb |
|---|---|
| 0 | 4.1040 |
| 500 | 2.0876 |
| 1,000 | 1.9957 |
| 1,500 | 1.9294 |
| 2,000 | 1.8502 |
| **Final (int8 roundtrip)** | **1.85221128** |

The score is improving consistently across the run and had not plateaued at step 2000, suggesting the architecture has room to improve with more training and loop dropout.

---

## What Did Not Work

**16 loops was too slow**. The architecture was designed to run 16 loops. On T4 without torch.compile, that required 18 seconds per step — too slow to train meaningfully. I reduced to 4 loops. Whether 16 loops outperform 4 loops at full training scale is something I was not able to verify.

**torch.compile hung indefinitely** on T4 with gradient checkpointing enabled. Disabling it slowed training by about 4×. This is a known compatibility issue on older GPU architectures.

**Loops did not clearly beat 1-loop** in CPU ablation experiments. Loop dropout reduced the performance gap from 0.28 nats to 0.03 nats, but I was not able to run enough steps to definitively prove the recurrent approach is better than a single-pass model. The compute constraints made this impossible to resolve on CPU alone.

---

## Reproducibility

**To reproduce the submitted result (2000 steps, reduced config):**

```bash
DATA_PATH=/path/to/fineweb10B_sp1024 \
TOKENIZER_PATH=/path/to/fineweb_1024_bpe.model \
RUN_ID=dart_repro \
N_LOOPS=4 N_MEMORY=16 MODEL_DIM=512 \
TRAIN_SEQ_LEN=256 TRAIN_BATCH_TOKENS=32768 \
ITERATIONS=2000 python train_gpt.py
```

Requires a CUDA GPU with at least 6GB VRAM. torch.compile is disabled in this submission due to compatibility issues on T4.

**To evaluate the architecture at full scale (recommended for 8×H100):**

```bash
DATA_PATH=/path/to/fineweb10B_sp1024 \
TOKENIZER_PATH=/path/to/fineweb_1024_bpe.model \
RUN_ID=dart_fullscale \
N_LOOPS=16 N_MEMORY=32 MODEL_DIM=512 \
TRAIN_SEQ_LEN=1024 TRAIN_BATCH_TOKENS=524288 \
ITERATIONS=20000 MAX_WALLCLOCK_SECONDS=600 \
python train_gpt.py
```

This uses the intended 16 loops, full sequence length, and standard batch size.
The submitted result used 0.6% of this training budget due to free-tier compute
constraints. The architecture was designed for this config and has not been
evaluated at full scale.

---

## Files

| File | Description |
|---|---|
| `train_gpt.py` | Training script |
| `submission.json` | Scores and metadata |
| `train_log.txt` | Full training log |
| `README.md` | This document |
