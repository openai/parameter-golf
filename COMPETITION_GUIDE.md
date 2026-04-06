# OpenAI Parameter Golf — Complete Competition Guide & Winning Strategy

**Prepared for:** Shamhith Kamasani
**Date:** March 23, 2026
**Competition Deadline:** April 30, 2026 (4:59:59 PM PST)
**Time Remaining:** ~38 days

---

## 1. What Is Parameter Golf?

Parameter Golf is OpenAI's "Model Craft Challenge" — an open research competition where participants must train the **best possible language model** that fits inside a **16 MB artifact** (weights + training code combined) and trains in **under 10 minutes on 8×H100 GPUs**. Performance is measured by **bits-per-byte (BPB)** compression on a held-out FineWeb validation set. Lower BPB = better model = higher ranking.

The name comes from golf: just as golfers aim for the fewest strokes, participants aim for the fewest bits per byte — achieving maximum language modeling performance with minimal parameters.

**Official Repo:** https://github.com/openai/parameter-golf
**Official Page:** https://openai.com/index/parameter-golf/
**Discord:** OpenAI Discord → #parameter-golf-discussions & #parameter-golf-announcements
**Compute Credits Portal:** https://modelcraft.runpod.io/

---

## 2. Hard Constraints (Non-Negotiable Rules)

| Constraint | Specification |
|---|---|
| **Artifact Size** | ≤ 16,000,000 bytes (decimal MB, not MiB). Includes code + compressed model weights |
| **Training Time** | ≤ 10 minutes on 8×H100 SXM GPUs |
| **Evaluation Time** | ≤ 10 minutes on 8×H100 SXM GPUs (separate from training) |
| **Dataset** | Fixed FineWeb validation set (first 50k documents) |
| **Metric** | val_bpb (bits-per-byte, tokenizer-agnostic) |
| **Self-contained** | No network calls, no external downloads, no training data access during evaluation |
| **Statistical Significance** | New SOTA must beat current best by ≥ 0.005 nats at p < 0.01 |

**What's Allowed:**
- Any PyTorch-compatible library (FlashAttention, custom CUDA kernels, etc.)
- Any evaluation sequence length
- Offline hyperparameter tuning
- Test-time training on validation tokens you've *already evaluated*
- Any tokenizer (but custom tokenizers get extra scrutiny)

**What's Prohibited:**
- Accessing training data during evaluation
- Embedding/"paying for" validation set data in the 16MB via "paid prefix"
- Test-time training on validation tokens *not yet evaluated*
- Sneaking in extra compute through brute-force seed search
- Network calls during evaluation

---

## 3. Current Leaderboard (as of March 23, 2026)

| Rank | Run | BPB Score | Author | Key Techniques |
|---|---|---|---|---|
| 1 | 11L EMA + GPTQ-lite + warmdown3500 | **1.1228** | signalrush | GPTQ-lite clip search, EMA, warmdown3500, QAT@0.15 |
| 2 | 11L Partial RoPE + LN Scale + EMA + XSA4 | **1.1248** | jfprincz | Partial RoPE (16/64), layerwise LN scale, XSA last 4 layers |
| 3 | 11L XSA4 + EMA + Int6 MLP3x | **1.1271** | jfprincz | XSA on last 4 layers, EMA replacing SWA, int6, 3× MLP |
| 4 | 11L Efficient Partial XSA | **1.1307** | unnir | Efficient Partial XSA on deepest 3 layers, FA3, SWA120 |
| 5 | 10L Int5-MLP + BigramHash | **1.1428** | thwu1 | Mixed int5/int6, BigramHash(10240), SWA, WD=0.04 |
| — | **Baseline** | **1.2244** | — | 9 layers, 512 dim, 1024 vocab, tied embeddings, 4 KV heads |

The performance gap from baseline to SOTA is **~0.1 BPB**, achieved through layered innovations in architecture, quantization, and evaluation tricks.

---

## 4. The Technology Stack: What Top Submissions Use

### 4.1 Architecture Innovations

**More Layers (9 → 11):** Nearly every top submission has moved from the baseline 9 layers to 10–11 layers, fitting them in the 16MB budget through aggressive quantization.

**XSA (Extended Self-Attention):** Applied to the last 3–4 layers only ("Partial XSA"), this allows deeper layers to attend over a longer context window without the memory/compute cost of applying it everywhere.

**Partial RoPE:** Instead of full Rotary Position Embeddings on all dimensions, top submissions use RoPE on only 16 out of 64 dimensions, saving parameters while retaining positional information.

**SmearGate:** A gating mechanism that helps information flow, used by several mid-tier submissions.

**BigramHash:** A character-level bigram hashing feature that gives the model cheap access to sub-word statistics, acting as a powerful low-cost input feature.

**3× MLP Width:** Expanding the MLP hidden dimension to 3× the model dimension, then quantizing aggressively to fit in budget.

### 4.2 Quantization & Compression

**Int6 Quantization:** The dominant quantization level — 6-bit integer weights offer the best BPB-per-byte tradeoff.

**Mixed Precision Strategy:** The critical insight from top submissions: keep the **tied embedding matrix in fp16** (~1MB cost) because it's used for both input and output — quantization errors compound in both directions. Quantize everything else aggressively.

**GPTQ-lite:** The current #1 uses a lightweight variant of GPTQ (post-training quantization with calibration), which is more accurate than naive round-to-nearest quantization.

**QAT (Quantization-Aware Training):** Training with simulated quantization so the model learns to be robust to low-precision weights. Applied at different points during training (e.g., QAT@0.15 means enabling QAT at 15% through training).

**Zstandard Compression (level 22):** After quantization, weights are compressed with zstd at level 22, which squeezes int6 data significantly tighter than zlib — enough to fit 1–2M more parameters.

### 4.3 Training Techniques

**EMA (Exponential Moving Average):** Replacing SWA in top submissions. Maintains a running average of weights that is smoother and generalizes better.

**SWA (Stochastic Weight Averaging):** An earlier technique where model weights are averaged over the last portion of training. Being replaced by EMA in newer submissions.

**Muon Optimizer with Weight Decay:** Used instead of or alongside Adam. WD=0.04 appears to be a sweet spot.

**OrthoInit (Orthogonal Initialization):** Better weight initialization that helps training converge faster in the limited 10-minute window.

**Warmdown Schedule:** A learning rate schedule where the final portion of training uses a cooldown — warmdown3500 (3500 steps of warmdown) is used by the current #1.

### 4.4 Evaluation Tricks

**Sliding Window Evaluation:** Instead of evaluating on fixed-length chunks, use a sliding window with stride < sequence length, so each token benefits from maximum context. This alone dropped BPB significantly.

**Longer Evaluation Sequence Length:** Since the 10-minute eval budget is separate from training, you can evaluate with much longer sequences than you trained with.

**Test-Time Training (TTT):** The most aggressive evaluation technique — you fine-tune the model on validation tokens you've already scored, then use the adapted model on subsequent tokens. LoRA-based TTT has already been demonstrated. Note: the GEPA + AdamW TTT approach has reached 1.0672 BPB but is under extra scrutiny.

---

## 5. Prizes & Incentives

**$1,000,000 in RunPod Compute Credits:** OpenAI is distributing credits to help participants train. Apply via the form at https://openai.com/index/parameter-golf/#credit-form. You need an OpenAI/ChatGPT-linked email to apply.

**Job Opportunities:** In June 2026, OpenAI plans to hire a small cohort of early-career researchers (undergrads, recent graduates, Olympiad medalists). Standout participants may be invited to interview.

**Public Recognition:** Winning approaches will be featured publicly.

---

## 6. Step-by-Step Battle Plan to Win

### Phase 1: Setup & Baseline (Days 1–2)

1. **Fork the repo** (already done — in your `parameter_golf` folder)
2. **Set up local dev environment** (Mac with Apple Silicon):
   ```bash
   cd parameter_golf
   python3 -m venv .venv
   source .venv/bin/activate
   pip install mlx numpy sentencepiece huggingface-hub datasets tqdm
   python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1
   ```
3. **Run the baseline** locally to understand the code:
   ```bash
   RUN_ID=mlx_smoke ITERATIONS=200 TRAIN_BATCH_TOKENS=8192 \
   VAL_LOSS_EVERY=0 VAL_BATCH_SIZE=8192 python3 train_gpt_mlx.py
   ```
4. **Apply for RunPod credits** at https://openai.com/index/parameter-golf/#credit-form with your OpenAI-linked email
5. **Join the OpenAI Discord** → #parameter-golf-discussions
6. **Read every top submission's README** in the `records/` folder — this is the most valuable learning material

### Phase 2: Reproduce the SOTA (Days 3–7)

1. **Set up a 1×H100 pod on RunPod** using the official template: https://console.runpod.io/deploy?template=y5cejece4j&ref=nl2r56th
2. **Reproduce the current top entries** — don't innovate yet, just prove you can match existing scores
3. **Build ablation infrastructure** — create scripts that let you toggle techniques on/off and measure deltas
4. **Understand the budget:** Map out exactly how many bytes each component uses (embedding, layers, MLP, compressed overhead)

### Phase 3: Low-Hanging Fruit (Days 7–14)

Implement the "core five" techniques that every competitive submission uses:

1. **fp16 Tied Embeddings** (keep embedding matrix in full precision)
2. **Int6 Quantization + Zstd-22** compression
3. **Increase to 10–11 layers** (made possible by quantization savings)
4. **3× MLP width** (wider MLPs quantize better and learn more)
5. **Sliding Window Evaluation** (free BPB improvement at eval time)

Expected result: **~1.15–1.16 BPB** (top 10 territory)

### Phase 4: Competitive Edge (Days 14–25)

Now layer in the advanced techniques, **one at a time with ablations**:

1. **EMA** (replace SWA — current top submissions use EMA)
2. **QAT** (quantization-aware training — experiment with when to enable it)
3. **BigramHash** (cheap sub-word features)
4. **XSA on last 3–4 layers** (extended attention for deeper layers)
5. **Partial RoPE** (16/64 dimensions)
6. **GPTQ-lite post-training quantization** (calibration-based quantization)
7. **Warmdown learning rate schedule** (experiment with duration)
8. **Muon optimizer** with weight decay tuning

**Critical methodology:** Run ablations by removing one technique at a time. Interaction effects are real — a technique that helps in isolation can hurt when combined with others.

Expected result: **~1.12–1.13 BPB** (top 3 territory)

### Phase 5: Frontier Exploration (Days 25–35)

This is where you try to break through the frontier:

1. **Test-Time Training (TTT):** Implement LoRA-based TTT within the eval budget. This is the biggest remaining gap — TTT submissions have hit 1.07 BPB
2. **Novel architectures:** Depth recurrence, parameter tying, mixture of experts at this scale
3. **Custom CUDA kernels / megakernels:** Fuse operations to squeeze more training into 10 minutes
4. **Novel tokenizers:** Risky (extra scrutiny) but a different tokenizer could unlock better BPB
5. **Precision budgeting:** Spend fp16 only where quantization error hurts most (tied embeddings, late-layer keys)

### Phase 6: Submission Polish (Days 35–38)

1. **Run 3+ seeds** to demonstrate statistical significance (p < 0.01, ≥ 0.005 nat improvement)
2. **Verify on 8×H100 SXM** — this is the exact hardware used for judging
3. **Write a clear README** explaining every technique and its contribution
4. **Create submission.json** with your name, GitHub ID, and val_bpb
5. **Test that your code runs from a clean state** in the records folder
6. **Submit PR** adding your folder to `/records/track_10min_16mb/`

---

## 7. Where to Train

| Platform | SKU | Cost | Use Case |
|---|---|---|---|
| **Local Mac (Apple Silicon)** | M1/M2/M3 | Free | Code iteration, smoke tests, architecture experiments |
| **RunPod 1×H100** | H100 SXM | ~$2.50/hr | Development and single-GPU experiments |
| **RunPod 8×H100** | H100 SXM | ~$20/hr | Final submission runs and verification |
| **Free Credits** | Via OpenAI | $0 | Apply at openai.com/index/parameter-golf/#credit-form |

**Strategy:** Do 80% of your work on 1×H100 or locally. Only scale to 8×H100 for final verification runs and submission attempts.

---

## 8. Submission Format Checklist

Your PR should add a single folder to `records/track_10min_16mb/` containing:

- [ ] **README.md** — Detailed explanation of your approach, every technique used, and ablation results
- [ ] **submission.json** — `{ "name": "Your Name", "github_id": "your_github", "val_bpb": 1.XXXX, ... }`
- [ ] **Training logs** — At least 3 runs showing statistical significance
- [ ] **train_gpt.py** — Your complete training script (must compile and run from the records folder)
- [ ] **requirements.txt** — Any additional dependencies beyond the base environment

---

## 9. Key Resources

- **GitHub Repo:** https://github.com/openai/parameter-golf
- **RunPod Template:** https://console.runpod.io/deploy?template=y5cejece4j&ref=nl2r56th
- **Compute Credits Application:** https://openai.com/index/parameter-golf/#credit-form
- **RunPod × OpenAI Portal:** https://modelcraft.runpod.io/
- **Challenge Terms & Conditions:** https://cdn.openai.com/pdf/d5caec5a-ee81-419d-b0d7-39f1424d819c/OpenAI%20Model%20Craft_%20Parameter%20Golf%20Challenge%20Terms%20and%20Conditions.pdf
- **Participant Form (optional, for recruiting):** https://jobs.ashbyhq.com/openai/form/open-ai-challenge-parameter-golf
- **Discord:** OpenAI Discord → #parameter-golf-discussions
- **NanoGPT Speedrun (inspiration):** https://github.com/KellerJordan/modded-nanogpt
- **DeepWiki Analysis:** https://deepwiki.com/openai/parameter-golf

---

## 10. Verification Notes

All facts in this report were cross-referenced against:

1. The official GitHub README at https://github.com/openai/parameter-golf (fetched March 23, 2026)
2. OpenAI's official announcement at https://openai.com/index/parameter-golf/
3. The official Terms & Conditions PDF
4. Community analysis from GitHub Issues (#140, #158) and DeepWiki
5. Multiple independent news sources (The Decoder, Inc., AiToolsClub, i10x.ai)

Leaderboard data is sourced directly from the repo README as of March 23, 2026, and may change as new submissions are accepted.
