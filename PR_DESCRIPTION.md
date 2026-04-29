## S0/PR1851 + Cap Tokenizer + LQER + Global TTT (val_bpb = 1.0713)

A joint effort by **Billy Li** and **Tim Shen**, with thanks to **Xingyuan Ding** for additional experiments and **Bill (Yiyuan) Li** for meaningful discussions on tokenizers.

I started looking at this challenge around 4/20. The merged leaderboard hadn't changed much by then, but the volume of PRs and improvements was absolutely overwhelming. I cleaned up my thoughts and followed a systematic procedure — tackling the problem piece by piece: **data → tokenization → model architecture → optimizer → quantization → test-time compute**.

A more detailed write-up is at: https://www.junchengbillyli.com/llm-notes.html

---

### Results

**Best result: quant+TTT val_bpb = 1.0713, artifact ≈ 16.09 MB** (seed 1337, 8xH100 SXM, 10min wallclock).

| Seed | Steps | EMA BPB | Quant BPB | Quant+TTT BPB | Artifact |
|------|-------|---------|-----------|---------------|----------|
| 1337 | 4733 | 1.0746 | 1.0832 | **1.0713** | 16.09 MB |
| 42 | 4741 | 1.0752 | 1.0834 | **1.0718** | 16.09 MB |
| 999 | 4775 | 1.0740 | 1.0845 | *(running)* | 16.09 MB |

Script: `final_s0_pr1851_mod_gptq_v2.py` (3143 lines, 31 KB compressed).

---

### Data & Tokenization

There was a clear trend across PRs that smaller vocabs have a low ceiling — 8192 seems to be the sweet spot for all the later successful submissions. But relying on the default SentencePiece tokenizer is not the best idea.

**What we tried:**
- **Vocabulary pruning:** Thought tokenizing full words could be wasteful given the time/compute limits. Tried pruning long words that could be covered by combinations of shorter subword tokens. This did not help (+0.001 BPB).
- **Case folding (lowercasing + capital token):** Lowercasing everything and treating the leading capital letter as a special token — this helped. This is the "cap tokenizer" (SP8192, effective vocab 7972 after folding).
- **Data normalization:** Getting rid of long URLs and anything rare/difficult in the FineWeb dataset.

**Key insight:** The fact that 1024-token vocabs plateau quickly tells us the network tends to stall if tokenization is too easy. The tokenization needs to make the task hard enough for the model to keep learning.

---

### Model Architecture

When I first saw the 9-layer implementation, I thought it was pretty standard. Depth recurrence was clearly proven effective within the community. From there:

- **GQA → MHA:** Considered replacing GQA (group size 2) back to MHA to trade a bit more parameters for better performance.
- **Local attention heads:** Implemented fancy local attention — failed horrendously, since the implementation is inherently inefficient and could never utilize the Flash Attention 3 ecosystem.
- **DeepSeek Engrams, value embeddings, embedding factorizations:** None worked within the 10-minute wall clock. None of these are as fast as a vanilla attention + MLP combo.
- **The only thing that helps is making the MLPs wider.** All other architectural tweaks don't see ROI.

**Final architecture:** 11L x 512d, 8 heads / 4 KV heads, MLP 4x, tied embeddings (vocab 7972), logit softcap 30.0, partial RoPE (16/64 dims), layer looping (layers 3–5, 2 loops enabled at 35% of training), parallel residuals from layer 8+, skip gates (U-Net connections).

---

### Optimizer

We first ablated Muon vs. AdamW. I thought AdamW wouldn't lag Muon too much on a relatively small dataset — this is not true. **AdamW consistently lags Muon in our experiments.**

We then looked into Muon to see what could be improved. The all_reduce communication overhead was something I aimed to reduce, but eventually by the 0427 trick, I was only able to squeeze out ~0.0005 BPB gain.

**Final config:** Muon (Polar-Express Newton-Schulz, 5 backend steps) for matrix params (lr=0.026, momentum=0.97, wd=0.095), AdamW for embeddings (lr=0.6, wd=0.085) and scalars (lr=0.02). Gradient clipping 0.3, warmdown 75%.

---

### Quantization

Quantization was a bit of a black box, though we've done it before. My intuition was that group quantization should produce a more stable estimate of all parameters and be better suited for GPTQ. However, GPTQ's group statistics also take additional space, which pushes the submission file to go oversize — **the gain does not justify its cost.**

From intuition QAT should work better, but I never got a successful QAT run.

**Final config:**
- GPTQ int6 for all attention + MLP weight matrices (16 calibration batches)
- GPTQ int8 for tied embeddings
- LQER error correction: rank 4, int4 factors, asymmetric (group 64), applied to top-3 highest-error layers
- Brotli compression

---

### Test-Time Compute

This is absolutely the backdoor lottery ticket. The main theme is to **align the trained distribution with the test-time distribution.**

**Final TTT config:**
- Phased global TTT: 1 phase, 2000 prefix docs, cosine LR (peak 0.001)
- 215 gradient chunks over 48K suffix docs (32K tokens/chunk)
- LoRA rank 96 on K, O, and MLP projections (Adam, beta1=0, beta2=0.999, wd=1.0)
- TTT consistently drops BPB by ~0.01 from the quantized baseline

---

### Reproduction

```bash
pip install brotli sentencepiece
pip install flash_attn_3 --no-deps --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/

SEED=1337 torchrun --standalone --nproc_per_node=8 final_s0_pr1851_mod_gptq_v2.py
```

### Files

| File | Description |
|------|-------------|
| `final_s0_pr1851_mod_gptq_v2.py` | Final training script |
| `logs/*20260429*.log` | Training logs for all 04/29 runs |
| `train_gpt_s0_pr1851_mod.py` | Earlier annotated PR #1851 exploration |
| `train_gpt_s9.py` | Prior S9 stack (bank-mode + Polar-Express Muon) |
| `train_gpt_s9_caseops_lqer.py` | Prior cap tokenizer variant |
