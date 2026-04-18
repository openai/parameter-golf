# SP8192 · 3-Layer Recurrence · Parallel Residuals · QK-Gain 5.5 · SGD-TTT

> **Track:** 10 min / 16 MB &nbsp;|&nbsp; **Hardware:** 8×H100 SXM &nbsp;|&nbsp; **Date:** 2026-04-18

---

## Results

| Eval | BPB |
|------|-----|
| Pre-quantization (post-EMA) | 1.0914 |
| Quantized (standard) | 1.1023 |
| Quantized + Sliding Window | 1.0857 |
| Quantized + Sliding Window + SGD-TTT | *pending official 3-seed eval* |

Expected with SGD-TTT: **~1.082–1.084**
Artifact size: **15.97 MB** &nbsp;·&nbsp; Training time: **~588 s**

---

## What This Submission Does

### Architecture

A compact GPT trained on FineWeb with SP8192 tokenization. The core design stacks several
complementary ideas to squeeze more representational power within the 16 MB artifact budget:

**Depth Recurrence (layers 3–5, ×2 loops)**
Layers 3, 4, and 5 are visited twice per forward pass, giving 17 effective layers from an
11-layer model. Looping activates at 35% of training, allowing the model to first learn
stable representations before iterating them.

**Parallel Residuals (layers 7+)**
From layer 7 onward, each block receives both its standard input and the original embedding
stream as an additive residual. This gives deeper layers a direct path to low-level features
without gradient degradation.

**QK-Gain 5.5**
A learned per-head scalar multiplied onto queries before attention. Initialized at 5.5
(up from 5.25 in prior work), continuing the monotonically improving trend observed across
submissions using this technique.

**Recur-Alpha**
A learned scalar per recurrent block, initialized to zero, that adds the first-visit
activation to subsequent recurrence passes. At init it has no effect; during training it
learns how much "carry" helps each layer. This costs 6 parameters total.

**Quantization**
Full-Hessian GPTQ int6 on weight matrices, int8 on embeddings, with sigma-clipping at
12.85σ (SDClip). Final artifact: 15.97 MB.

---

### Training Setup

| Setting | Value |
|---------|-------|
| Tokenizer | SP8192 SentencePiece BPE |
| Layers / Dim / Heads | 11 / 512 / 8 (GQA, 4 KV heads) |
| RoPE | 16 partial dims, base 10000 |
| Optimizer | MuonEq-R (matrix params) + Adam (scalars/embeddings) |
| Weight decay | MuonEq 0.095, Adam 0.02 |
| LR schedule | Linear warmup (20 steps) → cosine warmdown (72%) |
| EMA decay | 0.9965 |
| Training tokens | ~3.2B (wall-clock capped at 600 s) |
| Batch tokens | 786,432 / step |

---

### Test-Time Training (SGD-TTT)

At inference, the model is adapted per validation chunk using the same data it is about
to be scored on — all within the rules of Issue #1017:

1. **Causal** — sliding window (stride 64, seq 2048) is unchanged; no future tokens seen
2. **Normalized distribution** — standard softmax throughout
3. **Score before update** — the chunk is fully scored under `torch.no_grad()` before
   any `.backward()` is called
4. **Single pass** — each token is scored exactly once

SGD optimizer with momentum 0.9, cosine-annealed learning rate, 3 epochs per chunk.

---

## Compute

Developed entirely under the **RunPod Quick Start Grant ($25 GPU credit)**.

| Run | Cost |
|-----|------|
| Training (8×H100, ~10 min) | ~$3.50 |
| Validation + debug iterations | ~$18.00 |
| **Total** | **~$21.50 / $25** |

---

## Future Work: LoRA-TTT

### The Idea

Instead of updating all ~30M base model parameters per TTT chunk (as in standard SGD-TTT),
attach small LoRA adapters to attention Q/V projections (rank 4) and the MLP gate (rank 2).
Only ~65K parameters are updated per chunk, allowing 12 TTT epochs instead of 3 at lower
per-step cost. Adapters are reset to zero at each chunk boundary, so no state persists
across chunks (fully compliant with TTT rules).

**Size overhead:** ~128 KB at fp16 — adapters are runtime-only and not serialized to the
artifact.

### Implementation Status

The implementation is complete and present in the submitted `train_gpt.py`. A `try/except`
block runs LoRA-TTT first; if it fails, base SGD-TTT runs automatically as fallback.

### Known Blocker

`RuntimeError: Inference tensors cannot be saved for backward` during the LoRA backward
pass. The error traces to `flash_attn_interface.flash_attn_func` (Flash Attention 3), which
appears to mark its output tensor with the inference-mode bit internally, even when called
outside of `torch.inference_mode()`. This prevents autograd from saving the tensor for
the LoRA gradient computation.

**Fix applied (unverified due to exhausted compute budget):**
```python
y = flash_attn_3_func(q, k, v, causal=True).clone()
```
The `.clone()` forces a new allocation outside any inference context. This is in the
submitted code. A try/except ensures the submission never fails due to this bug.

**Alternative fix path:**
```python
# Replace flash_attn_3 with SDPA during TTT training phase only
if torch.is_grad_enabled():
    y = F.scaled_dot_product_attention(
        q.transpose(1,2), k.transpose(1,2), v.transpose(1,2), is_causal=True
    ).transpose(1,2).contiguous()
else:
    y = flash_attn_3_func(q, k, v, causal=True)
```

**Expected gain if resolved:** +0.004–0.006 BPB over base TTT → projected ~1.076–1.079,
which would challenge the current #1 position.

---

## Attribution

Builds on the SP8192 tokenizer and base architecture established across prior submissions
in this track. QK-Gain, depth recurrence, and parallel residual ideas were developed
iteratively through the public submission history.
