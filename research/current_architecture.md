# Current Model Architecture — Parameter Golf SOTA

**Reference:** `2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT`  
**Code:** `train_gpt_sota.py` on branch `research` (commit `01e6fcf`)  
**Score:** 1.0810 bpb (official leaderboard SOTA as of 2026-04-19)

---

## Overview

A decoder-only transformer with several non-standard features layered on top of a standard causal attention backbone: depth recurrence (re-running a block stack multiple times), parallel residuals (a second MLP path added at later layers), Test-Time Training (TTT) applied at eval, and INT6 quantization for the final submission.

---

## Tokenizer

- **SentencePiece BPE**, vocab size **8192** (`SP8192`)
- Trained on FineWeb; tokenizer at `/workspace/data/tokenizers/fineweb_8192_bpe.model`
- Data: `fineweb10B_sp8192/` (~10B tokens)

---

## Model Dimensions

| Param | Value |
|-------|-------|
| Layers (`n_layer`) | 11 |
| Model dim (`n_embd`) | 512 |
| Heads (`n_head`) | 8 (head dim = 64) |
| MLP ratio | 4× (hidden = 2048) |
| Context length | 1024 tokens |
| Depth recurrence loops | 3 (layers run 3× total) |
| Parallel residual start | Layer 7 |

---

## Architecture Components

### Attention: CausalSelfAttention
- Standard multi-head causal self-attention with RoPE positional encoding
- **QK gain**: initialized at `qk_gain_init=5.25` (learnable scalar on Q and K projections, not a fixed temperature). This was the key difference from the default 5.0 — worth ~0.002 bpb.
- Sliding window attention at eval time (reduces memory, minimal quality impact)

### MLP: LeakyReLU²
- Two-layer MLP with activation `LeakyReLU(x)²` (squared leaky relu)
- Applied to both primary and parallel residual paths

### Depth Recurrence (3-Layer Recur)
- The 11-layer block stack is run **3 times** sequentially with shared weights
- Effective depth = 33 layers at the cost of 11 layers of parameters
- Recurrence starts at `frac ≥ 0.35` of training (introduced gradually to stabilize)

### Parallel Residuals (ParResid)
- From layer 7 onward: each block outputs `attn(x) + mlp(x) + mlp2(x)` instead of sequential attn→mlp
- `mlp2` is an additional MLP with its own weights (doubles MLP capacity at later layers)
- Skip gates (`skip_attn`, `skip_mlp`) learn to interpolate between residual and transformed path

### No BigramHash
- Our code has a `BigramHashEmbedding` (L440-474) that is **disabled** via `BIGRAM_VOCAB_SIZE=0`
- The SOTA submission never included it; it's an experimental addition in our codebase only

---

## Optimizer Stack

### Muon (MuonEq-R variant)
- Applied to all weight matrices (Q, K, V, O projections; MLP weights)
- Newton-Schulz orthogonalization of the gradient: `g ← NS(g)` at each step
- Momentum warmup over first ~455 steps
- No weight decay on Muon params

### AdamW
- Applied to embedding, output projection, LayerNorm scales/biases, scalar params
- Standard hyperparams (β1=0.9, β2=0.95, ε=1e-8)
- Weight decay on matrix params, none on vectors/scalars

### LR Schedule
- Warmup → constant → warmdown (cosine to 0)
- Warmdown starts at `frac ≈ 0.75` of training
- `warmdown_frac` controls the length of the tail

### SDClip (Gradient/Weight Clipping)
- Clips per-matrix update norms with learnable sigma parameters (`matrix_sigma`, `embed_sigma`)
- Stabilizes training without a fixed global clip value

---

## Test-Time Training (TTT) — `TTT_ENABLED=1`

Applied at **evaluation only** (not during training). This is what makes it "Legal TTT" in the submission name.

**How it works:**
1. At eval time, for each chunk of context, run a few gradient steps of SGD on the model using the chunk itself as training data (self-supervised next-token prediction)
2. The model adapts its weights temporarily to the specific eval document
3. After the chunk, weights are reset to the trained checkpoint

**Details:**
- SGD with momentum, cosine LR schedule across chunks
- Score-before-update: compute loss on chunk before updating, then update for next chunk
- Gradient clip applied within TTT loop
- `ttt_hash_buckets=16384` and `ttt_hash_embed=True` appear in the SOTA config log but are **dead config** — no code references them in either SOTA or our implementation. Ignore when reading logs.

**Cost:** TTT eval adds ~6 min to wall time vs. a standard forward pass eval. Training is unaffected.

---

## Quantization (GPTQ INT6)

Applied after training to produce the final submission artifact:

1. **Hessian collection**: run one forward pass with hooks to collect per-layer activation statistics (Hessians)
2. **INT6 quantization**: quantize all weight matrices to 6-bit with `clip_sigmas` controlling per-channel clipping thresholds
3. **INT8 for embeddings/output**: embedding and lm_head stay at INT8 (too sensitive for INT6)
4. Result compressed with brotli-11 for submission

GPTQ is a post-hoc step — does not affect training dynamics.

---

## EMA

- Exponential moving average of weights maintained throughout training
- Final submission uses the EMA weights (not the last-step weights)
- EMA applied before GPTQ quantization

---

## Training Run (8×H100, ~10 min)

| Phase | Steps | Notes |
|-------|-------|-------|
| Momentum warmup | 0–455 | Muon momentum ramped up |
| Main training | 455–3412 | Constant LR |
| Warmdown | 3412–4550 | Cosine decay to 0 |

- Total steps: ~4550
- Batch size: large (fills 8×H100 memory via gradient accumulation)
- Wall time breakdown: 10 min train + 1 min EMA/quant + 25s quant eval + 2 min sliding-window eval + 6 min TTT eval ≈ **~20 min total**

---

## Checkpoints (our extension, not in SOTA submission)

Phase-boundary saves emitted when `CKPT_DIR` is set. Not in the original SOTA code — added by us for hotstart experiments:

| Checkpoint | When |
|-----------|------|
| `ckpt_event_step455.pt` | Momentum warmup end |
| `ckpt_pre_recurrence_stepN.pt` | Recurrence phase starts (frac ≥ 0.35) |
| `ckpt_warmdown_start_stepN.pt` | Warmdown begins |
| `ckpt_event_step{10/25/50/75%}.pt` | Milestone steps (CKPT_STEPS) |
| `ckpt_final_pre_ema_stepN.pt` | End of training loop |
| `ckpt_final_post_ema_stepN.pt` | After EMA applied |

~1 GB each, ~9 GB total. These only save — they do NOT change training dynamics.
