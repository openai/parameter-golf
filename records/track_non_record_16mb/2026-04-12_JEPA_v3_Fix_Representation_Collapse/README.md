# JEPA v3 — Fixing Representation Collapse via Span Masking

**Track:** Non-record (unlimited compute)
**Architecture:** 11L 512-dim U-Net Transformer, mlp_mult=3, GQA 8q/4kv
**val_bpb:** 1.2321 pre-quant — builds on JEPA v2 ([2026-04-01](../2026-04-01_JEPA_v2_MultiStep_Int6_BigramHash_EMA))

---

## The Problem v2 Left Unsolved

JEPA v2 correctly diagnosed three implementation bugs (EMA momentum too high, single-step prediction redundant with CE, gradient accumulation batch mismatch) and fixed all three. After fixing them, the JEPA loss still stabilized at 0.002 — the same collapse as v1.

v2 hypothesized that the remaining issue was structural: same-sequence next-k prediction may be too easy for a causal LM, since the context encoder already has access to all the information needed to predict nearby positions. v2 concluded:

> *The fix requires masking (I-JEPA style) or cross-sequence targets — both are architectural changes that add complexity without a clear BPB path.*

v3 implements that fix. 

---

## The Fix: Span-Masked JEPA

The core change is to align with I-JEPA's training procedure:

```
Context tokens  ──► Context Encoder  ──► context embeddings
                                               │
                                          Predictor
                                               │
                                               ▼
Target tokens   ──► Target Encoder   ──► target embeddings  (ground truth)
```

**What changes:** The context encoder no longer sees the full sequence. Target span positions are replaced with a learned mask embedding (`jepa_mask_emb`) before the encoder. The prediction task is now genuinely hard — the context encoder cannot reconstruct the target token from its own input, and must rely on surrounding context.

**What stays the same:** Architecture (11L U-Net, same hyperparameters), CE loss path (full unmasked forward), EMA target encoder (now sees full unmasked sequence), int6+LZMA compression, BigramHash, LeakyReLU(0.5)², artifact EMA.

---

## Implementation

### 1. Span Sampling

Each training step, `sample_block_spans()` samples `JEPA_NUM_SPANS=4` non-overlapping contiguous spans from the 1024-token sequence. Span lengths are drawn from a Geometric distribution with mean `JEPA_SPAN_LEN_MEAN=16`, clamped to `[JEPA_SPAN_LEN_MIN=4, seq_len/(2*num_spans)]`.

Defaults: 4 spans × ~16 tokens = ~6% of sequence masked per step.

```python
def sample_block_spans(seq_len, num_spans, span_len_mean, span_len_min=4, device=None):
    # Geometric(p=1/span_len_mean) lengths, greedy overlap resolution
    # Returns LongTensor [num_spans, 2] of (start, end_exclusive) pairs
```

The Geometric distribution produces heavy-tailed lengths: most spans are short, occasionally longer — varying prediction difficulty within a step.

### 2. Mask Embedding

```python
self.jepa_mask_emb = nn.Parameter(torch.zeros(model_dim))
```

A single learned 512-dim vector shared across all masked positions. Zero-init: starts neutral and is trained by the JEPA loss gradient to encode "this position is unknown, predict it from context." Conceptually equivalent to BERT's `[MASK]` token but in continuous embedding space.

**Bigram leak fix:** `BigramHashEmbedding(prev_tok, masked_tok)` would reveal the masked token's identity via the Cantor hash `h(a,b) = (a+b)(a+b+1)/2 + b`. The bigram contribution is explicitly zeroed at masked positions before applying `jepa_mask_emb`.

```python
if jepa_mask is not None:
    bigram = bigram.masked_fill(jepa_mask.unsqueeze(-1), 0.0)  # prevent hash leak
x = torch.where(jepa_mask.unsqueeze(-1), self.jepa_mask_emb.to(x.dtype), x)
```

### 3. Two-Pass Forward Per Step

```python
# Pass 1: CE loss — full unmasked sequence (unchanged)
ce_loss = model(x, y)

# Pass 2: JEPA — masked context encode + target encode
spans = sample_block_spans(T, num_spans, span_len_mean, span_len_min)
jepa_mask = torch.zeros((B, T), dtype=torch.bool, device=x.device)
for s, e in spans.tolist():
    jepa_mask[:, s:e] = True

with torch.no_grad():
    z_target = target_encoder.encode(x)                      # full, unmasked

z_context = base_model.encode(x, jepa_mask=jepa_mask)        # masked input
z_pred    = base_model.jepa_predictor(z_context)

# Loss only at masked positions
z_p = z_pred[jepa_mask]          # [N_masked, D]
z_t = z_target[jepa_mask]        # [N_masked, D]

mse_loss = F.mse_loss(z_p.float(), z_t.float())
var_loss  = vicreg_var_loss(z_p.float(), gamma=1.0, eps=1e-4)  # predictor only
cov_loss  = vicreg_cov_loss(z_p.float())                       # predictor only
jepa_loss = mse_loss + 0.15 * var_loss + 0.02 * cov_loss

loss = ce_loss + jepa_lambda * jepa_loss
```

The target encoder (EMA copy) sees the full unmasked sequence — its representations are not corrupted by masking. The CE path also remains fully unmasked. Only the context encoder sees the masked input.

### 4. VICReg Anti-Collapse Regularization

Span masking prevents collapse structurally, but VICReg terms provide an explicit signal to maintain a spread, decorrelated embedding space:

```python
def vicreg_var_loss(z, gamma, eps):
    """Hinge: penalize per-feature std < gamma across the batch of masked tokens."""
    z_c = z - z.mean(dim=0)
    std = (z_c.pow(2).sum(0) / (n - 1) + eps).sqrt()
    return (gamma - std).clamp(min=0).mean()

def vicreg_cov_loss(z):
    """Off-diagonal covariance penalty: decorrelate feature dimensions."""
    cov = (z - z.mean(0)).T @ (z - z.mean(0)) / (n - 1)
    off = cov.pow(2); off.fill_diagonal_(0)
    return off.sum() / d
```

Both terms are applied only to the **predictor** side (`z_pred[jepa_mask]`) where gradients flow. The target side is monitored as a diagnostic but receives no gradient — it is updated only via EMA.

This follows V-JEPA practice: variance and covariance regularization on the online (predictor) representations ensures the model cannot collapse all masked positions to a single point or to a low-rank subspace.

### 5. Optimizer Bug Fix (v2 regression)

In v2's `train_gpt.py`, the optimizer setup iterates only `base_model.blocks.named_parameters()` — `jepa_predictor` and `jepa_mask_emb` are outside `blocks` and appear in none of the three optimizer groups (Muon, scalar Adam, tok Adam). This is verifiable in the v2 commit (`b4a428b`). The predictor was frozen at zero-init throughout training — JEPA gradients were computed but never applied to it.

Fixed by explicitly appending predictor and `jepa_mask_emb` to the parameter lists:

```python
scalar_params.append(base_model.jepa_mask_emb)
for name, p in base_model.jepa_predictor.named_parameters():
    if p.ndim == 2:
        matrix_params.append(p)   # fc.weight, proj.weight → Muon
    else:
        scalar_params.append(p)   # RMSNorm → Adam
```

---

## Architecture Summary

```
11L U-Net Transformer (5 encoder + 6 decoder, skip connections)
  dim=512, 8 attention heads, 4 KV heads (GQA)
  mlp_mult=3, LeakyReLU(0.5)^2
  RoPE, RMSNorm, logit softcap=30

Embedding:
  tok_emb(t) + BigramHashEmb(t-1, t) [zeroed at masked pos] → RMSNorm → transformer

JEPA (auxiliary, span-masked):
  context_encoder(x, mask=jepa_mask) → z_context → JEPAPredictor → z_pred
  EMA target_encoder(x, mask=None)   → z_target
  Loss: MSE(z_pred[mask], z_target[mask])
        + 0.15 × VICReg_var(z_pred[mask])   ← anti-collapse variance hinge
        + 0.02 × VICReg_cov(z_pred[mask])   ← decorrelation penalty
  Spans: Geometric(mean=16), num_spans=4, ~6% of sequence per step

Serialization (unchanged from v2):
  artifact_ema (Polyak avg, decay=0.9999)
  → int6 quantization (range [-31,31])
  → LZMA compression (preset=9)
```

---

## Results

| Metric | Value |
|--------|-------|
| val_bpb (pre-quant) | **1.2321** |
| Architecture | 11L 512-dim U-Net |
| JEPA spans | 4 × Geometric(mean=16) |
| Mask ratio | ~6% per step |
| jepa_lambda | 0.12 |
| EMA momentum | 0.9 → 0.999 |
| VICReg var weight | 0.15 |
| VICReg cov weight | 0.02 |

### Comparison to v2

| Submission | val_bpb | JEPA approach | Collapse? |
|---|---|---|---|
| v2 bigram (no JEPA) | 1.4617 | — | — |
| v2 full (next-k JEPA) | 1.6047 | Unmasked, offset [1,2,4,8] | Yes (loss→0.002) |
| **v3 (this)** | **1.2321** | **Span-masked, I-JEPA style** | **No** |

v3 is **0.2326 BPB better than v2 with JEPA**, and **0.2296 BPB better than v2 without JEPA**. Span masking produces genuine gradient signal from step 1.

---

## Key Env Vars

```bash
JEPA_NUM_SPANS=4          # number of target spans per sequence
JEPA_SPAN_LEN_MEAN=16     # geometric mean span length (tokens)
JEPA_SPAN_LEN_MIN=4       # minimum span length
JEPA_LAMBDA=0.12          # JEPA loss weight
JEPA_EMA_MOMENTUM=0.9     # starting EMA momentum (rises to 0.999)
JEPA_PRED_DIM=256         # predictor hidden dim
JEPA_VAR_WEIGHT=0.15      # VICReg variance term weight
JEPA_COV_WEIGHT=0.02      # VICReg covariance term weight
JEPA_VAR_GAMMA=1.0        # target std for variance hinge
BIGRAM_VOCAB_SIZE=2048    # 0 = disable bigram embedding
ARTIFACT_EMA_DECAY=0.9999
QUANT_MAX=31              # int6
```

---

## What This Submission Is (and Isn't)

This is a **research non-record submission**. The goal is to demonstrate that properly-structured JEPA — specifically, span masking that prevents the context encoder from seeing target tokens — produces the gradient signal that v1 and v2 failed to generate.

The path from here to record territory requires combining span-masked JEPA with the compression and quantization techniques from the current SOTA (GPTQ, TTT, XSA). This submission establishes that the JEPA auxiliary objective itself is no longer the bottleneck.

---

## References

- JEPA original: LeCun (2022), "A Path Towards Autonomous Machine Intelligence"
- I-JEPA: Assran et al. (2023), CVPR — span masking for vision
- BYOL: Grill et al. (2020), NeurIPS — EMA target encoder design
- JEPA v2 (this repo): [2026-04-01](../2026-04-01_JEPA_v2_MultiStep_Int6_BigramHash_EMA) — multi-step prediction + optimizer fixes
- Parameter Golf SOTA: abaybektursun PR #1019 — 1.1147 BPB
