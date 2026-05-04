# Winning Techniques in Parameter Golf — Deep Research
# Compiled: 2026-05-04 | From competition PRs + papers

## 1. LQER — Low-Rank Quantization Error Recovery

**Source:** ICML 2024, Zhang et al. | https://arxiv.org/abs/2402.02446
**GitHub:** https://github.com/ChengZhang-98/lqer

### What It Does
After GPTQ quantization, there's a residual error: `W_quant = W + E` where E is the quantization error matrix. LQER decomposes E into a low-rank approximation: `E ≈ A × B` where A: [d_in × rank], B: [rank × d_out].

These low-rank matrices A, B are stored alongside the quantized weights. At inference:
```python
# Instead of: y = x @ W_quant
# LQER does:  y = x @ (W_quant + A @ B)
#             y = x @ W_quant + (x @ A) @ B  ← extra rank-r matmul
```

### Why It Works for Parameter Golf
- Rank-4 adapter: 4 × (512 + 1536) × 2 bytes = 16KB per layer per weight = ~10% extra params
- But fits within 16MB artifact budget because it replaces random error with structured correction
- Competition uses rank-4, asymmetric LQER (A and B have different ranks for in vs out)

### Key Tuning
```
LQER_RANK = 4           # rank of correction matrix
LQER_ASYMMETRIC = True  # different rank for rows vs cols (better for rectangular weights)
```

### Performance
- Near-lossless W4A8 quantization
- Paper shows 1.36× fewer hardware resources than prior SOTA
- In competition context: ~0.003-0.005 BPB improvement over vanilla GPTQ

---

## 2. AWQ-lite — Activation-Aware Weight Quantization

**Source:** MLSys 2024 Best Paper | https://arxiv.org/abs/2306.00978
**GitHub:** https://github.com/mit-han-lab/llm-awq

### What It Does
Not all weights are equally important. Key insight: **the 1% of weights aligned with large activation channels cause most quantization error.**

AWQ finds these "salient" channels by looking at activation magnitudes, then applies a per-channel scale factor before quantization:

```python
# Standard GPTQ: quantize W directly → uniform error across all channels
# AWQ: find salient channels, scale them UP before quant (then scale output down)
#      non-salient channels: quantized at low precision
#      salient channels: effectively quantized at higher effective precision

# Scale computation (simplified):
scales = activation_magnitudes.pow(0.5)  # larger activation → larger scale
W_scaled = W * scales[None, :]           # scale the salient input dimension
W_quant = gptq_quantize(W_scaled)        # quantize scaled weights
# At inference: output = (input / scales) @ W_quant  ← equivalent to unscaled
```

### AWQ-lite in Competition
The "lite" version doesn't do the full per-layer optimization. Instead:
- Collects activation statistics during a forward pass on calibration data
- Applies smooth per-channel scales to reduce quantization sensitivity
- Compatible with GPTQ (AWQ scales first, then GPTQ quantizes)

### Why It Works Together with GPTQ + LQER
```
Pipeline:
1. AWQ scaling    → reduces sensitivity of salient channels
2. GPTQ quant     → minimize column-wise reconstruction error (with Hessian)
3. LQER correction → add low-rank residual to recover remaining error
```
Each stage handles different aspects of the quantization error.

---

## 3. Asymmetric Logit Rescale

**Source:** PR #1923 in parameter-golf

### What It Does
Standard softmax has a symmetric cap: `logit_softcap = scalar`. 
AsymLogit uses different caps for positive and negative logits:
```python
# Standard:
logit_clipped = logit_softcap * torch.tanh(logits / logit_softcap)

# AsymLogit:
pos_cap = softcap_pos  # different scale for positive logits
neg_cap = softcap_neg  # different scale for negative logits
logit_clipped = torch.where(
    logits > 0,
    pos_cap * torch.tanh(logits / pos_cap),
    neg_cap * torch.tanh(logits / neg_cap)
)
```

### Why It Matters for TTT
During TTT, the LoRA adapters learn asymmetric logit distributions. When the underlying logit rescaling is also asymmetric, the TTT adaptation is more expressive.

From PR #1923: "3-phase per-doc LoRA learns asymmetric logit distributions during TTT eval that the symmetric `logit_softcap` scalar cannot capture, but `softcap_pos`/`softcap_neg` can."

**Effect: ~0.003 BPB improvement when combined with AWQ-lite quantization**

---

## 4. SmearGate

**Source:** Original in competition, PR #1787 lineage, BOS-fixed in PR #1855

### What It Does
A per-token "smearing" operation in the residual stream:
```python
gate_param = nn.Parameter(torch.zeros(1))
gate = torch.sigmoid(gate_param)

# Forward:
def smear_gate(x, input_ids, BOS_ID):
    not_bos = (input_ids[:, 1:] != BOS_ID).to(x.dtype).unsqueeze(-1)
    x = torch.cat([
        x[:, :1],
        x[:, 1:] + gate * x[:, :-1] * not_bos  # BOS fix: don't leak across docs
    ], dim=1)
    return x
```

### Why It Works
- Allows the model to "look back" by 1 token at essentially zero parameter cost
- gate=0 at init → identity, so it can't hurt early training
- not_bos mask prevents document leakage (important for compliance)
- Effectively extends receptive field by 1 token for free

**Effect: ~0.002 BPB improvement**

---

## 5. GPTQ Calibration Batches: The Hidden Lever

**Key finding from PR #2135:**

GPTQ calibration quality depends critically on calibration batch count:
| Batches | BPB (3-seed mean) | Notes |
|---------|-------------------|-------|
| 16 | 1.06110 | PR #2130 baseline |
| 32 | **1.05651** | PR #2135 final SOTA |
| ~0.004 BPB gap | | Just from calibration batches |

### Why More Batches Help
- GPTQ computes column-wise Hessians: H = X^T X
- More batches → better estimate of activation statistics
- Especially important for rare tokens in early vocab layers

### Practical Limits
- Our v9 used random calibration with 128 batches (different approach)
- PR #2135 uses 32 training-set batches (not random)
- Using training data for calibration = better activations = better H estimate
- Target: 32-64 batches from actual training data distribution

---

## 6. Phased TTT — Score-First Test-Time Training

**Status:** LEGAL under competition rules (confirmed in merged PRs)

### Protocol (3-phase implementation)
```
For each validation document d:
  Phase 1 (prefix, gradient=0):
    - Score tokens 1..N/3 under inference_mode  ← score first!
    - Accumulate gradients but don't apply
  Phase 2 (transition):
    - Score tokens N/3..2N/3 with gradient enabled
    - Apply LoRA gradient step
  Phase 3 (suffix, gradient=1):
    - Train on tokens 2N/3..N
    - Score tokens simultaneously (score-first guaranteed)
```

### LoRA Configuration
```
TTT_LORA_RANK = 1-4   # typical: 1 for K, 1 for V, 0 for Q (no_qv mask)
TTT_LOCAL_LR = base_lr * 0.75  # 0.75 multiplier beats 1.0
TTT_MASK = no_qv      # don't update Q/V LoRA, only K
EVAL_SEQ_LEN = 2560   # longer eval context = better TTT
```

### Impact: 0.012 BPB gain
From PR #2135: pre-quant 1.061 → post-quant 1.069 → post-TTT **1.057**
The quantization hurts 0.008 BPB, TTT recovers 0.012 BPB → net gain!

### Key Implementation Note
TTT is EVAL-ONLY. Training uses normal forward passes. TTT adapts the model during evaluation on each test document. The LoRA weights are reset between documents.

---

## 7. Per-Group lrzip + Brotli Compression

**Source:** PR #1855 - added per-group compression

### Why It Beats Plain Brotli
- Int6 weight groups have different entropy profiles:
  - QK weights: lower magnitude, higher entropy
  - V/MLP weights: higher magnitude, more clusterable
- **Per-group approach**: sorts rows by L1 similarity before compressing
  - Adjacent rows are numerically close → better delta compression
  - Permutation indices stored as uint16 + brotli

### Pipeline
```
1. Bucket tensors by role (qo_bank, kv_bank, mlp_up, mlp_down, etc.)
2. For "hot" 2D groups (attn.c_q, mlp.fc, tok_emb):
   a. Compute L1 pairwise similarity
   b. Sort rows to maximize adjacency
   c. Store sort permutation as uint16
3. Compress each group with lrzip -z -L9 (ZPAQ context-mixing)
4. Fall back to brotli for residuals, scales, LQER factors
```

### Result
~280KB smaller artifact than plain brotli-11

### Dependency
```bash
apt-get install lrzip   # must be installed before training script runs
```

---

## 8. CaseOps Tokenizer

**Source:** PR #1729 (romeerp)

### What It Does
Case-sensitive byte-level tokenization with "operations" that signal case patterns.

Standard byte-level: just raw bytes → case info baked into tokens
CaseOps: separates case pattern from content → more efficient for alphabetic text

This likely:
1. Reduces unique token count for alphabetic characters
2. Enables better n-gram modeling across case variants
3. Compresses better (case patterns are predictable)

### SP8192
"Softmax Partition 8192" — likely partitions the vocabulary into 8192 groups for:
- More efficient softmax over large vocab
- Or: embedding dimension tied to 8192

---

## 9. Progressive Context Growth

**Source:** PR #2014

### Schedule
```
TRAIN_SEQ_SCHEDULE=1024@0.100,2048@0.700,3072@1.000

Meaning:
  Steps 0 - 10%: sequence length = 1024
  Steps 10% - 70%: sequence length = 2048  
  Steps 70% - 100%: sequence length = 3072
```

### Why It Works
- Short sequences early: cheaper per-step, more steps in same time
- Long sequences late: better long-range context for final fine-tuning
- Net effect: more total gradient updates with better final context

### Our Version
We use fixed sequence length throughout. Progressive schedule could give us extra steps early.

---

## 10. Polar Express Newton-Schulz Muon

**Source:** Competition, part of PR #1855 baseline

### What It Is
An optimized Muon (Momentum Update with Orthogonality Normalization) optimizer:
- Standard Muon: iterative Newton-Schulz for matrix orthogonalization
- Polar Express: vectorized Newton-Schulz that runs on all layers simultaneously
- Less CUDA synchronization overhead

### Performance Impact
Our v9 already uses batched Muon (similar concept). The "Polar Express" variant may be faster due to better batching across layers.

---

## Summary: What to Implement for Round 2

| Technique | Estimated BPB gain | Complexity | Dependencies |
|-----------|-------------------|------------|--------------|
| TTT (fixed) | **0.012** | Medium | torch.compile fix |
| GPTQ 32 batches | **0.004** | Low | Just change constant |
| LQER rank-4 | ~0.004 | Medium | Post-quant step |
| SmearGate | ~0.002 | Low | 10 lines of code |
| AWQ-lite | ~0.003 | Medium | Pre-quant scaling |
| Progressive context | ~0.002 | Low | Schedule parameter |
| Per-group lrzip | ~0.001 | Low-Medium | apt-get lrzip |
| AsymLogit | ~0.002 | Low | 5 lines of code |
| **TOTAL** | **~0.030** | — | — |

Our current gap from SOTA is ~0.063 BPB. The above addresses ~0.030 of it. 
Remaining ~0.033 gap: CaseOps tokenizer, architecture differences (SP8192, parallel residuals).
