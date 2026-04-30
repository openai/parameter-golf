## Record: 10L LeakyReLU + Gated Attention + Value Residual (val_bpb: 1.1454)

**val_bpb: 1.1454 ± 0.0011** (sliding window stride=64, 3-seed mean) | **15.65 MB** (mean) | 8xH100 SXM, 600s

### Key Innovations Over PR #583

Three orthogonal improvements stacked on top of PR #583's 10-layer baseline. Each was validated individually before combining; only changes that produced ≥0.001 BPB improvement on a single 600s seed were kept.

| Change | PR #583 | This | Solo Δ BPB | Stacked Δ BPB |
|--------|---------|------|-----------:|--------------:|
| **MLP activation** | `relu(x).square()` | `leaky_relu(x, 0.5).square()` | -0.0024 | (component) |
| **Attention output gate** | None | `out * sigmoid(gate)`, gate ∈ ℝ^(1,1,d), bias init +2 | -0.0010 | (component) |
| **Value Residual Learning (VRL)** | None | layer i>0 blends `v = α·v + (1-α)·v_first`, α learnable scalar init 0.9 | -0.0010 | (component) |
| **Total (stacked)** | **1.1489** | **1.1454** | — | **-0.0035 BPB** |

### LeakyReLU(0.5)² Activation

The MLP uses `(0.5x if x<0 else x)²` instead of `(max(0,x))²`. Both halves of the activation pass non-zero gradients, so dead-neuron fraction drops without changing the squared-output shape. We measured -0.0024 BPB at 600s with no other changes, no parameter cost, and no step-time regression.

### Gated Attention (GA)

After the attention output projection, we element-wise multiply by `sigmoid(gate)` where `gate` is a learnable parameter of shape `(1, 1, model_dim)` initialized at +2 (so `sigmoid(2) ≈ 0.88` at step 0 — close to identity). The gate is excluded from Muon (treated as a control tensor) and never quantized. Cost: 11 × 512 = 5,632 fp32 control params (~22 KB). Solo improvement: -0.0010 BPB.

### Value Residual Learning (VRL)

Inspired by the value-residual stream from recent transformer literature: the first attention block stores its V tensor (`v_first`), and every subsequent block blends its own V with `v_first` via a learnable scalar `α` (init 0.9):

```
v_block_i = α · v_current + (1 - α) · v_first   for i > 0
```

The blending happens *before* the GQA repeat-interleave so `v_first` remains shape `(B, num_kv_heads, T, head_dim)`. `α` is a per-block scalar (10 scalars total) that goes to the AdamW scalar group with weight decay. Solo improvement: -0.0010 BPB.

### Methodology: 9-Feature Ablation Study

Rather than stacking everything at once (which previously caused us to ship a worse 1.1524 with feature interactions), we tested **each candidate feature individually for one 600s seed against a clean reproduction of PR #583**, then kept only the winners:

| Feature | rt_bpb | vs Control 1.1482 | Decision |
|---------|-------:|------------------:|----------|
| (control reproduction) | 1.1482 | — | matches PR #583 |
| LeakyReLU(0.5)² | **1.1458** | **-0.0024** | **KEEP** |
| Gated Attention | **1.1472** | **-0.0010** | **KEEP** |
| Partial RoPE 16/64 | 1.1489 | +0.0007 | DROP |
| GPTQ-lite clip search | 1.1489 | +0.0007 | DROP |
| Cross-Sequence Attention (last 3) | 1.1537 | +0.0055 | DROP |
| **Value Residual Learning** | **1.1472** | **-0.0010** | **KEEP** |
| 11 layers | 1.1431 | -0.0051 | DROP (×6 size retries, all over 16 MB cap) |
| EMA + LATE_QAT_FRAC=0.5 | 1.1949 | +0.0467 | DROP (EMA shadow weights post-QAT eval poorly) |

The 11-layer experiment is informative: it produces the lowest BPB of any single change but cannot fit under the 16 MB cap with `MLP_MULT≥3.0` and a learned bigram, even after dropping bigram_vocab to 2048 and bigram_dim to 64 (16,329,703 bytes — 330 KB over). We chose not to compromise MLP capacity to land it.

### Results (3 seeds, 8xH100 SXM, 600s, sliding eval stride=64)

| Seed | Steps | val_loss | Sliding BPB | Artifact |
|------|------:|---------:|------------:|---------:|
| **1337** | 5453 | 1.9320 | **1.1442** | 15.63 MB |
| 42 | 5452 | 1.9336 | 1.1452 | 15.68 MB |
| 7 | 5441 | 1.9364 | 1.1468 | 15.64 MB |

**Mean: 1.1454 | Std: 0.0011** | Submitted: seed 1337 (best)

### Architecture (from PR #583)

- 10 transformer layers, 512-dim, 8 heads (4 KV heads, GQA)
- 3.0× MLP expansion (1536 hidden), `leaky_relu(0.5)²` activation
- U-Net skip connections (5 encoder, 5 decoder)
- Full RoPE (head_dim=64), base=50000
- Gated Attention output gate (sigmoid, bias init +2)
- Value Residual Learning (α init 0.9, layers 1–9 blend with layer-0 V)
- LN Scale Factor per RMSNorm
- SmearGate + BigramHash (4096 buckets, dim=128)
- Tied embeddings, logit softcap=30.0

### Training

- Muon optimizer (matrices): lr=0.035, momentum=0.99 (warmup 0.92→0.99 over 1500 steps), WD=0.045
- AdamW (embeddings, lm_head, scalars): tied embed lr=0.045, scalar lr=0.035, β=(0.9, 0.95), eps=1e-8, WD=0.01
- 786,432 token batches, seq_len=2048, grad clip 0.35, warmdown 2000 steps
- Late QAT (int6 STE) from step 0; SWA off; EMA off
- Compile: `torch.compile(dynamic=False, fullgraph=True)` + DDP
- 8 H100 80 GB SXM, ~107–110 ms/step, ~5450 steps in 600 s wallclock

### Quantization & Submission

- MLP weights → int5 per-row (signed 5-bit, fp16 scales)
- Attention weights → int6 per-row (signed 6-bit, fp16 scales)
- Token embedding → fp16 passthrough
- Control tensors (scales, gates, residual mixers, q_gain, smear, ln_scale, bigram.scale, attn_gate, v_alpha) → fp32 passthrough
- zstd level 22 compression on the quantized state dict
- Final submission: 15.59–15.68 MB across seeds (≥316 KB headroom under cap)

### Reproduction

```
RUN_ID=v4 SEED=1337 MAX_WALLCLOCK_SECONDS=600 \
  NUM_LAYERS=10 MLP_MULT=3.0 BIGRAM_VOCAB_SIZE=4096 BIGRAM_DIM=128 \
  EMA_ENABLED=0 SWA_ENABLED=0 ROPE_DIMS=0 LATE_QAT_FRAC=0.0 INT5_MLP=1 \
  LEAKY_RELU=1 GATED_ATTN=1 GPTQ_CLIP_SEARCH=0 VRL_ENABLED=1 \
  MATRIX_LR=0.035 TIED_EMBED_LR=0.045 SCALAR_LR=0.035 \
  MUON_WEIGHT_DECAY=0.045 WARMDOWN_ITERS=2000 GRAD_CLIP_NORM=0.35 \
  PASSTHROUGH_MAX_NUMEL=16384 SLIDING_EVAL=1 EVAL_STRIDE=64 EVAL_BATCH_SEQS=32 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### Author

- **suchihype** (primary)
- Co-authored with **Claude Opus 4.7** (1M context) and **OpenAI Codex** (Codex CLI)
