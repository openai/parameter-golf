# Parameter Golf — Battle Plan

> **Mission**: Train the best tiny language model that fits in 16 MB, in under 10 minutes on 8xH100s.
> **Our Edge**: Quantum-inspired tensor compression + classical dequantization tricks — a novel angle nobody on the leaderboard is using yet.

---

## The Arena

```
 Target       Current SOTA     Baseline
  ???    <--- 1.1194 bpb <--- 1.2244 bpb
  |              |                |
  |     LeakyReLU² + TTT         |
  |     + Parallel Muon     Naive 9-layer
  |                          512-dim GPT
  |
  Our goal: push below 1.11
```

| Constraint         | Limit                                      |
|--------------------|--------------------------------------------|
| Artifact size      | 16,000,000 bytes (model + code, compressed) |
| Training time      | 10 minutes on 8x H100 SXM                  |
| Eval time          | 10 minutes (separate)                       |
| Metric             | Bits-per-byte (BPB) — lower is better       |
| Statistical rigor  | 3+ seeds, p < 0.01, beat SOTA by 0.005 nats |
| Deadline           | April 30, 2026                               |

---

## Our Secret Weapons

### 1. Tensor Network Decomposition (MPS / Tucker)

The core insight: treat weight matrices as **entangled quantum systems** and decompose them into shared low-rank cores.

```
Traditional weight matrix W (d x d):
  ┌─────────────────────┐
  │                     │    d² parameters
  │     Full Matrix     │    Expensive. Redundant.
  │                     │
  └─────────────────────┘

Tensor Train / MPS decomposition:
  ┌───┐   ┌───┐   ┌───┐
  │ G1├───┤ G2├───┤ G3│    Shared low-rank cores
  └───┘   └───┘   └───┘    50-200x compression
                            Minimal accuracy loss
```

**Where to apply:**
- Attention Q/K/V projection matrices — highest parameter count
- FFN up/down projections — largest single layers
- Use `tensorly` library for seamless PyTorch integration

### 2. Dequantization Tricks (Tang / Kerenidis)

Ewin Tang (2018) showed that "quantum speedups" for ML are actually achievable classically with clever **sampling + sketching**:

```
Quantum approach:          Classical dequantized approach:
  |ψ⟩ = Σ αᵢ|i⟩    →    Sample rows/cols proportional to norm
  Superposition            Low-rank sketch of gradient matrices
  over matrix entries      Same asymptotic speedup, no qubits needed
```

**For us this means:**
- Faster gradient updates via importance sampling
- Better low-rank approximations inside attention
- Smarter weight initialization from SVD sketches

### 3. Aggressive Quantization Stack

```
Training (bfloat16)
      │
      ▼
Tensor decomposition (compress structure)
      │
      ▼
4-bit quantization (compress values)
      │
      ▼
Pruning (remove near-zero cores)
      │
      ▼
zlib/zstd compression (squeeze bits)
      │
      ▼
≤ 16 MB artifact ✓
```

---

## The 7-Step Execution Plan

```
Step    Task                          Dev GPU        Target
─────────────────────────────────────────────────────────────
 1      Baseline                      RTX 5070 Ti    ~1.22 bpb
        Run stock train_gpt.py on
        FineWeb, confirm baseline

 2      Tensorize                     RTX 5070 Ti    ~1.18 bpb
        MPS/Tucker on attn + FFN
        via tensorly, measure
        compression vs accuracy

 3      Smart Init                    RTX 5070 Ti    ~1.16 bpb
        He init + low-rank SVD
        seeds for faster convergence
        in limited training time

 4      Train Fast                    RTX 5070 Ti    iterate
        torch.compile + bfloat16
        + gradient accumulation
        Optimize on small shards

 5      Compress                      RTX 5070 Ti    fit 16MB
        4-bit quant + pruning +
        knowledge distillation
        from larger checkpoints

 6      Dequant Tricks                RTX 5070 Ti    ~1.12 bpb
        Tang-style sampling for
        gradients and attention
        weight approximations

 7      Submit                        8x H100        < 1.11 bpb
        3+ seeds, full FineWeb
        training, PR to OpenAI
─────────────────────────────────────────────────────────────
```

---

## Hardware Strategy

```
┌──────────────────────────────────┐
│  Development: RTX 5070 Ti        │
│  16 GB GDDR7                     │
│  ─────────────────────────       │
│  • Small batch + grad accum      │
│  • 1-2 FineWeb shards            │
│  • torch.compile + bf16          │
│  • Rapid iteration sandbox       │
│  • If it's fast here, it         │
│    FLIES on 8x H100              │
└──────────────┬───────────────────┘
               │ code transfers directly
               ▼
┌──────────────────────────────────┐
│  Competition: 8x H100 SXM       │
│  80 GB HBM3 each (640 GB total) │
│  ─────────────────────────       │
│  • Full 80-shard training        │
│  • DDP across 8 GPUs             │
│  • 15-20x throughput vs local    │
│  • Final submission runs here    │
└──────────────────────────────────┘
```

---

## What the Leaderboard Is Doing (and What They're NOT)

### Techniques already explored by top teams:
- LeakyReLU² activation
- Test-Time Training (TTT)
- Parallel Muon optimizer
- Int6/Int8 quantization + QAT
- Partial RoPE, EMA, extended warmdown
- 3x MLP expansion, bigram hash embeddings

### Our novel angle (not yet on the leaderboard):
- **Tensor network decomposition** — nobody is doing MPS/Tucker on weights
- **Dequantized gradient sampling** — Tang's tricks are unexplored here
- **Quantum-inspired initialization** — variational circuit-style SVD seeds

This is our competitive moat. The top teams are squeezing BPB with architecture tweaks and training recipes. We're attacking the **parameter representation itself**.

---

## Architecture Baseline (What We're Modifying)

```
Input tokens
     │
     ▼
┌─────────────┐
│  Embedding   │  vocab=1024, dim=512, tied with output
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Encoder x4  │  RMSNorm → GQA Attention (RoPE) → ReLU² FFN
└──────┬──────┘  + residual connections
       │
       │ skip connections
       ▼
┌─────────────┐
│  Decoder x5  │  Same blocks + skip from encoder
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   LM Head    │  → logits → softmax → BPB
└─────────────┘

Optimizer: Muon (matrix params) + Adam (embeddings, scalars)
Schedule: 20-step warmup → train → 1200-step cosine warmdown
```

---

## Key Files

| File | Role |
|------|------|
| `train_gpt.py` | Main training script — **this is what we modify** |
| `train_gpt_mlx.py` | Apple Silicon version (not for us) |
| `data/cached_challenge_fineweb.py` | Download FineWeb shards |
| `data/tokenizer_specs.json` | Tokenizer configs (sp1024 default) |
| `records/track_10min_16mb/` | Leaderboard — study these for ideas |

---

## Quick Start Commands

```bash
# 1. Download data (1 shard for dev, 80 for full)
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1

# 2. Run baseline (single GPU dev)
RUN_ID=baseline torchrun --standalone --nproc_per_node=1 train_gpt.py

# 3. Full competition run (8x H100)
RUN_ID=submission torchrun --standalone --nproc_per_node=8 train_gpt.py
```

---

*Last updated: March 26, 2026*
