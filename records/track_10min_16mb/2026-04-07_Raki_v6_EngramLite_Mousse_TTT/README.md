# Raki v6: EngramLite + Mousse + Progressive Depth Recurrence + Score-First TTT

**val_bpb = 1.1026 (SEED=1337) | 15.95 MB | 8×H100 SXM | 590s training + 382s eval**

> Single seed submission due to compute budget constraints. We respectfully request consideration.

---

*A personal note: Being part of this challenge meant everything. My fiancée Virginia and I were supposed to go on vacation — but I spent that budget on H100 runs instead. She still sits next to me at 3 AM saying "keep going." This score is for her.*

## Abstract

Building on our previous Raki v5 submission (1.1047 BPB), we introduce three new components that collectively push performance to **1.1026 BPB**: **EngramLite** (multi-head gated bigram+trigram hash replacing legacy BigramHash), **Mousse optimizer** (diagonal curvature-aware Muon preconditioning), and **Progressive Depth Recurrence** (phased activation of recurrence layers for training stability). We also explored LoRA-based TTT as an alternative to full-weight TTT but found full-weight adaptation marginally superior on our architecture.

## Results

| Stage | val_loss | val_bpb | Notes |
|-------|----------|---------|-------|
| Pre-quantization (EMA) | 1.9126 | 1.1328 | 5,667 steps, 590s wallclock |
| Post-quantization (int6 GPTQ, qmax=42) | 1.9250 | 1.1401 | Quant gap: 0.0073 |
| Sliding window (stride=64) | 1.8638 | 1.1038 | Full context scoring |
| **Score-first TTT (3 epochs)** | **1.8617** | **1.1026** | Legal backward-looking |
| Artifact size | — | — | 15,948,298 bytes (99.7% of 16 MB) |

## Delta from Raki v5 (1.1047 → 1.1026)

| Change | Impact | Notes |
|--------|--------|-------|
| BigramHash(1536) → EngramLite(3072, 2-head, bigram+trigram) | −0.003 | Multi-order n-gram hashing with sigmoid gating |
| Muon → Mousse (diagonal curvature EMA) | −0.002 | Kronecker-factored preconditioning before NS5 |
| Fixed recurrence (step 2000) → Progressive (1500→3000) | −0.001 | Phase 1: layers 4,5 at step 1500, Phase 2: full at step 3000 |
| Recurrence layers 3,4,5 → 4,5 | neutral | Fewer repeated layers, more training stability |
| LoRA TTT (rank-4 adapters) | +0.001 worse | Full-weight TTT still superior on this architecture |

## Experimental Log: LoRA TTT Investigation

We investigated LoRA-based TTT as a potential improvement over full-weight TTT, motivated by the hypothesis that depth recurrence creates weight-coupling that makes full-parameter updates suboptimal.

| TTT Variant | val_bpb | Notes |
|-------------|---------|-------|
| Full-weight AdamW, lr=0.01, 3ep, reset=0 | **1.1026** | Best result |
| Full-weight AdamW, lr=0.003, 5ep, reset=1 | 1.1033 | Per-chunk reset hurts |
| Full-weight SGD, lr=0.002, mom=0.9, reset=0 | 1.1058 | SGD worse on our architecture |
| Full-weight SGD, lr=0.002, freeze=2, reset=1 | 1.1027 | Marginal |
| LoRA rank-4 AdamW, lr=0.02, 3ep, reset=0 | 1.1033 | Doesn't beat full-weight |
| Freeze recurrence blocks (4,5) only | 1.1027 | No improvement |

**Finding:** Contrary to expectations from Issue #140 ("TTT fundamentally conflicts with depth recurrence"), full-weight AdamW TTT with birikimli (non-reset) adaptation remains optimal for our architecture. The recurrence conflict is mitigated by the per-block adaptive LR schedule and moderate learning rate.

## Contributions

### 1. EngramLite: Multi-Head Gated N-gram Hash
Replaces legacy BigramHash(1536, 128d) with a multi-order hashing scheme:
- 4 unrolled hash computations: bigram×2 + trigram×2 (no Python loops for torch.compile)
- Shared embedding table (3072 buckets, 112d)
- Sigmoid gate with learned bias (initialized at −1.0 for conservative start)
- Projected to vocab_size logits, added as residual

### 2. Mousse Optimizer: Curvature-Aware Muon
Extends Muon with diagonal-only Kronecker curvature estimation (O(rows+cols) storage):
```
L_diag = diag(G @ G^T),  R_diag = diag(G^T @ G)
G_preconditioned = G * L_diag^{-1/2} * R_diag^{-1/2}
```
Applied with EMA smoothing (β=0.95) before Newton-Schulz iteration. Combined with MuonEq-R row normalization.

### 3. Progressive Depth Recurrence
Instead of activating all recurrence layers at once:
- **Phase 1 (step 1500):** Layers 4,5 repeated — gentle introduction
- **Phase 2 (step 3000):** Full recurrence active
This avoids the training instability observed when recurrence activates abruptly.

### 4. Auto-QMax Artifact Packing (from Raki v5)
Binary search over qmax ∈ [31, 127], landing at qmax=42 for this run. Every unused byte in the 16MB budget is wasted precision.

### 5. Adaptive Markov Curriculum (from Raki v5)
Bigram-surprise-weighted loss scaling (RAKI_POWER=0.10), steering capacity toward tokens that statistical n-gram methods cannot predict.

## Architecture

| Component | Configuration |
|-----------|---------------|
| Transformer | 11 layers, 512d, 8 heads, 4 KV heads |
| MLP | 4× expansion, LeakyReLU(0.5)² activation |
| Depth Recurrence | Layers 4,5 repeated once (13 effective layers) |
| Progressive Recurrence | Phase 1 at step 1500, Phase 2 at step 3000 |
| Parallel Residuals | Dual-lane attention/MLP from layer 7, learned merge gate |
| XSA | All 11 layers (value-orthogonal projection) |
| Partial RoPE | 16 of 64 head dimensions |
| LN Scale | 1/√(layer_idx + 1) per-layer normalization |
| EngramLite | 3072 buckets, 112d, bigram+trigram, 2 heads, sigmoid gate |
| Value Embedding | 128d shared, applied at layers 9–10 |
| Skip Gates | Learned sigmoid gating on U-Net connections |
| Logit Softcap | 30.0 (tanh-based) |

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Mousse (matrices) + AdamW (scalars/embeddings) |
| Matrix LR | 0.025 |
| Weight Decay | 0.090 (Muon/embed), 0.02 (Adam) |
| Momentum | 0.99 (warmup 0.92→0.99 over 1,500 steps) |
| Batch Tokens | 786,432 |
| Sequence Length | 1,024 (SP1024 tokenizer) |
| Late QAT | Last 200 steps, int6 STE + dynamo reset |
| Warmdown | 66.7% cosine decay |
| EMA | 0.997 decay |

## Reproduce

```bash
pip install sentencepiece brotli
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10

VOCAB_SIZE=1024 TRAIN_SEQ_LEN=1024 EVAL_SEQ_LEN=1024 \
MUON_WD=0.090 EMBED_WD=0.090 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 EMA_DECAY=0.997 EVAL_STRIDE=64 \
RAKI_POWER=0.10 \
DTTT_ENABLED=0 TTT_ENABLED=1 TTT_LR=0.01 TTT_EPOCHS=3 \
TTT_CHUNK_TOKENS=32768 TTT_RESET_PER_CHUNK=0 \
ENGRAM_ENABLED=1 MOUSSE_ENABLED=1 \
CAUTIOUS_ENABLED=0 SDCLIP_ENABLED=0 \
HADAMARD_ENABLED=0 CATALYTIC_ENABLED=0 \
LATE_QAT=1 GPTQ_ENABLED=1 \
GPTQ_RESERVE_SECONDS=10 EMBED_BITS=8 EMBED_CLIP_SIGMAS=20.0 \
MAX_WALLCLOCK_SECONDS=600 ITERATIONS=20000 WARMUP_STEPS=20 \
VAL_LOSS_EVERY=4000 SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Credits

- PR #1339 (@bigbag), PR #1204 (@msisovic) — Depth Recurrence, Parallel Residuals
- PR #549 (@abaybektursun) — Score-first TTT framework, LeakyReLU²
- PR #1331, #1260 (@dexhunter) — MuonEq-R
- PR #287 (@jfprincz) — Partial RoPE, LN Scale
- PR #198 (@unnir), PR #374 (@signalrush) — XSA, EMA, GPTQ-lite
- PR #803 — Complementary Training (inspiration for Markov Curriculum)
- Mousse (arXiv:2603.09697) — Curvature-aware Muon preconditioning
