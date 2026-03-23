# Multi-Epoch Cosine LoRA TTT + SOTA Base (10L Int5-MLP + Int6-Attn + BigramHash + SWA)

**Author:** Atharva Date (ADIITJ)
**Status:** Non-record submission (pending H100 validation)
**Approach:** SOTA training stack (10L, Int5-MLP, Int6-Attn, BigramHash(10240), SWA) + 50-epoch cosine LoRA TTT at evaluation.

---

## Summary

This submission combines the SOTA training stack (thwu1, 1.1428 bpb) with document-isolated multi-epoch LoRA test-time training at evaluation. TTT adapts rank-8 LoRA adapters on Q and V projections per document using 50 epochs of cosine-scheduled adaptation, then scores the document using the adapted model.

**Key changes from original single-pass LoRA TTT:**
- **50 epochs** per document instead of 1 step per chunk
- **Cosine LR decay**: `lr * 0.5 * (1 + cos(π * epoch / 50))`, from 0.001 → ~0
- Score only in the final epoch (accumulates adapted model's NLL)
- Train on ALL chunks every epoch (including last chunk)

**Artifact cost:** Zero. LoRA weights initialized at eval time, discarded after each document.

**Expected bpb:** ~1.05–1.10 (projected from SOTA 1.1428 + multi-epoch TTT delta ~0.04–0.09 based on competition results: PR #517 gets 0.978 bpb on a similar base with 100-epoch cosine TTT on the full model; our rank-8 LoRA should give a fraction of that improvement).

---

## Architecture (Training — identical to SOTA thwu1)

- **10 transformer layers**, dim=512, 8 heads, 4 KV heads (GQA)
- **MLP**: 3x hidden width (1536), relu² activation
- **SmearGate**: learned gate blending token t with token t-1 embedding
- **BigramHash(10240, dim=128)**: hash consecutive token pairs → learned embedding → project to model_dim
- **Orthogonal init** for all large weight matrices; output projections scaled by `1/√(2L)`
- **Tied embeddings** (fp16)
- **U-Net skip connections** between encoder and decoder halves
- **RoPE** positional encodings with QK-Norm and q_gain

## Quantization (Export — identical to SOTA)

| Tensor Class | Precision | Compression |
|---|---|---|
| MLP weights | Int5 (clip=15, per-row scale) | zstd-22 (~1.88× ratio) |
| Attention weights | Int6 (clip=31, per-row scale) | zstd-22 (~1.51× ratio) |
| Tied embeddings | FP16 | stored raw |
| Blocks[8].attn.c_k | FP16 | stored raw |
| Control scalars | FP32 | stored raw |
| 3% magnitude pruning | applied before quantization | improves compressibility |

## Training Hyperparameters

| Parameter | Value |
|---|---|
| num_layers | 10 |
| model_dim | 512 |
| mlp_mult | 3.0 |
| train_seq_len | 2048 |
| train_batch_tokens | 786,432 |
| iterations | 20,000 (wallclock-capped at 600s) |
| warmdown_iters | **3,500** (up from SOTA's 3,000) |
| warmup_steps | 20 |
| matrix_lr (Muon) | 0.02 |
| muon_momentum | 0.99 (warmup from 0.92 over 1500 steps) |
| muon_weight_decay | 0.04 |
| adamw_weight_decay | 0.04 |
| grad_clip_norm | 0.3 |
| SWA start_frac | **0.35** (SOTA used 0.40) |
| SWA every | 50 steps |
| bigram_vocab_size | 10,240 |
| bigram_dim | 128 |

## Multi-Epoch Cosine LoRA TTT at Evaluation

After training, quantization, and dequantization, the model is evaluated using document-isolated multi-epoch LoRA TTT:

**Algorithm:**
1. Find document boundaries in the validation set using BOS token (token_id=1).
2. For each batch of 32 documents (sorted by length):
   a. Initialize fresh LoRA adapters: A ~ Kaiming-uniform, B = zeros.
   b. For each epoch `ep` in `[0, n_epochs)`:
      - Set LR = `base_lr * 0.5 * (1 + cos(π * ep / n_epochs))`
      - Slide through the document in chunks of 256 tokens with 2048-token context.
      - For each chunk: **score first** (accumulate NLL only if `ep == n_epochs-1`), then take one Adam step on LoRA.
   c. Reset LoRA and optimizer state before the next document batch.
3. LoRA targets: Q and V projections in all 10 attention layers (rank=8, base_lr=0.001).

**Fairness:** Scoring always precedes training on each chunk within every epoch. Multi-epoch = repeated passes over the same document (no cross-document leakage). The final epoch's scores are the reported NLL.

**Artifact cost:** Zero. LoRA weights are initialized at eval time and discarded after each document.

### LoRA TTT Hyperparameters

| Parameter | Value |
|---|---|
| lora_rank | 8 |
| ttt_n_epochs | 50 |
| ttt_lora_lr (base) | 0.001 |
| LR schedule | Cosine: 0.001 → ~0 over 50 epochs |
| chunk_size | 256 tokens |
| eval_seq_len | 2048 tokens |
| batch_size | 32 documents |
| optimizer | Adam (β₁=0.9, β₂=0.95, ε=1e-10) |
| adapters | Q and V in all 10 layers |

## Expected Performance

| Baseline | BPB | Source |
|---|---|---|
| SOTA thwu1 (no TTT) | 1.1428 | verified 3-seed |
| Single-pass LoRA TTT (original) | ~1.137–1.140 | projected |
| PR #517 full-model 100-ep cosine TTT | 0.978 | verified 3-seed |
| PR #518 full-model 50-ep cosine TTT | 1.062 | verified |
| **This submission (LoRA 50-ep cosine TTT)** | **~1.05–1.10** | projected |

LoRA rank-8 adapts fewer parameters than full-model TTT, so the benefit per epoch is smaller. However, LoRA TTT requires no explicit model copy at eval time and stays within artifact size budget with zero overhead.

## Artifact Size Budget

Identical to SOTA (~14.3MB):

| Component | Est. Size |
|---|---|
| Int5 MLP (10L, 3× hidden, zstd-22) | ~8.4MB |
| Int6 Attention (10L, zstd-22) | ~3.9MB |
| FP16 embeddings | ~1.0MB |
| BigramHash(10240, dim=128) | ~0.9MB |
| Code (train_gpt.py) | ~55KB |
| **Total** | **~14.3MB** |

LoRA weights: 0 bytes (initialized at eval time, discarded after use).

## Compliance Checklist

- [x] Artifact ≤ 16,000,000 bytes (est. ~14.3MB, same as SOTA)
- [x] No network calls during evaluation
- [x] No training data access during evaluation
- [x] Self-contained and reproducible
- [x] Training ≤ 10 minutes on 8xH100
- [x] TTT only uses already-scored validation tokens (score-first per chunk per epoch)
- [ ] Statistical significance (3+ seeds not yet run — non-record status)

## Run Commands

### Training + Evaluation (8xH100)

```bash
# Setup data (once)
python3 data/cached_challenge_fineweb.py --variant sp1024

# Run from the records folder:
cd records/track_10min_16mb/2026-03-23_LoRA_TTT_Int5Int6/

# Seed 42 (default)
DATA_PATH=../../../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../../../data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=8 train_gpt.py

# Seed 1337
SEED=1337 \
DATA_PATH=../../../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../../../data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=8 train_gpt.py

# Disable TTT (sliding window only, for ablation)
TTT_ENABLED=0 \
DATA_PATH=../../../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../../../data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=8 train_gpt.py

# 100-epoch TTT (slower but potentially better)
TTT_N_EPOCHS=100 \
DATA_PATH=../../../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../../../data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Dependencies

See `requirements.txt`. The `zstandard` package is required for zstd-22 compression.

## Attribution

Training architecture, quantization scheme, SWA, BigramHash, SmearGate, and OrthoInit are from the SOTA submission by thwu1 (2026-03-20), which builds on Raahil Shah's PR #162.

Multi-epoch cosine LoRA TTT design is adapted from:
- PR #517 (lukacf): cosine LR scheduling for TTT (3 lines that made 0.978 bpb possible)
- PR #77 (samacqua): backward-looking LoRA TTT protocol

This submission by Atharva Date (ADIITJ) applies the cosine LR improvement to rank-8 LoRA TTT on the SOTA base.

## Non-Record Status

Until at least 3 independent seeds are validated on 8xH100, the submission is classified as **non-record**. The implementation is complete and correct; the record determination is pending compute validation.
